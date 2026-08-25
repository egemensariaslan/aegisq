"""The dashboard: experiment payloads, JSON safety, routing and asset serving.

The panels are only as trustworthy as the numbers behind them, so these tests
check the *shape and meaning* of what each experiment returns -- that the
noiseless reference really is noiseless, that the retention metric is
scale-free, that nothing non-finite escapes into JSON -- rather than merely that
an endpoint answers 200.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from aegisq.ui import experiments
from aegisq.ui.server import ASSETS, JSON_ROUTES, build_handler, json_safe, serve


def get(url):
    """Fetch a URL, returning the response object even for error statuses."""
    try:
        return urlopen(url)
    except HTTPError as error:
        return error


# ----------------------------------------------------------------------
class TestJSONSafety:
    def test_non_finite_floats_become_null(self):
        assert json_safe(float("nan")) is None
        assert json_safe(float("inf")) is None
        assert json_safe(float("-inf")) is None

    def test_finite_values_pass_through(self):
        assert json_safe(1.5) == 1.5
        assert json_safe(0) == 0
        assert json_safe("text") == "text"

    def test_nested_structures_are_cleaned(self):
        payload = {"a": [1.0, float("nan")], "b": {"c": float("inf")}}
        cleaned = json_safe(payload)
        assert cleaned == {"a": [1.0, None], "b": {"c": None}}
        assert "NaN" not in json.dumps(cleaned)

    def test_every_experiment_survives_strict_json(self):
        """json.dumps emits bare NaN happily; JSON.parse in a browser does not."""
        for payload in (
            experiments.catalog(4),
            experiments.zne_curve(n_qubits=3, n_layers=2),
            experiments.symmetry_scan(n_qubits=4, strengths=[0.01]),
            experiments.noise_sweep(n_qubits=3, n_layers=1, strengths=[0.0, 0.01],
                                    ansatze=["local_entangler"]),
        ):
            text = json.dumps(json_safe(payload), allow_nan=False)
            assert "NaN" not in text and "Infinity" not in text


# ----------------------------------------------------------------------
class TestExperiments:
    def test_catalog_lists_each_ansatz_once(self):
        data = experiments.catalog(6)
        names = [row["name"] for row in data["rows"]]
        assert len(names) == len(set(names)), "aliases must not produce duplicate rows"
        assert {"local_entangler", "strongly_entangling"} <= set(names)
        assert all(row["role"] in ("aegisq", "baseline") for row in data["rows"])

    def test_zne_panel_reports_a_real_ground_truth(self):
        data = experiments.zne_curve(n_qubits=4, n_layers=3, noise=0.01, trials=6)
        assert abs(data["truth"]) <= 1.0, "an expectation value cannot exceed 1"
        assert len(data["points"]) == len(data["params"]["scale_factors"])
        assert data["trials"] == 6
        # raw_error is a mean of per-trial |measured - truth|, so by the triangle
        # inequality it can only be >= the gap between the trial-averaged values.
        assert data["raw_error"] >= abs(data["points"][0]["y"] - data["truth"]) - 1e-9
        assert data["raw_error_std"] >= 0
        for fit in data["fits"]:
            assert fit["residual"] == pytest.approx(abs(fit["estimate"] - data["truth"]))
            assert fit["estimate_std"] >= 0
            assert 0.0 <= fit["coverage"] <= 1.0

    def test_zne_single_trial_matches_the_exact_error_identity(self):
        """With one trial there is no averaging, so the old exact identity holds."""
        data = experiments.zne_curve(n_qubits=4, n_layers=3, noise=0.01, trials=1)
        assert data["trials"] == 1
        assert data["truth_std"] == 0.0
        assert data["raw_error"] == pytest.approx(abs(data["points"][0]["y"] - data["truth"]))

    def test_zne_measurements_decay_as_noise_is_amplified(self):
        data = experiments.zne_curve(n_qubits=4, n_layers=3, noise=0.02,
                                     scale_factors=(1, 2, 3))
        magnitudes = [abs(p["y"]) for p in data["points"]]
        assert magnitudes[0] > magnitudes[-1], "folding must amplify the noise"

    def test_zne_recovers_bias_in_the_regime_it_is_meant_for(self):
        data = experiments.zne_curve(n_qubits=4, n_layers=3, noise=0.01,
                                     scale_factors=(1, 2, 3))
        best = max(fit["bias_reduction"] for fit in data["fits"])
        assert best > 0.5

    def test_richardson_variance_cost_grows_with_more_scale_factors(self):
        """The reason the default is three points and not five."""
        few = experiments.zne_curve(n_layers=2, scale_factors=(1, 2))
        many = experiments.zne_curve(n_layers=2, scale_factors=(1, 1.5, 2, 2.5, 3))
        cost = lambda d: next(f["variance_cost"] for f in d["fits"] if f["key"] == "richardson")  # noqa: E731
        assert cost(many) > 10 * cost(few)

    def test_plateau_series_carry_a_decay_rate(self):
        data = experiments.plateau_scan(qubit_counts=(4, 6), n_layers=2, n_samples=5,
                                        ansatze=["local_entangler", "equivariant"])
        assert len(data["series"]) == 2
        for series in data["series"]:
            assert len(series["points"]) == 2
            assert all(point["y"] >= 0 for point in series["points"])
            assert math.isfinite(series["per_qubit_factor"])

    def test_noise_sweep_metric_is_scale_free(self):
        """Retention must start at 1 and fall, whatever the output magnitudes."""
        data = experiments.noise_sweep(n_qubits=4, n_layers=2,
                                       strengths=[0.0, 0.01, 0.05],
                                       ansatze=["local_entangler", "strongly_entangling"],
                                       with_zne=False)
        for series in data["series"]:
            values = [point["y"] for point in series["points"]]
            assert values[0] == pytest.approx(1.0), "zero noise must retain the whole signal"
            assert values[-1] < values[0], "retention must fall as noise rises"
            assert all(v <= 1.05 for v in values)

    def test_noise_sweep_includes_a_mitigated_series(self):
        data = experiments.noise_sweep(n_qubits=3, n_layers=2, strengths=[0.0, 0.02],
                                       ansatze=["local_entangler"], with_zne=True)
        assert any(series["mitigated"] for series in data["series"])

    def test_noise_sweep_reports_trial_spread(self):
        data = experiments.noise_sweep(n_qubits=3, n_layers=2, strengths=[0.0, 0.02],
                                       ansatze=["local_entangler"], with_zne=False, trials=4)
        assert data["trials"] == 4
        series = data["series"][0]
        assert series["points"][0]["err"] == 0.0, "no noise, nothing to vary across trials"
        assert series["points"][1]["err"] >= 0.0
        assert all("err" in point for point in series["points"])

    def test_symmetry_panel_reports_zero_noiseless_leakage(self):
        data = experiments.symmetry_scan(n_qubits=4, n_layers=2, strengths=[0.01, 0.03])
        assert data["symmetry"] == "particle_number"
        assert data["noiseless_leakage"] < 1e-9
        assert data["sector_size"] < data["space_size"]
        for row in data["rows"]:
            assert 0.0 <= row["accepted"] <= 1.0
            assert row["leakage"] + row["accepted"] == pytest.approx(1.0, abs=1e-6)

    def test_symmetry_leakage_grows_with_noise(self):
        data = experiments.symmetry_scan(n_qubits=4, n_layers=2, strengths=[0.002, 0.05])
        assert data["rows"][0]["leakage"] < data["rows"][-1]["leakage"]

    def test_training_stream_emits_start_epochs_and_done(self):
        events = list(experiments.training_stream(
            n_qubits=3, n_layers=1, epochs=2, noise_name="depolarizing",
            models=["local_entangler"], samples=40,
        ))
        assert events[0]["event"] == "start"
        assert events[-1]["event"] == "done"
        epochs = [e for e in events if e["event"] == "epoch"]
        assert [e["epoch"] for e in epochs] == [0, 1, 2], "epoch 0 is the untrained baseline"
        for record in epochs:
            assert 0.0 <= record["test_accuracy"] <= 1.0

    def test_every_panel_names_the_command_that_reproduces_it(self):
        for payload in (
            experiments.zne_curve(n_qubits=3, n_layers=1),
            experiments.plateau_scan(qubit_counts=(4, 6), n_samples=3),
            experiments.symmetry_scan(n_qubits=4, strengths=[0.01]),
        ):
            assert payload["command"].startswith("python3 run.py")
            assert payload["params"]


# ----------------------------------------------------------------------
class TestAssets:
    def test_every_referenced_asset_exists(self):
        index = (ASSETS / "index.html").read_text(encoding="utf-8")
        for name in ("app.css", "app.js"):
            assert name in index, f"index.html should reference {name}"
        # charts.js is reached as an ES module import from app.js, not from the page.
        assert "charts.js" in (ASSETS / "app.js").read_text(encoding="utf-8")
        for name in ("app.css", "app.js", "charts.js", "index.html"):
            assert (ASSETS / name).is_file()

    def test_the_page_makes_no_external_requests(self):
        """A CDN reference would break the offline promise and the CSP."""
        for name in ("index.html", "app.js", "charts.js", "app.css"):
            text = (ASSETS / name).read_text(encoding="utf-8")
            for marker in ("http://", "https://", "//cdn.", "unpkg", "googleapis"):
                if marker in ("http://", "https://"):
                    # URLs are allowed inside comments and the SVG namespace only.
                    offenders = [
                        line for line in text.splitlines()
                        if marker in line
                        and "www.w3.org" not in line
                        and not line.strip().startswith(("*", "/*", "//", "#"))
                    ]
                    assert not offenders, f"{name} reaches outside: {offenders[:2]}"
                else:
                    assert marker not in text


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(quiet=True))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class TestServer:
    def test_index_is_served(self, live_server):
        with urlopen(f"{live_server}/") as response:
            body = response.read().decode("utf-8")
        assert response.status == 200
        assert "AegisQ" in body

    @pytest.mark.parametrize("name", ["app.css", "app.js", "charts.js"])
    def test_assets_are_served_with_a_sane_type(self, live_server, name):
        with urlopen(f"{live_server}/assets/{name}") as response:
            assert response.status == 200
            assert response.headers["Content-Type"].split("/")[0] in ("text", "application")

    def test_directory_traversal_is_refused(self, live_server):
        response = get(f"{live_server}/assets/../../../etc/passwd")
        assert response.status == 404
        response.close()

    def test_environment_endpoint_reports_versions(self, live_server):
        with urlopen(f"{live_server}/api/env") as response:
            payload = json.load(response)
        assert {"aegisq", "pennylane", "torch", "python"} <= set(payload)

    def test_cheap_endpoints_answer(self, live_server):
        for route in ("/api/catalog?qubits=4", "/api/zne?qubits=3&layers=1"):
            with urlopen(f"{live_server}{route}") as response:
                assert response.status == 200
                json.load(response)

    def test_unknown_routes_return_json_404(self, live_server):
        response = get(f"{live_server}/api/nope")
        assert response.status == 404
        assert "error" in json.load(response)
        response.close()

    def test_responses_carry_a_content_security_policy(self, live_server):
        with urlopen(f"{live_server}/") as response:
            policy = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in policy

    def test_every_json_route_is_reachable(self, live_server):
        for route in JSON_ROUTES:
            with urlopen(f"{live_server}{route}?qubits=3&layers=1&samples=3") as response:
                assert response.status == 200, route


class TestBinding:
    def test_refuses_a_non_loopback_host(self):
        """The endpoints run simulations on request; that must stay local."""
        with pytest.raises(ValueError, match="loopback"):
            serve(host="0.0.0.0", open_browser=False)


class TestDesignSystem:
    """Each named style must redefine the whole token contract.

    A style that omits a token silently inherits the previous theme's value.
    That is precisely how the data series on the flagship chart ended up drawn
    in dark-theme ink on a light background -- invisible, and invisible in a
    way no single screenshot reveals.
    """

    STYLES = ("instrument", "paper", "bento", "terminal")

    @staticmethod
    def _blocks():
        import re

        css = (ASSETS / "app.css").read_text(encoding="utf-8")
        found = {}
        for name in TestDesignSystem.STYLES:
            match = re.search(r'\[data-style="%s"\] \{(.*?)\n\}' % name, css, re.S)
            assert match, f"no token block for style {name!r}"
            found[name] = dict(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", match.group(1)))
        return found

    def test_every_style_defines_the_same_tokens(self):
        blocks = self._blocks()
        contract = set(blocks["instrument"])
        for name, tokens in blocks.items():
            assert set(tokens) == contract, (
                f"style {name!r} differs by {set(tokens) ^ contract}"
            )

    def test_every_style_defines_a_full_series_palette(self):
        for name, tokens in self._blocks().items():
            for index in range(1, 7):
                assert f"--c{index}" in tokens, f"{name} is missing --c{index}"
            assert "--ink-line" in tokens, f"{name} must define the data ink colour"

    def test_series_colours_are_distinct_within_a_style(self):
        """Two series sharing a colour is a legibility bug, not a grouping."""
        for name, tokens in self._blocks().items():
            series = [tokens[f"--c{i}"].strip().lower() for i in range(1, 7)]
            assert len(set(series)) == len(series), f"{name} repeats a series colour"

    def test_data_ink_contrasts_with_the_background(self):
        """The measured-data line must never be drawn in the background colour."""
        for name, tokens in self._blocks().items():
            assert tokens["--ink-line"].strip().lower() != tokens["--bg"].strip().lower()

    def test_the_switcher_offers_every_style(self):
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        for name in self.STYLES:
            assert f'id: "{name}"' in app, f"{name} is not offered in the switcher"

    def test_charts_read_the_palette_from_tokens(self):
        charts = (ASSETS / "charts.js").read_text(encoding="utf-8")
        assert "--c${i}" in charts or "--c${index}" in charts, (
            "series colours must come from the active style's tokens"
        )

    def test_panels_register_a_redraw_for_style_changes(self):
        """Charts are drawn, not styled: switching themes must repaint them."""
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert app.count("registerRedraw(") >= 6

    def test_color_scheme_is_set_explicitly_per_style(self):
        """Chromium gives :root's color-scheme priority over a later,
        equal-specificity attribute-selector rule regardless of source order --
        confirmed empirically, not merely suspected -- so the stylesheet alone
        cannot be trusted to pick light vs dark per style. app.js must set it
        via an inline style, which is unambiguous, on every style switch.
        """
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "documentElement.style.colorScheme" in app
        assert '"instrument"' in app and '"terminal"' in app  # the dark set

    def test_every_style_declares_a_css_color_scheme_fallback(self):
        """A CSS-level fallback for the no-JS case, even though JS is authoritative."""
        css = (ASSETS / "app.css").read_text(encoding="utf-8")
        assert css.count("color-scheme:") >= 4

    # -- WCAG contrast -------------------------------------------------
    @staticmethod
    def _relative_luminance(hex_colour: str) -> float:
        hex_colour = hex_colour.strip().lstrip("#")
        if len(hex_colour) == 3:
            hex_colour = "".join(c * 2 for c in hex_colour)
        r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (0, 2, 4))

        def linearise(channel: float) -> float:
            return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

        r, g, b = linearise(r), linearise(g), linearise(b)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @classmethod
    def _contrast_ratio(cls, a: str, b: str) -> float:
        """WCAG 2.1 contrast ratio (1:1 to 21:1) between two hex colours."""
        la, lb = sorted([cls._relative_luminance(a), cls._relative_luminance(b)], reverse=True)
        return (la + 0.05) / (lb + 0.05)

    def test_semantic_colours_meet_contrast_aa(self):
        """WCAG 2.1 AA requires 4.5:1 for normal text.

        ``--good``/``--warn``/``--bad`` carry the pass/fail verdicts in every
        stat card (e.g. "+58.3% of error removed"), so a reader who cannot
        resolve them loses the panel's headline number, not just a decoration.
        This check is why Bento's original semantic palette (2.5-3.0:1 against
        its own card background) got darkened -- it looked fine on a
        well-lit monitor and failed outright for a low-vision reader.
        """
        blocks = self._blocks()
        failures = []
        for style, tokens in blocks.items():
            bg = tokens["--bg"].strip()
            surface2 = tokens["--surface-2"].strip()
            for name in ("--good", "--warn", "--bad", "--text", "--muted"):
                colour = tokens[name].strip()
                for background, bg_label in ((bg, "bg"), (surface2, "surface-2")):
                    ratio = self._contrast_ratio(colour, background)
                    if ratio < 4.5:
                        failures.append(f"{style}.{name} on --{bg_label}: {ratio:.2f}:1")
        assert not failures, "below WCAG AA (4.5:1):\n" + "\n".join(failures)

    def test_accent_meets_contrast_aa_large_text_minimum(self):
        """--accent labels headings and links; large/UI-scale text needs 3:1 minimum."""
        blocks = self._blocks()
        for style, tokens in blocks.items():
            ratio = self._contrast_ratio(tokens["--accent"].strip(), tokens["--bg"].strip())
            assert ratio >= 3.0, f"{style}.--accent on --bg is only {ratio:.2f}:1"


class TestEditorialChrome:
    """Guards against regressing back to the decorative chrome this dashboard
    deliberately does not have: numbered section badges, a full-width command
    block under every chart, and a coloured-dot theme switcher. These were
    all removed in one pass after review, and nothing stops a later edit from
    reintroducing one piecemeal.
    """

    def test_headings_carry_no_numeric_index_prefix(self):
        index_html = (ASSETS / "index.html").read_text(encoding="utf-8")
        assert 'class="index"' not in index_html

    def test_no_full_width_reproduce_block_under_charts(self):
        index_html = (ASSETS / "index.html").read_text(encoding="utf-8")
        assert 'class="repro"' not in index_html
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "function repro(" not in app

    def test_every_experiment_panel_has_a_copy_command_button(self):
        index_html = (ASSETS / "index.html").read_text(encoding="utf-8")
        for panel_id in ("zne", "plateau", "sweep", "symmetry", "train"):
            assert f'id="{panel_id}-copy"' in index_html, f"{panel_id} has no copy button"
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert app.count("copyButton(") >= 5

    def test_comma_separated_list_fields_use_chip_controls(self):
        """Register widths and noise-strength lists are edited as chips, not
        typed as raw CSV into a text box."""
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert 'type: "chips"' in app
        assert app.count('type: "chips"') == 3  # widths, sweep strengths, symmetry strengths

    def test_style_switcher_has_no_colour_swatches(self):
        """A segmented control distinguishes the active style by position and
        a filled pill, not by a separate coloured dot per option."""
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "chip-dot" not in app
        css = (ASSETS / "app.css").read_text(encoding="utf-8")
        assert ".chip-dot" not in css

    def test_no_blocking_native_dialogs(self):
        """window.alert/confirm/prompt block the whole page until dismissed --
        a failed clipboard write must degrade silently (the command stays
        available on the button's title tooltip), not pop a modal.
        """
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        for blocking in ("window.alert(", "window.confirm(", "window.prompt("):
            assert blocking not in app, f"{blocking} blocks the page until dismissed"

    def test_style_is_deep_linkable_via_url(self):
        """?style=X should be resolvable at load time (e.g. for a screenshot
        or a shared link to a specific look) without requiring a click."""
        app = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "URLSearchParams(location.search)" in app
        assert '.get("style")' in app

    def test_wide_tables_scroll_within_their_own_box(self):
        """A table with many numeric columns can be wider than a phone
        viewport; it must not force the whole page horizontally scrollable.
        Found via a real headless-Chrome layout probe, not by inspection --
        the table's own scrollWidth (761px) was making the entire page wider
        than a 390px viewport.
        """
        index_html = (ASSETS / "index.html").read_text(encoding="utf-8")
        for table_id in ("plateau-table", "catalog-table"):
            assert f'<div class="table-scroll"><table id="{table_id}">' in index_html
        css = (ASSETS / "app.css").read_text(encoding="utf-8")
        assert ".table-scroll" in css and "overflow-x: auto" in css
