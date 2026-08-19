"""The zero-prerequisite launcher at the repository root.

``run.py`` is what a new user types first, so its logic is worth testing even
though its slow paths (downloading uv, building an environment) cannot run in a
unit suite.  Everything here is offline: the network-touching branches are
checked through their decision logic, not by exercising them.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    """Import run.py by path -- it lives outside the package on purpose."""
    spec = importlib.util.spec_from_file_location("aegisq_run_launcher", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


class TestPackaging:
    def test_launcher_sits_at_the_repository_root(self):
        assert (ROOT / "run.py").is_file()
        assert launcher.ROOT == ROOT

    def test_importing_it_does_not_run_anything(self):
        """The __main__ guard must hold, or importing would start a bootstrap."""
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source

    def test_it_is_not_shipped_inside_the_package(self):
        assert not (ROOT / "src" / "aegisq" / "run.py").exists()


class TestInterpreterSupport:
    def test_supported_versions_match_the_project_metadata(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'requires-python = ">=3.10"' in pyproject
        assert launcher.SUPPORTED[0] == (3, 10)

    def test_the_interpreter_running_the_tests_is_supported(self):
        assert launcher._interpreter_is_supported() is True
        assert sys.version_info[:2] in launcher.SUPPORTED

    def test_preferred_version_is_one_we_support(self):
        major, minor = (int(part) for part in launcher.PREFERRED.split("."))
        assert (major, minor) in launcher.SUPPORTED


class TestReleaseAssets:
    def test_this_platform_has_a_prebuilt_uv(self):
        key = (sys.platform, platform.machine().lower())
        assert key in launcher._UV_ASSETS, f"no uv asset mapped for {key}"

    @pytest.mark.parametrize(
        "key",
        [("darwin", "arm64"), ("darwin", "x86_64"),
         ("linux", "x86_64"), ("linux", "aarch64"), ("win32", "amd64")],
    )
    def test_the_common_platforms_are_covered(self, key):
        assert key in launcher._UV_ASSETS

    def test_archive_extensions_match_the_platform(self):
        for (system, _), asset in launcher._UV_ASSETS.items():
            expected = ".zip" if system == "win32" else ".tar.gz"
            assert asset.endswith(expected), f"{asset} is wrong for {system}"

    def test_release_url_points_at_the_official_project(self):
        assert launcher._UV_RELEASE.startswith("https://github.com/astral-sh/uv/releases/")


class TestStrategies:
    def test_finding_uv_is_side_effect_free(self):
        found = launcher._find_uv()
        assert found is None or Path(found).exists()
        assert not launcher.BOOTSTRAP_VENV.exists(), "probing must not create anything"

    def test_current_interpreter_strategy_handles_the_installed_case(self):
        """The suite runs where aegisq is importable, so this is the fast path."""
        with pytest.raises(SystemExit) as exit_info:
            launcher._try_current_interpreter(["--version"])
        assert exit_info.value.code == 0

    def test_pip_fallback_declines_an_unsupported_interpreter(self, monkeypatch):
        monkeypatch.setattr(launcher, "_interpreter_is_supported", lambda: False)
        assert launcher._try_pip([]) is None

    def test_strategies_are_ordered_cheapest_first(self):
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        order = source.index("_try_current_interpreter, _try_project_venv, _try_uv, _try_pip")
        assert order > 0, "main() should try an existing install before building one"

    def test_failure_message_names_the_supported_versions(self, capsys):
        assert launcher._report_failure() == 1
        message = capsys.readouterr().err
        assert "3.10, 3.11, 3.12, 3.13" in message
        assert "venv" in message and "uv" in message
