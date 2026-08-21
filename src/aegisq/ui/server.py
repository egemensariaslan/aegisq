"""A dependency-free dashboard server.

Built on :mod:`http.server` rather than a web framework so that ``pip install
aegisq`` still pulls in nothing but the quantum stack.  The front end is plain
HTML, CSS and JavaScript with hand-drawn SVG charts -- no CDN, no build step,
and it works with the network cable pulled.

The server binds to the loopback interface only.  It executes experiments on
request, which is exactly what makes it useful and also why it must never be
exposed beyond the machine it runs on.
"""

from __future__ import annotations

import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qs, urlparse

from . import experiments

__all__ = ["serve", "build_handler", "json_safe"]

ASSETS = Path(__file__).resolve().parent / "assets"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None``.

    ``json.dumps`` happily emits ``NaN`` and ``Infinity``, which are not valid
    JSON and make ``JSON.parse`` throw in the browser.  Several honest results
    are non-finite -- an infinite post-selection overhead at total decoherence,
    an undefined decay rate when a variance is exactly zero -- so they are
    converted here rather than papered over upstream.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


# ----------------------------------------------------------------------
# request parameter helpers
# ----------------------------------------------------------------------
def _number(query: dict, key: str, default: float) -> float:
    try:
        return float(query[key][0])
    except (KeyError, IndexError, ValueError):
        return default


def _integer(query: dict, key: str, default: int) -> int:
    return int(_number(query, key, default))


def _numbers(query: dict, key: str, default: list) -> list:
    try:
        raw = query[key][0]
    except (KeyError, IndexError):
        return list(default)
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return list(default)
    return values or list(default)


def _strings(query: dict, key: str, default: list) -> list:
    try:
        raw = query[key][0]
    except (KeyError, IndexError):
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(default)


# ----------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------
def _route_catalog(query: dict) -> dict:
    return experiments.catalog(
        n_qubits=max(2, min(_integer(query, "qubits", 6), 12)),
        n_layers=max(1, min(_integer(query, "layers", 1), 8)),
    )


def _route_zne(query: dict) -> dict:
    return experiments.zne_curve(
        n_qubits=max(2, min(_integer(query, "qubits", 4), 8)),
        n_layers=max(1, min(_integer(query, "layers", 3), 8)),
        noise=max(0.0, min(_number(query, "noise", 0.02), 0.2)),
        scale_factors=_numbers(query, "scales", [1.0, 2.0, 3.0]),
        seed=_integer(query, "seed", 7),
        observable=_integer(query, "observable", 0),
        trials=max(1, min(_integer(query, "trials", 8), 30)),
    )


def _route_plateau(query: dict) -> dict:
    counts = [int(value) for value in _numbers(query, "qubits", [4, 6, 8, 10])]
    counts = sorted({max(2, min(value, 12)) for value in counts})
    return experiments.plateau_scan(
        qubit_counts=counts,
        n_layers=max(1, min(_integer(query, "layers", 4), 8)),
        n_samples=max(2, min(_integer(query, "samples", 30), 200)),
        ansatze=_strings(query, "ansatze", ["strongly_entangling", "basic_entangler",
                                            "local_entangler", "equivariant"]),
        seed=_integer(query, "seed", 0),
    )


def _route_noise_sweep(query: dict) -> dict:
    return experiments.noise_sweep(
        n_qubits=max(2, min(_integer(query, "qubits", 4), 8)),
        n_layers=max(1, min(_integer(query, "layers", 3), 8)),
        strengths=_numbers(query, "strengths", [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04]),
        seed=_integer(query, "seed", 5),
        trials=max(1, min(_integer(query, "trials", 5), 20)),
    )


def _route_symmetry(query: dict) -> dict:
    return experiments.symmetry_scan(
        n_qubits=max(2, min(_integer(query, "qubits", 4), 8)),
        n_layers=max(1, min(_integer(query, "layers", 3), 8)),
        strengths=_numbers(query, "strengths", [0.002, 0.005, 0.01, 0.02, 0.05]),
        seed=_integer(query, "seed", 0),
    )


def _route_env(query: dict) -> dict:
    import platform
    import sys as _sys

    import pennylane
    import torch

    import aegisq

    return {
        "aegisq": aegisq.__version__,
        "pennylane": pennylane.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "executable": _sys.executable,
    }


JSON_ROUTES: dict[str, Callable[[dict], dict]] = {
    "/api/env": _route_env,
    "/api/catalog": _route_catalog,
    "/api/zne": _route_zne,
    "/api/plateau": _route_plateau,
    "/api/noise-sweep": _route_noise_sweep,
    "/api/symmetry": _route_symmetry,
}


def _stream_training(query: dict) -> Iterator[dict]:
    return experiments.training_stream(
        n_qubits=max(2, min(_integer(query, "qubits", 4), 8)),
        n_layers=max(1, min(_integer(query, "layers", 3), 8)),
        epochs=max(1, min(_integer(query, "epochs", 12), 60)),
        noise_name=_strings(query, "noise", ["hardware_like"])[0],
        models=_strings(query, "models", ["local_entangler", "strongly_entangling"]),
        dataset=_strings(query, "dataset", ["two_moons"])[0],
        samples=max(40, min(_integer(query, "samples", 120), 400)),
        seed=_integer(query, "seed", 0),
    )


STREAM_ROUTES: dict[str, Callable[[dict], Iterator[dict]]] = {
    "/api/train": _stream_training,
}


# ----------------------------------------------------------------------
def build_handler(quiet: bool = True) -> type[BaseHTTPRequestHandler]:
    """Create the request handler class for the dashboard."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AegisQ"
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: D102
            if not quiet:
                super().log_message(*args)

        # -- helpers ------------------------------------------------
        def _send(self, status: int, body: bytes, content_type: str,
                  extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # The page is entirely self-contained; forbid outbound requests so a
            # dashboard rendering local measurements cannot phone anywhere.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'",
            )
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(json_safe(payload)).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _send_asset(self, name: str) -> None:
            path = (ASSETS / name).resolve()
            # Defence in depth: only ever serve files from the assets directory.
            if not path.is_file() or ASSETS not in path.parents:
                self._send_json({"error": "not found"}, status=404)
                return
            self._send(200, path.read_bytes(),
                       _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

        # -- routing ------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route, query = parsed.path, parse_qs(parsed.query)

            if route in ("/", "/index.html"):
                return self._send_asset("index.html")
            if route in JSON_ROUTES:
                try:
                    return self._send_json(JSON_ROUTES[route](query))
                except Exception as error:  # surfaced in the panel, not swallowed
                    return self._send_json(
                        {"error": f"{type(error).__name__}: {error}"}, status=500
                    )
            if route in STREAM_ROUTES:
                return self._stream(STREAM_ROUTES[route], query)
            if route.startswith("/assets/"):
                return self._send_asset(route[len("/assets/"):])
            return self._send_json({"error": "not found"}, status=404)

        do_HEAD = do_GET

        def _stream(self, factory: Callable[[dict], Iterator[dict]], query: dict) -> None:
            """Server-sent events, so a long run reports progress as it happens."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for event in factory(query):
                    payload = json.dumps(json_safe(event))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the reader navigated away; nothing to clean up
            except Exception as error:
                try:
                    failure = json.dumps({"event": "error",
                                          "message": f"{type(error).__name__}: {error}"})
                    self.wfile.write(f"data: {failure}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    pass
            self.close_connection = True

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          quiet: bool = True) -> int:
    """Run the dashboard until interrupted."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        # The endpoints run arbitrary simulations on request; that is fine for a
        # local tool and not fine on a shared interface.
        raise ValueError(
            f"refusing to bind {host!r}: the dashboard executes experiments and is "
            "intended for loopback only"
        )

    httpd = ThreadingHTTPServer((host, port), build_handler(quiet=quiet))
    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}/"

    print(f"AegisQ dashboard running at {url}")
    print("Every panel runs the real library on this machine. Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0
