#!/usr/bin/env python3
"""Zero-prerequisite launcher for AegisQ.

    git clone <repo> && cd aegisq
    python3 run.py

That is the whole setup.  This script needs nothing but a Python 3 interpreter:
it locates or installs the tooling, builds an isolated environment, and runs the
AegisQ command line inside it.  Any arguments are forwarded, so
``python3 run.py benchmark --quick`` works exactly like ``aegisq benchmark
--quick``.

Why a launcher rather than "pip install and run"?  A console script only lands
on ``PATH`` inside the environment it was installed into, and the interpreter
that happens to be called ``python3`` is often one PyTorch has no wheels for
(3.14 at the time of writing) or one the OS forbids installing into (Homebrew,
Debian).  Rather than make that the reader's problem, this script resolves it.

Everything it creates lives inside the repository and is removed by deleting it.
Nothing is installed system-wide and nothing needs elevated permissions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOTSTRAP_VENV = ROOT / ".aegisq-bootstrap"

#: Interpreters this project is tested against and PyTorch publishes wheels for.
SUPPORTED = ((3, 10), (3, 11), (3, 12), (3, 13))
#: Version uv installs when it has to fetch one.
PREFERRED = "3.12"

BIN = "Scripts" if os.name == "nt" else "bin"
EXE = ".exe" if os.name == "nt" else ""


def _say(message: str) -> None:
    print(f"[run.py] {message}", flush=True)


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(ROOT), **kwargs)


def _announce_launcher() -> None:
    """Tell the CLI how it was invoked so its printed hints stay copy-pasteable."""
    launcher = "python run.py" if os.name == "nt" else "python3 run.py"
    os.environ.setdefault("AEGISQ_LAUNCHER", launcher)


# ----------------------------------------------------------------------
# strategy 1: already installed
# ----------------------------------------------------------------------
def _try_current_interpreter(argv: list[str]) -> int | None:
    """Run in-process when this interpreter can already import the package."""
    try:
        from aegisq.cli import main
    except ImportError:
        return None
    return main(argv)


def _try_project_venv(argv: list[str]) -> int | None:
    """Reuse a virtual environment a previous run (or the user) already built."""
    for candidate in (ROOT / ".venv", BOOTSTRAP_VENV):
        python = candidate / BIN / f"python{EXE}"
        if not python.exists():
            continue
        probe = _run(
            [str(python), "-c", "import aegisq"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return _run([str(python), "-m", "aegisq", *argv]).returncode
    return None


# ----------------------------------------------------------------------
# strategy 2: uv
# ----------------------------------------------------------------------
def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / BIN / f"uv{EXE}",
        Path.home() / ".cargo" / BIN / f"uv{EXE}",
        BOOTSTRAP_VENV / BIN / f"uv{EXE}",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _install_uv_with_pip() -> str | None:
    """Install uv into a throwaway venv inside the repository.

    Deliberately via PyPI rather than a piped remote install script: the same
    trust boundary the project's own dependencies already sit behind, no shell
    execution of downloaded text, and nothing written outside this directory.
    """
    try:
        import venv

        venv.EnvBuilder(with_pip=True, clear=True).create(BOOTSTRAP_VENV)
    except Exception as error:  # pragma: no cover - depends on host python
        # Seen in the wild on a Homebrew Python whose pyexpat fails to load:
        # ensurepip dies, so pip is unusable no matter what we ask of it.
        _say(f"this interpreter cannot bootstrap pip ({type(error).__name__})")
        return None

    python = BOOTSTRAP_VENV / BIN / f"python{EXE}"
    result = _run(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "uv"]
    )
    if result.returncode != 0:
        return None
    uv = BOOTSTRAP_VENV / BIN / f"uv{EXE}"
    return str(uv) if uv.exists() else None


#: uv release assets, keyed by (sys.platform prefix, machine).
_UV_ASSETS = {
    ("darwin", "arm64"): "uv-aarch64-apple-darwin.tar.gz",
    ("darwin", "aarch64"): "uv-aarch64-apple-darwin.tar.gz",
    ("darwin", "x86_64"): "uv-x86_64-apple-darwin.tar.gz",
    ("linux", "x86_64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("linux", "amd64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("linux", "aarch64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
    ("linux", "arm64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
    ("win32", "amd64"): "uv-x86_64-pc-windows-msvc.zip",
    ("win32", "x86_64"): "uv-x86_64-pc-windows-msvc.zip",
    ("win32", "arm64"): "uv-aarch64-pc-windows-msvc.zip",
}

_UV_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download/"


def _install_uv_from_release() -> str | None:
    """Fetch the official uv binary using only the standard library.

    The last line of defence: it needs neither pip nor a working ``ensurepip``,
    which is what makes it able to rescue a broken host interpreter.  A signed
    release artefact is downloaded and unpacked; no downloaded text is executed.
    """
    import platform

    key = (sys.platform, platform.machine().lower())
    asset = _UV_ASSETS.get(key)
    if asset is None:
        _say(f"no prebuilt uv for {key[0]}/{key[1]}")
        return None

    import io
    import urllib.request

    url = _UV_RELEASE + asset
    _say(f"downloading uv from {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except Exception as error:
        _say(f"download failed: {error}")
        return None

    destination = BOOTSTRAP_VENV / BIN
    destination.mkdir(parents=True, exist_ok=True)
    binary = destination / f"uv{EXE}"
    try:
        if asset.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                member = next(n for n in archive.namelist() if n.endswith(f"uv{EXE}"))
                binary.write_bytes(archive.read(member))
        else:
            import tarfile

            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                member = next(m for m in archive.getmembers() if m.name.endswith("/uv"))
                extracted = archive.extractfile(member)
                if extracted is None:
                    return None
                binary.write_bytes(extracted.read())
    except Exception as error:
        _say(f"could not unpack uv: {error}")
        return None

    binary.chmod(0o755)
    return str(binary)


def _install_uv() -> str | None:
    _say("setting up uv (a fast Python package manager) in .aegisq-bootstrap/")
    return _install_uv_with_pip() or _install_uv_from_release()


def _try_uv(argv: list[str]) -> int | None:
    uv = _find_uv() or _install_uv()
    if uv is None:
        return None
    _say("preparing the environment with uv (first run downloads PyTorch, ~1 minute)")
    command = [uv, "run"]
    if not _interpreter_is_supported():
        # uv can fetch its own interpreter, so a too-new system Python is fine.
        command += ["--python", PREFERRED]
    result = _run([*command, "aegisq", *argv])
    return result.returncode


# ----------------------------------------------------------------------
# strategy 3: plain venv + pip
# ----------------------------------------------------------------------
def _interpreter_is_supported() -> bool:
    return sys.version_info[:2] in SUPPORTED


def _try_pip(argv: list[str]) -> int | None:
    """Last resort for hosts where uv is unavailable but this Python will do."""
    if not _interpreter_is_supported():
        return None
    _say("uv unavailable; falling back to venv + pip (first run takes a few minutes)")
    import venv

    target = ROOT / ".venv"
    if not (target / BIN / f"python{EXE}").exists():
        venv.EnvBuilder(with_pip=True).create(target)
    python = target / BIN / f"python{EXE}"
    install = _run(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-e", "."]
    )
    if install.returncode != 0:
        return None
    return _run([str(python), "-m", "aegisq", *argv]).returncode


# ----------------------------------------------------------------------
def _report_failure() -> int:
    version = ".".join(str(part) for part in sys.version_info[:3])
    supported = ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED)
    print(
        "\n".join(
            [
                "",
                "[run.py] Could not prepare an environment automatically.",
                "",
                f"  This interpreter is Python {version}; AegisQ needs one of: {supported}.",
                "  Automatic setup needs either uv on PATH or network access to PyPI.",
                "",
                "  To do it by hand with a supported interpreter:",
                "",
                "    python3.12 -m venv .venv",
                f"    .venv/{BIN}/pip install -e .",
                f"    .venv/{BIN}/aegisq",
                "",
                "  Or install uv (https://docs.astral.sh/uv/) and re-run this script.",
                "",
            ]
        ),
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _announce_launcher()

    # Ordered cheapest-first: an existing install short-circuits everything.
    for strategy in (_try_current_interpreter, _try_project_venv, _try_uv, _try_pip):
        code = strategy(argv)
        if code is not None:
            return code
    return _report_failure()


if __name__ == "__main__":
    raise SystemExit(main())
