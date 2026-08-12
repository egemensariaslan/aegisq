"""Command-line interface.

``aegisq`` with no arguments runs a self-contained demo that exercises every
part of the library and prints what it measured -- the fastest way to see
whether the install works and what the thing actually does.  Subcommands expose
the heavier experiments individually.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Sequence

__all__ = ["main"]

_BULLET = "  - "


# ----------------------------------------------------------------------
# presentation helpers
# ----------------------------------------------------------------------
def _supports_colour() -> bool:
    return sys.stdout.isatty() and sys.platform != "win32"


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap(text, "1")

    def dim(self, text: str) -> str:
        return self._wrap(text, "2")

    def good(self, text: str) -> str:
        return self._wrap(text, "32")

    def warn(self, text: str) -> str:
        return self._wrap(text, "33")


STYLE = _Style(_supports_colour())


def _heading(index: int | None, title: str) -> None:
    label = f"{index}. {title}" if index is not None else title
    print()
    print(STYLE.bold(label))
    print(STYLE.dim("-" * max(len(label), 12)))


def invocation() -> str:
    """How the user reached us, so printed hints are commands they can actually run.

    ``run.py`` exports ``AEGISQ_LAUNCHER`` because a reader who bootstrapped
    through it has no ``aegisq`` on ``PATH`` -- telling them to type one would
    be the same mistake the launcher exists to prevent.
    """
    return os.environ.get("AEGISQ_LAUNCHER", "aegisq")


def _row(label: str, value: str, width: int = 34) -> None:
    # Pad to the column, but never let a long label swallow the gap.
    padding = max(width - len(label), 2)
    print(f"  {label}{' ' * padding}{value}")


def _cell(text: str, width: int, style: Callable[[str], str] | None = None) -> str:
    """Right-align to ``width`` before styling, so ANSI codes never skew the layout."""
    padded = text.rjust(width)
    return style(padded) if style else padded


# ----------------------------------------------------------------------
# demo steps
# ----------------------------------------------------------------------
def _step_environment(step: int | None = None) -> None:
    import pennylane as qml
    import torch

    import aegisq

    _heading(step, "Environment")
    _row("aegisq", aegisq.__version__)
    _row("pennylane", qml.__version__)
    _row("torch", torch.__version__)
    _row("python", sys.version.split()[0])


def _step_catalog(step: int | None = None, *, n_qubits: int = 6) -> None:
    from aegisq import QuantumLayer
    from aegisq.layers import ANSATZE

    _heading(step, f"Layer catalog (per layer, {n_qubits} qubits)")
    print(f"  {'ansatz':<24}{'2q gates':>10}{'params':>9}{'2q depth':>11}  conserves")
    seen: set[type] = set()
    for name, cls in ANSATZE.items():
        if cls in seen:
            continue  # skip aliases
        seen.add(cls)
        layer = QuantumLayer(n_qubits, n_layers=1, ansatz=name)
        conserves = layer.ansatz.symmetry or "-"
        print(
            f"  {name:<24}{layer.two_qubit_gates:>10}{layer.n_circuit_parameters:>9}"
            f"{layer.circuit_depth:>11}  {conserves}"
        )


def _step_drop_in(step: int | None = None) -> None:
    import torch
    from torch import nn

    from aegisq import ZNE, QuantumLayer

    _heading(step, "nn.Sequential integration")
    layer = QuantumLayer(4, n_layers=2, noise="hardware_like", seed=0)
    model = nn.Sequential(nn.Linear(8, 4), ZNE(layer, scale_factors=(1, 2, 3)), nn.Linear(4, 2))
    out = model(torch.randn(6, 8))
    out.sum().backward()

    named = [name for name, _ in model.named_parameters()]
    circuit_grad = layer.weights.grad
    _row("model output", f"{tuple(out.shape)}  requires_grad={out.requires_grad}")
    _row("registered parameters", ", ".join(named))
    _row(
        "circuit gradient",
        STYLE.good(
            f"finite={bool(torch.isfinite(circuit_grad).all())}  "
            f"norm={float(circuit_grad.norm()):.4f}"
        ),
    )
    _row("circuit runs per forward", str(model[1].circuit_evaluations))


def _step_gradient_chain(step: int | None = None) -> None:
    """The critical claim: mitigation must differentiate as the linear map it is."""
    import torch

    from aegisq import ZNE, QuantumLayer

    _heading(step, "Gradients through mitigation")
    x = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
    scales = (1.0, 2.0, 3.0)
    layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=3, dtype=torch.float64)
    mitigated = ZNE(layer, scale_factors=scales)

    mitigated(x).sum().backward()
    through_zne = layer.weights.grad.clone()

    expected = torch.zeros_like(through_zne)
    for coefficient, scale in zip(mitigated.extrapolator.coefficients, scales):
        layer.weights.grad = None
        layer.run_at_scale(x, scale, "global").sum().backward()
        expected += float(coefficient) * layer.weights.grad

    deviation = float((through_zne - expected).abs().max())
    tone = STYLE.good if deviation < 1e-9 else STYLE.warn
    _row("ZNE grad vs sum(c_i * scaled grad)", tone(f"max deviation {deviation:.2e}"))

    grads = []
    for method in ("backprop", "parameter-shift"):
        probe = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=11,
                             diff_method=method, dtype=torch.float64)
        probe(x).sum().backward()
        grads.append(probe.weights.grad.clone())
    shift_deviation = float((grads[0] - grads[1]).abs().max())
    _row("backprop vs parameter-shift", f"max deviation {shift_deviation:.2e}")


def _step_zne(step: int | None = None, *, n_qubits: int = 4, n_layers: int = 3) -> None:
    import torch

    from aegisq import ZNE, QuantumLayer
    from aegisq.benchmark import mitigation_bias

    _heading(step, "Zero-noise extrapolation")
    x = torch.randn(12, n_qubits, generator=torch.Generator().manual_seed(0))
    print(f"  {'noise p':>9}{'raw error':>12}{'richardson':>12}{'exponential':>14}{'linear':>10}")
    for p in (0.005, 0.01, 0.02, 0.04):
        ideal = QuantumLayer(n_qubits, n_layers=n_layers, noise=None, seed=7)
        noisy = QuantumLayer(n_qubits, n_layers=n_layers, noise=p, seed=7)
        with torch.no_grad():
            reference, unmitigated = ideal(x), noisy(x)
            cells = []
            for name in ("richardson", "exponential", "linear"):
                scales = (1.0, 2.0) if name == "linear" else (1.0, 2.0, 3.0)
                estimate = ZNE(noisy, scale_factors=scales, extrapolate=name)(x)
                cells.append(mitigation_bias(estimate, unmitigated, reference)["bias_reduction"])
            raw = float((unmitigated - reference).abs().mean())
        print(
            f"  {p:>9}{raw:>12.4f}{cells[0]:>12.1%}{cells[1]:>14.1%}{cells[2]:>10.1%}"
        )
    print(STYLE.dim("  fraction of error removed; negative = overshot"))


def _step_symmetry(step: int | None = None, *, n_qubits: int = 4) -> None:
    import torch

    from aegisq import QuantumLayer, SymmetryVerification

    _heading(step, "Symmetry verification")
    x = torch.randn(6, n_qubits - 1, generator=torch.Generator().manual_seed(0))
    out_of_sector = [i for i in range(2**n_qubits) if bin(i).count("1") != 2]

    def build(noise):
        return QuantumLayer(
            n_qubits, n_layers=3, ansatz="particle_conserving", encoding="excitation",
            measurement="probs", noise=noise, seed=0,
        )

    clean = build(None)
    with torch.no_grad():
        clean_leak = float(clean(x)[:, out_of_sector].sum(-1).mean())
    _row("circuit symmetry", str(clean.symmetry))
    _row("noiseless leakage out of sector", f"{clean_leak:.2e}")
    print(f"  {'noise p':>9}{'leakage':>10}{'accepted':>11}{'raw error':>12}{'verified':>11}")

    z = 1.0 - 2.0 * torch.tensor(
        [[(i >> (n_qubits - 1 - w)) & 1 for w in range(n_qubits)] for i in range(2**n_qubits)],
        dtype=torch.get_default_dtype(),
    )
    with torch.no_grad():
        ideal = clean(x) @ z
        for p in (0.005, 0.01, 0.02):
            layer = build(p)
            verified = SymmetryVerification(layer)
            probs = layer(x)
            print(
                f"  {p:>9}{float(probs[:, out_of_sector].sum(-1).mean()):>10.3f}"
                f"{float(verified.sector_weight(x).mean()):>11.3f}"
                f"{float(((probs @ z) - ideal).abs().mean()):>12.4f}"
                f"{float((verified(x) - ideal).abs().mean()):>11.4f}"
            )
    print(STYLE.dim("  shot overhead = 1/accepted"))


def _step_plateau(step: int | None = None, *, qubit_counts: Sequence[int] = (4, 8),
                  n_samples: int = 15) -> None:
    from aegisq.benchmark import barren_plateau_scan

    _heading(step, f"Barren plateau: gradient variance, {qubit_counts[0]} to "
                   f"{qubit_counts[-1]} qubits")
    print(f"  {'ansatz':<24}" + "".join(f"n={n:<11}" for n in qubit_counts) + f"{'decay':>7}")
    for ansatz in ("strongly_entangling", "basic_entangler", "local_entangler", "equivariant"):
        scan = barren_plateau_scan(
            qubit_counts, n_layers=4, n_samples=n_samples, seed=0, ansatz=ansatz
        )
        cells = "".join(f"{scan[n].variance:<13.2e}" for n in qubit_counts)
        decay = scan[qubit_counts[-1]].variance / scan[qubit_counts[0]].variance
        marker = _cell(f"{decay:.3f}", 7, STYLE.good if decay > 0.1 else STYLE.warn)
        print(f"  {ansatz:<24}{cells}{marker}")
    print(STYLE.dim("  decay = var(widest)/var(narrowest)"))


def _step_train(step: int | None = None, *, epochs: int = 8) -> None:
    import torch
    from torch import nn

    from aegisq import QuantumLayer
    from aegisq.benchmark import accuracy, get_dataset

    _heading(step, "Training under hardware-like noise")
    torch.manual_seed(0)
    data = get_dataset("two_moons", n_samples=140, n_features=4, seed=0)
    model = nn.Sequential(
        QuantumLayer(4, n_layers=3, noise="hardware_like", data_reupload=True, seed=0),
        nn.Linear(4, 2),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        start = accuracy(model(data.x_test), data.y_test)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(data))
        for begin in range(0, len(data), 16):
            index = order[begin : begin + 16]
            optimiser.zero_grad()
            loss_fn(model(data.x_train[index]), data.y_train[index]).backward()
            optimiser.step()
        if epoch % max(epochs // 3, 1) == 0 or epoch == epochs:
            with torch.no_grad():
                train = accuracy(model(data.x_train), data.y_train)
                test = accuracy(model(data.x_test), data.y_test)
            _row(f"epoch {epoch}", f"train {train:.3f}   test {test:.3f}")
    with torch.no_grad():
        final = accuracy(model(data.x_test), data.y_test)
    _row("test accuracy", STYLE.good(f"{start:.3f} -> {final:.3f}"))


def _step_caveat(step: int | None = None) -> None:
    _heading(step, "Known limits")
    print(_BULLET + "LocalEntangler's default CZ entangler is diagonal; a local_z read-out")
    print("    cannot fit globally correlated targets. 4-qubit parity: 0.569, against")
    print("    1.000 for entangler='cnot' or measurement='global_z'.")
    print(_BULLET + "PermutationEquivariant has 3 parameters per layer and needs data with")
    print("    that symmetry; 0.71-0.76 on two_moons against 0.94-0.97 for the baselines.")
    print(_BULLET + "Density-matrix simulation is 4^n in memory: about 10-12 qubits here.")
    run = invocation()
    print(STYLE.dim(f"\n  {run} choose-ansatz    {run} benchmark"))


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    print(STYLE.bold("AegisQ -- noise-resilient quantum layers for PyTorch + PennyLane"))
    print(STYLE.dim("Measured locally. Nothing is written to disk."))

    steps: list[Callable[[int], None]] = [
        _step_environment,
        _step_catalog,
        _step_drop_in,
        _step_gradient_chain,
        _step_zne,
        _step_symmetry,
        lambda step: _step_plateau(step, n_samples=8 if args.quick else 15),
        lambda step: _step_train(step, epochs=4 if args.quick else 8),
        _step_caveat,
    ]
    for index, step in enumerate(steps, start=1):
        step(index)

    print()
    print(STYLE.dim(f"done in {time.perf_counter() - started:.1f}s"))
    run = invocation()
    print(STYLE.dim(f"next: {run} --help   |   {run} benchmark --quick"))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from aegisq.benchmark import default_noise_levels, get_dataset, standard_benchmark
    from aegisq.noise import depolarizing, noiseless

    epochs, seeds, samples = args.epochs, args.seeds, args.samples
    noise_levels = default_noise_levels()
    if args.quick:
        # The composite profiles are by far the slowest to simulate (a channel
        # per wire per gate), so a quick run drops to two and shortens training.
        epochs, seeds, samples = 5, 1, 80
        noise_levels = {"noiseless": noiseless(), "depolarizing": depolarizing(0.01)}

    data = get_dataset(args.dataset, n_samples=samples, n_features=args.qubits, seed=0)
    result = standard_benchmark(
        n_qubits=args.qubits,
        n_layers=args.layers,
        dataset=data,
        epochs=epochs,
        seeds=tuple(range(seeds)),
        include_zne=not args.no_zne,
        noise_levels=noise_levels,
        verbose=not args.quiet,
    )
    print()
    print(result.summary())
    if args.csv:
        print(f"\nwrote {result.to_csv(args.csv)}")
    return 0


def cmd_plateau(args: argparse.Namespace) -> int:
    _step_plateau(qubit_counts=args.qubits, n_samples=args.samples)
    return 0


def cmd_zne(args: argparse.Namespace) -> int:
    _step_zne(n_qubits=args.qubits, n_layers=args.layers)
    return 0


def cmd_symmetry(args: argparse.Namespace) -> int:
    _step_symmetry(n_qubits=args.qubits)
    return 0


def cmd_choose_ansatz(args: argparse.Namespace) -> int:
    from aegisq.benchmark import NoiseBenchmark, get_dataset, quantum_model
    from aegisq.noise import depolarizing

    configurations = {
        "LocalEntangler, CZ, local_z (defaults)": {},
        "LocalEntangler, CNOT, local_z": {"ansatz_kwargs": {"entangler": "cnot"}},
        "LocalEntangler, ZZ, local_z": {"ansatz_kwargs": {"entangler": "zz"}},
        "LocalEntangler, CZ, local_zz": {"measurement": "local_zz"},
        "LocalEntangler, CZ, global_z": {"measurement": "global_z"},
        "BasicEntangler (reference)": {"ansatz": "basic_entangler"},
    }
    data = get_dataset("parity", n_samples=120, n_features=4, seed=0)
    result = NoiseBenchmark(
        {
            name: quantum_model(4, 2, n_layers=3, data_reupload=True, **kwargs)
            for name, kwargs in configurations.items()
        },
        {"depolarizing": depolarizing(0.01)},
        data,
        epochs=args.epochs,
        seeds=tuple(range(args.seeds)),
        measure_gradient_variance=False,
        verbose=not args.quiet,
    )
    records = result.run()
    print()
    print(STYLE.bold("4-qubit parity under depolarizing noise"))
    width = max(len(name) for name in configurations) + 2
    for model in records.models:
        scores = [record.test_accuracy for record in records if record.model == model]
        mean = sum(scores) / len(scores)
        cell = STYLE.good(f"{mean:.3f}") if mean > 0.9 else STYLE.warn(f"{mean:.3f}")
        print(f"  {model:<{width}}{cell}   {[round(s, 3) for s in scores]}")
    print(
        STYLE.dim(
            "\n  CZ and IsingZZ are diagonal: they write correlations into phase, which a"
            "\n  single-qubit <Z_i> read-out cannot see. CNOT moves parity into the"
            "\n  computational basis; a global Z...Z observable reads it off directly."
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .ui.server import serve

    return serve(host=args.host, port=args.port, open_browser=not args.no_browser,
                 quiet=not args.verbose)


def cmd_info(args: argparse.Namespace) -> int:
    import aegisq
    from aegisq.layers import available
    from aegisq.mitigation import EXTRAPOLATORS
    from aegisq.noise import PRESETS
    from aegisq.benchmark import DATASETS

    print(STYLE.bold(f"aegisq {aegisq.__version__}"))
    catalog = available()
    for kind in ("ansatz", "encoding", "measurement"):
        _heading(None, kind)
        print("  " + ", ".join(catalog[kind]))
    _heading(None, "noise presets")
    print("  " + ", ".join(sorted(PRESETS)))
    _heading(None, "extrapolators")
    print("  " + ", ".join(sorted(EXTRAPOLATORS)))
    _heading(None, "datasets")
    print("  " + ", ".join(sorted(DATASETS)))
    _step_catalog()
    return 0


# ----------------------------------------------------------------------
def _int_list(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if len(values) < 2:
        raise argparse.ArgumentTypeError("give at least two qubit counts, e.g. 4,6,8")
    return values


def build_parser() -> argparse.ArgumentParser:
    import aegisq

    parser = argparse.ArgumentParser(
        prog="aegisq",
        description=(
            "Noise-resilient quantum neural network layers for PyTorch + PennyLane. "
            "Run with no arguments for a self-contained demo."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aegisq {aegisq.__version__}")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="run every capability end to end (default)")
    demo.add_argument("--quick", action="store_true", help="fewer samples and epochs")
    demo.set_defaults(func=cmd_demo)

    bench = subparsers.add_parser("benchmark", help="resilient layers vs standard templates")
    bench.add_argument("--quick", action="store_true",
                       help="two noise levels, one seed, 5 epochs (~15 seconds)")
    bench.add_argument("--qubits", type=int, default=4)
    bench.add_argument("--layers", type=int, default=3)
    bench.add_argument("--dataset", default="two_moons",
                       choices=["two_moons", "circles", "parity", "linearly_separable"])
    bench.add_argument("--samples", type=int, default=120, help="dataset size")
    bench.add_argument("--epochs", type=int, default=10)
    bench.add_argument("--seeds", type=int, default=2, help="repeat each cell this many times")
    bench.add_argument("--no-zne", action="store_true", help="skip the mitigated model (faster)")
    bench.add_argument("--csv", help="write per-run records to this path")
    bench.add_argument("--quiet", action="store_true")
    bench.set_defaults(func=cmd_benchmark)

    plateau = subparsers.add_parser("plateau", help="gradient variance vs register width")
    plateau.add_argument("--qubits", type=_int_list, default=[4, 6, 8],
                         help="comma-separated widths, e.g. 4,6,8,10")
    plateau.add_argument("--samples", type=int, default=30)
    plateau.set_defaults(func=cmd_plateau)

    zne = subparsers.add_parser("zne", help="zero-noise extrapolation bias reduction")
    zne.add_argument("--qubits", type=int, default=4)
    zne.add_argument("--layers", type=int, default=3)
    zne.set_defaults(func=cmd_zne)

    symmetry = subparsers.add_parser("symmetry", help="symmetry verification and leakage")
    symmetry.add_argument("--qubits", type=int, default=4)
    symmetry.set_defaults(func=cmd_symmetry)

    choose = subparsers.add_parser(
        "choose-ansatz", help="where the resilient layers win, and where they do not"
    )
    choose.add_argument("--epochs", type=int, default=15)
    choose.add_argument("--seeds", type=int, default=2)
    choose.add_argument("--quiet", action="store_true")
    choose.set_defaults(func=cmd_choose_ansatz)

    ui = subparsers.add_parser(
        "serve", help="open the interactive dashboard in a browser"
    )
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--host", default="127.0.0.1", help="loopback only, by design")
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ui.add_argument("--verbose", action="store_true", help="log every request")
    ui.set_defaults(func=cmd_serve)

    info = subparsers.add_parser("info", help="list every registered component")
    info.set_defaults(func=cmd_info)

    return parser


#: Subcommand run when the user types a bare ``aegisq``.
DEFAULT_COMMAND = "demo"

COMMANDS = ("demo", "benchmark", "plateau", "zne", "symmetry", "choose-ansatz",
            "serve", "info")


def normalise_argv(argv: Sequence[str]) -> list[str]:
    """Insert the default subcommand when the user named none.

    Top-level flags such as ``--help`` and ``--version`` must reach the root
    parser untouched, so they suppress the insertion.
    """
    argv = list(argv)
    if any(arg in COMMANDS for arg in argv):
        return argv
    if any(arg in ("-h", "--help", "--version") for arg in argv):
        return argv
    return [DEFAULT_COMMAND, *argv]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(normalise_argv(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
