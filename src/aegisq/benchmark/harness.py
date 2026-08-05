"""The noise benchmark harness.

Trains a set of models across a set of noise profiles under identical
conditions -- same data, same seeds, same optimiser, same epoch budget -- and
reports test accuracy alongside the trainability diagnostics that explain it.

The comparison the library exists to make is one call:

>>> from aegisq.benchmark import standard_benchmark
>>> result = standard_benchmark(n_qubits=4, epochs=5)   # doctest: +SKIP
>>> print(result.summary())                             # doctest: +SKIP
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from ..layers.quantum_layer import QuantumLayer
from ..mitigation.zne import ZNE
from ..noise import NoiseSpec, depolarizing, hardware_like, noiseless, thermal_relaxation
from .datasets import Dataset, get_dataset
from .metrics import accuracy, gradient_variance

__all__ = [
    "ModelSpec",
    "RunRecord",
    "BenchmarkResult",
    "NoiseBenchmark",
    "quantum_model",
    "resilient_vs_baseline",
    "standard_benchmark",
    "default_noise_levels",
]

ModelFactory = Callable[[NoiseSpec], nn.Module]


# ----------------------------------------------------------------------
# specification
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """A named model builder, parameterised by the noise profile it runs under."""

    name: str
    factory: ModelFactory
    #: Marks an AegisQ-designed model as opposed to a standard-template baseline.
    resilient: bool = True


def quantum_model(
    n_qubits: int,
    n_classes: int = 2,
    *,
    n_layers: int = 2,
    zne: Mapping | None = None,
    head_bias: bool = True,
    **layer_kwargs,
) -> ModelFactory:
    """Build a factory for ``QuantumLayer -> Linear`` hybrid classifiers.

    The only classical trainable part is the final linear read-out mapping
    expectation values to logits.  There is deliberately no classical front-end:
    a trainable encoder before the circuit would let a benchmark "recover" from
    a circuit that had stopped contributing anything.

    ``zne``, when given, is a keyword mapping forwarded to
    :class:`~aegisq.mitigation.ZNE` to wrap the layer.
    """

    def factory(noise: NoiseSpec) -> nn.Module:
        layer = QuantumLayer(n_qubits, n_layers=n_layers, noise=noise, **layer_kwargs)
        block: nn.Module = ZNE(layer, **dict(zne)) if zne is not None else layer
        return nn.Sequential(
            block, nn.Linear(layer.out_features, n_classes, bias=head_bias)
        )

    return factory


def resilient_vs_baseline(
    n_qubits: int,
    n_classes: int = 2,
    *,
    n_layers: int = 2,
    include_zne: bool = True,
    **layer_kwargs,
) -> list[ModelSpec]:
    """The canonical model set: AegisQ layers against the standard PennyLane templates."""
    specs = [
        ModelSpec(
            "BasicEntangler (baseline)",
            quantum_model(
                n_qubits, n_classes, n_layers=n_layers, ansatz="basic_entangler", **layer_kwargs
            ),
            resilient=False,
        ),
        ModelSpec(
            "StronglyEntangling (baseline)",
            quantum_model(
                n_qubits,
                n_classes,
                n_layers=n_layers,
                ansatz="strongly_entangling",
                **layer_kwargs,
            ),
            resilient=False,
        ),
        ModelSpec(
            "LocalEntangler",
            quantum_model(
                n_qubits, n_classes, n_layers=n_layers, ansatz="local_entangler", **layer_kwargs
            ),
        ),
        ModelSpec(
            "PermutationEquivariant",
            quantum_model(
                n_qubits, n_classes, n_layers=n_layers, ansatz="equivariant", **layer_kwargs
            ),
        ),
    ]
    if include_zne:
        specs.append(
            ModelSpec(
                "LocalEntangler + ZNE",
                quantum_model(
                    n_qubits,
                    n_classes,
                    n_layers=n_layers,
                    ansatz="local_entangler",
                    zne={"scale_factors": (1.0, 2.0, 3.0), "extrapolate": "richardson"},
                    **layer_kwargs,
                ),
            )
        )
    return specs


def default_noise_levels() -> dict[str, NoiseSpec]:
    """The profiles named in the AegisQ specification, at a comparable strength."""
    return {
        "noiseless": noiseless(),
        "depolarizing": depolarizing(0.01),
        "thermal": thermal_relaxation(t1=50_000, t2=30_000, gate_time=400),
        "hardware_like": hardware_like(1.0),
    }


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------
@dataclass
class RunRecord:
    """One (model, noise profile, seed) training run."""

    model: str
    noise: str
    noise_strength: float
    seed: int
    test_accuracy: float
    train_accuracy: float
    final_loss: float
    gradient_variance: float | None
    n_parameters: int
    n_circuit_parameters: int
    two_qubit_gates: int
    circuit_evaluations: int
    seconds: float
    resilient: bool
    loss_curve: list[float] = field(default_factory=list, repr=False)


class BenchmarkResult:
    """Container for :class:`RunRecord`s with reporting helpers."""

    def __init__(self, records: Sequence[RunRecord], dataset: Dataset) -> None:
        self.records = list(records)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def models(self) -> list[str]:
        seen: dict[str, None] = {}
        for record in self.records:
            seen.setdefault(record.model, None)
        return list(seen)

    @property
    def noise_levels(self) -> list[str]:
        seen: dict[str, None] = {}
        for record in self.records:
            seen.setdefault(record.noise, None)
        return list(seen)

    def to_dicts(self) -> list[dict]:
        return [asdict(record) for record in self.records]

    def to_csv(self, path: str) -> str:
        """Write every run to ``path`` (loss curves excluded) and return the path."""
        rows = [{k: v for k, v in row.items() if k != "loss_curve"} for row in self.to_dicts()]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def pivot(self, metric: str = "test_accuracy") -> dict[str, dict[str, float]]:
        """``{model: {noise: mean metric over seeds}}``."""
        buckets: dict[str, dict[str, list[float]]] = {}
        for record in self.records:
            value = getattr(record, metric)
            if value is None:
                continue
            buckets.setdefault(record.model, {}).setdefault(record.noise, []).append(
                float(value)
            )
        return {
            model: {noise: sum(vals) / len(vals) for noise, vals in noises.items()}
            for model, noises in buckets.items()
        }

    def _table(self, metric: str, fmt: str) -> str:
        table = self.pivot(metric)
        if not table:
            return f"(no data for {metric})"
        noises = self.noise_levels
        label_width = max(len(m) for m in table) + 2
        columns = [max(len(n), 11) for n in noises]
        header = "  ".join(n.rjust(w) for n, w in zip(noises, columns))
        lines = [f"{'model'.ljust(label_width)}{header}"]
        lines.append("-" * len(lines[0]))
        for model in self.models:
            if model not in table:
                continue
            cells = []
            for noise, width in zip(noises, columns):
                value = table[model].get(noise)
                cells.append(("-" if value is None else format(value, fmt)).rjust(width))
            lines.append(f"{model.ljust(label_width)}" + "  ".join(cells))
        return "\n".join(lines)

    def summary(self) -> str:
        """A printable report: accuracy, gradient variance and circuit cost."""
        blocks = [
            f"AegisQ noise benchmark -- dataset {self.dataset.name!r} "
            f"({len(self.dataset)} train / {len(self.dataset.x_test)} test, "
            f"{self.dataset.n_features} features)",
            "",
            "test accuracy",
            self._table("test_accuracy", ".3f"),
            "",
            "gradient variance at initialisation",
            self._table("gradient_variance", ".2e"),
            "",
            "circuit cost",
        ]
        cost_width = max(len(m) for m in self.models) + 2
        blocks.append(
            f"{'model'.ljust(cost_width)}{'2q gates':>10}{'circuit params':>16}"
            f"{'runs/forward':>14}"
        )
        blocks.append("-" * (cost_width + 40))
        for model in self.models:
            record = next(r for r in self.records if r.model == model)
            blocks.append(
                f"{model.ljust(cost_width)}{record.two_qubit_gates:>10}"
                f"{record.n_circuit_parameters:>16}{record.circuit_evaluations:>14}"
            )
        return "\n".join(blocks)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BenchmarkResult(runs={len(self.records)}, models={len(self.models)}, "
            f"noise_levels={len(self.noise_levels)})"
        )


# ----------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------
def _as_noise_levels(levels) -> dict[str, NoiseSpec]:
    if isinstance(levels, Mapping):
        return {str(k): v for k, v in levels.items()}
    out: dict[str, NoiseSpec] = {}
    for item in levels:
        if isinstance(item, tuple):
            label, spec = item
        else:
            label, spec = item.describe(), item
        label = str(label)
        if label in out:
            # Two profiles that describe identically would otherwise overwrite
            # each other and silently shrink the sweep.
            suffix = 2
            while f"{label} ({suffix})" in out:
                suffix += 1
            label = f"{label} ({suffix})"
        out[label] = spec
    return out


def _count_circuit_evaluations(model: nn.Module) -> int:
    return sum(m.circuit_evaluations for m in model.modules() if isinstance(m, ZNE)) or 1


def _find_layer(model: nn.Module) -> QuantumLayer:
    for module in model.modules():
        if isinstance(module, QuantumLayer):
            return module
    raise TypeError("benchmark models must contain a QuantumLayer")


class NoiseBenchmark:
    """Train every model under every noise profile and collect the results.

    Parameters
    ----------
    models:
        :class:`ModelSpec` list, or a ``{name: factory}`` mapping.
    noise_levels:
        ``{label: NoiseSpec}`` mapping, or an iterable of specs / ``(label, spec)`` pairs.
    dataset:
        A :class:`~aegisq.benchmark.Dataset`, or a name to build one from.
        Its feature count must match the layer's ``in_features``.
    epochs, lr, batch_size:
        Optimiser budget.  Kept small by default -- these are density-matrix
        simulations, and every extra epoch costs real seconds.
    seeds:
        Repeat each cell once per seed and average.  More than one seed is
        strongly advised before drawing conclusions from a noisy simulation.
    measure_gradient_variance:
        Also sample the gradient variance at initialisation for each cell.
    """

    def __init__(
        self,
        models: Sequence[ModelSpec] | Mapping[str, ModelFactory],
        noise_levels: Mapping[str, NoiseSpec] | Iterable,
        dataset: Dataset | str = "two_moons",
        *,
        epochs: int = 15,
        lr: float = 0.05,
        batch_size: int = 16,
        seeds: Sequence[int] = (0,),
        measure_gradient_variance: bool = True,
        gradient_samples: int = 12,
        verbose: bool = True,
    ) -> None:
        if isinstance(models, Mapping):
            models = [ModelSpec(name, factory) for name, factory in models.items()]
        self.models = list(models)
        if not self.models:
            raise ValueError("at least one model is required")
        self.noise_levels = _as_noise_levels(noise_levels)
        if not self.noise_levels:
            raise ValueError("at least one noise level is required")
        self.dataset = get_dataset(dataset) if isinstance(dataset, str) else dataset
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seeds = tuple(seeds)
        self.measure_gradient_variance = measure_gradient_variance
        self.gradient_samples = int(gradient_samples)
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _train_once(
        self, spec: ModelSpec, noise: NoiseSpec, seed: int
    ) -> tuple[nn.Module, list[float]]:
        torch.manual_seed(seed)
        model = spec.factory(noise)
        layer = _find_layer(model)
        if layer.in_features != self.dataset.n_features:
            raise ValueError(
                f"model {spec.name!r} encodes {layer.in_features} features but dataset "
                f"{self.dataset.name!r} has {self.dataset.n_features}. Build the dataset "
                f"with n_features={layer.in_features}."
            )
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        x, y = self.dataset.x_train, self.dataset.y_train
        losses: list[float] = []
        for _ in range(self.epochs):
            permutation = torch.randperm(len(x))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(x), self.batch_size):
                index = permutation[start : start + self.batch_size]
                optimiser.zero_grad()
                loss = loss_fn(model(x[index]), y[index])
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return model, losses

    def _gradient_variance(self, spec: ModelSpec, noise: NoiseSpec, seed: int) -> float | None:
        try:
            stats = gradient_variance(
                lambda: spec.factory(noise),
                n_samples=self.gradient_samples,
                seed=seed,
                x=self.dataset.x_train[0],
            )
        except Exception:  # pragma: no cover - diagnostics must never fail a run
            return None
        return stats.variance

    def run(self) -> BenchmarkResult:
        """Execute the sweep and return the collected records."""
        records: list[RunRecord] = []
        total = len(self.models) * len(self.noise_levels) * len(self.seeds)
        done = 0
        for spec in self.models:
            for noise_label, noise in self.noise_levels.items():
                for seed in self.seeds:
                    done += 1
                    if self.verbose:
                        print(
                            f"[{done}/{total}] {spec.name} | {noise_label} | seed={seed}",
                            flush=True,
                        )
                    started = time.perf_counter()
                    model, losses = self._train_once(spec, noise, seed)
                    with torch.no_grad():
                        test_logits = model(self.dataset.x_test)
                        train_logits = model(self.dataset.x_train)
                    elapsed = time.perf_counter() - started
                    layer = _find_layer(model)
                    variance = (
                        self._gradient_variance(spec, noise, seed)
                        if self.measure_gradient_variance
                        else None
                    )
                    records.append(
                        RunRecord(
                            model=spec.name,
                            noise=noise_label,
                            noise_strength=noise.strength,
                            seed=seed,
                            test_accuracy=accuracy(test_logits, self.dataset.y_test),
                            train_accuracy=accuracy(train_logits, self.dataset.y_train),
                            final_loss=losses[-1] if losses else float("nan"),
                            gradient_variance=variance,
                            n_parameters=sum(
                                p.numel() for p in model.parameters() if p.requires_grad
                            ),
                            n_circuit_parameters=layer.n_circuit_parameters,
                            two_qubit_gates=layer.two_qubit_gates,
                            circuit_evaluations=_count_circuit_evaluations(model),
                            seconds=elapsed,
                            resilient=spec.resilient,
                            loss_curve=losses,
                        )
                    )
        return BenchmarkResult(records, self.dataset)


def standard_benchmark(
    n_qubits: int = 4,
    *,
    n_layers: int = 2,
    dataset: Dataset | str = "two_moons",
    epochs: int = 15,
    seeds: Sequence[int] = (0,),
    include_zne: bool = True,
    noise_levels: Mapping[str, NoiseSpec] | None = None,
    verbose: bool = True,
    **layer_kwargs,
) -> BenchmarkResult:
    """Run the canonical resilient-versus-baseline comparison.

    Builds the dataset with the right feature count for ``n_qubits``, assembles
    :func:`resilient_vs_baseline` models, and sweeps
    :func:`default_noise_levels`.

    A 4-qubit, 4-model, 4-noise-level sweep at 15 epochs takes a few minutes on
    a laptop; the ZNE model costs one extra circuit run per scale factor.
    """
    if isinstance(dataset, str):
        dataset = get_dataset(dataset, n_features=n_qubits)
    benchmark = NoiseBenchmark(
        resilient_vs_baseline(
            n_qubits, dataset.n_classes, n_layers=n_layers, include_zne=include_zne,
            **layer_kwargs,
        ),
        noise_levels or default_noise_levels(),
        dataset,
        epochs=epochs,
        seeds=seeds,
        verbose=verbose,
    )
    return benchmark.run()
