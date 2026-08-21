"""Experiments backing the dashboard, each returning plain JSON-serialisable data.

Every function here re-runs the real library on demand -- nothing is precomputed
or cached from a previous session, so a number on the screen is a number this
machine just measured.  Each result carries the parameters and seed that
produced it, which is what makes a panel reproducible from the command line.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from ..benchmark import accuracy, barren_plateau_scan, get_dataset
from ..layers import ANSATZE, QuantumLayer
from ..mitigation import ZNE, get_extrapolator
from ..noise import NoiseSpec, depolarizing, get_preset, hardware_like

__all__ = [
    "catalog",
    "zne_curve",
    "plateau_scan",
    "noise_sweep",
    "symmetry_scan",
    "training_stream",
    "ANSATZ_LABELS",
]

#: Display names and roles for the ansaetze the dashboard compares.
ANSATZ_LABELS: dict[str, dict[str, Any]] = {
    "strongly_entangling": {"label": "StronglyEntangling", "role": "baseline"},
    "basic_entangler": {"label": "BasicEntangler", "role": "baseline"},
    "local_entangler": {"label": "LocalEntangler", "role": "aegisq"},
    "equivariant": {"label": "PermutationEquivariant", "role": "aegisq"},
    "particle_conserving": {"label": "ParticleConserving", "role": "aegisq"},
    "z2_equivariant": {"label": "Z2Equivariant", "role": "aegisq"},
}


def _fixed_batch(n_features: int, batch: int = 12, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, n_features, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------
def catalog(n_qubits: int = 6, n_layers: int = 1) -> dict:
    """Per-layer circuit cost for every registered ansatz."""
    rows = []
    seen: set[type] = set()
    for name, cls in ANSATZE.items():
        if cls in seen:
            continue
        seen.add(cls)
        layer = QuantumLayer(n_qubits, n_layers=n_layers, ansatz=name)
        meta = ANSATZ_LABELS.get(name, {"label": name, "role": "aegisq"})
        rows.append(
            {
                "name": name,
                "label": meta["label"],
                "role": meta["role"],
                "two_qubit_gates": layer.two_qubit_gates,
                "parameters": layer.n_circuit_parameters,
                "depth": layer.circuit_depth,
                "symmetry": layer.ansatz.symmetry,
            }
        )
    rows.sort(key=lambda row: (row["role"] != "aegisq", row["label"]))
    return {"n_qubits": n_qubits, "n_layers": n_layers, "rows": rows}


# ----------------------------------------------------------------------
def _lagrange_curve(scales: Sequence[float], values: Sequence[float], grid: np.ndarray):
    """Evaluate the interpolating polynomial Richardson uses, across a grid."""
    result = np.zeros_like(grid)
    for i, (li, vi) in enumerate(zip(scales, values)):
        basis = np.ones_like(grid)
        for j, lj in enumerate(scales):
            if i != j:
                basis *= (grid - lj) / (li - lj)
        result += vi * basis
    return result


def _polynomial_curve(scales, values, grid, degree: int):
    coefficients = np.polyfit(np.asarray(scales), np.asarray(values), degree)
    return np.polyval(coefficients, grid)


def _exponential_curve(scales, values, grid, asymptote: float = 0.0):
    centred = np.asarray(values) - asymptote
    sign = np.sign(centred[0]) or 1.0
    magnitude = sign * centred
    if np.any(magnitude <= 1e-12):
        return None
    slope, intercept = np.polyfit(np.asarray(scales), np.log(magnitude), 1)
    if slope > 0:
        return None
    return asymptote + sign * np.exp(intercept + slope * grid)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and standard deviation (0 for a single value)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return float(array.mean()) if array.size else float("nan"), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def zne_curve(
    n_qubits: int = 4,
    n_layers: int = 3,
    noise: float = 0.02,
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
    seed: int = 7,
    observable: int = 0,
    trials: int = 8,
) -> dict:
    """The extrapolation itself: measured points, fitted models, and the truth.

    The circuit weights are fixed (one ``seed``), but the classical input is
    resampled ``trials`` times so every reported number is a mean over
    independent draws rather than the outcome for one input. Reported spread is
    the sample standard deviation across those draws, propagated through to the
    plotted error bars and the bias-reduction figures -- a single point
    estimate is not a claim this panel is willing to make.
    """
    scale_factors = [float(s) for s in scale_factors]
    trials = max(1, int(trials))
    ideal_layer = QuantumLayer(n_qubits, n_layers=n_layers, noise=None, seed=seed)
    noisy_layer = QuantumLayer(n_qubits, n_layers=n_layers, noise=noise, seed=seed)
    grid = np.linspace(0.0, max(scale_factors) * 1.05, 120)

    definitions: list[tuple[str, str, Callable[[Sequence[float]], Any]]] = [
        ("richardson", "Richardson", lambda m: _lagrange_curve(scale_factors, m, grid)),
        ("linear", "Linear", lambda m: _polynomial_curve(scale_factors, m, grid, 1)),
        ("exponential", "Exponential", lambda m: _exponential_curve(scale_factors, m, grid)),
    ]
    extrapolators = {key: get_extrapolator(key, scale_factors) for key, _, _ in definitions}

    truths: list[float] = []
    raw_errors: list[float] = []
    measured_trials: list[list[float]] = []
    fit_estimates: dict[str, list[float]] = {key: [] for key, _, _ in definitions}
    fit_reductions: dict[str, list[float]] = {key: [] for key, _, _ in definitions}
    fit_curves: dict[str, list[np.ndarray]] = {key: [] for key, _, _ in definitions}

    with torch.no_grad():
        for trial in range(trials):
            x = _fixed_batch(n_qubits, batch=1, seed=1000 * seed + trial)[0]
            truth = float(ideal_layer(x)[observable])
            measured = [
                float(noisy_layer.run_at_scale(x, scale, "global")[observable])
                for scale in scale_factors
            ]
            stacked = torch.tensor(measured, dtype=torch.float64).reshape(-1, 1)
            raw_error = abs(measured[0] - truth)
            truths.append(truth)
            raw_errors.append(raw_error)
            measured_trials.append(measured)

            for key, _, curve_fn in definitions:
                estimate = float(extrapolators[key](stacked)[0])
                fit_estimates[key].append(estimate)
                if raw_error > 1e-9:
                    fit_reductions[key].append(1.0 - abs(estimate - truth) / raw_error)
                curve = curve_fn(measured)
                if curve is not None:
                    fit_curves[key].append(curve)

    truth_mean, truth_std = _mean_std(truths)
    raw_error_mean, raw_error_std = _mean_std(raw_errors)
    measured_array = np.asarray(measured_trials)  # (trials, n_scales)
    points = [
        {"x": s, "y": float(measured_array[:, j].mean()),
         "err": float(measured_array[:, j].std(ddof=1)) if trials > 1 else 0.0}
        for j, s in enumerate(scale_factors)
    ]

    fits = []
    for key, label, _ in definitions:
        estimate_mean, estimate_std = _mean_std(fit_estimates[key])
        reduction_mean, reduction_std = _mean_std(fit_reductions[key]) if fit_reductions[key] else (0.0, 0.0)
        curves = fit_curves[key]
        curve_payload = None
        if curves:
            stacked_curves = np.stack(curves)
            curve_payload = [
                {"x": float(a), "y": float(b)}
                for a, b in zip(grid, stacked_curves.mean(axis=0))
            ]
        fits.append(
            {
                "key": key,
                "label": label,
                "estimate": estimate_mean,
                "estimate_std": estimate_std,
                "residual": abs(estimate_mean - truth_mean),
                "bias_reduction": reduction_mean,
                "bias_reduction_std": reduction_std,
                "coverage": len(fit_curves[key]) / trials,
                # How much sampling variance this coefficient set costs. It grows
                # fast with closely spaced scale factors, and is the reason a
                # five-point Richardson fit can be far worse than a three-point one.
                "variance_cost": getattr(extrapolators[key], "noise_amplification", float("nan")),
                "curve": curve_payload,
            }
        )

    return {
        "params": {
            "n_qubits": n_qubits, "n_layers": n_layers, "noise": noise,
            "scale_factors": scale_factors, "seed": seed, "observable": observable,
            "trials": trials,
        },
        "command": (
            f"python3 run.py zne --qubits {n_qubits} --layers {n_layers}"
        ),
        "truth": truth_mean,
        "truth_std": truth_std,
        "raw_error": raw_error_mean,
        "raw_error_std": raw_error_std,
        "trials": trials,
        "points": points,
        "fits": fits,
        "observable_label": f"<Z_{observable}>",
    }


# ----------------------------------------------------------------------
def plateau_scan(
    qubit_counts: Sequence[int] = (4, 6, 8, 10),
    n_layers: int = 4,
    n_samples: int = 30,
    ansatze: Sequence[str] = ("strongly_entangling", "basic_entangler",
                             "local_entangler", "equivariant"),
    seed: int = 0,
) -> dict:
    """Gradient variance against register width, with a fitted decay rate."""
    qubit_counts = [int(n) for n in qubit_counts]
    series = []
    for ansatz in ansatze:
        scan = barren_plateau_scan(
            qubit_counts, n_layers=n_layers, n_samples=n_samples, seed=seed, ansatz=ansatz
        )
        points = [
            {
                "x": n,
                "y": scan[n].variance,
                "max": scan[n].max_variance,
                "dead": scan[n].dead_parameter_fraction,
                "parameters": scan[n].n_parameters,
            }
            for n in qubit_counts
        ]
        variances = np.array([p["y"] for p in points])
        positive = variances > 0
        # A straight line through log(variance) is an exponential decay; its
        # slope is the per-qubit factor, which is the barren-plateau signature.
        if positive.sum() >= 2:
            slope = float(
                np.polyfit(np.array(qubit_counts)[positive], np.log(variances[positive]), 1)[0]
            )
            per_qubit = float(np.exp(slope))
        else:
            per_qubit = float("nan")
        meta = ANSATZ_LABELS.get(ansatz, {"label": ansatz, "role": "aegisq"})
        series.append(
            {
                "key": ansatz,
                "label": meta["label"],
                "role": meta["role"],
                "points": points,
                "decay": points[-1]["y"] / points[0]["y"] if points[0]["y"] > 0 else float("nan"),
                "per_qubit_factor": per_qubit,
            }
        )
    return {
        "params": {
            "qubit_counts": qubit_counts, "n_layers": n_layers,
            "n_samples": n_samples, "seed": seed,
        },
        "command": (
            f"python3 run.py plateau --qubits {','.join(str(n) for n in qubit_counts)} "
            f"--samples {n_samples}"
        ),
        "series": series,
    }


# ----------------------------------------------------------------------
def noise_sweep(
    n_qubits: int = 4,
    n_layers: int = 3,
    strengths: Sequence[float] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04),
    ansatze: Sequence[str] = ("strongly_entangling", "basic_entangler",
                             "local_entangler", "equivariant"),
    seed: int = 5,
    with_zne: bool = True,
    trials: int = 5,
) -> dict:
    """Signal retained by each ansatz as noise rises.

    The metric is deliberately scale-free.  Depolarizing noise contracts
    expectation values towards zero, ``E_noisy ~ f * E_ideal``, so the least
    squares estimate

        f = sum(E_noisy * E_ideal) / sum(E_ideal^2)

    is the fraction of the signal that survived: 1.0 is untouched, 0.0 is
    destroyed.  A raw ``mean|E_noisy - E_ideal|`` would instead reward whichever
    ansatz happens to produce expectation values near zero -- a deeply entangling
    circuit has little to lose and would score as the most "robust", which is an
    artifact of the metric rather than a property of the circuit.

    Each point is computed on ``trials`` independent random-input batches (fresh
    draws, not a bootstrap of one batch) and reported as mean ± sample standard
    deviation, so the curve is a distribution of the retained-signal estimator
    rather than the value one batch happened to produce.
    """
    strengths = [float(s) for s in strengths]
    trials = max(1, int(trials))
    series = []

    def retention(noisy: torch.Tensor, ideal: torch.Tensor) -> float:
        denominator = float((ideal * ideal).sum())
        if denominator <= 1e-12:
            return float("nan")
        return float((noisy * ideal).sum()) / denominator

    def build_series(ansatz: str, label: str, role: str, mitigate: bool):
        reference_layer = QuantumLayer(n_qubits, n_layers=n_layers, ansatz=ansatz,
                                       noise=None, seed=seed)
        points = []
        for strength in strengths:
            if strength == 0.0:
                points.append({"x": 0.0, "y": 1.0, "err": 0.0})
                continue
            layer = QuantumLayer(n_qubits, n_layers=n_layers, ansatz=ansatz,
                                 noise=depolarizing(strength), seed=seed)
            wrapped = ZNE(layer, scale_factors=(1.0, 2.0, 3.0)) if mitigate else layer
            retentions = []
            with torch.no_grad():
                for trial in range(trials):
                    x = _fixed_batch(n_qubits, seed=2000 * seed + trial)
                    reference = reference_layer(x)
                    estimate = wrapped(x)
                    retentions.append(retention(estimate, reference))
            mean, std = _mean_std(retentions)
            points.append({"x": strength, "y": mean, "err": std})
        return {"key": ansatz + ("_zne" if mitigate else ""), "label": label,
                "role": role, "points": points, "mitigated": mitigate}

    for ansatz in ansatze:
        meta = ANSATZ_LABELS.get(ansatz, {"label": ansatz, "role": "aegisq"})
        series.append(build_series(ansatz, meta["label"], meta["role"], mitigate=False))

    if with_zne:
        series.append(
            build_series("local_entangler", "LocalEntangler + ZNE", "mitigated", mitigate=True)
        )

    return {
        "params": {"n_qubits": n_qubits, "n_layers": n_layers,
                   "strengths": strengths, "seed": seed, "trials": trials},
        "command": "python3 run.py benchmark",
        "trials": trials,
        "series": series,
    }


# ----------------------------------------------------------------------
def symmetry_scan(
    n_qubits: int = 4,
    n_layers: int = 3,
    strengths: Sequence[float] = (0.002, 0.005, 0.01, 0.02, 0.05),
    seed: int = 0,
) -> dict:
    """Leakage out of the conserved sector, and what post-selecting it recovers."""
    from ..mitigation import SymmetryVerification

    strengths = [float(s) for s in strengths]

    def build(noise):
        return QuantumLayer(
            n_qubits, n_layers=n_layers, ansatz="particle_conserving",
            encoding="excitation", measurement="probs", noise=noise, seed=seed,
        )

    clean = build(None)
    x = _fixed_batch(clean.in_features, batch=8)
    n_particles = clean.encoding.particles(n_qubits)
    out_of_sector = [i for i in range(2**n_qubits) if bin(i).count("1") != n_particles]
    z_table = 1.0 - 2.0 * torch.tensor(
        [[(i >> (n_qubits - 1 - w)) & 1 for w in range(n_qubits)] for i in range(2**n_qubits)],
        dtype=torch.get_default_dtype(),
    )

    with torch.no_grad():
        ideal = clean(x) @ z_table
        clean_leakage = float(clean(x)[:, out_of_sector].sum(-1).mean())
        rows = []
        for strength in strengths:
            layer = build(strength)
            verified = SymmetryVerification(layer)
            probs = layer(x)
            rows.append(
                {
                    "x": strength,
                    "leakage": float(probs[:, out_of_sector].sum(-1).mean()),
                    "accepted": float(verified.sector_weight(x).mean()),
                    "raw_error": float(((probs @ z_table) - ideal).abs().mean()),
                    "verified_error": float((verified(x) - ideal).abs().mean()),
                }
            )
    for row in rows:
        row["overhead"] = 1.0 / row["accepted"] if row["accepted"] > 0 else float("inf")
        row["improvement"] = (
            1.0 - row["verified_error"] / row["raw_error"] if row["raw_error"] > 0 else 0.0
        )

    return {
        "params": {"n_qubits": n_qubits, "n_layers": n_layers,
                   "n_particles": n_particles, "seed": seed},
        "command": f"python3 run.py symmetry --qubits {n_qubits}",
        "symmetry": clean.symmetry,
        "noiseless_leakage": clean_leakage,
        "sector_size": 2**n_qubits - len(out_of_sector),
        "space_size": 2**n_qubits,
        "rows": rows,
    }


# ----------------------------------------------------------------------
def training_stream(
    n_qubits: int = 4,
    n_layers: int = 3,
    epochs: int = 12,
    noise_name: str = "hardware_like",
    models: Sequence[str] = ("local_entangler", "strongly_entangling"),
    dataset: str = "two_moons",
    samples: int = 120,
    seed: int = 0,
    learning_rate: float = 0.05,
    batch_size: int = 16,
) -> Iterator[dict]:
    """Train the given ansaetze side by side, yielding one event per epoch."""
    spec: NoiseSpec = (
        hardware_like() if noise_name == "hardware_like" else get_preset(noise_name)
    )
    data = get_dataset(dataset, n_samples=samples, n_features=n_qubits, seed=seed)

    built = []
    for ansatz in models:
        torch.manual_seed(seed)
        layer = QuantumLayer(n_qubits, n_layers=n_layers, ansatz=ansatz,
                             noise=spec, data_reupload=True, seed=seed)
        head = torch.nn.Linear(layer.out_features, data.n_classes)
        model = torch.nn.Sequential(layer, head)
        built.append(
            {
                "key": ansatz,
                "label": ANSATZ_LABELS.get(ansatz, {}).get("label", ansatz),
                "role": ANSATZ_LABELS.get(ansatz, {}).get("role", "aegisq"),
                "model": model,
                "optimiser": torch.optim.Adam(model.parameters(), lr=learning_rate),
                "two_qubit_gates": layer.two_qubit_gates,
            }
        )

    yield {
        "event": "start",
        "epochs": epochs,
        "noise": spec.describe(),
        "dataset": {"name": data.name, "train": len(data), "test": len(data.x_test)},
        "series": [
            {"key": b["key"], "label": b["label"], "role": b["role"],
             "two_qubit_gates": b["two_qubit_gates"]}
            for b in built
        ],
    }

    loss_fn = torch.nn.CrossEntropyLoss()
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        for entry in built:
            if epoch > 0:  # epoch 0 reports the untrained baseline
                order = torch.randperm(len(data))
                for begin in range(0, len(data), batch_size):
                    index = order[begin : begin + batch_size]
                    entry["optimiser"].zero_grad()
                    loss_fn(entry["model"](data.x_train[index]), data.y_train[index]).backward()
                    entry["optimiser"].step()
            with torch.no_grad():
                train_logits = entry["model"](data.x_train)
                test_logits = entry["model"](data.x_test)
                yield {
                    "event": "epoch",
                    "key": entry["key"],
                    "epoch": epoch,
                    "train_accuracy": accuracy(train_logits, data.y_train),
                    "test_accuracy": accuracy(test_logits, data.y_test),
                    "loss": float(loss_fn(train_logits, data.y_train)),
                    "elapsed": time.perf_counter() - started,
                }
    yield {"event": "done", "elapsed": time.perf_counter() - started}
