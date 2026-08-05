"""Benchmark harness, datasets and trainability metrics."""

from .datasets import (
    DATASETS,
    Dataset,
    circles,
    get_dataset,
    linearly_separable,
    parity,
    two_moons,
)
from .harness import (
    BenchmarkResult,
    ModelSpec,
    NoiseBenchmark,
    RunRecord,
    default_noise_levels,
    quantum_model,
    resilient_vs_baseline,
    standard_benchmark,
)
from .metrics import (
    GradientStats,
    accuracy,
    barren_plateau_scan,
    gradient_variance,
    mitigation_bias,
)

__all__ = [
    "Dataset",
    "DATASETS",
    "get_dataset",
    "two_moons",
    "circles",
    "parity",
    "linearly_separable",
    "NoiseBenchmark",
    "BenchmarkResult",
    "ModelSpec",
    "RunRecord",
    "quantum_model",
    "resilient_vs_baseline",
    "standard_benchmark",
    "default_noise_levels",
    "accuracy",
    "gradient_variance",
    "GradientStats",
    "barren_plateau_scan",
    "mitigation_bias",
]
