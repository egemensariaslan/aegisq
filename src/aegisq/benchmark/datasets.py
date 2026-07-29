"""Small synthetic classification datasets.

Deliberately dependency-free (numpy only) so that ``pip install aegisq`` pulls in
nothing beyond the quantum stack.  Every generator embeds its intrinsic 2-D
structure into ``n_features`` dimensions and rescales to roughly ``[-pi, pi]``,
the range where angle encodings are injective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "Dataset",
    "two_moons",
    "circles",
    "parity",
    "linearly_separable",
    "DATASETS",
    "get_dataset",
]


@dataclass(frozen=True)
class Dataset:
    """A train/test split held as torch tensors."""

    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    n_classes: int

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    def __len__(self) -> int:
        return int(self.x_train.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, train={len(self)}, test={len(self.x_test)}, "
            f"n_features={self.n_features}, n_classes={self.n_classes})"
        )


def _embed(x: np.ndarray, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Lift 2-D data into ``n_features`` dimensions with a random isometry.

    Padding with zeros would leave most qubits carrying no signal; a random
    isometry spreads the information across the whole register, which is what a
    fair width comparison needs.
    """
    if n_features == x.shape[1]:
        return x
    if n_features < x.shape[1]:
        return x[:, :n_features]
    projection = rng.normal(size=(x.shape[1], n_features))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    return x @ projection


def _rescale(x: np.ndarray, limit: float = np.pi) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.abs(x).max(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return limit * x / scale


def _split(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    test_fraction: float,
    rng: np.random.Generator,
) -> Dataset:
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_test = max(1, int(round(len(x) * test_fraction)))
    n_train = len(x) - n_test
    if n_train < 1:
        raise ValueError("test_fraction leaves no training samples")
    to_x = lambda a: torch.as_tensor(a, dtype=torch.get_default_dtype())  # noqa: E731
    to_y = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    return Dataset(
        name=name,
        x_train=to_x(x[:n_train]),
        y_train=to_y(y[:n_train]),
        x_test=to_x(x[n_train:]),
        y_test=to_y(y[n_train:]),
        n_classes=n_classes,
    )


def two_moons(
    n_samples: int = 200,
    *,
    n_features: int = 2,
    noise: float = 0.15,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Dataset:
    """Two interleaved half-circles -- the standard non-linear toy problem."""
    rng = np.random.default_rng(seed)
    n_a = n_samples // 2
    n_b = n_samples - n_a
    theta_a = rng.uniform(0, np.pi, n_a)
    theta_b = rng.uniform(0, np.pi, n_b)
    outer = np.stack([np.cos(theta_a), np.sin(theta_a)], axis=1)
    inner = np.stack([1 - np.cos(theta_b), 0.5 - np.sin(theta_b)], axis=1)
    x = np.concatenate([outer, inner]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])
    return _split("two_moons", _rescale(_embed(x, n_features, rng)), y, 2, test_fraction, rng)


def circles(
    n_samples: int = 200,
    *,
    n_features: int = 2,
    noise: float = 0.1,
    factor: float = 0.5,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Dataset:
    """Concentric rings; not linearly separable in any number of dimensions."""
    rng = np.random.default_rng(seed)
    n_a = n_samples // 2
    n_b = n_samples - n_a
    theta_a = rng.uniform(0, 2 * np.pi, n_a)
    theta_b = rng.uniform(0, 2 * np.pi, n_b)
    outer = np.stack([np.cos(theta_a), np.sin(theta_a)], axis=1)
    inner = factor * np.stack([np.cos(theta_b), np.sin(theta_b)], axis=1)
    x = np.concatenate([outer, inner]) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])
    return _split("circles", _rescale(_embed(x, n_features, rng)), y, 2, test_fraction, rng)


def parity(
    n_samples: int = 200,
    *,
    n_features: int = 4,
    flip_probability: float = 0.0,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Dataset:
    """Bitstring parity -- maximally non-local, and a natural quantum target.

    Every feature matters equally, so a model that has lost coherence across the
    register cannot fake it: accuracy collapses to chance rather than degrading
    gracefully.  That makes it the sharpest of these datasets for exposing noise.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, n_features))
    y = bits.sum(axis=1) % 2
    if flip_probability > 0:
        flips = rng.random(n_samples) < flip_probability
        y = np.where(flips, 1 - y, y)
    x = np.pi * (bits.astype(float) - 0.5)
    return _split("parity", x, y.astype(int), 2, test_fraction, rng)


def linearly_separable(
    n_samples: int = 200,
    *,
    n_features: int = 4,
    margin: float = 0.5,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Dataset:
    """An easy control task: whatever a model cannot do here is not the data's fault."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n_features)
    w /= np.linalg.norm(w)
    x = rng.normal(size=(n_samples, n_features))
    score = x @ w
    keep = np.abs(score) > margin
    x, score = x[keep], score[keep]
    y = (score > 0).astype(int)
    return _split("linearly_separable", _rescale(x), y, 2, test_fraction, rng)


DATASETS = {
    "two_moons": two_moons,
    "circles": circles,
    "parity": parity,
    "linearly_separable": linearly_separable,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    """Build a dataset by name, e.g. ``get_dataset("parity", n_features=4)``."""
    try:
        factory = DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}") from None
    return factory(**kwargs)
