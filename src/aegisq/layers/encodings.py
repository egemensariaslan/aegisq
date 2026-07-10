"""Classical-data encodings.

Each encoding declares how many classical features it consumes for a given
register width and whether it survives unitary folding.  ``foldable`` is not a
detail: :class:`~aegisq.mitigation.ZNE` in ``folding="global"`` mode inverts the
whole circuit, and a state-preparation instruction has no inverse to apply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import pennylane as qml
import torch

from .ansatz import _all_bonds

__all__ = [
    "Encoding",
    "AngleEncoding",
    "DenseAngleEncoding",
    "IQPEncoding",
    "AmplitudeEncoding",
    "ExcitationEncoding",
]

_ROTATIONS = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}


class Encoding(ABC):
    """Base class for a data-encoding block."""

    name: str = "encoding"
    #: Symmetry the encoding respects, matched against the ansatz symmetry.
    symmetry: str | None = None
    #: Whether the block can be inverted by ``fold_global``.
    foldable: bool = True
    #: Whether the block may be re-applied between ansatz layers (data re-uploading).
    repeatable: bool = True

    @abstractmethod
    def in_features(self, n_wires: int) -> int:
        """Number of classical features consumed."""

    @abstractmethod
    def apply(self, x: torch.Tensor, wires: Sequence) -> None:
        """Queue the encoding gates for a ``(..., in_features)`` input."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class AngleEncoding(Encoding):
    """One rotation per wire: the standard, cheapest encoding.

    ``scaling`` multiplies the features before they become angles; keeping data
    inside roughly ``[-pi, pi]`` avoids the aliasing that makes distinct inputs
    collide.
    """

    name = "angle"

    def __init__(self, *, rotation: Literal["X", "Y", "Z"] = "Y", scaling: float = 1.0) -> None:
        rotation = rotation.upper()
        if rotation not in _ROTATIONS:
            raise ValueError(f"rotation must be one of X, Y, Z; got {rotation!r}")
        self.rotation = rotation
        self.scaling = scaling

    def in_features(self, n_wires: int) -> int:
        return n_wires

    def apply(self, x: torch.Tensor, wires: Sequence) -> None:
        gate = _ROTATIONS[self.rotation]
        for i, wire in enumerate(wires):
            gate(self.scaling * x[..., i], wires=wire)


class DenseAngleEncoding(Encoding):
    """Two features per wire, loaded as an ``RY``/``RZ`` pair.

    Doubles the classical capacity of the register at the cost of one extra
    single-qubit gate per wire -- cheap next to any two-qubit gate.
    """

    name = "dense_angle"

    def __init__(self, *, scaling: float = 1.0) -> None:
        self.scaling = scaling

    def in_features(self, n_wires: int) -> int:
        return 2 * n_wires

    def apply(self, x: torch.Tensor, wires: Sequence) -> None:
        n = len(wires)
        for i, wire in enumerate(wires):
            qml.RY(self.scaling * x[..., i], wires=wire)
        for i, wire in enumerate(wires):
            qml.RZ(self.scaling * x[..., n + i], wires=wire)


class IQPEncoding(Encoding):
    """Entangling IQP-style encoding (:class:`pennylane.IQPEmbedding`).

    Harder to simulate classically than product encodings, at the cost of a ring
    of two-qubit gates per repetition -- noticeably more noise-exposed, which is
    exactly the trade-off the benchmark harness is meant to quantify.
    """

    name = "iqp"

    def __init__(self, *, repeats: int = 1, scaling: float = 1.0) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.repeats = repeats
        self.scaling = scaling

    def in_features(self, n_wires: int) -> int:
        return n_wires

    def apply(self, x: torch.Tensor, wires: Sequence) -> None:
        qml.IQPEmbedding(self.scaling * x, wires=wires, n_repeats=self.repeats)


class AmplitudeEncoding(Encoding):
    """Load ``2**n`` features into the amplitudes of the register.

    Exponentially compact, but implemented as a state preparation: it is *not*
    foldable, so pair it with ``folding="noise"`` when using
    :class:`~aegisq.mitigation.ZNE`.
    """

    name = "amplitude"
    foldable = False
    repeatable = False

    def __init__(self, *, normalize: bool = True, pad_with: float = 0.0) -> None:
        self.normalize = normalize
        self.pad_with = pad_with

    def in_features(self, n_wires: int) -> int:
        return 2**n_wires

    def apply(self, x: torch.Tensor, wires: Sequence) -> None:
        qml.AmplitudeEmbedding(
            x, wires=wires, normalize=self.normalize, pad_with=self.pad_with
        )


class ExcitationEncoding(Encoding):
    """Particle-number-preserving encoding via Givens rotations.

    Prepares a reference state of Hamming weight ``n_particles`` using ``X``
    gates -- foldable, unlike ``BasisState`` -- then loads each feature into a
    :class:`pennylane.SingleExcitation` on a nearest-neighbour bond.  Combined
    with :class:`~aegisq.layers.ParticleConserving`, the *entire* circuit
    conserves particle number, so any measured leakage out of the sector is
    attributable to noise.
    """

    name = "excitation"
    symmetry = "particle_number"
    repeatable = False

    def __init__(
        self, *, n_particles: int | None = None, ring: bool = False, scaling: float = 1.0
    ) -> None:
        self.n_particles = n_particles
        self.ring = ring
        self.scaling = scaling

    def particles(self, n_wires: int) -> int:
        return self.n_particles if self.n_particles is not None else max(1, n_wires // 2)

    def in_features(self, n_wires: int) -> int:
        return len(_all_bonds(n_wires, ring=self.ring))

    def apply(self, x: torch.Tensor, wires: Sequence) -> None:
        n_wires = len(wires)
        k = self.particles(n_wires)
        if not 0 < k < n_wires:
            raise ValueError(
                f"n_particles must satisfy 0 < k < n_wires; got k={k}, n_wires={n_wires}"
            )
        for wire in wires[:k]:
            qml.PauliX(wires=wire)
        for idx, (i, j) in enumerate(_all_bonds(n_wires, ring=self.ring)):
            qml.SingleExcitation(self.scaling * x[..., idx], wires=[wires[i], wires[j]])
