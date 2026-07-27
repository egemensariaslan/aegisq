"""Symmetry verification: mitigation that costs nothing extra to run.

If every gate in a circuit conserves a quantity, then any measured violation of
that quantity is noise -- there is no other explanation.  :class:`SymmetryVerification`
turns that statement into an estimator: it discards the probability mass sitting
outside the allowed sector and renormalises what remains.

Unlike zero-noise extrapolation this needs no extra circuit executions, only a
richer measurement (the basis-state distribution instead of a handful of
expectation values).  The two techniques compose: verify the symmetry, then
extrapolate what the symmetry could not catch.

Requires a layer whose circuit actually has the symmetry -- pair
:class:`~aegisq.layers.ParticleConserving` with ``encoding="excitation"`` and
``measurement="probs"``.
"""

from __future__ import annotations

import warnings
from typing import Literal

import torch
from torch import nn

from ..layers.quantum_layer import QuantumLayer

__all__ = ["SymmetryVerification", "sector_mask", "SymmetryName"]

SymmetryName = Literal["particle_number", "parity"]


def sector_mask(
    n_qubits: int,
    symmetry: SymmetryName,
    *,
    n_particles: int | None = None,
    parity: int = 0,
) -> torch.Tensor:
    """Boolean mask over the ``2**n`` basis states allowed by ``symmetry``.

    Index ``b`` follows PennyLane's convention: wire 0 is the most significant bit.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    indices = torch.arange(2**n_qubits)
    popcount = torch.zeros_like(indices)
    for bit in range(n_qubits):
        popcount = popcount + ((indices >> bit) & 1)
    if symmetry == "particle_number":
        if n_particles is None:
            raise ValueError("particle_number verification requires n_particles")
        if not 0 <= n_particles <= n_qubits:
            raise ValueError(f"n_particles must lie in [0, {n_qubits}], got {n_particles}")
        return popcount == n_particles
    if symmetry == "parity":
        if parity not in (0, 1):
            raise ValueError(f"parity must be 0 or 1, got {parity}")
        return (popcount % 2) == parity
    raise ValueError(f"unknown symmetry {symmetry!r}")


def _z_table(n_qubits: int) -> torch.Tensor:
    """``(2**n, n)`` table of ``<Z_i>`` eigenvalues for each basis state."""
    indices = torch.arange(2**n_qubits)
    bits = torch.stack(
        [(indices >> (n_qubits - 1 - wire)) & 1 for wire in range(n_qubits)], dim=1
    )
    return 1.0 - 2.0 * bits.to(torch.float64)


class SymmetryVerification(nn.Module):
    """Post-select a layer's output onto its symmetry sector.

    The wrapped layer must return the full basis-state distribution
    (``measurement="probs"``).  The module reports the same ``<Z_i>`` vector a
    :class:`~aegisq.layers.LocalZ` measurement would, but computed from the
    renormalised in-sector distribution.

    Parameters
    ----------
    layer:
        A :class:`~aegisq.layers.QuantumLayer` measuring probabilities.
    symmetry:
        ``"particle_number"`` or ``"parity"``.  Inferred from the layer when the
        encoding and ansatz agree on a conserved quantity.
    n_particles:
        Sector to keep for particle-number verification.  Inferred from an
        :class:`~aegisq.layers.ExcitationEncoding` when available.
    eps:
        Floor on the retained probability mass, guarding the renormalisation
        against a division by zero at extreme noise.

    Examples
    --------
    >>> import torch
    >>> from aegisq import QuantumLayer, SymmetryVerification
    >>> layer = QuantumLayer(
    ...     4, n_layers=2, ansatz="particle_conserving", encoding="excitation",
    ...     measurement="probs", noise="depolarizing", seed=0,
    ... )
    >>> verified = SymmetryVerification(layer)
    >>> verified(torch.randn(2, layer.in_features)).shape
    torch.Size([2, 4])
    """

    def __init__(
        self,
        layer: QuantumLayer,
        *,
        symmetry: SymmetryName | None = None,
        n_particles: int | None = None,
        parity: int = 0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if not getattr(layer.measurement, "returns_probabilities", False):
            raise ValueError(
                "SymmetryVerification needs the basis-state distribution; build the layer "
                "with measurement='probs'."
            )
        symmetry = symmetry or self._infer_symmetry(layer)
        if symmetry == "particle_number" and n_particles is None:
            n_particles = self._infer_particles(layer)

        if layer.noise.is_noiseless:
            warnings.warn(
                "SymmetryVerification on a noiseless layer is a no-op: there is no "
                "leakage to remove. It is also the one configuration where PennyLane's "
                "probability backprop can return NaN gradients at exactly-zero "
                "probabilities.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.layer = layer
        self.symmetry: SymmetryName = symmetry
        self.n_particles = n_particles
        self.parity = parity
        self.eps = float(eps)

        mask = sector_mask(
            layer.n_qubits, symmetry, n_particles=n_particles, parity=parity
        ).to(torch.float64)
        self.register_buffer("_mask", mask, persistent=False)
        self.register_buffer("_z", _z_table(layer.n_qubits), persistent=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_symmetry(layer: QuantumLayer) -> SymmetryName:
        symmetry = layer.symmetry
        if symmetry == "particle_number":
            return "particle_number"
        raise ValueError(
            "could not infer a conserved quantity for this layer. Pair a symmetry-preserving "
            "ansatz with a matching encoding (e.g. ansatz='particle_conserving', "
            "encoding='excitation'), or pass symmetry= explicitly."
        )

    @staticmethod
    def _infer_particles(layer: QuantumLayer) -> int:
        particles = getattr(layer.encoding, "particles", None)
        if particles is None:
            raise ValueError(
                "n_particles could not be inferred from the encoding; pass it explicitly."
            )
        return particles(layer.n_qubits)

    # ------------------------------------------------------------------
    @property
    def in_features(self) -> int:
        return self.layer.in_features

    @property
    def out_features(self) -> int:
        return self.layer.n_qubits

    def sector_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Probability mass surviving the symmetry check, in ``[0, 1]``.

        A direct, calibration-free readout of how much the circuit was corrupted:
        ``1.0`` means no detectable error, and the post-selection overhead of the
        estimator is ``1 / sector_weight``.
        """
        probs = self.layer(x)
        return (probs * self._mask.to(probs.dtype)).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.layer(x)
        mask = self._mask.to(probs.dtype)
        z = self._z.to(probs.dtype)
        kept = probs * mask
        weight = kept.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return (kept @ z) / weight

    def extra_repr(self) -> str:
        bits = [f"symmetry={self.symmetry!r}"]
        if self.symmetry == "particle_number":
            bits.append(f"n_particles={self.n_particles}")
        else:
            bits.append(f"parity={self.parity}")
        return ", ".join(bits)
