"""Zero-noise extrapolation as a ``torch.nn.Module``.

:class:`ZNE` wraps a :class:`~aegisq.layers.QuantumLayer`, runs it at several
amplified noise levels and combines the results into a zero-noise estimate.  It
is a *module*, not a decorator: the wrapped layer stays a submodule, so its
parameters register with the optimiser exactly once, ``state_dict`` round-trips,
and ``model.to(...)`` / ``model.train()`` propagate as usual.

Gradient flow is the whole point.  Each scaled forward pass returns a tensor
with a live ``grad_fn``; stacking them and applying the extrapolator adds only
ordinary tensor ops, so ``loss.backward()`` reaches the circuit parameters
through *every* scale factor at once.  A parameter-shift gradient is likewise
taken of the mitigated estimator rather than of one unmitigated pass.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from ..layers.quantum_layer import FoldingMode, QuantumLayer
from .extrapolation import Extrapolator, get_extrapolator

__all__ = ["ZNE", "zne"]


class ZNE(nn.Module):
    """Error-mitigated execution wrapper around a quantum layer.

    Parameters
    ----------
    layer:
        The :class:`~aegisq.layers.QuantumLayer` to mitigate.
    scale_factors:
        Noise amplification factors.  Must include ``1.0`` in practice -- the
        unamplified circuit is the anchor of the fit.
    extrapolate:
        ``"richardson"`` (default), ``"polynomial"``, ``"linear"``,
        ``"exponential"``, or an :class:`~aegisq.mitigation.Extrapolator`.
    folding:
        ``"global"`` amplifies noise by unitary folding, ``U -> U U^dag U``,
        which is how ZNE is done on real hardware and requires no knowledge of
        the noise model.  ``"noise"`` instead rescales the simulated channel
        rates directly -- the "virtual" variant: much cheaper, exact by
        construction, and available only on a simulator.

        The two modes mitigate different error budgets, and both are right about
        their own.  Folding lengthens the circuit, so it amplifies *gate* noise
        only; readout error happens once at measurement whatever the fold
        factor, and no amount of extrapolation will remove it -- exactly as on
        hardware, where readout error needs its own calibration. Virtual
        scaling amplifies readout error too, so it can extrapolate that away
        as well.
    clamp:
        Optional ``(low, high)`` bounds applied to the mitigated output.
        Extrapolation is an unbounded operation and can overshoot the physical
        range of an expectation value; clamping restores it at the cost of
        zeroing the gradient outside the range.

    Examples
    --------
    >>> import torch
    >>> from aegisq import QuantumLayer, ZNE
    >>> layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=0)
    >>> mitigated = ZNE(layer, scale_factors=(1, 2, 3))
    >>> out = mitigated(torch.randn(4, 3))
    >>> out.shape
    torch.Size([4, 3])
    """

    def __init__(
        self,
        layer: QuantumLayer,
        *,
        scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
        extrapolate: str | Extrapolator | type[Extrapolator] = "richardson",
        folding: FoldingMode = "global",
        clamp: tuple[float, float] | None = None,
        **extrapolator_kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(layer, nn.Module):
            raise TypeError(f"ZNE wraps an nn.Module layer, got {type(layer).__name__}")
        if not hasattr(layer, "run_at_scale"):
            raise TypeError(
                f"{type(layer).__name__} does not expose run_at_scale(); ZNE needs a layer "
                "that can execute at an amplified noise level (e.g. aegisq.QuantumLayer)"
            )
        if folding not in ("global", "noise"):
            raise ValueError(f"folding must be 'global' or 'noise', got {folding!r}")
        if clamp is not None and clamp[0] >= clamp[1]:
            raise ValueError(f"clamp bounds must satisfy low < high, got {clamp}")

        self.layer = layer
        self.folding: FoldingMode = folding
        self.clamp = clamp
        self.extrapolator = get_extrapolator(extrapolate, scale_factors, **extrapolator_kwargs)
        self.scale_factors = self.extrapolator.scale_factors

        if 1.0 not in self.scale_factors:
            raise ValueError(
                f"scale_factors should include 1.0 (the unamplified circuit); got "
                f"{self.scale_factors}"
            )
        encoding = getattr(layer, "encoding", None)
        if folding == "global" and encoding is not None and not encoding.foldable:
            raise ValueError(
                f"encoding {encoding.name!r} cannot be unitary-folded; "
                "use folding='noise' or a rotation-based encoding."
            )

    # ------------------------------------------------------------------
    @property
    def in_features(self) -> int:
        return self.layer.in_features

    @property
    def out_features(self) -> int:
        return self.layer.out_features

    @property
    def circuit_evaluations(self) -> int:
        """Circuit runs per forward pass -- the cost multiplier of mitigation."""
        return len(self.scale_factors)

    # ------------------------------------------------------------------
    def scaled_values(self, x: torch.Tensor) -> torch.Tensor:
        """The raw ``(n_scales, batch, out_features)`` stack before extrapolation.

        Useful for diagnostics: plotting these against the scale factors shows
        directly whether the noise model is in the regime the extrapolator
        assumes.
        """
        return torch.stack(
            [self.layer.run_at_scale(x, s, self.folding) for s in self.scale_factors],
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.extrapolator(self.scaled_values(x))
        if self.clamp is not None:
            out = out.clamp(self.clamp[0], self.clamp[1])
        return out

    def unmitigated(self, x: torch.Tensor) -> torch.Tensor:
        """The plain noisy forward pass, for a like-for-like comparison."""
        return self.layer(x)

    def extra_repr(self) -> str:
        bits = [f"folding={self.folding!r}", f"circuit_evaluations={self.circuit_evaluations}"]
        if self.clamp is not None:
            bits.append(f"clamp={self.clamp}")
        return ", ".join(bits)


def zne(layer: QuantumLayer, **kwargs) -> ZNE:
    """Functional shorthand for ``ZNE(layer, **kwargs)``."""
    return ZNE(layer, **kwargs)
