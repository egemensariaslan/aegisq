"""Zero-noise extrapolators, written as differentiable ``nn.Module``s.

The design rests on one observation: for a *fixed* set of scale factors, both
Richardson and least-squares polynomial extrapolation to zero noise are linear
maps on the measured expectation values, with coefficients that depend only on
the scale factors.  Precomputing those coefficients once turns mitigation into a
single ``sum(c_i * E_i)`` -- a plain tensor contraction that autograd traverses
like any other, with no numerical fitting inside the training loop and no risk
of severing the graph.

The exponential model is genuinely non-linear, so it is expressed with
``torch.log``/``torch.exp`` instead; it too stays inside autograd.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

__all__ = [
    "Extrapolator",
    "RichardsonExtrapolator",
    "PolynomialExtrapolator",
    "LinearExtrapolator",
    "ExponentialExtrapolator",
    "EXTRAPOLATORS",
    "get_extrapolator",
]


def _validate_scale_factors(scale_factors: Sequence[float]) -> tuple[float, ...]:
    factors = tuple(float(s) for s in scale_factors)
    if len(factors) < 2:
        raise ValueError("zero-noise extrapolation needs at least two scale factors")
    if len(set(factors)) != len(factors):
        raise ValueError(f"scale factors must be distinct, got {factors}")
    if any(s < 1.0 for s in factors):
        raise ValueError(
            f"scale factors must be >= 1 (noise can be amplified, not removed), got {factors}"
        )
    return factors


class Extrapolator(nn.Module):
    """Maps a stack of noise-scaled results to the zero-noise estimate.

    ``forward`` takes a tensor of shape ``(n_scales, ...)`` and returns ``(...)``.
    """

    def __init__(self, scale_factors: Sequence[float]) -> None:
        super().__init__()
        self.scale_factors = _validate_scale_factors(scale_factors)

    def extra_repr(self) -> str:
        return f"scale_factors={self.scale_factors}"


class _LinearExtrapolator(Extrapolator):
    """Shared machinery for extrapolators that are a fixed linear combination."""

    def __init__(self, scale_factors: Sequence[float], coefficients: torch.Tensor) -> None:
        super().__init__(scale_factors)
        self.register_buffer("coefficients", coefficients.to(torch.float64), persistent=False)

    @property
    def noise_amplification(self) -> float:
        """``sum |c_i|`` -- the factor by which shot noise is amplified.

        Richardson extrapolation is unbiased in the noise model but not free:
        this number is how much sampling variance the mitigation costs, and it
        grows quickly with the number of scale factors.  Watch it when running
        with finite ``shots``.
        """
        return float(self.coefficients.abs().sum())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] != len(self.scale_factors):
            raise ValueError(
                f"expected leading dimension {len(self.scale_factors)} "
                f"(one entry per scale factor), got {values.shape[0]}"
            )
        coeffs = self.coefficients.to(values.dtype).reshape(-1, *([1] * (values.dim() - 1)))
        return (coeffs * values).sum(dim=0)


class RichardsonExtrapolator(_LinearExtrapolator):
    """Exact polynomial interpolation through every point, evaluated at zero noise.

    With ``n`` scale factors this fits the unique degree ``n-1`` polynomial and
    cancels the first ``n-1`` orders of the noise expansion.  It uses every data
    point and makes no fitting choices, which makes it the right default -- at
    the price of the largest :attr:`noise_amplification` of the family.
    """

    name = "richardson"

    def __init__(self, scale_factors: Sequence[float]) -> None:
        factors = _validate_scale_factors(scale_factors)
        # Lagrange basis polynomials evaluated at lambda = 0.
        coeffs = []
        for i, li in enumerate(factors):
            c = 1.0
            for j, lj in enumerate(factors):
                if i != j:
                    c *= lj / (lj - li)
            coeffs.append(c)
        super().__init__(factors, torch.tensor(coeffs, dtype=torch.float64))


class PolynomialExtrapolator(_LinearExtrapolator):
    """Least-squares polynomial fit of a chosen degree, evaluated at zero noise.

    A degree below ``len(scale_factors) - 1`` averages over the extra points
    instead of interpolating them, trading a little bias for markedly lower
    variance -- usually the better deal when the expectation values come from a
    finite shot budget.
    """

    name = "polynomial"

    def __init__(self, scale_factors: Sequence[float], *, degree: int = 2) -> None:
        factors = _validate_scale_factors(scale_factors)
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if degree > len(factors) - 1:
            raise ValueError(
                f"degree {degree} needs at least {degree + 1} scale factors, "
                f"got {len(factors)}"
            )
        self.degree = degree
        scales = torch.tensor(factors, dtype=torch.float64)
        vandermonde = torch.stack([scales**k for k in range(degree + 1)], dim=1)
        # Row 0 of the pseudo-inverse maps the data onto the constant term,
        # which is exactly the fitted value at lambda = 0.
        coeffs = torch.linalg.pinv(vandermonde)[0]
        super().__init__(factors, coeffs)

    def extra_repr(self) -> str:
        return f"scale_factors={self.scale_factors}, degree={self.degree}"


class LinearExtrapolator(PolynomialExtrapolator):
    """Degree-1 fit: the cheapest and most variance-tolerant extrapolator."""

    name = "linear"

    def __init__(self, scale_factors: Sequence[float]) -> None:
        super().__init__(scale_factors, degree=1)


class ExponentialExtrapolator(Extrapolator):
    """Fit ``E(lambda) = asymptote + B * exp(-C * lambda)``.

    Depolarizing noise damps an expectation value geometrically in the number of
    noisy gates, so an exponential is the physically motivated model, and it
    extrapolates far more gracefully than a polynomial when the noise is strong.
    The fit is linearised by regressing ``log|E - asymptote|`` on ``lambda``,
    which keeps it closed-form -- hence differentiable, hence safe inside a
    training loop.

    The default ``asymptote=0`` is correct for a traceless observable under
    depolarizing noise, whose expectation decays towards the maximally mixed
    value.

    Robustness
    ----------
    The log transform is only defined while ``E - asymptote`` keeps one sign.
    An expectation value that crosses zero between scale factors -- routine for
    a ``<Z_i>`` on a trained circuit -- would otherwise send the fitted
    intercept, and with it the estimate, to absurd values.  Each output element
    is therefore checked against the model's own assumptions (constant sign,
    magnitude decaying with noise) and falls back to a linear extrapolation
    where they fail.  :meth:`validity` reports the fraction of elements the
    exponential model actually handled.
    """

    name = "exponential"

    def __init__(
        self,
        scale_factors: Sequence[float],
        *,
        asymptote: float = 0.0,
        eps: float = 1e-9,
        max_amplification: float = 10.0,
    ) -> None:
        super().__init__(scale_factors)
        if max_amplification < 1.0:
            raise ValueError("max_amplification must be >= 1")
        self.asymptote = float(asymptote)
        self.eps = float(eps)
        self.max_amplification = float(max_amplification)
        scales = torch.tensor(self.scale_factors, dtype=torch.float64)
        design = torch.stack([torch.ones_like(scales), scales], dim=1)
        pinv = torch.linalg.pinv(design)
        # Rows 0 and 1 give the intercept and slope of log|E - asymptote|.
        self.register_buffer("_intercept_coeffs", pinv[0], persistent=False)
        self.register_buffer("_slope_coeffs", pinv[1], persistent=False)
        # Fallback used wherever the exponential model does not apply.
        self._fallback = LinearExtrapolator(self.scale_factors)
        # The least-noisy point is the best available anchor for the sign.
        self._anchor = int(min(range(len(scale_factors)), key=lambda i: self.scale_factors[i]))

    def _contract(self, coeffs: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        shaped = coeffs.to(values.dtype).reshape(-1, *([1] * (values.dim() - 1)))
        return (shaped * values).sum(dim=0)

    def _fit(self, values: torch.Tensor):
        centred = values - self.asymptote
        # Sign is piecewise constant; detaching it discards no gradient
        # information while keeping the log argument positive.
        sign = torch.sign(centred[self._anchor]).detach()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        magnitude = sign * centred
        log_magnitude = torch.log(torch.clamp(magnitude, min=self.eps))
        intercept = self._contract(self._intercept_coeffs, log_magnitude)
        slope = self._contract(self._slope_coeffs, log_magnitude)
        # A fit driven by a near-zero measurement can claim an arbitrarily large
        # zero-noise value. Bound the amplification relative to the least-noisy
        # point, in log space so nothing ever overflows.
        ceiling = log_magnitude[self._anchor] + math.log(self.max_amplification)
        # The model holds only where every point keeps the anchor's sign, the
        # magnitude does not *grow* with the noise, and the extrapolation stays
        # within that bound.
        valid = (magnitude > self.eps).all(dim=0) & (slope <= 0) & (intercept <= ceiling)
        return sign, torch.minimum(intercept, ceiling), valid

    def validity(self, values: torch.Tensor) -> float:
        """Fraction of output elements the exponential model was applicable to."""
        _, _, valid = self._fit(values)
        return float(valid.to(torch.float64).mean())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] != len(self.scale_factors):
            raise ValueError(
                f"expected leading dimension {len(self.scale_factors)} "
                f"(one entry per scale factor), got {values.shape[0]}"
            )
        sign, intercept, valid = self._fit(values)
        estimate = self.asymptote + sign * torch.exp(intercept)
        return torch.where(valid, estimate, self._fallback(values))

    def extra_repr(self) -> str:
        return f"scale_factors={self.scale_factors}, asymptote={self.asymptote}"


EXTRAPOLATORS: dict[str, type[Extrapolator]] = {
    "richardson": RichardsonExtrapolator,
    "polynomial": PolynomialExtrapolator,
    "linear": LinearExtrapolator,
    "exponential": ExponentialExtrapolator,
}


def get_extrapolator(
    name: str | Extrapolator | type[Extrapolator],
    scale_factors: Sequence[float],
    **kwargs,
) -> Extrapolator:
    """Build an extrapolator from a name, a class or an existing instance."""
    if isinstance(name, Extrapolator):
        return name
    if isinstance(name, type) and issubclass(name, Extrapolator):
        return name(scale_factors, **kwargs)
    try:
        cls = EXTRAPOLATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown extrapolator {name!r}; available: {sorted(EXTRAPOLATORS)}"
        ) from None
    return cls(scale_factors, **kwargs)
