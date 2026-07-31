"""Trainability and fidelity diagnostics.

Accuracy alone hides the failure mode that matters on NISQ hardware: a model can
score well while its gradients are already too small to survive sampling noise.
:func:`gradient_variance` measures that directly, and
:func:`barren_plateau_scan` traces how it scales with register width -- the
signature that separates a merely hard landscape from an exponentially flat one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from ..layers.quantum_layer import QuantumLayer

__all__ = [
    "accuracy",
    "gradient_variance",
    "GradientStats",
    "barren_plateau_scan",
    "mitigation_bias",
]


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct predictions for class ``logits`` of shape ``(batch, n_classes)``."""
    if logits.dim() == 1 or logits.shape[-1] == 1:
        predictions = (logits.reshape(-1) > 0).long()
    else:
        predictions = logits.argmax(dim=-1)
    return float((predictions == targets.reshape(-1)).to(torch.float64).mean())


@dataclass
class GradientStats:
    """Sampling statistics of circuit-parameter gradients over random initialisations."""

    #: Headline trainability number: the mean per-parameter gradient variance,
    #: or the variance of a single parameter when ``param_index`` was given.
    variance: float
    #: Largest per-parameter variance -- the best-case direction.
    max_variance: float
    mean: float
    abs_mean: float
    n_samples: int
    n_qubits: int
    n_layers: int
    n_parameters: int
    per_parameter_variance: list[float] = field(default_factory=list, repr=False)

    @property
    def std(self) -> float:
        return self.variance**0.5

    @property
    def dead_parameter_fraction(self) -> float:
        """Share of angles whose gradient never moves -- structurally decoupled, not flat.

        A rotation that commutes through everything downstream of it (a final
        ``RZ`` before a ``Z`` measurement, say) has an identically zero
        derivative for algebraic reasons.  Those angles are not evidence of a
        barren plateau, and this number says how much of the circuit they are.
        """
        if not self.per_parameter_variance:
            return 0.0
        dead = sum(1 for v in self.per_parameter_variance if v <= 1e-16)
        return dead / len(self.per_parameter_variance)


def gradient_variance(
    layer_factory: Callable[[], torch.nn.Module],
    *,
    n_samples: int = 30,
    param_index: int | tuple[int, ...] | None = None,
    x: torch.Tensor | None = None,
    cost: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0,
) -> GradientStats:
    """Variance of one parameter's gradient over random weight initialisations.

    ``layer_factory`` is called once per sample and must return a freshly
    initialised module (the randomness has to come from the initialisation, not
    from re-using one set of weights).  The default cost is the mean output,
    which for a ``local_z`` measurement is the average local magnetisation.

    A variance that shrinks exponentially with qubit count is a barren plateau;
    one that shrinks polynomially is merely a hard optimisation problem.

    With ``param_index`` left at ``None`` the reported variance is the mean over
    *all* circuit angles.  Pinning one index instead -- the classic formulation --
    is available but treacherous: a rotation that commutes through everything
    downstream of it has an identically zero derivative for algebraic reasons,
    and singling one out would report a barren plateau where there is only a
    symmetry.  Averaging keeps the measurement honest, and
    :attr:`GradientStats.dead_parameter_fraction` says how many such angles the
    circuit has.
    """
    if n_samples < 2:
        raise ValueError("gradient_variance needs at least two samples")
    torch.manual_seed(seed)
    cost = cost or (lambda out: out.mean())

    rows: list[torch.Tensor] = []
    reference: torch.nn.Module | None = None
    for _ in range(n_samples):
        module = layer_factory()
        reference = module
        weights = _circuit_weights(module)
        sample_x = x
        if sample_x is None:
            sample_x = torch.zeros(_in_features(module), dtype=weights.dtype)
        (grad,) = torch.autograd.grad(cost(module(sample_x)), weights)
        if param_index is None:
            row = grad.reshape(-1)
        elif isinstance(param_index, int):
            row = grad.reshape(-1)[param_index].reshape(1)
        else:
            row = grad[tuple(param_index)].reshape(1)
        rows.append(row.detach().to(torch.float64))

    samples = torch.stack(rows)  # (n_samples, n_selected_parameters)
    per_parameter = samples.var(dim=0, unbiased=True)
    inner = _inner_layer(reference)
    return GradientStats(
        variance=float(per_parameter.mean()),
        max_variance=float(per_parameter.max()),
        mean=float(samples.mean()),
        abs_mean=float(samples.abs().mean()),
        n_samples=n_samples,
        n_qubits=inner.n_qubits,
        n_layers=inner.n_layers,
        n_parameters=inner.n_circuit_parameters,
        per_parameter_variance=[float(v) for v in per_parameter],
    )


def _inner_layer(module: torch.nn.Module | None) -> QuantumLayer:
    """Find the :class:`QuantumLayer` inside a possibly-wrapped module."""
    if module is None:
        raise ValueError("no module was produced by the factory")
    if isinstance(module, QuantumLayer):
        return module
    for child in module.modules():
        if isinstance(child, QuantumLayer):
            return child
    raise TypeError(f"{type(module).__name__} contains no QuantumLayer")


def _circuit_weights(module: torch.nn.Module) -> torch.Tensor:
    return _inner_layer(module).weights


def _in_features(module: torch.nn.Module) -> int:
    return _inner_layer(module).in_features


def barren_plateau_scan(
    qubit_counts: Sequence[int],
    *,
    n_layers: int = 4,
    n_samples: int = 30,
    seed: int = 0,
    **layer_kwargs,
) -> dict[int, GradientStats]:
    """Gradient variance as a function of register width.

    Every keyword is forwarded to :class:`~aegisq.layers.QuantumLayer`, so a
    single call compares e.g. ``measurement="local_z"`` against
    ``measurement="global_z"`` on identical circuits.

    Examples
    --------
    >>> scan = barren_plateau_scan([2, 3], n_layers=2, n_samples=5,
    ...                            ansatz="equivariant")
    >>> sorted(scan)
    [2, 3]
    """
    results: dict[int, GradientStats] = {}
    for n_qubits in qubit_counts:
        kwargs = dict(layer_kwargs)
        kwargs.pop("seed", None)

        def factory(n: int = n_qubits, kw: dict = kwargs) -> QuantumLayer:
            return QuantumLayer(n, n_layers=n_layers, **kw)

        results[n_qubits] = gradient_variance(factory, n_samples=n_samples, seed=seed)
    return results


def mitigation_bias(
    mitigated: torch.Tensor, noisy: torch.Tensor, ideal: torch.Tensor
) -> dict[str, float]:
    """Compare a mitigated estimate against the noisy and noiseless references.

    Returns the mean absolute error of each estimator plus ``bias_reduction``,
    the fraction of the noisy circuit's error that mitigation removed.  A value
    of ``1.0`` is perfect recovery; a negative value means the mitigation made
    matters worse, which does happen when the extrapolation model is wrong for
    the noise at hand.
    """
    noisy_error = float((noisy - ideal).abs().mean())
    mitigated_error = float((mitigated - ideal).abs().mean())
    reduction = 1.0 - mitigated_error / noisy_error if noisy_error > 0 else 0.0
    return {
        "noisy_error": noisy_error,
        "mitigated_error": mitigated_error,
        "bias_reduction": reduction,
    }
