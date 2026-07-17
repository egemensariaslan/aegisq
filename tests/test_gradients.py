"""Differentiability guarantees.

The library's central promise is that wrapping a circuit in a mitigation routine
never severs the autograd graph, and that PennyLane's parameter-shift rule keeps
agreeing with backprop once noise channels and folding are in play.  These tests
check that against independent references (finite differences, and the explicit
linear combination ZNE is supposed to compute) rather than merely asserting that
a gradient exists.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aegisq import QuantumLayer, SymmetryVerification, ZNE


def _finite_difference(module, x, index, eps=1e-4):
    """Central-difference derivative of ``sum(module(x))`` w.r.t. one weight."""
    with torch.no_grad():
        flat = module.weights.view(-1) if hasattr(module, "weights") else None
        assert flat is not None
        original = flat[index].clone()
        flat[index] = original + eps
        plus = float(module(x).sum())
        flat[index] = original - eps
        minus = float(module(x).sum())
        flat[index] = original
    return (plus - minus) / (2 * eps)


class TestLayerGradients:
    @pytest.mark.parametrize(
        "ansatz",
        ["local_entangler", "particle_conserving", "equivariant",
         "z2_equivariant", "basic_entangler", "strongly_entangling"],
    )
    def test_every_ansatz_produces_finite_non_zero_gradients(self, ansatz):
        layer = QuantumLayer(4, n_layers=2, ansatz=ansatz, noise="depolarizing", seed=0)
        out = layer(torch.linspace(-1, 1, layer.in_features))
        out.sum().backward()
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()
        assert layer.weights.grad.abs().max() > 0

    def test_backprop_matches_finite_differences_under_noise(self):
        layer = QuantumLayer(3, n_layers=2, noise="hardware_like", seed=0,
                             dtype=torch.float64)
        x = torch.tensor([0.4, -0.9, 1.1], dtype=torch.float64)
        layer(x).sum().backward()
        analytic = layer.weights.grad.view(-1)
        for index in (0, 3, analytic.numel() - 1):
            assert float(analytic[index]) == pytest.approx(
                _finite_difference(layer, x, index), abs=1e-5
            )

    def test_parameter_shift_agrees_with_backprop_on_a_noisy_circuit(self):
        """The critical compatibility claim: same circuit, same noise, two diff methods."""
        x = torch.tensor([0.3, -0.6, 0.9], dtype=torch.float64)
        grads = []
        for method in ("backprop", "parameter-shift"):
            layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=11,
                                 diff_method=method, dtype=torch.float64)
            layer(x).sum().backward()
            grads.append(layer.weights.grad.clone())
        assert torch.allclose(grads[0], grads[1], atol=1e-6)

    def test_gradient_survives_a_classical_sandwich(self):
        layer = QuantumLayer(4, seed=0, noise="depolarizing")
        model = nn.Sequential(nn.Linear(6, 4), layer, nn.Linear(4, 2))
        model(torch.randn(4, 6)).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())

    def test_training_reduces_the_loss(self):
        torch.manual_seed(0)
        layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=0)
        model = nn.Sequential(layer, nn.Linear(3, 2))
        x = torch.randn(24, 3)
        y = (x[:, 0] + x[:, 1] > 0).long()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            first = float(loss_fn(model(x), y))
        for _ in range(25):
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            assert float(loss_fn(model(x), y)) < first


class TestZNEGradients:
    @pytest.mark.parametrize("folding", ["global", "noise"])
    @pytest.mark.parametrize("extrapolate", ["richardson", "linear", "exponential"])
    def test_gradients_flow_through_mitigation(self, folding, extrapolate):
        layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=0)
        mitigated = ZNE(layer, scale_factors=(1, 2, 3), extrapolate=extrapolate,
                        folding=folding)
        out = mitigated(torch.tensor([0.5, -0.5, 1.0]))
        out.sum().backward()
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()
        assert layer.weights.grad.abs().max() > 0

    def test_richardson_gradient_equals_the_combination_of_scaled_gradients(self):
        """A linear extrapolator must differentiate as a linear map -- verified term by term."""
        x = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
        scales = (1.0, 2.0, 3.0)

        layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=3,
                             dtype=torch.float64)
        mitigated = ZNE(layer, scale_factors=scales)
        mitigated(x).sum().backward()
        combined = layer.weights.grad.clone()

        expected = torch.zeros_like(combined)
        coefficients = mitigated.extrapolator.coefficients
        for coefficient, scale in zip(coefficients, scales):
            layer.weights.grad = None
            layer.run_at_scale(x, scale, "global").sum().backward()
            expected += float(coefficient) * layer.weights.grad

        assert torch.allclose(combined, expected, atol=1e-9)

    def test_parameters_are_registered_once(self):
        layer = QuantumLayer(3, seed=0, noise="depolarizing")
        mitigated = ZNE(layer)
        names = [name for name, _ in mitigated.named_parameters()]
        assert names == ["layer.weights"]

    def test_mitigated_model_trains(self):
        torch.manual_seed(0)
        layer = QuantumLayer(3, n_layers=1, noise="depolarizing", seed=0)
        model = nn.Sequential(ZNE(layer, scale_factors=(1, 2)), nn.Linear(3, 2))
        x = torch.randn(16, 3)
        y = (x[:, 0] > 0).long()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            first = float(loss_fn(model(x), y))
        for _ in range(20):
            optimiser.zero_grad()
            loss_fn(model(x), y).backward()
            optimiser.step()
        with torch.no_grad():
            assert float(loss_fn(model(x), y)) < first

    def test_shot_based_mitigated_gradients_stay_finite(self):
        layer = QuantumLayer(2, n_layers=1, noise="depolarizing", shots=200, seed=0)
        mitigated = ZNE(layer, scale_factors=(1, 2))
        mitigated(torch.tensor([0.3, 0.8])).sum().backward()
        assert torch.isfinite(layer.weights.grad).all()


class TestSymmetryVerificationGradients:
    def _layer(self, **kwargs):
        return QuantumLayer(
            4, n_layers=2, ansatz="particle_conserving", encoding="excitation",
            measurement="probs", noise="depolarizing", seed=0, **kwargs
        )

    def test_gradients_flow_through_post_selection(self):
        layer = self._layer()
        verified = SymmetryVerification(layer)
        out = verified(torch.tensor([0.6, -0.2, 0.9]))
        # A uniform sum over <Z_i> is constant inside a fixed-particle sector,
        # so weight the outputs to get a cost that actually varies.
        (out * torch.tensor([1.0, -1.0, 0.5, 0.25])).sum().backward()
        assert layer.weights.grad is not None
        assert torch.isfinite(layer.weights.grad).all()
        assert layer.weights.grad.abs().max() > 0

    def test_composes_with_zne(self):
        layer = self._layer()
        verified = SymmetryVerification(layer)
        model = nn.Sequential(verified, nn.Linear(4, 2))
        model(torch.randn(3, layer.in_features)).sum().backward()
        assert torch.isfinite(layer.weights.grad).all()
