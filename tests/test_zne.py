"""Extrapolator mathematics and end-to-end zero-noise extrapolation."""

from __future__ import annotations

import math

import pytest
import torch

from aegisq import QuantumLayer, ZNE
from aegisq.benchmark.metrics import mitigation_bias
from aegisq.mitigation import (
    ExponentialExtrapolator,
    LinearExtrapolator,
    PolynomialExtrapolator,
    RichardsonExtrapolator,
    get_extrapolator,
)


class TestExtrapolatorMath:
    def test_richardson_is_exact_on_polynomial_data(self):
        """Three points determine a quadratic exactly, so E(0) must be recovered."""
        scales = (1.0, 2.0, 3.0)
        poly = lambda s: 0.7 - 0.25 * s + 0.04 * s**2  # noqa: E731
        values = torch.tensor([[poly(s)] for s in scales], dtype=torch.float64)
        estimate = RichardsonExtrapolator(scales)(values)
        assert float(estimate[0]) == pytest.approx(poly(0.0), abs=1e-12)

    def test_linear_is_exact_on_linear_data(self):
        scales = (1.0, 3.0, 5.0)
        line = lambda s: -0.2 + 0.13 * s  # noqa: E731
        values = torch.tensor([[line(s)] for s in scales], dtype=torch.float64)
        assert float(LinearExtrapolator(scales)(values)[0]) == pytest.approx(-0.2, abs=1e-12)

    def test_polynomial_degree_matches_richardson_at_full_degree(self):
        scales = (1.0, 2.0, 4.0)
        values = torch.rand(3, 5, dtype=torch.float64)
        full = PolynomialExtrapolator(scales, degree=2)(values)
        assert torch.allclose(full, RichardsonExtrapolator(scales)(values), atol=1e-10)

    def test_exponential_is_exact_on_exponential_decay(self):
        scales = (1.0, 2.0, 3.0)
        decay = lambda s: 0.8 * math.exp(-0.35 * s)  # noqa: E731
        values = torch.tensor([[decay(s)] for s in scales], dtype=torch.float64)
        estimate = ExponentialExtrapolator(scales)(values)
        assert float(estimate[0]) == pytest.approx(0.8, abs=1e-10)

    def test_exponential_handles_negative_expectation_values(self):
        scales = (1.0, 2.0, 3.0)
        decay = lambda s: -0.6 * math.exp(-0.4 * s)  # noqa: E731
        values = torch.tensor([[decay(s)] for s in scales], dtype=torch.float64)
        assert float(ExponentialExtrapolator(scales)(values)[0]) == pytest.approx(
            -0.6, abs=1e-10
        )

    def test_exponential_falls_back_when_its_model_does_not_apply(self):
        """A sign change between scale factors breaks the log fit; the result must stay sane."""
        scales = (1.0, 2.0, 3.0)
        values = torch.tensor([[0.5], [-0.01], [0.2]], dtype=torch.float64)
        extrapolator = ExponentialExtrapolator(scales)
        estimate = extrapolator(values)
        assert torch.isfinite(estimate).all()
        assert abs(float(estimate[0])) < 5.0
        assert extrapolator.validity(values) == 0.0

    def test_exponential_reports_full_validity_on_clean_decay(self):
        scales = (1.0, 2.0, 3.0)
        values = torch.tensor([[0.8, 0.5], [0.6, 0.4], [0.45, 0.32]], dtype=torch.float64)
        assert ExponentialExtrapolator(scales).validity(values) == 1.0

    def test_richardson_coefficients_sum_to_one(self):
        """The zero-noise estimate must reproduce a constant signal exactly."""
        for scales in [(1.0, 2.0), (1.0, 2.0, 3.0), (1.0, 3.0, 5.0, 7.0)]:
            assert float(RichardsonExtrapolator(scales).coefficients.sum()) == pytest.approx(
                1.0, abs=1e-10
            )

    def test_noise_amplification_grows_with_more_scale_factors(self):
        few = RichardsonExtrapolator((1.0, 2.0)).noise_amplification
        many = RichardsonExtrapolator((1.0, 2.0, 3.0, 4.0, 5.0)).noise_amplification
        assert many > few > 1.0

    def test_batched_and_multi_output_shapes_are_preserved(self):
        values = torch.rand(3, 7, 4)
        assert RichardsonExtrapolator((1.0, 2.0, 3.0))(values).shape == (7, 4)

    def test_rejects_bad_scale_factors(self):
        with pytest.raises(ValueError, match="at least two"):
            RichardsonExtrapolator((1.0,))
        with pytest.raises(ValueError, match="distinct"):
            RichardsonExtrapolator((1.0, 1.0))
        with pytest.raises(ValueError, match=">= 1"):
            RichardsonExtrapolator((0.5, 1.0))

    def test_rejects_wrong_stack_size(self):
        with pytest.raises(ValueError, match="leading dimension"):
            RichardsonExtrapolator((1.0, 2.0))(torch.rand(3, 2))

    def test_degree_must_fit_the_data(self):
        with pytest.raises(ValueError, match="needs at least"):
            PolynomialExtrapolator((1.0, 2.0), degree=2)

    def test_get_extrapolator_accepts_name_class_and_instance(self):
        instance = LinearExtrapolator((1.0, 2.0))
        assert isinstance(get_extrapolator("richardson", (1.0, 2.0)), RichardsonExtrapolator)
        assert isinstance(
            get_extrapolator(LinearExtrapolator, (1.0, 2.0)), LinearExtrapolator
        )
        assert get_extrapolator(instance, (1.0, 2.0)) is instance
        with pytest.raises(KeyError, match="available"):
            get_extrapolator("nope", (1.0, 2.0))


class TestZNEModule:
    def test_shapes_and_passthrough_properties(self):
        layer = QuantumLayer(3, noise="depolarizing", seed=0)
        mitigated = ZNE(layer, scale_factors=(1, 2, 3))
        assert mitigated.in_features == layer.in_features
        assert mitigated.out_features == layer.out_features
        assert mitigated.circuit_evaluations == 3
        assert mitigated(torch.zeros(4, 3)).shape == (4, 3)

    def test_scaled_values_are_ordered_by_noise(self):
        """Folding must actually amplify the noise, or there is nothing to extrapolate from."""
        layer = QuantumLayer(3, n_layers=2, noise="depolarizing", seed=0)
        mitigated = ZNE(layer, scale_factors=(1, 2, 3))
        with torch.no_grad():
            values = mitigated.scaled_values(torch.tensor([0.5, -0.3, 0.8]))
        magnitudes = values.abs().sum(dim=-1)
        assert magnitudes[0] > magnitudes[1] > magnitudes[2]

    @pytest.mark.parametrize("folding", ["global", "noise"])
    def test_mitigation_moves_towards_the_noiseless_value(self, folding):
        x = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
        ideal = QuantumLayer(4, n_layers=3, noise=None, seed=7)
        noisy = QuantumLayer(4, n_layers=3, noise=0.01, seed=7)
        assert torch.allclose(ideal.weights, noisy.weights)
        with torch.no_grad():
            report = mitigation_bias(
                ZNE(noisy, scale_factors=(1, 2, 3), folding=folding)(x),
                noisy(x),
                ideal(x),
            )
        assert report["bias_reduction"] > 0.5

    def test_folding_amplifies_gate_noise_only(self):
        """Readout error happens once at measurement, so folding cannot reach it."""
        from aegisq.noise import readout

        layer = QuantumLayer(3, n_layers=2, noise=readout(0.05), seed=0)
        x = torch.tensor([0.5, -0.3, 0.8])
        with torch.no_grad():
            folded = ZNE(layer, scale_factors=(1, 2, 3), folding="global").scaled_values(x)
            virtual = ZNE(layer, scale_factors=(1, 2, 3), folding="noise").scaled_values(x)
        assert torch.allclose(folded[0], folded[1], atol=1e-6)
        assert torch.allclose(folded[0], folded[2], atol=1e-6)
        assert not torch.allclose(virtual[0], virtual[2], atol=1e-6)

    def test_clamp_bounds_the_output(self):
        layer = QuantumLayer(3, n_layers=3, noise=0.05, seed=0)
        mitigated = ZNE(layer, scale_factors=(1, 3, 5), clamp=(-1.0, 1.0))
        with torch.no_grad():
            out = mitigated(torch.randn(6, 3))
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_unmitigated_matches_the_bare_layer(self):
        layer = QuantumLayer(3, noise="depolarizing", seed=0)
        mitigated = ZNE(layer)
        x = torch.tensor([0.1, 0.2, 0.3])
        with torch.no_grad():
            assert torch.allclose(mitigated.unmitigated(x), layer(x))

    def test_requires_the_unamplified_circuit(self):
        layer = QuantumLayer(2, noise="depolarizing", seed=0)
        with pytest.raises(ValueError, match="should include 1.0"):
            ZNE(layer, scale_factors=(2, 3))

    def test_rejects_unknown_folding_mode(self):
        layer = QuantumLayer(2, noise="depolarizing", seed=0)
        with pytest.raises(ValueError, match="folding must be"):
            ZNE(layer, folding="magic")

    def test_state_preparation_encodings_cannot_be_folded(self):
        layer = QuantumLayer(3, encoding="amplitude", noise="depolarizing", seed=0)
        with pytest.raises(ValueError, match="cannot be unitary-folded"):
            ZNE(layer, folding="global")

    def test_state_preparation_encodings_work_with_virtual_scaling(self):
        layer = QuantumLayer(3, encoding="amplitude", noise="depolarizing", seed=0)
        mitigated = ZNE(layer, folding="noise", scale_factors=(1, 2))
        assert mitigated(torch.randn(2, layer.in_features)).shape == (2, 3)

    def test_virtual_scaling_needs_a_noise_profile(self):
        layer = QuantumLayer(2, noise=None, seed=0)
        with pytest.raises(ValueError, match="has no noise profile"):
            ZNE(layer, folding="noise", scale_factors=(1, 2))(torch.zeros(2))

    def test_rejects_a_layer_that_cannot_scale_its_noise(self):
        with pytest.raises(TypeError, match="run_at_scale"):
            ZNE(torch.nn.Linear(2, 2))
