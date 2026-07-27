"""Symmetry-preserving circuits and symmetry-verification mitigation."""

from __future__ import annotations

import warnings

import pytest
import torch

from aegisq import QuantumLayer, SymmetryVerification
from aegisq.mitigation.symmetry import sector_mask


def _layer(**kwargs):
    defaults = dict(
        n_layers=2,
        ansatz="particle_conserving",
        encoding="excitation",
        measurement="probs",
        noise="depolarizing",
        seed=0,
    )
    defaults.update(kwargs)
    return QuantumLayer(4, **defaults)


class TestSectorMask:
    def test_particle_number_selects_the_right_hamming_weight(self):
        mask = sector_mask(4, "particle_number", n_particles=2)
        assert int(mask.sum()) == 6  # C(4, 2)
        assert all(bin(i).count("1") == 2 for i, keep in enumerate(mask) if keep)

    def test_parity_splits_the_space_in_half(self):
        assert int(sector_mask(4, "parity", parity=0).sum()) == 8
        assert int(sector_mask(4, "parity", parity=1).sum()) == 8

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="requires n_particles"):
            sector_mask(3, "particle_number")
        with pytest.raises(ValueError, match=r"must lie in \[0, 3\]"):
            sector_mask(3, "particle_number", n_particles=9)
        with pytest.raises(ValueError, match="unknown symmetry"):
            sector_mask(3, "spin")


class TestSymmetryPreservation:
    def test_noiseless_circuit_never_leaves_its_sector(self):
        layer = _layer(noise=None, n_layers=4)
        with torch.no_grad():
            probs = layer(torch.tensor([1.1, -0.4, 0.9]))
        leakage = float(probs[[i for i in range(16) if bin(i).count("1") != 2]].sum())
        assert leakage < 1e-9

    def test_noise_is_what_causes_leakage(self):
        clean = _layer(noise=None, n_layers=3)
        noisy = _layer(noise=0.02, n_layers=3)
        x = torch.tensor([0.8, 0.2, -0.5])
        with torch.no_grad():
            out_of_sector = [i for i in range(16) if bin(i).count("1") != 2]
            assert float(clean(x)[out_of_sector].sum()) < 1e-9
            assert float(noisy(x)[out_of_sector].sum()) > 0.01

    def test_angle_encoding_forfeits_the_symmetry_claim(self):
        assert _layer(encoding="angle").symmetry is None
        assert _layer().symmetry == "particle_number"


class TestSymmetryVerification:
    def test_output_shape_and_range(self):
        verified = SymmetryVerification(_layer())
        out = verified(torch.randn(5, 3))
        assert out.shape == (5, 4)
        assert out.min() >= -1.0 - 1e-9 and out.max() <= 1.0 + 1e-9

    def test_infers_symmetry_and_particle_count(self):
        verified = SymmetryVerification(_layer())
        assert verified.symmetry == "particle_number"
        assert verified.n_particles == 2

    def test_sector_weight_detects_corruption(self):
        x = torch.tensor([0.5, -0.9, 0.3])
        with torch.no_grad():
            light = SymmetryVerification(_layer(noise=0.002)).sector_weight(x)
            heavy = SymmetryVerification(_layer(noise=0.05)).sector_weight(x)
        assert 0.0 < float(heavy) < float(light) <= 1.0

    def test_verification_recovers_the_noiseless_expectation(self):
        """Post-selection should land closer to the ideal value than the raw noisy one."""
        x = torch.randn(6, 3, generator=torch.Generator().manual_seed(1))
        with warnings.catch_warnings():  # the noiseless reference is a deliberate no-op
            warnings.simplefilter("ignore", RuntimeWarning)
            ideal = SymmetryVerification(_layer(noise=None))
        noisy_layer = _layer(noise=0.02)
        verified = SymmetryVerification(noisy_layer)
        z = verified._z.to(torch.get_default_dtype())
        with torch.no_grad():
            reference = ideal(x)
            unmitigated = noisy_layer(x) @ z  # plain <Z_i>, no post-selection
            mitigated = verified(x)
        assert (mitigated - reference).abs().mean() < (unmitigated - reference).abs().mean()

    def test_requires_a_probability_measurement(self):
        with pytest.raises(ValueError, match="measurement='probs'"):
            SymmetryVerification(_layer(measurement="local_z"))

    def test_reports_when_no_symmetry_can_be_inferred(self):
        with pytest.raises(ValueError, match="could not infer"):
            SymmetryVerification(_layer(encoding="angle", ansatz="local_entangler"))

    def test_warns_on_a_noiseless_layer(self):
        with pytest.warns(RuntimeWarning, match="no-op"):
            SymmetryVerification(_layer(noise=None))

    def test_explicit_parity_verification(self):
        layer = QuantumLayer(3, n_layers=1, measurement="probs", noise=0.01, seed=0)
        verified = SymmetryVerification(layer, symmetry="parity", parity=0)
        assert verified(torch.zeros(2, 3)).shape == (2, 3)
