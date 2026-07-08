"""Noise specification, scaling semantics and PennyLane realisation."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
import torch

from aegisq.noise import NoiseSpec, get_preset, hardware_like, noiseless
from aegisq.noise.spec import _scale_damping, _scale_depolarizing, _scale_flip


class TestValidation:
    def test_rejects_out_of_range_probability(self):
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            NoiseSpec(depolarizing_1q=1.5)

    def test_rejects_half_specified_thermal(self):
        with pytest.raises(ValueError, match="both t1 and t2"):
            NoiseSpec(t1=100.0)

    def test_rejects_t2_above_2t1(self):
        with pytest.raises(ValueError, match="t2 must not exceed"):
            NoiseSpec(t1=100.0, t2=300.0)

    def test_rejects_negative_scale_factor(self):
        with pytest.raises(ValueError, match="non-negative"):
            NoiseSpec(depolarizing_1q=0.1).scaled(-1.0)


class TestScaling:
    def test_noiseless_stays_noiseless(self):
        assert noiseless().scaled(5.0).is_noiseless

    def test_scale_one_is_identity(self):
        spec = hardware_like()
        assert spec.scaled(1.0) is spec

    def test_composed_scaling_matches_channel_composition(self):
        """The exact rate of two composed depolarizing channels, checked on a density matrix."""
        p = 0.07
        p2 = _scale_depolarizing(p, 2.0)

        def evolve(rates):
            dev = qml.device("default.mixed", wires=1)

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(0)
                for rate in rates:
                    qml.DepolarizingChannel(rate, wires=0)
                return qml.expval(qml.PauliX(0))

            return float(circuit())

        assert evolve([p, p]) == pytest.approx(evolve([p2]), abs=1e-12)

    def test_composed_scaling_stays_physical(self):
        for factor in (1.0, 3.0, 10.0, 100.0):
            assert 0.0 <= _scale_depolarizing(0.4, factor) <= 0.75 + 1e-12
            assert 0.0 <= _scale_flip(0.3, factor) <= 0.5 + 1e-12
            assert 0.0 <= _scale_damping(0.3, factor) <= 1.0 + 1e-12

    def test_linear_mode_is_first_order(self):
        spec = NoiseSpec(depolarizing_1q=0.01, scale_mode="linear")
        assert spec.scaled(3.0).depolarizing_1q == pytest.approx(0.03)

    def test_composed_and_linear_agree_to_first_order(self):
        p = 1e-5
        composed = NoiseSpec(depolarizing_1q=p).scaled(3.0).depolarizing_1q
        linear = NoiseSpec(depolarizing_1q=p, scale_mode="linear").scaled(3.0).depolarizing_1q
        assert composed == pytest.approx(linear, rel=1e-3)

    def test_scaling_is_monotone(self):
        spec = hardware_like()
        strengths = [spec.scaled(f).strength for f in (1.0, 2.0, 3.0, 5.0)]
        assert strengths == sorted(strengths)

    def test_thermal_scaling_lengthens_the_gate(self):
        spec = get_preset("thermal_relaxation")
        assert spec.scaled(3.0).gate_time == pytest.approx(spec.gate_time * 3.0)

    def test_mul_operator(self):
        spec = NoiseSpec(depolarizing_1q=0.01, scale_mode="linear")
        assert (spec * 2.0).depolarizing_1q == pytest.approx(0.02)
        assert (2.0 * spec).depolarizing_1q == pytest.approx(0.02)


class TestNoiseModel:
    def test_noiseless_builds_no_model(self):
        assert noiseless().to_noise_model() is None

    def test_model_applies_channels_and_damps_expectation(self):
        spec = NoiseSpec(depolarizing_1q=0.1, depolarizing_2q=0.2)
        dev = qml.device("default.mixed", wires=2)

        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        clean = qml.QNode(circuit, dev)
        noisy = qml.add_noise(clean, spec.to_noise_model())
        assert float(clean()) == pytest.approx(1.0, abs=1e-9)
        assert 0.0 < float(noisy()) < 1.0

    def test_readout_noise_biases_measurement_only(self):
        spec = NoiseSpec(readout=0.1)
        dev = qml.device("default.mixed", wires=1)

        def circuit():
            return qml.expval(qml.PauliZ(0))

        noisy = qml.add_noise(qml.QNode(circuit, dev), spec.to_noise_model())
        # |0> with a 10% bit flip gives <Z> = 0.9*1 + 0.1*(-1).
        assert float(noisy()) == pytest.approx(0.8, abs=1e-9)

    def test_two_qubit_gates_accumulate_error_on_both_wires(self):
        spec = NoiseSpec(depolarizing_2q=0.2)
        dev = qml.device("default.mixed", wires=2)

        def circuit():
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        noisy = qml.add_noise(qml.QNode(circuit, dev), spec.to_noise_model())
        z0, z1 = (float(v) for v in noisy())
        assert z0 < 1.0 and z1 < 1.0

    def test_state_preparation_is_not_noisy(self):
        """State prep models an idealised reset; folding cannot invert it either."""
        spec = NoiseSpec(depolarizing_1q=0.3)
        dev = qml.device("default.mixed", wires=1)

        def circuit():
            qml.BasisState(np.array([1]), wires=[0])
            return qml.expval(qml.PauliZ(0))

        noisy = qml.add_noise(qml.QNode(circuit, dev), spec.to_noise_model())
        assert float(noisy()) == pytest.approx(-1.0, abs=1e-9)


class TestPresets:
    @pytest.mark.parametrize(
        "name",
        ["noiseless", "depolarizing", "dephasing", "amplitude_damping",
         "thermal_relaxation", "readout", "hardware_like"],
    )
    def test_every_preset_builds(self, name):
        spec = get_preset(name)
        assert isinstance(spec, NoiseSpec)
        assert isinstance(spec.describe(), str)

    def test_unknown_preset_lists_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            get_preset("does_not_exist")

    def test_hardware_like_scale_amplifies(self):
        assert hardware_like(2.0).strength > hardware_like(1.0).strength
