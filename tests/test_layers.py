"""QuantumLayer construction, the nn.Module contract, and the ansatz catalog."""

from __future__ import annotations

import copy
import io

import pennylane as qml
import pytest
import torch
from torch import nn

from aegisq import QuantumLayer
from aegisq.layers import ANSATZE, ENCODINGS, MEASUREMENTS, LocalEntangler, available
from aegisq.layers.registry import resolve_ansatz, resolve_encoding, resolve_measurement

ANSATZ_NAMES = sorted(set(ANSATZE))
ENCODING_NAMES = sorted(set(ENCODINGS))
MEASUREMENT_NAMES = sorted(set(MEASUREMENTS))


class TestConstruction:
    @pytest.mark.parametrize("name", ANSATZ_NAMES)
    def test_every_ansatz_runs_and_declares_its_shape(self, name):
        layer = QuantumLayer(4, n_layers=2, ansatz=name, seed=0)
        assert layer.weights.shape == layer.ansatz.weight_shape(2, 4)
        out = layer(torch.zeros(3, layer.in_features))
        assert out.shape == (3, layer.out_features)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("name", ENCODING_NAMES)
    def test_every_encoding_consumes_its_declared_features(self, name):
        layer = QuantumLayer(4, n_layers=1, encoding=name, seed=0)
        out = layer(torch.zeros(2, layer.in_features))
        assert out.shape == (2, layer.out_features)

    @pytest.mark.parametrize("name", MEASUREMENT_NAMES)
    def test_every_measurement_declares_its_width(self, name):
        layer = QuantumLayer(3, n_layers=1, measurement=name, seed=0)
        out = layer(torch.zeros(2, layer.in_features))
        assert out.shape[-1] == layer.out_features

    def test_custom_observable_list(self):
        layer = QuantumLayer(3, measurement=[qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)])
        assert layer.out_features == 2
        assert layer(torch.zeros(2, 3)).shape == (2, 2)

    def test_rejects_bad_sizes(self):
        with pytest.raises(ValueError, match="n_qubits"):
            QuantumLayer(0)
        with pytest.raises(ValueError, match="n_layers"):
            QuantumLayer(2, n_layers=0)

    def test_unknown_name_lists_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            QuantumLayer(2, ansatz="nope")

    def test_pure_state_device_rejects_noise(self):
        with pytest.raises(ValueError, match="cannot apply noise channels"):
            QuantumLayer(2, noise="depolarizing", device="default.qubit")

    def test_noise_accepts_several_spellings(self):
        assert QuantumLayer(2, noise=0.01).noise.depolarizing_1q == pytest.approx(0.01)
        assert not QuantumLayer(2, noise="hardware_like").noise.is_noiseless
        assert QuantumLayer(2, noise={"depolarizing_1q": 0.02}).noise.depolarizing_1q == 0.02
        assert QuantumLayer(2, noise=None).noise.is_noiseless

    def test_state_prep_encoding_cannot_be_reuploaded(self):
        with pytest.raises(ValueError, match="cannot be re-uploaded"):
            QuantumLayer(3, encoding="amplitude", data_reupload=True)


class TestForward:
    def test_unbatched_input_returns_unbatched_output(self):
        layer = QuantumLayer(3, seed=0)
        assert layer(torch.zeros(3)).shape == (3,)

    @pytest.mark.parametrize("batch", [1, 2, 7])
    def test_batching(self, batch):
        layer = QuantumLayer(3, seed=0)
        assert layer(torch.zeros(batch, 3)).shape == (batch, 3)

    def test_wrong_feature_count_is_reported_clearly(self):
        layer = QuantumLayer(3, seed=0)
        with pytest.raises(ValueError, match="expects 3 input features"):
            layer(torch.zeros(2, 5))

    def test_rejects_rank_three_input(self):
        layer = QuantumLayer(3, seed=0)
        with pytest.raises(ValueError, match=r"expects a \(batch"):
            layer(torch.zeros(2, 2, 3))

    def test_output_matches_module_dtype(self):
        layer = QuantumLayer(2, seed=0, dtype=torch.float32)
        assert layer(torch.zeros(2, 2)).dtype == torch.float32

    def test_float64_layers_simulate_in_double_precision(self):
        """PennyLane takes the density-matrix dtype from torch's global default.

        The last RZ commutes through the CZ layer and the Z measurement, so its
        angle cannot change the output.  At single precision the noisy simulator
        drifts by ~1e-7 anyway; at double it is exactly invariant.
        """
        layer = QuantumLayer(3, n_layers=2, noise="hardware_like", seed=0,
                             dtype=torch.float64)
        x = torch.tensor([0.4, -0.9, 1.1], dtype=torch.float64)
        flat = layer.weights.view(-1)
        with torch.no_grad():
            baseline = float(layer(x).sum())
            original = flat[-1].clone()
            for delta in (1e-7, 1e-5, 1e-3):
                flat[-1] = original + delta
                assert float(layer(x).sum()) == baseline
            flat[-1] = original
        assert torch.get_default_dtype() is not torch.float64  # scope was restored

    def test_noise_damps_expectation_values(self):
        clean = QuantumLayer(3, n_layers=3, seed=0)
        noisy = QuantumLayer(3, n_layers=3, seed=0, noise="depolarizing")
        x = torch.linspace(-1, 1, 3)
        with torch.no_grad():
            assert noisy(x).abs().sum() < clean(x).abs().sum()

    def test_data_reupload_changes_the_function(self):
        plain = QuantumLayer(3, n_layers=2, seed=0)
        reupload = QuantumLayer(3, n_layers=2, seed=0, data_reupload=True)
        x = torch.tensor([0.4, -0.7, 1.1])
        with torch.no_grad():
            assert not torch.allclose(plain(x), reupload(x))

    def test_trainable_input_scaling_is_a_parameter(self):
        layer = QuantumLayer(3, seed=0, trainable_input_scaling=True)
        assert "input_scaling" in dict(layer.named_parameters())
        layer(torch.ones(2, 3)).sum().backward()
        assert layer.input_scaling.grad is not None

    def test_shots_produce_sampled_estimates(self):
        layer = QuantumLayer(2, seed=0, shots=200, noise="depolarizing")
        assert layer.diff_method == "parameter-shift"
        x = torch.tensor([0.3, 0.9])
        first, second = layer(x), layer(x)
        assert first.shape == (2,)
        # Finite sampling means two runs disagree; an analytic device would not.
        assert not torch.allclose(first, second)


class TestModuleContract:
    def test_registers_exactly_one_circuit_parameter_tensor(self):
        layer = QuantumLayer(4, n_layers=2, seed=0)
        names = [n for n, _ in layer.named_parameters()]
        assert names == ["weights"]
        assert layer.n_circuit_parameters == layer.weights.numel()

    def test_composes_inside_sequential(self):
        layer = QuantumLayer(4, seed=0)
        model = nn.Sequential(nn.Linear(6, 4), layer, nn.Linear(4, 2))
        out = model(torch.randn(5, 6))
        assert out.shape == (5, 2)
        out.sum().backward()
        assert layer.weights.grad is not None

    def test_state_dict_round_trip(self):
        source = QuantumLayer(3, n_layers=2, seed=0)
        target = QuantumLayer(3, n_layers=2, seed=1)
        assert not torch.allclose(source.weights, target.weights)
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        target.load_state_dict(torch.load(buffer, weights_only=True))
        assert torch.allclose(source.weights, target.weights)
        x = torch.tensor([0.2, 0.4, 0.6])
        with torch.no_grad():
            assert torch.allclose(source(x), target(x))

    def test_deepcopy_keeps_the_circuit_working(self):
        layer = QuantumLayer(3, seed=0, noise="depolarizing")
        clone = copy.deepcopy(layer)
        x = torch.tensor([0.1, 0.2, 0.3])
        with torch.no_grad():
            assert torch.allclose(layer(x), clone(x))

    def test_seed_makes_initialisation_reproducible(self):
        assert torch.allclose(QuantumLayer(3, seed=5).weights, QuantumLayer(3, seed=5).weights)

    def test_repr_names_the_configuration(self):
        text = repr(QuantumLayer(3, ansatz="equivariant", noise="depolarizing"))
        assert "equivariant" in text and "n_qubits=3" in text

    def test_draw_renders_noise_channels(self):
        layer = QuantumLayer(2, n_layers=1, noise="depolarizing")
        drawing = layer.draw()
        assert "DepolarizingChannel" in drawing


class TestAnsatzProperties:
    def test_local_entangler_uses_only_adjacent_wires(self):
        layer = QuantumLayer(5, n_layers=2, ansatz="local_entangler")
        tape = qml.workflow.construct_tape(layer._qnode())(
            torch.zeros(1, 5), layer.weights
        )
        for op in tape.operations:
            if len(op.wires) == 2:
                assert abs(op.wires[0] - op.wires[1]) == 1

    def test_ring_option_adds_the_wrap_around_bond(self):
        closed = QuantumLayer(5, n_layers=1, ansatz="local_entangler",
                              ansatz_kwargs={"ring": True})
        open_chain = QuantumLayer(5, n_layers=1, ansatz="local_entangler")
        assert closed.two_qubit_gates == open_chain.two_qubit_gates + 1

    def test_equivariant_parameter_count_is_width_independent(self):
        counts = {n: QuantumLayer(n, n_layers=3, ansatz="equivariant").n_circuit_parameters
                  for n in (3, 5, 8)}
        assert len(set(counts.values())) == 1

    def test_equivariant_is_invariant_under_cyclic_input_shift(self):
        layer = QuantumLayer(4, n_layers=2, ansatz="equivariant", seed=0)
        x = torch.tensor([0.3, -0.8, 1.2, 0.05])
        with torch.no_grad():
            base = layer(x)
            shifted = layer(x.roll(1))
        assert torch.allclose(base.roll(1), shifted, atol=1e-6)

    def test_local_ansatz_is_shallower_than_the_baseline(self):
        local = QuantumLayer(8, n_layers=3, ansatz="local_entangler")
        baseline = QuantumLayer(8, n_layers=3, ansatz="strongly_entangling")
        assert local.circuit_depth < baseline.circuit_depth

    def test_particle_conserving_circuit_stays_in_its_sector(self):
        layer = QuantumLayer(
            4, n_layers=3, ansatz="particle_conserving", encoding="excitation",
            measurement="probs", seed=0,
        )
        assert layer.symmetry == "particle_number"
        with torch.no_grad():
            probs = layer(torch.tensor([0.7, -1.3, 0.2]))
        occupied = [i for i, p in enumerate(probs) if p > 1e-9]
        assert all(bin(i).count("1") == 2 for i in occupied)

    def test_symmetry_is_only_claimed_when_encoding_agrees(self):
        mismatched = QuantumLayer(4, ansatz="particle_conserving", encoding="angle")
        assert mismatched.symmetry is None


class TestRegistry:
    def test_available_lists_all_three_kinds(self):
        catalog = available()
        assert set(catalog) == {"ansatz", "encoding", "measurement"}
        assert "local_entangler" in catalog["ansatz"]

    def test_resolve_accepts_name_class_and_instance(self):
        instance = LocalEntangler(entangler="cnot")
        assert resolve_ansatz("local_entangler").name == "local_entangler"
        assert isinstance(resolve_ansatz(LocalEntangler), LocalEntangler)
        assert resolve_ansatz(instance) is instance

    def test_kwargs_with_an_instance_are_rejected(self):
        with pytest.raises(TypeError, match="already-built"):
            resolve_ansatz(LocalEntangler(), entangler="cz")

    def test_resolve_encoding_and_measurement(self):
        assert resolve_encoding("angle").in_features(4) == 4
        assert resolve_measurement("global_z").out_features(4) == 1
        assert resolve_measurement([qml.PauliZ(0)]).out_features(1) == 1
