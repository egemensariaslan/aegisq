"""Datasets, trainability metrics and the benchmark harness."""

from __future__ import annotations

import pytest
import torch

from aegisq import QuantumLayer
from aegisq.benchmark import (
    DATASETS,
    ModelSpec,
    NoiseBenchmark,
    accuracy,
    barren_plateau_scan,
    get_dataset,
    gradient_variance,
    mitigation_bias,
    quantum_model,
    resilient_vs_baseline,
)
from aegisq.noise import depolarizing, noiseless


class TestDatasets:
    @pytest.mark.parametrize("name", sorted(DATASETS))
    def test_every_dataset_builds_a_usable_split(self, name):
        data = get_dataset(name, n_samples=60, n_features=4, seed=0)
        assert data.n_features == 4
        assert data.x_train.dim() == 2 and data.x_test.dim() == 2
        assert len(data) > 0 and len(data.x_test) > 0
        assert set(data.y_train.tolist()) <= {0, 1}
        assert torch.isfinite(data.x_train).all()

    def test_features_stay_inside_the_encodable_range(self):
        data = get_dataset("two_moons", n_samples=80, n_features=5, seed=1)
        assert data.x_train.abs().max() <= torch.pi + 1e-6

    def test_seed_is_reproducible(self):
        a = get_dataset("circles", n_samples=40, seed=3)
        b = get_dataset("circles", n_samples=40, seed=3)
        assert torch.allclose(a.x_train, b.x_train)

    def test_parity_labels_are_the_actual_parity(self):
        data = get_dataset("parity", n_samples=40, n_features=4, seed=0)
        bits = (data.x_train > 0).long()
        assert torch.equal(bits.sum(dim=1) % 2, data.y_train)

    def test_unknown_dataset_lists_alternatives(self):
        with pytest.raises(KeyError, match="available"):
            get_dataset("mnist")


class TestMetrics:
    def test_accuracy_on_logits_and_binary_scores(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.5]])
        assert accuracy(logits, torch.tensor([0, 1, 0])) == pytest.approx(1.0)
        assert accuracy(logits, torch.tensor([1, 1, 1])) == pytest.approx(1 / 3)

    def test_gradient_variance_is_positive_and_reports_the_circuit(self):
        stats = gradient_variance(
            lambda: QuantumLayer(3, n_layers=2, ansatz="local_entangler"),
            n_samples=8,
            seed=0,
        )
        assert stats.variance > 0
        assert stats.max_variance >= stats.variance
        assert stats.n_qubits == 3 and stats.n_layers == 2
        assert stats.std == pytest.approx(stats.variance**0.5)
        assert len(stats.per_parameter_variance) == stats.n_parameters

    def test_gradient_variance_finds_the_layer_inside_a_wrapper(self):
        factory = quantum_model(3, 2, n_layers=1)
        stats = gradient_variance(lambda: factory(noiseless()), n_samples=5, seed=0)
        assert stats.n_qubits == 3

    def test_resilient_ansaetze_resist_the_barren_plateau(self):
        """The library's core trainability claim, measured rather than asserted.

        Between 4 and 8 qubits the standard template loses most of its gradient
        variance, the localised ansatz loses much less, and the equivariant
        ansatz -- whose parameter count does not grow with width -- keeps
        essentially all of it.
        """
        def decay(ansatz: str) -> float:
            scan = barren_plateau_scan([4, 8], n_layers=4, n_samples=20, seed=0,
                                       ansatz=ansatz)
            return scan[8].variance / scan[4].variance

        baseline = decay("strongly_entangling")
        localised = decay("local_entangler")
        equivariant = decay("equivariant")

        assert baseline < 0.05         # more than an order of magnitude of variance, gone
        assert localised > 4 * baseline
        assert equivariant > 0.5       # essentially width-independent

    def test_barren_plateau_scan_covers_every_width(self):
        scan = barren_plateau_scan([2, 3, 4], n_layers=2, n_samples=6, ansatz="equivariant")
        assert sorted(scan) == [2, 3, 4]
        assert all(stats.variance >= 0 for stats in scan.values())

    def test_gradient_variance_needs_several_samples(self):
        with pytest.raises(ValueError, match="at least two"):
            gradient_variance(lambda: QuantumLayer(2), n_samples=1)

    def test_mitigation_bias_reports_the_recovered_fraction(self):
        ideal = torch.zeros(4)
        noisy = torch.full((4,), 0.4)
        perfect = mitigation_bias(ideal.clone(), noisy, ideal)
        assert perfect["bias_reduction"] == pytest.approx(1.0)
        worse = mitigation_bias(torch.full((4,), 0.8), noisy, ideal)
        assert worse["bias_reduction"] < 0


class TestHarness:
    def _dataset(self, n_features=3):
        return get_dataset("two_moons", n_samples=40, n_features=n_features, seed=0)

    def test_end_to_end_sweep(self):
        data = self._dataset()
        result = NoiseBenchmark(
            {
                "local": quantum_model(3, 2, n_layers=1, ansatz="local_entangler"),
                "baseline": quantum_model(3, 2, n_layers=1, ansatz="basic_entangler"),
            },
            {"clean": noiseless(), "noisy": depolarizing(0.02)},
            data,
            epochs=2,
            gradient_samples=4,
            verbose=False,
        ).run()

        assert len(result) == 4
        assert result.models == ["local", "baseline"]
        assert result.noise_levels == ["clean", "noisy"]
        for record in result:
            assert 0.0 <= record.test_accuracy <= 1.0
            assert record.two_qubit_gates > 0
            assert record.seconds > 0
            assert len(record.loss_curve) == 2

    def test_summary_and_pivot_report_every_cell(self):
        data = self._dataset()
        result = NoiseBenchmark(
            [ModelSpec("only", quantum_model(3, 2, n_layers=1))],
            {"clean": noiseless()},
            data,
            epochs=1,
            measure_gradient_variance=False,
            verbose=False,
        ).run()
        text = result.summary()
        assert "test accuracy" in text and "only" in text and "clean" in text
        assert result.pivot("test_accuracy")["only"]["clean"] == result.records[0].test_accuracy

    def test_csv_export_round_trips(self, tmp_path):
        data = self._dataset()
        result = NoiseBenchmark(
            [ModelSpec("only", quantum_model(3, 2, n_layers=1))],
            [depolarizing(0.01)],
            data,
            epochs=1,
            measure_gradient_variance=False,
            verbose=False,
        ).run()
        path = result.to_csv(str(tmp_path / "runs.csv"))
        contents = open(path, encoding="utf-8").read()
        assert "test_accuracy" in contents and "loss_curve" not in contents

    def test_zne_models_report_their_extra_cost(self):
        data = self._dataset()
        result = NoiseBenchmark(
            {
                "plain": quantum_model(3, 2, n_layers=1),
                "mitigated": quantum_model(3, 2, n_layers=1, zne={"scale_factors": (1, 2, 3)}),
            },
            {"noisy": depolarizing(0.01)},
            data,
            epochs=1,
            measure_gradient_variance=False,
            verbose=False,
        ).run()
        costs = {r.model: r.circuit_evaluations for r in result}
        assert costs == {"plain": 1, "mitigated": 3}

    def test_feature_mismatch_is_reported_clearly(self):
        with pytest.raises(ValueError, match="Build the dataset with n_features=5"):
            NoiseBenchmark(
                {"wide": quantum_model(5, 2, n_layers=1)},
                {"clean": noiseless()},
                self._dataset(3),
                epochs=1,
                verbose=False,
            ).run()

    def test_resilient_vs_baseline_marks_the_reference_models(self):
        specs = resilient_vs_baseline(3, 2, n_layers=1)
        assert {s.name for s in specs if not s.resilient} == {
            "BasicEntangler (baseline)",
            "StronglyEntangling (baseline)",
        }

    def test_rejects_empty_configuration(self):
        with pytest.raises(ValueError, match="at least one model"):
            NoiseBenchmark({}, {"clean": noiseless()}, self._dataset())
        with pytest.raises(ValueError, match="at least one noise level"):
            NoiseBenchmark({"m": quantum_model(3, 2)}, {}, self._dataset())
