"""Show where the resilient layers win, and where they do not.

Noise resilience is bought with expressivity, and the bill comes due on targets
that depend on every qubit at once.  This trains the same localised layer with
different entanglers and read-outs on bitstring parity -- a maximally non-local
function -- and prints what each combination reaches.

    python examples/06_choosing_an_ansatz.py     # ~2 minutes
"""

from __future__ import annotations

from aegisq.benchmark import NoiseBenchmark, get_dataset, quantum_model
from aegisq.noise import depolarizing

CONFIGURATIONS = {
    "LocalEntangler, CZ, local_z (defaults)": {},
    "LocalEntangler, CNOT, local_z": {"ansatz_kwargs": {"entangler": "cnot"}},
    "LocalEntangler, ZZ, local_z": {"ansatz_kwargs": {"entangler": "zz"}},
    "LocalEntangler, CZ, local_zz": {"measurement": "local_zz"},
    "LocalEntangler, CZ, global_z": {"measurement": "global_z"},
    "BasicEntangler (reference)": {"ansatz": "basic_entangler"},
}


def main() -> None:
    data = get_dataset("parity", n_samples=120, n_features=4, seed=0)
    models = {
        name: quantum_model(4, 2, n_layers=3, data_reupload=True, **kwargs)
        for name, kwargs in CONFIGURATIONS.items()
    }
    result = NoiseBenchmark(
        models,
        {"depolarizing": depolarizing(0.01)},
        data,
        epochs=15,
        seeds=(0, 1),
        measure_gradient_variance=False,
        verbose=False,
    ).run()

    print("4-qubit parity under depolarizing noise, mean of 2 seeds\n")
    width = max(len(name) for name in CONFIGURATIONS) + 2
    for model in result.models:
        scores = [record.test_accuracy for record in result if record.model == model]
        mean = sum(scores) / len(scores)
        print(f"{model:<{width}}{mean:.3f}   {[round(s, 3) for s in scores]}")

    print(
        "\nCZ and IsingZZ are diagonal: they write correlations into phase, which a"
        "\nsingle-qubit <Z_i> read-out cannot see directly. CNOT moves parity into the"
        "\ncomputational basis, and a global Z...Z observable reads it off wholesale."
        "\nEither fix recovers the task; the default pairing does not."
    )


if __name__ == "__main__":
    main()
