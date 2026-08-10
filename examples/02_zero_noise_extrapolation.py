"""Measure how much bias zero-noise extrapolation actually removes.

Runs the same circuit with the same weights three ways -- noiseless (the truth),
noisy, and ZNE-mitigated -- and reports the recovered fraction of the error for
each extrapolator, across a sweep of noise strengths.

    python examples/02_zero_noise_extrapolation.py
"""

from __future__ import annotations

import torch

from aegisq import QuantumLayer, ZNE
from aegisq.benchmark import mitigation_bias

N_QUBITS = 4
N_LAYERS = 3
CONFIGURATIONS = [
    ("richardson", (1.0, 2.0, 3.0)),
    ("richardson", (1.0, 3.0, 5.0)),
    ("exponential", (1.0, 2.0, 3.0)),
    ("linear", (1.0, 2.0)),
]


def main() -> None:
    x = torch.randn(16, N_QUBITS, generator=torch.Generator().manual_seed(0))

    print(f"{N_QUBITS} qubits, {N_LAYERS} layers, depolarizing noise, global folding")
    print("(bias reduction: fraction of the noisy circuit's error removed)\n")

    for p in (0.005, 0.01, 0.02, 0.04):
        # Same seed on both layers, so the weights -- and hence the ideal
        # expectation values -- are identical.
        ideal = QuantumLayer(N_QUBITS, n_layers=N_LAYERS, noise=None, seed=7)
        noisy = QuantumLayer(N_QUBITS, n_layers=N_LAYERS, noise=p, seed=7)

        with torch.no_grad():
            reference, unmitigated = ideal(x), noisy(x)
            print(f"depolarizing p={p:<6} raw error {float((unmitigated - reference).abs().mean()):.4f}")
            for name, scales in CONFIGURATIONS:
                mitigated = ZNE(noisy, scale_factors=scales, extrapolate=name)(x)
                report = mitigation_bias(mitigated, unmitigated, reference)
                print(
                    f"    {name:<12} {str(scales):<12} "
                    f"error {report['mitigated_error']:.4f}  "
                    f"bias reduction {report['bias_reduction']:+.1%}"
                )
        print()

    # Mitigation is not free: Richardson amplifies sampling variance.
    print("shot-noise cost of the extrapolation:")
    for scales in [(1.0, 2.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0, 5.0)]:
        layer = QuantumLayer(N_QUBITS, noise=0.01, seed=0)
        wrapper = ZNE(layer, scale_factors=scales)
        print(
            f"    {str(scales):<22} circuit runs {wrapper.circuit_evaluations}  "
            f"variance amplification {wrapper.extrapolator.noise_amplification:.1f}x"
        )


if __name__ == "__main__":
    main()
