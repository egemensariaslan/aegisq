"""Use a conserved quantity as a free error detector.

A particle-conserving circuit cannot leave its Hamming-weight sector, so every
bit of probability found outside it was put there by noise.  Discarding that
mass costs no extra circuit executions -- only a richer measurement -- and the
retained weight doubles as a calibration-free error readout.

    python examples/04_symmetry_verification.py
"""

from __future__ import annotations

import torch

from aegisq import QuantumLayer, SymmetryVerification

N_QUBITS = 4


def build(noise) -> QuantumLayer:
    return QuantumLayer(
        N_QUBITS,
        n_layers=3,
        ansatz="particle_conserving",  # Givens rotations: conserve particle number
        encoding="excitation",         # ...and an encoding that also conserves it
        measurement="probs",           # verification needs the full distribution
        noise=noise,
        seed=0,
    )


def main() -> None:
    reference = build(None)
    print("circuit symmetry:", reference.symmetry)
    x = torch.randn(8, reference.in_features, generator=torch.Generator().manual_seed(0))

    out_of_sector = [i for i in range(2**N_QUBITS) if bin(i).count("1") != 2]
    z_table = 1.0 - 2.0 * torch.tensor(
        [[(i >> (N_QUBITS - 1 - w)) & 1 for w in range(N_QUBITS)]
         for i in range(2**N_QUBITS)],
        dtype=torch.get_default_dtype(),
    )

    with torch.no_grad():
        ideal = reference(x) @ z_table
        print(f"noiseless leakage out of the sector: {float(reference(x)[:, out_of_sector].sum(-1).mean()):.2e}\n")

        print(f"{'noise':>8}  {'leakage':>9}  {'kept':>6}  {'raw error':>10}  {'verified error':>15}")
        print("-" * 56)
        for p in (0.005, 0.01, 0.02, 0.05):
            layer = build(p)
            verified = SymmetryVerification(layer)
            probs = layer(x)
            leakage = float(probs[:, out_of_sector].sum(-1).mean())
            kept = float(verified.sector_weight(x).mean())
            raw_error = float(((probs @ z_table) - ideal).abs().mean())
            verified_error = float((verified(x) - ideal).abs().mean())
            print(
                f"{p:>8}  {leakage:>9.4f}  {kept:>6.3f}  "
                f"{raw_error:>10.4f}  {verified_error:>15.4f}"
            )

    print(
        "\n'kept' is the post-selection acceptance rate: 1/kept is the shot overhead "
        "\nthe technique costs on hardware."
    )


if __name__ == "__main__":
    main()
