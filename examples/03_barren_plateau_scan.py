"""Compare how gradient variance scales with register width.

The barren-plateau signature is not a small gradient at one width -- it is
gradient variance falling off a cliff as the register grows.  This scans four
ansaetze from 4 to 10 qubits and prints the decay factor for each.

    python examples/03_barren_plateau_scan.py
"""

from __future__ import annotations

from aegisq.benchmark import barren_plateau_scan

QUBIT_COUNTS = [4, 6, 8, 10]
ANSATZE = [
    ("StronglyEntangling (baseline)", "strongly_entangling"),
    ("BasicEntangler (baseline)", "basic_entangler"),
    ("LocalEntangler", "local_entangler"),
    ("PermutationEquivariant", "equivariant"),
]


def main() -> None:
    header = "  ".join(f"n={n:<11}" for n in QUBIT_COUNTS)
    print("mean per-parameter gradient variance at initialisation, 4 layers, 60 samples\n")
    print(f"{'ansatz':<32}{header}  decay  params  dead")
    print("-" * (32 + len(header) + 22))

    for label, ansatz in ANSATZE:
        scan = barren_plateau_scan(
            QUBIT_COUNTS, n_layers=4, n_samples=60, seed=0, ansatz=ansatz
        )
        cells = "  ".join(f"{scan[n].variance:<13.2e}" for n in QUBIT_COUNTS)
        decay = scan[QUBIT_COUNTS[-1]].variance / scan[QUBIT_COUNTS[0]].variance
        widest = scan[QUBIT_COUNTS[-1]]
        print(
            f"{label:<32}{cells}{decay:>6.3f}{widest.n_parameters:>8}"
            f"{widest.dead_parameter_fraction:>6.0%}"
        )

    print(
        "\ndecay = var(widest) / var(narrowest). A value near 1 means the ansatz stays"
        "\ntrainable as the register grows; a small one is a barren plateau."
        "\ndead = angles that commute through everything downstream, so their gradient is"
        "\nzero for algebraic reasons rather than because the landscape is flat."
    )


if __name__ == "__main__":
    main()
