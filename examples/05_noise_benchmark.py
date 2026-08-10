"""Run the full resilient-versus-baseline benchmark.

Trains every model under every noise profile with identical data, seeds and
optimiser budget, then prints accuracy, gradient variance and circuit cost side
by side.

    python examples/05_noise_benchmark.py           # ~5 minutes on a laptop
    python examples/05_noise_benchmark.py --quick   # ~1 minute
"""

from __future__ import annotations

import argparse

from aegisq.benchmark import get_dataset, standard_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer epochs and one seed")
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--dataset", default="two_moons")
    parser.add_argument("--csv", default=None, help="write per-run records to this path")
    args = parser.parse_args()

    data = get_dataset(args.dataset, n_samples=200, n_features=args.qubits, seed=0)
    result = standard_benchmark(
        n_qubits=args.qubits,
        n_layers=3,
        dataset=data,
        epochs=6 if args.quick else 20,
        seeds=(0,) if args.quick else (0, 1, 2),
    )

    print()
    print(result.summary())
    if args.csv:
        print(f"\nwrote {result.to_csv(args.csv)}")


if __name__ == "__main__":
    main()
