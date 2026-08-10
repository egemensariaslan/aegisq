"""Train a hybrid classical-quantum classifier under hardware-like noise.

Shows the drop-in claim: a ``QuantumLayer`` sits inside ``torch.nn.Sequential``
next to ordinary ``Linear`` layers and trains with a normal PyTorch loop.

    python examples/01_hybrid_classifier.py
"""

from __future__ import annotations

import torch
from torch import nn

from aegisq import QuantumLayer
from aegisq.benchmark import accuracy, get_dataset

N_QUBITS = 4


def main() -> None:
    torch.manual_seed(0)
    data = get_dataset("two_moons", n_samples=200, n_features=N_QUBITS, seed=0)

    model = nn.Sequential(
        QuantumLayer(
            N_QUBITS,
            n_layers=3,
            ansatz="local_entangler",   # shallow, nearest-neighbour entanglement
            measurement="local_z",      # local cost function
            noise="hardware_like",      # simulate a real device
            data_reupload=True,         # expressivity without extra entangling depth
            seed=0,
        ),
        nn.Linear(N_QUBITS, data.n_classes),
    )
    print(model, "\n")

    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        permutation = torch.randperm(len(data))
        for start in range(0, len(data), 16):
            index = permutation[start : start + 16]
            optimiser.zero_grad()
            loss = loss_fn(model(data.x_train[index]), data.y_train[index])
            loss.backward()
            optimiser.step()

        if epoch % 5 == 0:
            with torch.no_grad():
                train = accuracy(model(data.x_train), data.y_train)
                test = accuracy(model(data.x_test), data.y_test)
            print(f"epoch {epoch:3d}  train {train:.3f}  test {test:.3f}")


if __name__ == "__main__":
    main()
