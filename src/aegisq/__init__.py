"""AegisQ -- noise-resilient quantum layers for PyTorch + PennyLane.

Quick start
-----------
>>> import torch
>>> from aegisq import QuantumLayer, ZNE
>>> layer = QuantumLayer(4, n_layers=2, noise="hardware_like", seed=0)
>>> model = torch.nn.Sequential(
...     torch.nn.Linear(8, 4),
...     ZNE(layer, scale_factors=(1, 2, 3)),
...     torch.nn.Linear(4, 2),
... )
>>> model(torch.randn(16, 8)).shape
torch.Size([16, 2])
"""

from . import benchmark, layers, mitigation, noise
from .layers import (
    Ansatz,
    BasicEntanglerBaseline,
    Encoding,
    LocalEntangler,
    Measurement,
    ParticleConserving,
    PermutationEquivariant,
    QuantumLayer,
    StronglyEntanglingBaseline,
    Z2Equivariant,
    available,
)
from .mitigation import ZNE, Extrapolator, SymmetryVerification, get_extrapolator, zne
from .noise import NoiseSpec, get_preset

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # layers
    "QuantumLayer",
    "Ansatz",
    "LocalEntangler",
    "ParticleConserving",
    "PermutationEquivariant",
    "Z2Equivariant",
    "BasicEntanglerBaseline",
    "StronglyEntanglingBaseline",
    "Encoding",
    "Measurement",
    "available",
    # mitigation
    "ZNE",
    "zne",
    "SymmetryVerification",
    "Extrapolator",
    "get_extrapolator",
    # noise
    "NoiseSpec",
    "get_preset",
    # subpackages
    "layers",
    "mitigation",
    "noise",
    "benchmark",
]
