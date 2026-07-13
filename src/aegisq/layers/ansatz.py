"""Variational ansatz catalog.

An :class:`Ansatz` is a *circuit fragment*: it knows its parameter shape and how
to queue its gates onto a PennyLane tape.  It deliberately owns neither a device
nor a QNode, which is what lets the same object be folded for zero-noise
extrapolation, benchmarked against a baseline, or stacked inside a
:class:`~aegisq.layers.QuantumLayer`.

The catalog splits into two families:

``aegisq`` ansaetze
    :class:`LocalEntangler`, :class:`ParticleConserving`,
    :class:`PermutationEquivariant` and :class:`Z2Equivariant` -- shallow,
    nearest-neighbour and/or symmetry-restricted circuits designed to keep the
    accumulated error budget small.

baselines
    :class:`BasicEntanglerBaseline` and :class:`StronglyEntanglingBaseline` wrap
    the standard PennyLane templates so the benchmark harness can compare
    against them under an identical noise model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, Sequence

import pennylane as qml
import torch

InitStrategy = Literal["uniform", "normal", "small", "zeros"]

__all__ = [
    "Ansatz",
    "LocalEntangler",
    "ParticleConserving",
    "PermutationEquivariant",
    "Z2Equivariant",
    "BasicEntanglerBaseline",
    "StronglyEntanglingBaseline",
    "brick_bonds",
]


def brick_bonds(
    n_wires: int, *, ring: bool = False
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Even/odd nearest-neighbour bond layers of a 1-D brickwork pattern.

    Returns two lists of ``(i, j)`` index pairs.  Gates inside one list act on
    disjoint wires and therefore execute in a single hardware cycle -- the
    property that keeps the circuit's *depth* low even when the gate count is not.
    """
    if n_wires < 2:
        return [], []
    even = [(i, i + 1) for i in range(0, n_wires - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n_wires - 1, 2)]
    if ring and n_wires > 2:
        odd = odd + [(n_wires - 1, 0)]
    return even, odd


def _all_bonds(n_wires: int, *, ring: bool = False) -> list[tuple[int, int]]:
    even, odd = brick_bonds(n_wires, ring=ring)
    return even + odd


class Ansatz(ABC):
    """Base class for a parameterised circuit fragment."""

    #: Registry key, also used in benchmark reports.
    name: str = "ansatz"
    #: Symmetry the fragment leaves invariant, or ``None``.
    symmetry: str | None = None

    @abstractmethod
    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        """Shape of the trainable weight tensor this ansatz consumes."""

    @abstractmethod
    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        """Queue the gates for ``weights`` onto the active tape."""

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        """Number of two-qubit gates, i.e. the dominant noise cost."""
        return 0

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        """Two-qubit *depth* (parallel cycles), which is what T2 actually sees."""
        return 0

    # ------------------------------------------------------------------
    def init_weights(
        self,
        n_layers: int,
        n_wires: int,
        *,
        strategy: InitStrategy = "uniform",
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Draw an initial weight tensor.

        ``"uniform"`` samples the full ``[0, 2*pi)`` circle -- the convention used
        in the barren-plateau literature and therefore the honest default for
        benchmarking.  ``"small"`` concentrates near the identity, a cheap and
        effective plateau mitigation for deep circuits.
        """
        shape = self.weight_shape(n_layers, n_wires)
        if strategy == "uniform":
            w = torch.rand(shape, generator=generator, dtype=dtype) * (2 * math.pi)
        elif strategy == "normal":
            w = torch.randn(shape, generator=generator, dtype=dtype) * 0.1
        elif strategy == "small":
            w = (torch.rand(shape, generator=generator, dtype=dtype) - 0.5) * 0.2
        elif strategy == "zeros":
            w = torch.zeros(shape, dtype=dtype)
        else:
            raise ValueError(f"unknown init strategy {strategy!r}")
        return w

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


# ----------------------------------------------------------------------
# Resilient ansaetze
# ----------------------------------------------------------------------
class LocalEntangler(Ansatz):
    """Shallow brickwork ansatz with strictly nearest-neighbour entanglement.

    Every two-qubit gate connects physically adjacent wires, so no SWAP chain is
    needed to run it on a linear-connectivity device -- the routing overhead that
    silently multiplies the error budget of a long-range ansatz disappears.  The
    even and odd bond sub-layers each execute in one cycle, giving a two-qubit
    depth of ``2`` per layer regardless of the qubit count.

    Choosing the entangler matters more than it looks
    -------------------------------------------------
    ``"cz"`` and ``"zz"`` are *diagonal*: they build correlations in phase, which
    a single-qubit ``<Z_i>`` read-out sees only after the next layer's rotations
    bring them back into the measured basis.  On a target that is genuinely
    global -- bitstring parity being the extreme case -- the default pairing of
    ``entangler="cz"`` with ``measurement="local_z"`` trains to chance while the
    same layer with ``entangler="cnot"``, or with ``measurement="global_z"``,
    reaches 100%.  ``examples/06_choosing_an_ansatz.py`` reproduces the numbers.

    The default stays ``"cz"``: it is native on most superconducting hardware and
    holds gradient variance better as the register grows (measured decay from 4
    to 8 qubits: 0.20 for CZ against 0.11 for CNOT).  But if the function you are
    fitting depends on all qubits at once, change one of the two knobs.

    Parameters
    ----------
    entangler:
        ``"cz"`` (default) and ``"cnot"`` are parameter-free; ``"zz"`` uses a
        trainable ``IsingZZ`` per bond, which is the native two-qubit
        interaction on several hardware families and can be run at a smaller
        rotation angle -- and hence lower error -- than a full entangling gate.
    ring:
        Close the chain with a wrap-around bond.  Off by default: the closing
        bond is the one long-range gate a linear device cannot execute natively.
    rotations:
        Single-qubit rotation axes applied per layer, e.g. ``"YZ"`` (default)
        or ``"Y"`` for a leaner circuit.
    """

    name = "local_entangler"

    def __init__(
        self,
        *,
        entangler: Literal["cz", "cnot", "zz"] = "cz",
        ring: bool = False,
        rotations: str = "YZ",
    ) -> None:
        if entangler not in ("cz", "cnot", "zz"):
            raise ValueError(f"unknown entangler {entangler!r}")
        rotations = rotations.upper()
        if not rotations or any(axis not in "XYZ" for axis in rotations):
            raise ValueError(f"rotations must be a non-empty string over XYZ, got {rotations!r}")
        self.entangler = entangler
        self.ring = ring
        self.rotations = rotations

    def _n_rot(self, n_wires: int) -> int:
        return n_wires * len(self.rotations)

    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        n_bond_params = len(_all_bonds(n_wires, ring=self.ring)) if self.entangler == "zz" else 0
        return (n_layers, self._n_rot(n_wires) + n_bond_params)

    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        n_wires = len(wires)
        rot_gates = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
        even, odd = brick_bonds(n_wires, ring=self.ring)
        for layer in weights:
            idx = 0
            for axis in self.rotations:
                gate = rot_gates[axis]
                for w in range(n_wires):
                    gate(layer[idx], wires=wires[w])
                    idx += 1
            for bonds in (even, odd):
                for i, j in bonds:
                    if self.entangler == "cz":
                        qml.CZ(wires=[wires[i], wires[j]])
                    elif self.entangler == "cnot":
                        qml.CNOT(wires=[wires[i], wires[j]])
                    else:
                        qml.IsingZZ(layer[idx], wires=[wires[i], wires[j]])
                        idx += 1

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        return n_layers * len(_all_bonds(n_wires, ring=self.ring))

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (2 if n_wires > 2 else 1)


class ParticleConserving(Ansatz):
    """U(1)-symmetric ansatz built from Givens rotations.

    Every gate commutes with the total number operator ``N = sum_i (1 - Z_i)/2``,
    so the state is confined to a single Hamming-weight sector of dimension
    ``C(n, k)`` instead of the full ``2**n``.  Two things follow.  The
    variational manifold is exponentially smaller, which flattens the landscape
    far less aggressively than an unconstrained ansatz.  And because *noise* is
    the only process that can move weight out of the sector, the symmetry
    doubles as an error detector -- see
    :class:`aegisq.mitigation.SymmetryVerification`.

    Pair this with ``encoding="excitation"`` and ``init_state`` on
    :class:`~aegisq.layers.QuantumLayer` to keep the symmetry exact end to end.
    """

    name = "particle_conserving"
    symmetry = "particle_number"

    def __init__(self, *, ring: bool = False, phase_rotations: bool = True) -> None:
        self.ring = ring
        self.phase_rotations = phase_rotations

    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        n_phase = n_wires if self.phase_rotations else 0
        return (n_layers, n_phase + len(_all_bonds(n_wires, ring=self.ring)))

    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        n_wires = len(wires)
        even, odd = brick_bonds(n_wires, ring=self.ring)
        for layer in weights:
            idx = 0
            if self.phase_rotations:
                # RZ is diagonal, hence trivially number-conserving.
                for w in range(n_wires):
                    qml.RZ(layer[idx], wires=wires[w])
                    idx += 1
            for bonds in (even, odd):
                for i, j in bonds:
                    qml.SingleExcitation(layer[idx], wires=[wires[i], wires[j]])
                    idx += 1

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        return n_layers * len(_all_bonds(n_wires, ring=self.ring))

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (2 if n_wires > 2 else 1)


class PermutationEquivariant(Ansatz):
    """Cyclically equivariant ansatz with a qubit-count-independent parameter budget.

    All wires share the same rotation angle and all bonds share the same
    coupling, so a layer costs three parameters no matter how wide the register
    is.  The circuit commutes with cyclic qubit permutations, which is the right
    inductive bias whenever the input features carry no privileged ordering
    (graphs, particle sets, translation-invariant signals).

    Two practical consequences: the optimisation runs in a ``O(n_layers)``
    dimensional space rather than an ``O(n_layers * n_wires)`` one, which keeps
    gradient variance measurable at widths where an unconstrained ansatz has
    already flattened; and weight sharing means a coherent miscalibration on one
    wire cannot be absorbed independently, so fits stay honest under noise.
    """

    name = "equivariant"
    symmetry = "cyclic_permutation"

    def __init__(self, *, ring: bool = True, entangler: Literal["zz", "cz"] = "zz") -> None:
        if entangler not in ("zz", "cz"):
            raise ValueError(f"unknown entangler {entangler!r}")
        self.ring = ring
        self.entangler = entangler

    @property
    def _params_per_layer(self) -> int:
        return 3 if self.entangler == "zz" else 2

    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        return (n_layers, self._params_per_layer)

    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        n_wires = len(wires)
        even, odd = brick_bonds(n_wires, ring=self.ring)
        for layer in weights:
            for w in wires:
                qml.RX(layer[0], wires=w)
            for w in wires:
                qml.RZ(layer[1], wires=w)
            for bonds in (even, odd):
                for i, j in bonds:
                    if self.entangler == "zz":
                        qml.IsingZZ(layer[2], wires=[wires[i], wires[j]])
                    else:
                        qml.CZ(wires=[wires[i], wires[j]])

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        return n_layers * len(_all_bonds(n_wires, ring=self.ring))

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (2 if n_wires > 2 else 1)


class Z2Equivariant(Ansatz):
    """Ansatz commuting with the global spin-flip operator ``X^{\\otimes n}``.

    Built from ``RX`` rotations and ``IsingZZ`` couplings, both of which commute
    with ``X^{\\otimes n}``.  The conserved quantity is the ``X``-basis parity,
    which makes the circuit a natural fit for label-symmetric classification
    (swapping every input sign must swap the prediction) and, like
    :class:`ParticleConserving`, exposes a parity check that noise -- and only
    noise -- can violate.
    """

    name = "z2_equivariant"
    symmetry = "x_parity"

    def __init__(self, *, ring: bool = False) -> None:
        self.ring = ring

    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        return (n_layers, n_wires + len(_all_bonds(n_wires, ring=self.ring)))

    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        n_wires = len(wires)
        even, odd = brick_bonds(n_wires, ring=self.ring)
        for layer in weights:
            idx = 0
            for w in range(n_wires):
                qml.RX(layer[idx], wires=wires[w])
                idx += 1
            for bonds in (even, odd):
                for i, j in bonds:
                    qml.IsingZZ(layer[idx], wires=[wires[i], wires[j]])
                    idx += 1

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        return n_layers * len(_all_bonds(n_wires, ring=self.ring))

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (2 if n_wires > 2 else 1)


# ----------------------------------------------------------------------
# Standard baselines
# ----------------------------------------------------------------------
class BasicEntanglerBaseline(Ansatz):
    """:class:`pennylane.BasicEntanglerLayers`, wrapped as an :class:`Ansatz`.

    Included as a reference point: its closing CNOT ring is a long-range gate on
    a linear device, and it entangles every wire on every layer.
    """

    name = "basic_entangler"

    def __init__(self, *, rotation: Literal["X", "Y", "Z"] = "X") -> None:
        self._rotation = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[rotation.upper()]

    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        return (n_layers, n_wires)

    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        qml.BasicEntanglerLayers(weights, wires=wires, rotation=self._rotation)

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (n_wires if n_wires > 2 else 1)

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        # The CNOT ring is applied as a sequential chain, not a brickwork.
        return n_layers * (n_wires if n_wires > 2 else 1)


class StronglyEntanglingBaseline(Ansatz):
    """:class:`pennylane.StronglyEntanglingLayers`, wrapped as an :class:`Ansatz`.

    The canonical expressive ansatz, and the canonical noise victim: its
    ``range``-shifted CNOT pattern places long-range gates on most layers.
    """

    name = "strongly_entangling"

    def __init__(self, *, ranges: Sequence[int] | None = None) -> None:
        self.ranges = ranges

    def weight_shape(self, n_layers: int, n_wires: int) -> tuple[int, ...]:
        return (n_layers, n_wires, 3)

    def apply(self, weights: torch.Tensor, wires: Sequence) -> None:
        qml.StronglyEntanglingLayers(weights, wires=wires, ranges=self.ranges)

    def two_qubit_gates(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (n_wires if n_wires > 2 else 1)

    def circuit_depth(self, n_layers: int, n_wires: int) -> int:
        return n_layers * (n_wires if n_wires > 2 else 1)
