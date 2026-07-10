"""Measurement strategies.

The choice made here is not cosmetic.  A *global* observable such as
``Z_0 ... Z_{n-1}`` has a gradient whose variance decays exponentially in the
qubit count for a wide class of random circuits, while *local* observables keep
that variance polynomially bounded for shallow ansaetze.  Reaching for
``"local_z"`` (the default) is the single cheapest barren-plateau mitigation
available, which is why AegisQ makes it explicit rather than hiding it in a
circuit definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import Sequence

import pennylane as qml

__all__ = [
    "Measurement",
    "LocalZ",
    "GlobalZ",
    "LocalZZ",
    "Probabilities",
    "ObservableList",
]


class Measurement(ABC):
    """Base class for the terminal measurement block of a layer."""

    name: str = "measurement"
    #: ``True`` if the output is a probability vector rather than expectations.
    returns_probabilities: bool = False

    @abstractmethod
    def out_features(self, n_wires: int) -> int:
        """Length of the vector returned per sample."""

    @abstractmethod
    def apply(self, wires: Sequence):
        """Return the PennyLane measurement process(es) for ``wires``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class LocalZ(Measurement):
    """``<Z_i>`` on every wire -- one output feature per qubit."""

    name = "local_z"

    def out_features(self, n_wires: int) -> int:
        return n_wires

    def apply(self, wires: Sequence):
        return [qml.expval(qml.PauliZ(w)) for w in wires]


class GlobalZ(Measurement):
    """The single global parity ``<Z_0 ... Z_{n-1}>``.

    Provided mainly as the *hard* comparison case for barren-plateau studies.
    """

    name = "global_z"

    def out_features(self, n_wires: int) -> int:
        return 1

    def apply(self, wires: Sequence):
        obs = reduce(lambda a, b: a @ b, (qml.PauliZ(w) for w in wires))
        return [qml.expval(obs)]


class LocalZZ(Measurement):
    """Nearest-neighbour correlators ``<Z_i Z_{i+1}>``.

    Two-body but still local: it sees entanglement that :class:`LocalZ` misses
    without paying the exponential gradient cost of :class:`GlobalZ`.
    """

    name = "local_zz"

    def __init__(self, *, ring: bool = False) -> None:
        self.ring = ring

    def _bonds(self, n_wires: int) -> list[tuple[int, int]]:
        bonds = [(i, i + 1) for i in range(n_wires - 1)]
        if self.ring and n_wires > 2:
            bonds.append((n_wires - 1, 0))
        return bonds

    def out_features(self, n_wires: int) -> int:
        return max(len(self._bonds(n_wires)), 1)

    def apply(self, wires: Sequence):
        bonds = self._bonds(len(wires))
        if not bonds:
            return [qml.expval(qml.PauliZ(wires[0]))]
        return [qml.expval(qml.PauliZ(wires[i]) @ qml.PauliZ(wires[j])) for i, j in bonds]


class Probabilities(Measurement):
    """The full ``2**n`` computational-basis distribution.

    Required by :class:`~aegisq.mitigation.SymmetryVerification`, which needs
    per-bitstring weights to test the symmetry sector.  Only practical for
    small registers.
    """

    name = "probs"
    returns_probabilities = True

    def out_features(self, n_wires: int) -> int:
        return 2**n_wires

    def apply(self, wires: Sequence):
        return qml.probs(wires=wires)


class ObservableList(Measurement):
    """Expectation values of a user-supplied list of observables."""

    name = "custom"

    def __init__(self, observables: Sequence) -> None:
        observables = list(observables)
        if not observables:
            raise ValueError("ObservableList requires at least one observable")
        self.observables = observables

    def out_features(self, n_wires: int) -> int:
        return len(self.observables)

    def apply(self, wires: Sequence):
        return [qml.expval(obs) for obs in self.observables]
