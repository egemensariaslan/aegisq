"""Name-based lookup for ansaetze, encodings and measurements.

Strings keep experiment configuration serialisable -- a benchmark sweep, a YAML
config or a CLI flag can name a layer without importing it.  ``resolve_*``
accepts a name, a class or an already-built instance, so the typed API and the
string API stay interchangeable.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, TypeVar

from .ansatz import (
    Ansatz,
    BasicEntanglerBaseline,
    LocalEntangler,
    ParticleConserving,
    PermutationEquivariant,
    StronglyEntanglingBaseline,
    Z2Equivariant,
)
from .encodings import (
    AmplitudeEncoding,
    AngleEncoding,
    DenseAngleEncoding,
    Encoding,
    ExcitationEncoding,
    IQPEncoding,
)
from .measurements import GlobalZ, LocalZ, LocalZZ, Measurement, ObservableList, Probabilities

__all__ = [
    "ANSATZE",
    "ENCODINGS",
    "MEASUREMENTS",
    "register_ansatz",
    "register_encoding",
    "register_measurement",
    "resolve_ansatz",
    "resolve_encoding",
    "resolve_measurement",
    "available",
]

T = TypeVar("T")

ANSATZE: dict[str, type[Ansatz]] = {
    "local_entangler": LocalEntangler,
    "local": LocalEntangler,
    "particle_conserving": ParticleConserving,
    "equivariant": PermutationEquivariant,
    "permutation_equivariant": PermutationEquivariant,
    "z2_equivariant": Z2Equivariant,
    "basic_entangler": BasicEntanglerBaseline,
    "strongly_entangling": StronglyEntanglingBaseline,
}

ENCODINGS: dict[str, type[Encoding]] = {
    "angle": AngleEncoding,
    "dense_angle": DenseAngleEncoding,
    "iqp": IQPEncoding,
    "amplitude": AmplitudeEncoding,
    "excitation": ExcitationEncoding,
}

MEASUREMENTS: dict[str, type[Measurement]] = {
    "local_z": LocalZ,
    "global_z": GlobalZ,
    "local_zz": LocalZZ,
    "probs": Probabilities,
    "probabilities": Probabilities,
}

#: Names that identify a standard, deliberately noise-fragile reference ansatz.
BASELINE_ANSATZE = frozenset({"basic_entangler", "strongly_entangling"})


def _resolve(value: Any, table: Mapping[str, type], base: type, kind: str, **kwargs) -> Any:
    if isinstance(value, base):
        if kwargs:
            raise TypeError(
                f"cannot pass {kind} keyword arguments alongside an already-built "
                f"{base.__name__} instance"
            )
        return value
    if inspect.isclass(value) and issubclass(value, base):
        return value(**kwargs)
    if isinstance(value, str):
        try:
            cls = table[value]
        except KeyError:
            raise KeyError(
                f"unknown {kind} {value!r}; available: {sorted(set(table))}"
            ) from None
        return cls(**kwargs)
    raise TypeError(f"cannot interpret {value!r} as a {kind}")


def resolve_ansatz(value: str | type[Ansatz] | Ansatz, **kwargs) -> Ansatz:
    """Turn a name, class or instance into an :class:`~aegisq.layers.Ansatz`."""
    return _resolve(value, ANSATZE, Ansatz, "ansatz", **kwargs)


def resolve_encoding(value: str | type[Encoding] | Encoding, **kwargs) -> Encoding:
    """Turn a name, class or instance into an :class:`~aegisq.layers.Encoding`."""
    return _resolve(value, ENCODINGS, Encoding, "encoding", **kwargs)


def resolve_measurement(value: Any, **kwargs) -> Measurement:
    """Turn a name, class, instance or observable list into a measurement."""
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        return ObservableList(value)
    return _resolve(value, MEASUREMENTS, Measurement, "measurement", **kwargs)


def _register(table: dict[str, type], base: type, kind: str) -> Callable:
    def decorator(name: str, cls: type | None = None):
        def _apply(klass: type) -> type:
            if not (inspect.isclass(klass) and issubclass(klass, base)):
                raise TypeError(f"{kind} must subclass {base.__name__}")
            table[name] = klass
            return klass

        return _apply(cls) if cls is not None else _apply

    return decorator


register_ansatz = _register(ANSATZE, Ansatz, "ansatz")
register_encoding = _register(ENCODINGS, Encoding, "encoding")
register_measurement = _register(MEASUREMENTS, Measurement, "measurement")


def available() -> dict[str, list[str]]:
    """Every registered name, grouped by kind."""
    return {
        "ansatz": sorted(set(ANSATZE)),
        "encoding": sorted(set(ENCODINGS)),
        "measurement": sorted(set(MEASUREMENTS)),
    }
