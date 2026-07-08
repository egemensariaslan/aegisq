"""Ready-made :class:`~aegisq.noise.NoiseSpec` profiles.

The presets cover the channels named in the AegisQ specification --
depolarizing, dephasing, thermal relaxation and readout error -- plus a
composite ``hardware_like`` profile whose defaults sit in the range reported by
current superconducting devices.
"""

from __future__ import annotations

from .spec import NoiseSpec, ScaleMode

__all__ = [
    "noiseless",
    "depolarizing",
    "dephasing",
    "amplitude_damping",
    "thermal_relaxation",
    "readout",
    "hardware_like",
    "PRESETS",
    "get_preset",
]


def noiseless() -> NoiseSpec:
    """An empty profile; circuits run on the exact statevector simulator."""
    return NoiseSpec()


def depolarizing(
    p: float = 0.01,
    *,
    two_qubit_factor: float = 5.0,
    scale_mode: ScaleMode = "composed",
) -> NoiseSpec:
    """Depolarizing noise after every gate.

    ``two_qubit_factor`` reflects the usual order-of-magnitude gap between one-
    and two-qubit gate infidelities.
    """
    return NoiseSpec(
        depolarizing_1q=p,
        depolarizing_2q=min(p * two_qubit_factor, 1.0),
        scale_mode=scale_mode,
    )


def dephasing(
    p: float = 0.02,
    *,
    two_qubit_factor: float = 3.0,
    scale_mode: ScaleMode = "composed",
) -> NoiseSpec:
    """Pure dephasing (phase damping) after every gate."""
    return NoiseSpec(
        dephasing_1q=p,
        dephasing_2q=min(p * two_qubit_factor, 1.0),
        scale_mode=scale_mode,
    )


def amplitude_damping(
    gamma: float = 0.02,
    *,
    two_qubit_factor: float = 3.0,
    scale_mode: ScaleMode = "composed",
) -> NoiseSpec:
    """Energy relaxation towards |0> after every gate."""
    return NoiseSpec(
        amplitude_damping_1q=gamma,
        amplitude_damping_2q=min(gamma * two_qubit_factor, 1.0),
        scale_mode=scale_mode,
    )


def thermal_relaxation(
    t1: float = 50_000.0,
    t2: float = 30_000.0,
    gate_time: float = 200.0,
    *,
    excited_population: float = 0.01,
    scale_mode: ScaleMode = "composed",
) -> NoiseSpec:
    """Combined T1/T2 relaxation.

    Times share an arbitrary unit; the defaults are nanoseconds and correspond
    to a 50 us T1, a 30 us T2 and a 200 ns gate.
    """
    return NoiseSpec(
        t1=t1,
        t2=t2,
        gate_time=gate_time,
        excited_population=excited_population,
        scale_mode=scale_mode,
    )


def readout(p: float = 0.03, *, scale_mode: ScaleMode = "composed") -> NoiseSpec:
    """Measurement bit-flip error only; the unitary part stays exact."""
    return NoiseSpec(readout=p, scale_mode=scale_mode)


def hardware_like(
    scale: float = 1.0,
    *,
    scale_mode: ScaleMode = "composed",
) -> NoiseSpec:
    """A composite superconducting-style profile.

    ``scale`` multiplies every error rate, so ``hardware_like(2.0)`` is a device
    twice as noisy as the nominal one.
    """
    base = NoiseSpec(
        depolarizing_1q=0.001,
        depolarizing_2q=0.01,
        dephasing_1q=0.002,
        dephasing_2q=0.01,
        amplitude_damping_1q=0.001,
        amplitude_damping_2q=0.005,
        readout=0.02,
        scale_mode=scale_mode,
    )
    return base.scaled(scale) if scale != 1.0 else base


PRESETS = {
    "noiseless": noiseless,
    "depolarizing": depolarizing,
    "dephasing": dephasing,
    "amplitude_damping": amplitude_damping,
    "thermal_relaxation": thermal_relaxation,
    "readout": readout,
    "hardware_like": hardware_like,
}


def get_preset(name: str, **kwargs) -> NoiseSpec:
    """Look up a preset by name, e.g. ``get_preset("depolarizing", p=0.05)``."""
    try:
        factory = PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown noise preset {name!r}; available: {sorted(PRESETS)}"
        ) from None
    return factory(**kwargs)
