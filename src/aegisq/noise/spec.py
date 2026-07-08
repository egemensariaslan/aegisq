"""Declarative hardware-noise specification and its PennyLane realisation.

A :class:`NoiseSpec` is a plain, picklable description of the noise channels a
circuit should be executed under.  It is deliberately decoupled from PennyLane
so that it can be *scaled* -- the operation that virtual zero-noise
extrapolation is built on -- without touching a device or a tape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import pennylane as qml

ScaleMode = Literal["composed", "linear"]

__all__ = ["NoiseSpec", "ScaleMode"]


def _scale_depolarizing(p: float, factor: float) -> float:
    """Exact effective rate of ``factor`` composed single-qubit depolarizing channels.

    PennyLane's convention is ``rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)``,
    whose non-identity Pauli-transfer eigenvalue is ``1 - 4p/3``.  Composing the
    channel ``lambda`` times raises that eigenvalue to the ``lambda``-th power.
    """
    if p <= 0.0:
        return 0.0
    lam = 1.0 - 4.0 * p / 3.0
    # ``lam`` is negative only for unphysical p > 3/4; clamp so ``**factor`` is real.
    lam = max(lam, 0.0)
    return min(0.75 * (1.0 - lam**factor), 1.0)


def _scale_flip(p: float, factor: float) -> float:
    """Effective rate of ``factor`` composed bit/phase-flip channels."""
    if p <= 0.0:
        return 0.0
    lam = max(1.0 - 2.0 * p, 0.0)
    return min(0.5 * (1.0 - lam**factor), 1.0)


def _scale_damping(gamma: float, factor: float) -> float:
    """Effective rate of ``factor`` composed amplitude/phase-damping channels."""
    if gamma <= 0.0:
        return 0.0
    return min(1.0 - (1.0 - gamma) ** factor, 1.0)


def _scale_linear(p: float, factor: float, cap: float = 1.0) -> float:
    return min(p * factor, cap) if p > 0.0 else 0.0


def _acts_on_n_wires(n: int) -> qml.BooleanFn:
    """Noise conditional matching every unitary operation acting on ``n`` wires.

    Selecting on arity rather than on a hard-coded gate list means the profile
    also covers folded/adjointed gates and any custom template a user drops in.
    State preparations are excluded: they model an idealised reset, and folding
    cannot invert them.
    """

    @qml.BooleanFn
    def cond(op: Any) -> bool:
        return (
            isinstance(op, qml.operation.Operation)
            and not isinstance(op, qml.operation.Channel)
            and not isinstance(op, qml.operation.StatePrepBase)
            and len(op.wires) == n
        )

    return cond


@dataclass(frozen=True)
class NoiseSpec:
    """A hardware noise profile applied to a variational circuit.

    All rates are per-gate probabilities in ``[0, 1]``.  Two-qubit rates are
    applied to *each* wire touched by a two-qubit gate, matching the way
    calibration data is usually reported.

    Parameters
    ----------
    depolarizing_1q, depolarizing_2q:
        Depolarizing probability after single- and two-qubit gates.
    dephasing_1q, dephasing_2q:
        Phase-damping (pure dephasing) probability after single- and two-qubit gates.
    amplitude_damping_1q, amplitude_damping_2q:
        Amplitude-damping (T1 relaxation) probability after gates.
    t1, t2, gate_time, excited_population:
        Thermal-relaxation parameters.  Enabled only when both ``t1`` and ``t2``
        are set.  ``t1``/``t2``/``gate_time`` share an arbitrary but consistent
        time unit (nanoseconds by convention).
    readout:
        Bit-flip probability applied immediately before every measurement.
    scale_mode:
        How :meth:`scaled` maps a noise-scale factor onto channel rates.
        ``"composed"`` (default) computes the exact rate of ``factor`` composed
        copies of the channel; ``"linear"`` uses the first-order ``p * factor``
        common in the ZNE literature.
    """

    depolarizing_1q: float = 0.0
    depolarizing_2q: float = 0.0
    dephasing_1q: float = 0.0
    dephasing_2q: float = 0.0
    amplitude_damping_1q: float = 0.0
    amplitude_damping_2q: float = 0.0
    t1: float | None = None
    t2: float | None = None
    gate_time: float = 100.0
    excited_population: float = 0.0
    readout: float = 0.0
    scale_mode: ScaleMode = "composed"

    def __post_init__(self) -> None:
        for field in (
            "depolarizing_1q",
            "depolarizing_2q",
            "dephasing_1q",
            "dephasing_2q",
            "amplitude_damping_1q",
            "amplitude_damping_2q",
            "readout",
            "excited_population",
        ):
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"NoiseSpec.{field} must lie in [0, 1], got {value}")
        if (self.t1 is None) != (self.t2 is None):
            raise ValueError("NoiseSpec requires both t1 and t2 or neither")
        if self.t1 is not None:
            if self.t1 <= 0 or self.t2 <= 0:
                raise ValueError("t1 and t2 must be positive")
            if self.t2 > 2 * self.t1:
                raise ValueError(f"t2 must not exceed 2*t1 (t1={self.t1}, t2={self.t2})")
            if self.gate_time <= 0:
                raise ValueError("gate_time must be positive")
        if self.scale_mode not in ("composed", "linear"):
            raise ValueError(f"unknown scale_mode {self.scale_mode!r}")

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def is_noiseless(self) -> bool:
        """True when the spec adds no channels at all."""
        return not (self.has_gate_noise or self.readout > 0.0)

    @property
    def has_gate_noise(self) -> bool:
        return (
            max(
                self.depolarizing_1q,
                self.depolarizing_2q,
                self.dephasing_1q,
                self.dephasing_2q,
                self.amplitude_damping_1q,
                self.amplitude_damping_2q,
            )
            > 0.0
            or self.t1 is not None
        )

    @property
    def strength(self) -> float:
        """A single scalar summarising the profile, useful for plot axes."""
        rates = [
            self.depolarizing_1q,
            self.depolarizing_2q,
            self.dephasing_1q,
            self.dephasing_2q,
            self.amplitude_damping_1q,
            self.amplitude_damping_2q,
            self.readout,
        ]
        thermal = 0.0 if self.t1 is None else self.gate_time / self.t1
        return max(max(rates), thermal)

    # ------------------------------------------------------------------
    # scaling
    # ------------------------------------------------------------------
    def scaled(self, factor: float) -> "NoiseSpec":
        """Return a copy whose channel rates correspond to ``factor`` times the noise.

        This is the *virtual* noise-scaling primitive used by
        :class:`aegisq.mitigation.ZNE` with ``folding="noise"``: instead of
        lengthening the circuit, the simulated error rates are amplified
        directly.  Readout noise is scaled as well so that a full
        error-budget sweep stays consistent.
        """
        if factor < 0:
            raise ValueError(f"noise scale factor must be non-negative, got {factor}")
        if factor == 1.0:
            return self
        if self.scale_mode == "linear":
            dep = _scale_linear
            flip = _scale_linear
            damp = _scale_linear
        else:
            dep = _scale_depolarizing
            flip = _scale_flip
            damp = _scale_damping
        return replace(
            self,
            depolarizing_1q=dep(self.depolarizing_1q, factor),
            depolarizing_2q=dep(self.depolarizing_2q, factor),
            dephasing_1q=damp(self.dephasing_1q, factor),
            dephasing_2q=damp(self.dephasing_2q, factor),
            amplitude_damping_1q=damp(self.amplitude_damping_1q, factor),
            amplitude_damping_2q=damp(self.amplitude_damping_2q, factor),
            gate_time=self.gate_time * factor if self.t1 is not None else self.gate_time,
            readout=flip(self.readout, factor),
        )

    def __mul__(self, factor: float) -> "NoiseSpec":
        return self.scaled(float(factor))

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # PennyLane realisation
    # ------------------------------------------------------------------
    def _gate_noise_fn(self, two_qubit: bool) -> Callable[..., None]:
        depolarizing = self.depolarizing_2q if two_qubit else self.depolarizing_1q
        dephasing = self.dephasing_2q if two_qubit else self.dephasing_1q
        damping = self.amplitude_damping_2q if two_qubit else self.amplitude_damping_1q
        t1, t2 = self.t1, self.t2
        gate_time = self.gate_time * (2.0 if two_qubit else 1.0)
        excited = self.excited_population

        def noise_fn(op: qml.operation.Operator, **_: Any) -> None:
            # Every channel is single-qubit and applied wire-by-wire, so a
            # two-qubit gate accumulates error on both of its wires.
            for wire in op.wires:
                if depolarizing > 0.0:
                    qml.DepolarizingChannel(depolarizing, wires=wire)
                if dephasing > 0.0:
                    qml.PhaseDamping(dephasing, wires=wire)
                if damping > 0.0:
                    qml.AmplitudeDamping(damping, wires=wire)
                if t1 is not None:
                    qml.ThermalRelaxationError(excited, t1, t2, gate_time, wires=wire)

        return noise_fn

    def _readout_noise_fn(self) -> Callable[..., None]:
        readout = self.readout

        def noise_fn(mp: Any, **_: Any) -> None:
            for wire in mp.wires:
                qml.BitFlip(readout, wires=wire)

        return noise_fn

    def to_noise_model(self) -> qml.NoiseModel | None:
        """Build the :class:`pennylane.NoiseModel` for this spec.

        Returns ``None`` when the spec is noiseless, which lets callers skip the
        :func:`pennylane.add_noise` transform entirely.
        """
        if self.is_noiseless:
            return None

        model_map: dict[Any, Any] = {}
        if self.has_gate_noise:
            model_map[_acts_on_n_wires(1)] = self._gate_noise_fn(two_qubit=False)
            model_map[_acts_on_n_wires(2)] = self._gate_noise_fn(two_qubit=True)

        meas_map: dict[Any, Any] = {}
        if self.readout > 0.0:
            meas_map[qml.noise.meas_eq(qml.expval)] = self._readout_noise_fn()
            meas_map[qml.noise.meas_eq(qml.probs)] = self._readout_noise_fn()

        return qml.NoiseModel(model_map, meas_map=meas_map or None)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Compact one-line human-readable summary."""
        if self.is_noiseless:
            return "noiseless"
        parts = []
        if self.depolarizing_1q or self.depolarizing_2q:
            parts.append(f"depol={self.depolarizing_1q:.3g}/{self.depolarizing_2q:.3g}")
        if self.dephasing_1q or self.dephasing_2q:
            parts.append(f"deph={self.dephasing_1q:.3g}/{self.dephasing_2q:.3g}")
        if self.amplitude_damping_1q or self.amplitude_damping_2q:
            parts.append(f"amp={self.amplitude_damping_1q:.3g}/{self.amplitude_damping_2q:.3g}")
        if self.t1 is not None:
            parts.append(f"T1={self.t1:.3g} T2={self.t2:.3g} tg={self.gate_time:.3g}")
        if self.readout:
            parts.append(f"readout={self.readout:.3g}")
        return " ".join(parts)
