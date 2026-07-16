"""The :class:`QuantumLayer` -- a ``torch.nn.Module`` wrapping a noisy QNode.

The layer owns everything a variational circuit needs (encoding, ansatz,
measurement, device, noise profile) and exposes it through the ordinary
``nn.Module`` contract: it registers its circuit parameters, accepts batched
tensors, and can be dropped straight into ``torch.nn.Sequential``.

Gradients flow through PennyLane's torch interface.  With ``diff_method="backprop"``
(the default on a simulator) the whole density-matrix evolution is autograd
tape; with ``diff_method="parameter-shift"`` the layer produces exact hardware
gradients and still returns a tensor with a live ``grad_fn``.  Nothing in the
class ever calls ``.detach()`` or ``.item()`` on a circuit output.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Literal, Sequence

import pennylane as qml
import torch
from torch import nn

from ..noise import NoiseSpec, get_preset
from ..noise.presets import depolarizing
from .ansatz import Ansatz, InitStrategy
from .encodings import Encoding
from .measurements import Measurement
from .registry import resolve_ansatz, resolve_encoding, resolve_measurement

__all__ = ["QuantumLayer", "FoldingMode"]

FoldingMode = Literal["global", "noise"]


def _resolve_noise(noise: Any) -> NoiseSpec:
    if noise is None:
        return NoiseSpec()
    if isinstance(noise, NoiseSpec):
        return noise
    if isinstance(noise, str):
        return get_preset(noise)
    if isinstance(noise, dict):
        return NoiseSpec(**noise)
    if isinstance(noise, (int, float)):
        return depolarizing(float(noise))
    raise TypeError(f"cannot interpret {noise!r} as a NoiseSpec")


@contextlib.contextmanager
def _simulation_precision(dtype: torch.dtype) -> Iterator[None]:
    """Run a block with torch's default dtype pinned to ``dtype``.

    PennyLane's mixed-state simulator builds its density matrix through
    ``torch``'s *global* default dtype, not from the dtype of the circuit
    parameters.  Without this, a layer constructed with ``dtype=torch.float64``
    would still be simulated in single precision and its expectation values
    would carry ~1e-7 of numerical noise -- invisible in training, but enough to
    swamp a finite-difference check or a careful zero-noise extrapolation.

    The scope only ever *raises* precision, and the default dtype is a
    process-wide setting, so it is restored on exit.  It covers the forward
    pass and -- for ``diff_method="backprop"``, where the backward pass merely
    replays the graph built here -- the gradient as well.  Parameter-shift
    re-executes its tapes during ``backward()``, outside this scope; that path
    is meant for finite-shot execution, where sampling noise dwarfs the
    difference.
    """
    previous = torch.get_default_dtype()
    if dtype != torch.float64 or previous == dtype:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _stack_results(result: Any) -> torch.Tensor:
    """Normalise a QNode return value to a single ``(batch, out_features)`` tensor."""
    if isinstance(result, (tuple, list)):
        return torch.stack([torch.as_tensor(r) for r in result], dim=-1)
    return torch.as_tensor(result)


class QuantumLayer(nn.Module):
    """A noise-aware variational quantum layer.

    Parameters
    ----------
    n_qubits:
        Register width.
    n_layers:
        Number of ansatz repetitions.
    ansatz:
        Name, class or instance from :mod:`aegisq.layers.ansatz`.  Defaults to
        the shallow nearest-neighbour :class:`~aegisq.layers.LocalEntangler`.
    encoding:
        Name, class or instance from :mod:`aegisq.layers.encodings`.
    measurement:
        Name, class, instance, or a list of PennyLane observables.  The default
        ``"local_z"`` is a local cost function, which keeps gradient variance
        far healthier than a global parity observable.
    noise:
        ``None`` for an exact simulation, a :class:`~aegisq.noise.NoiseSpec`, a
        preset name such as ``"hardware_like"``, a keyword dict, or a float
        (shorthand for a depolarizing profile with that one-qubit rate).
    device:
        PennyLane device name or instance.  Defaults to ``"default.mixed"``
        when a noise profile is present and ``"default.qubit"`` otherwise.
    diff_method:
        Passed to the QNode.  Defaults to ``"backprop"`` for analytic execution
        and ``"parameter-shift"`` when ``shots`` is set.
    shots:
        Finite sampling budget; ``None`` runs analytically.
    data_reupload:
        Re-apply the encoding before every ansatz layer.  Raises the expressivity
        of a shallow circuit without deepening the entangling structure.
    init:
        Weight initialisation: ``"uniform"`` over the full circle (default),
        ``"normal"``, ``"small"`` (near-identity, plateau-friendly) or ``"zeros"``.
    seed:
        Seed for the weight draw, making a benchmark run reproducible.
    trainable_input_scaling:
        Add a learnable per-feature scale applied before encoding.

    Examples
    --------
    >>> import torch
    >>> from aegisq import QuantumLayer
    >>> layer = QuantumLayer(4, n_layers=2, noise="hardware_like", seed=0)
    >>> model = torch.nn.Sequential(torch.nn.Linear(8, 4), layer, torch.nn.Linear(4, 2))
    >>> model(torch.randn(5, 8)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        *,
        ansatz: str | type[Ansatz] | Ansatz = "local_entangler",
        encoding: str | type[Encoding] | Encoding = "angle",
        measurement: Any = "local_z",
        noise: Any = None,
        device: Any = None,
        diff_method: str | None = None,
        shots: int | None = None,
        data_reupload: bool = False,
        init: InitStrategy = "uniform",
        seed: int | None = None,
        trainable_input_scaling: bool = False,
        dtype: torch.dtype | None = None,
        ansatz_kwargs: dict | None = None,
        encoding_kwargs: dict | None = None,
        measurement_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.wires = list(range(self.n_qubits))
        self.dtype = dtype or torch.get_default_dtype()

        self.ansatz = resolve_ansatz(ansatz, **(ansatz_kwargs or {}))
        self.encoding = resolve_encoding(encoding, **(encoding_kwargs or {}))
        self.measurement = resolve_measurement(measurement, **(measurement_kwargs or {}))
        self.noise = _resolve_noise(noise)

        self.data_reupload = bool(data_reupload)
        if self.data_reupload and not self.encoding.repeatable:
            raise ValueError(
                f"encoding {self.encoding.name!r} cannot be re-uploaded; it prepares a "
                "state rather than rotating one. Use 'angle', 'dense_angle' or 'iqp'."
            )

        self.in_features = self.encoding.in_features(self.n_qubits)
        self.out_features = self.measurement.out_features(self.n_qubits)

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        self.weights = nn.Parameter(
            self.ansatz.init_weights(
                self.n_layers,
                self.n_qubits,
                strategy=init,
                generator=generator,
                dtype=self.dtype,
            )
        )
        if trainable_input_scaling:
            self.input_scaling = nn.Parameter(torch.ones(self.in_features, dtype=self.dtype))
        else:
            self.register_parameter("input_scaling", None)

        self.shots = shots
        self.diff_method = diff_method or ("backprop" if shots is None else "parameter-shift")
        self._device_spec = device
        self._device = self._make_device(device)
        self._qnode_cache: dict[tuple[str, float], Any] = {}

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _make_device(self, device: Any):
        if device is not None and not isinstance(device, str):
            return device
        name = device or ("default.mixed" if not self.noise.is_noiseless else "default.qubit")
        if not self.noise.is_noiseless and name in ("default.qubit", "lightning.qubit"):
            raise ValueError(
                f"device {name!r} is a pure-state simulator and cannot apply noise channels; "
                "use 'default.mixed' (the default when a noise profile is given)."
            )
        return qml.device(name, wires=self.n_qubits)

    def _circuit(self, x: torch.Tensor, weights: torch.Tensor):
        if self.data_reupload:
            for layer in range(self.n_layers):
                self.encoding.apply(x, self.wires)
                self.ansatz.apply(weights[layer : layer + 1], self.wires)
        else:
            self.encoding.apply(x, self.wires)
            self.ansatz.apply(weights, self.wires)
        return self.measurement.apply(self.wires)

    def _qnode(self, scale: float = 1.0, folding: FoldingMode = "global"):
        """Build (and cache) the QNode for a given noise-scale factor."""
        key = (folding if scale != 1.0 else "base", float(scale))
        cached = self._qnode_cache.get(key)
        if cached is not None:
            return cached

        qnode = qml.QNode(
            self._circuit, self._device, interface="torch", diff_method=self.diff_method
        )
        spec = self.noise

        if scale != 1.0:
            if folding == "global":
                if not self.encoding.foldable:
                    raise ValueError(
                        f"encoding {self.encoding.name!r} contains a state preparation and "
                        "cannot be unitary-folded; use ZNE(..., folding='noise')."
                    )
                qnode = qml.noise.fold_global(qnode, scale)
            elif folding == "noise":
                if spec.is_noiseless:
                    raise ValueError(
                        "folding='noise' scales the simulated error rates, but this layer "
                        "has no noise profile; pass noise=... or use folding='global'."
                    )
                spec = spec.scaled(scale)
            else:
                raise ValueError(f"unknown folding mode {folding!r}")

        # Noise must be inserted *after* folding so the amplified gate count is
        # what accumulates error -- that is the entire point of the technique.
        noise_model = spec.to_noise_model()
        if noise_model is not None:
            qnode = qml.add_noise(qnode, noise_model)
        if self.shots is not None:
            qnode = qml.set_shots(qnode, self.shots)

        self._qnode_cache[key] = qnode
        return qnode

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _prepare_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(self.weights.dtype)
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(
                f"{type(self).__name__} expects a (batch, {self.in_features}) tensor, "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{type(self).__name__} expects {self.in_features} input features for "
                f"encoding {self.encoding.name!r} on {self.n_qubits} qubits, "
                f"got {x.shape[-1]}"
            )
        if self.input_scaling is not None:
            x = x * self.input_scaling
        return x, unbatched

    def run_at_scale(
        self, x: torch.Tensor, scale: float = 1.0, folding: FoldingMode = "global"
    ) -> torch.Tensor:
        """Evaluate the layer with the noise amplified by ``scale``.

        This is the primitive :class:`~aegisq.mitigation.ZNE` drives.  The
        returned tensor keeps its ``grad_fn``, so the extrapolation stacked on
        top of it stays part of the same autograd graph.
        """
        x, unbatched = self._prepare_input(x)
        with _simulation_precision(self.dtype):
            out = _stack_results(self._qnode(scale, folding)(x, self.weights))
        out = out.to(self.dtype)
        return out.squeeze(0) if unbatched else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the circuit at the layer's nominal noise level."""
        return self.run_at_scale(x, 1.0)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def n_circuit_parameters(self) -> int:
        """Number of trainable circuit angles (excluding input scaling)."""
        return int(self.weights.numel())

    @property
    def two_qubit_gates(self) -> int:
        """Two-qubit gates in the ansatz -- the leading term of the error budget."""
        return self.ansatz.two_qubit_gates(self.n_layers, self.n_qubits)

    @property
    def circuit_depth(self) -> int:
        """Two-qubit depth of the ansatz in parallel cycles."""
        return self.ansatz.circuit_depth(self.n_layers, self.n_qubits)

    @property
    def symmetry(self) -> str | None:
        """The symmetry conserved by the *whole* circuit, or ``None``.

        Only reported when encoding and ansatz agree; an ansatz that conserves
        particle number preceded by an angle encoding does not give the circuit
        a well-defined sector.
        """
        if self.ansatz.symmetry is not None and self.ansatz.symmetry == self.encoding.symmetry:
            return self.ansatz.symmetry
        return None

    def draw(self, x: torch.Tensor | None = None, *, scale: float = 1.0, **kwargs) -> str:
        """Return an ASCII drawing of the executed circuit, noise channels included."""
        if x is None:
            x = torch.zeros(self.in_features, dtype=self.weights.dtype)
        x, _ = self._prepare_input(x)
        return qml.draw(self._qnode(scale), **kwargs)(x[:1], self.weights)

    def extra_repr(self) -> str:
        bits = [
            f"n_qubits={self.n_qubits}",
            f"n_layers={self.n_layers}",
            f"ansatz={self.ansatz.name!r}",
            f"encoding={self.encoding.name!r}",
            f"measurement={self.measurement.name!r}",
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"noise={self.noise.describe()!r}",
        ]
        if self.shots is not None:
            bits.append(f"shots={self.shots}")
        if self.data_reupload:
            bits.append("data_reupload=True")
        return ", ".join(bits)

    # QNodes hold a device handle and are rebuilt on demand; keeping them out of
    # the pickle keeps ``copy.deepcopy(model)`` and checkpointing cheap.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_qnode_cache"] = {}
        state["_device"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._device is None:
            self._device = self._make_device(self._device_spec)
