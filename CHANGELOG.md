# Changelog

## 0.1.0

First release.

### Getting started

- `run.py`, a zero-prerequisite launcher at the repository root: `git clone … && cd aegisq`
  then `python3 run.py`. It needs nothing but a Python 3 interpreter and resolves the
  environment itself, trying an existing install, then an existing `.venv`, then uv (on
  `PATH`, via pip, or downloaded as an official release binary with the standard library
  alone), then plain `venv` + pip. Everything it creates stays inside the repository.
- `aegisq` console script and `python -m aegisq`. A bare invocation runs a ~25 second
  self-contained demo covering install check, layer catalog, `nn.Sequential` integration,
  gradient-chain verification, ZNE, symmetry verification, plateau scaling and a training run.
- Subcommands `info`, `plateau`, `zne`, `symmetry`, `benchmark` (with `--quick` and `--csv`)
  and `choose-ansatz`. All arguments are forwarded through `run.py`.
- GitHub Actions CI: the test suite, the console entry point, and a standalone wheel install
  across Ubuntu and macOS on Python 3.10-3.13, plus a separate job that runs `python3 run.py`
  in a bare checkout to keep the zero-prerequisite claim honest. `CONTRIBUTING.md` covers local
  setup and PR expectations.

### Dashboard

- `aegisq serve` starts a local dashboard: seven panels covering zero-noise extrapolation against
  ground truth, barren-plateau scaling, signal retention under noise, symmetry verification, live
  streamed training, the layer catalog, and the measured limits.
- Built on `http.server` with hand-written SVG charts -- no web framework, no CDN, no build step.
  Binds to loopback only and refuses anything else, since the endpoints execute simulations on
  request. Responses carry a `default-src 'self'` content-security policy.
- Non-finite results (an infinite post-selection overhead, an undefined decay rate) are converted
  to `null` before serialisation, because `json.dumps` emits bare `NaN` and `JSON.parse` rejects it.
- Four switchable design directions -- Instrument (dark minimalist), Paper (editorial/Swiss),
  Bento (modular grid) and Terminal (retro-futurist) -- each redefining the full token contract
  including chart series colours and, for Bento, the panel layout. The choice persists in
  `localStorage` and defaults from `prefers-color-scheme`. Contract tests assert every style
  defines all 40 tokens, keeps its six series colours distinct, and never draws data ink in the
  background colour.
- Every style sets `color-scheme` both in CSS and as an inline style set from JavaScript. Chromium
  gives a `:root` rule's `color-scheme` priority over a later, equal-specificity attribute-selector
  rule regardless of source order (confirmed empirically), so the stylesheet declaration alone is
  not sufficient; the inline style, set on every switch, is what actually controls it.
- ZNE and signal-retention panels run `trials` independent random-input draws (default 8 and 5)
  and report mean +/- sample standard deviation, with error bars drawn on the chart, rather than
  the result of one random input. `bias_reduction` and `estimate` are means over trials;
  `coverage` reports what fraction of trials produced a valid exponential fit.
- Semantic stat-card colours (`--good`/`--warn`/`--bad`) and body text meet WCAG 2.1 AA contrast
  (>= 4.5:1) against both card backgrounds in every style, checked by a dedicated contrast test
  that computes relative luminance from the CSS custom properties rather than eyeballing it --
  this is what caught Bento's original semantic palette sitting at 2.5-3.0:1.
- Keyboard focus gets an explicit `:focus-visible` ring (the browser default clashes with at
  least one of the four palettes), and every panel's status region is `aria-live="polite"` so a
  screen reader announces "running..." and the completion time without the user re-finding it.
- Stat-card grid uses a wider minimum column width so multi-line values (e.g. "-0.1823 +/- 0.287")
  do not wrap mid-number on a phone-width viewport.

### Layers

- `QuantumLayer`, a `torch.nn.Module` wrapping a PennyLane QNode: batched input, registered
  circuit parameters, `state_dict` round-trip, `deepcopy`, and direct use inside
  `torch.nn.Sequential`.
- Resilient ansaetze: `LocalEntangler` (brickwork, strictly nearest-neighbour),
  `ParticleConserving` (U(1) Givens rotations), `PermutationEquivariant` (width-independent
  parameter count), `Z2Equivariant`.
- Baselines wrapping `BasicEntanglerLayers` and `StronglyEntanglingLayers` for like-for-like
  comparison under an identical noise model.
- Encodings: `angle`, `dense_angle`, `iqp`, `amplitude`, `excitation`, plus data re-uploading.
- Measurements: `local_z` (default), `global_z`, `local_zz`, `probs`, or a custom observable list.
- String registry with `register_ansatz` / `register_encoding` / `register_measurement`.

### Mitigation

- `ZNE` with global unitary folding and virtual noise-rate scaling.
- Richardson, polynomial, linear and exponential extrapolators, all differentiable. The
  exponential model falls back to a linear fit per output element where its assumptions
  (constant sign, magnitude decaying with noise) do not hold.
- `noise_amplification` reports the sampling-variance cost of a chosen coefficient set.
- `SymmetryVerification` post-selects onto a conserved sector and exposes the acceptance rate
  through `sector_weight`.

### Noise

- `NoiseSpec` covering depolarizing, dephasing, amplitude damping, T1/T2 thermal relaxation and
  readout error, with presets including a composite `hardware_like` profile.
- Channels are selected by gate arity, so folded and adjointed gates and custom templates all
  accumulate error correctly while state preparations stay ideal.
- `spec.scaled(λ)` uses exact channel composition rather than a first-order approximation.

### Benchmark

- `NoiseBenchmark` / `standard_benchmark` sweep models against noise profiles under matched
  data, seeds and optimiser budget; results export to CSV or a printable summary.
- `gradient_variance`, `barren_plateau_scan` and `mitigation_bias` diagnostics.
- Dependency-free synthetic datasets: `two_moons`, `circles`, `parity`, `linearly_separable`.

### Notes

- `QuantumLayer(..., dtype=torch.float64)` pins torch's default dtype for the duration of a
  circuit call. PennyLane's mixed-state simulator takes its density-matrix dtype from that
  global, so without it a float64 layer would still simulate in single precision.
- `LocalEntangler`'s default `entangler="cz"` is diagonal, so a `measurement="local_z"`
  read-out cannot see globally-correlated targets. On 4-qubit parity the default pairing
  trains to chance while `entangler="cnot"` or `measurement="global_z"` reaches 100%.
  Documented on the class and reproduced by `examples/06_choosing_an_ansatz.py`; the default
  stays CZ because it is hardware-native and holds gradient variance better with width.
