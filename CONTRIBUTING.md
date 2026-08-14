# Contributing

## Setup

```bash
python3 run.py                 # builds the environment, runs the demo
uv run --extra dev pytest -q   # 257 tests, ~20 seconds
```

Or manage the environment yourself: `python3 -m venv .venv && pip install -e ".[dev]"`.

## Before opening a PR

- `uv run --extra dev pytest -q` passes.
- New behavior gets a test. This project has no untested public function by design —
  find the nearest existing test file (`tests/test_*.py`) and add to it rather than
  starting a new one, unless you're adding a genuinely new module.
- If you touch `src/aegisq/ui/`, run `python3 run.py serve` and click through the
  panel(s) you changed. The test suite checks the data layer and the DOM contract;
  it does not render pixels.
- If a change affects a number quoted in `README.md` (the benchmark tables, the ZNE
  bias-reduction figures, the plateau decay rates), re-run the command named next to
  that table and update the number. Stale numbers in a scientific README are worse
  than no numbers.

## Code style

- No comments explaining *what* code does — names should carry that. A comment earns
  its place only by explaining *why*: a non-obvious constraint, a workaround, a
  citation. See the existing modules for the calibration.
- Don't add abstractions, config flags, or defensive code for cases that can't occur.
  Three similar lines beat a premature helper.
- Match the existing tone in user-facing text (CLI output, dashboard copy, README):
  plain statements of what was measured, not narration about how rigorous the
  measurement was.

## Reporting a bug

Include the output of `python3 run.py info` (library/interpreter versions) and, if it's
a dashboard bug, the panel name and the query string from the failing request (visible
in the browser's network tab).
