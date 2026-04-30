# Unit Guide: `examples`

`examples` contains runnable scripts and sample input files for manual exploration and smoke testing.

## What This Unit Owns

- example CLI or script entry points,
- the canonical sample directive input,
- quick manual reproduction paths for major subsystems.

## Main Files

- `project/examples/danino_repro.json` is the canonical professor-workflow directive sample.
- `project/examples/example_literature_review.py` exercises the literature subsystem directly.
- `project/examples/example_monte_carlo.py` exercises the simulation subsystem directly.
- `project/examples/example_orchestrator.py` is a lightweight orchestration example.

## How To Use It

Use this directory when you need:

- a quick smoke test,
- a reproducible demo command,
- a starting point for debugging with real-looking inputs.

## Boundary Rules

- Example paths should reflect real contracts and real workflows.
- Do not let examples become a second implementation path with special behavior.

## Common Risks

- Updating production contracts without updating the examples.
- Treating sample directives as permanent truth instead of as maintained fixtures.
