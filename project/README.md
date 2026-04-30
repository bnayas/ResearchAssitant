# Project Package Guide

This directory contains the installable package, the frontend app, the example directives, and the test suite for the Research Assistant Platform.

Repo-level entry points:

- [../README.md](../README.md)
- [../ONBOARDING_DOC.md](../ONBOARDING_DOC.md)
- [AGENTS.md](AGENTS.md)

Subsystem guides:

- [../docs/units/research_platform.md](../docs/units/research_platform.md)
- [../docs/units/sim_tool.md](../docs/units/sim_tool.md)
- [../docs/units/literature_review.md](../docs/units/literature_review.md)
- [../docs/units/writer.md](../docs/units/writer.md)
- [../docs/units/reviewer.md](../docs/units/reviewer.md)
- [../docs/units/frontend.md](../docs/units/frontend.md)
- [../docs/units/comms.md](../docs/units/comms.md)

Support guides:

- [../docs/units/examples.md](../docs/units/examples.md)
- [../docs/units/tests.md](../docs/units/tests.md)
- [../docs/units/docker.md](../docs/units/docker.md)
- [../docs/units/scripts.md](../docs/units/scripts.md)
- [../docs/units/programs.md](../docs/units/programs.md)

## If You Are Working From `project/`

These are the equivalent commands when your shell is already in this directory:

```bash
uv sync --extra full
uv sync --extra api --extra full
uv run python -m pytest -q tests
node tests/PIControl.test.js
node tests/ProfessorWorkflowView.test.js
npm --prefix frontend install
npm --prefix frontend run build
uv run python -m sim_tool.api_server --host 127.0.0.1 --port 8001
uv run python -m sim_tool.run_directive examples/danino_repro.json
```

## Directory Guide

- `research_platform/` defines shared contracts, the artifact registry, assistant adapters, and the professor workflow runner.
- `sim_tool/` owns simulation design, generation, execution, analysis, the HTTP API server, and the directive CLI entry point.
- `literature_review/` owns target-paper lookup, article interrogation, related-literature search, scope filtering, and synthesis.
- `writer/` owns grounded drafting over stored artifacts.
- `reviewer/` owns the separate manuscript-review path.
- `frontend/` contains the React/Vite UI.
- `comms/` stores professor mailbox messages and attachment metadata.
- `examples/` contains runnable examples and the canonical directive sample.
- `tests/` contains both Python and Node-based verification.
- `docker/` and `scripts/` contain the launcher runtime image and support scripts.
- `programs/` is where generated workflow outputs are commonly written.

## Packaging Notes

- [`pyproject.toml`](pyproject.toml) in this directory is the package definition used by `uv`.
- The repo-root [`../pyproject.toml`](../pyproject.toml) only defines the workspace wrapper and points at this package.
