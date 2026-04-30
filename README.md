# Research Assistant Platform

This repository contains a multi-agent research platform built around a professor-style workflow:

1. a PI issues a directive,
2. the literature assistant finds and interrogates the target paper,
3. the coding assistant designs and runs the reproduction,
4. the platform reports progress as mailbox-style updates with attachments,
5. the writer produces a grounded review draft from the accumulated artifacts.

The active package lives in [`project/`](project/). The repo root is a workspace wrapper plus the main documentation entry point.

If you are new to the codebase, start with [ONBOARDING_DOC.md](ONBOARDING_DOC.md).

## What Lives Here

- [`project/`](project/) is the installable Python package, frontend app, examples, and tests.
- [`project/research_platform/`](project/research_platform/) is the cross-unit coordination layer.
- [`project/sim_tool/`](project/sim_tool/) is the coding and simulation subsystem.
- [`project/literature_review/`](project/literature_review/) is the article and related-literature subsystem.
- [`project/writer/`](project/writer/) is the grounded drafting subsystem.
- [`project/reviewer/`](project/reviewer/) is the journal-review subsystem.
- [`project/frontend/`](project/frontend/) is the React/Vite UI.
- [`project/comms/`](project/comms/) is the professor-mailbox persistence layer.

## Architecture In One Page

The current platform contract is intentionally strict:

- Only `research_platform` should know about multiple assistants at once.
- Peer subsystems should not import one another's internals.
- Cross-unit handoffs should use shared contracts and persisted artifacts.
- The professor workflow is the primary product path.

At a high level, the platform looks like this:

1. `research_platform.workflow` receives a PI directive.
2. `research_platform.assistants` adapts the literature, coding, and writer subsystems behind shared ports.
3. `research_platform.registry` persists artifacts so later phases can consume them without direct subsystem coupling.
4. `comms.pi_email` stores mailbox-style updates for the PI, including attachment metadata and saved files.
5. `frontend` renders that workflow as the "Professor Desk" UI.

## Quick Start

From the repo root:

```bash
uv sync --project project --extra full
uv sync --project project --extra api --extra full
npm --prefix project/frontend install
uv run --project project python -m pytest -q project/tests
node project/tests/PIControl.test.js
node project/tests/ProfessorWorkflowView.test.js
npm --prefix project/frontend run build
uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8001
```

Useful entry points:

- API server: `uv run --project project python -m sim_tool.api_server --host 127.0.0.1 --port 8001`
- Professor-workflow example: `uv run --project project python -m sim_tool.run_directive project/examples/danino_repro.json`
- Literature example: `uv run --project project python project/examples/example_literature_review.py "neutral model with environmental noise"`
- Simulation example: `uv run --project project python project/examples/example_monte_carlo.py`

## Common Development Tasks

Run a focused backend suite:

```bash
uv run --project project python -m pytest -q project/tests/test_professor_workflow.py
uv run --project project python -m pytest -q project/tests/test_literature_review.py
uv run --project project python -m pytest -q project/tests/test_writer_api.py
```

Run the frontend locally:

```bash
npm --prefix project/frontend run dev
```

Validate the directive workflow end to end:

```bash
uv run --project project python -m sim_tool.run_directive project/examples/danino_repro.json
```

## Documentation Map

General entry points:

- [ONBOARDING_DOC.md](ONBOARDING_DOC.md)
- [project/AGENTS.md](project/AGENTS.md)
- [project/README.md](project/README.md)

Core unit guides:

- [docs/units/research_platform.md](docs/units/research_platform.md)
- [docs/units/sim_tool.md](docs/units/sim_tool.md)
- [docs/units/literature_review.md](docs/units/literature_review.md)
- [docs/units/writer.md](docs/units/writer.md)
- [docs/units/reviewer.md](docs/units/reviewer.md)
- [docs/units/frontend.md](docs/units/frontend.md)
- [docs/units/comms.md](docs/units/comms.md)

Support unit guides:

- [docs/units/examples.md](docs/units/examples.md)
- [docs/units/tests.md](docs/units/tests.md)
- [docs/units/docker.md](docs/units/docker.md)
- [docs/units/scripts.md](docs/units/scripts.md)
- [docs/units/programs.md](docs/units/programs.md)

## Repository Layout

- `project/` contains the installable package and almost all day-to-day code.
- `docs/units/` contains one Markdown file per major unit or support area.
- `ONBOARDING_DOC.md` is the guided reading path for new contributors.
- `pyproject.toml` at the repo root is only the workspace wrapper.

## Runtime Notes

- Package metadata that matters for Python lives in [`project/pyproject.toml`](project/pyproject.toml).
- Generated workflow outputs usually land under [`project/programs/`](project/programs/) unless the caller chooses a different output directory.
- The repo supports both deterministic local tests and live LLM / literature API paths. Use deterministic tests first; use live runs when validating end-to-end behavior or external integrations.
