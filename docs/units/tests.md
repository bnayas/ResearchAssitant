# Unit Guide: `tests`

`tests` is the verification layer for the repository. When the prose and the code disagree, the tests are usually the better source of truth.

## What This Unit Owns

- deterministic backend verification,
- frontend logic verification outside the browser,
- regression coverage for the professor workflow and subsystem contracts.

## Test Groups

- `test_contract*.py` covers simulation contract rules.
- `test_tool.py`, `test_orchestrator.py`, and `test_api_server.py` cover the simulation facade and HTTP server behavior.
- `test_literature_review.py` covers literature matching, scope validation, and synthesis behavior.
- `test_writer.py` and `test_writer_api.py` cover writer behavior and the real service path over stored artifacts.
- `test_professor_workflow.py` covers the end-to-end professor workflow with injected assistants.
- `PIControl.test.js` and `ProfessorWorkflowView.test.js` cover frontend logic and contract normalization.

## How To Use It

- Read the focused test file before editing the corresponding subsystem.
- Prefer deterministic tests before live LLM or live literature runs.
- Add tests where a new contract or regression boundary is being introduced.

## Common Commands

From the repo root:

```bash
uv run --project project python -m pytest -q project/tests
node project/tests/PIControl.test.js
node project/tests/ProfessorWorkflowView.test.js
```

## Best Starting Points For New Developers

- `project/tests/test_professor_workflow.py`
- `project/tests/test_writer_api.py`
- `project/tests/test_api_server.py`
- `project/tests/test_orchestrator.py`
- `project/tests/test_literature_review.py`
