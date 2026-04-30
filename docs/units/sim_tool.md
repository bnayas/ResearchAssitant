# Unit Guide: `sim_tool`

`sim_tool` is the coding and simulation subsystem. It turns a research instruction into a simulation spec, runnable code, execution results, and analysis.

## What This Unit Owns

- the typed simulation lifecycle,
- simulation code generation,
- deterministic execution,
- result analysis and patch suggestions,
- the FastAPI server entry points,
- the CLI entry point for running a directive.

## Main Files

- `project/sim_tool/contract.py` defines the public simulation contracts.
- `project/sim_tool/tool.py` exposes `SimulationTool`, the subsystem facade.
- `project/sim_tool/orchestrator.py` runs the stateful simulation approval flow.
- `project/sim_tool/designer.py` produces candidate specs and clarification requests.
- `project/sim_tool/launcher.py` executes generated code deterministically.
- `project/sim_tool/analyst.py` inspects outputs and proposes safe follow-up actions.
- `project/sim_tool/api_server.py` exposes simulation, literature, writer, and directive endpoints over HTTP.
- `project/sim_tool/run_directive.py` is the CLI entry point for sample professor workflows.
- `project/sim_tool/research_directive.py` preserves the old directive-facing API surface while delegating to `research_platform`.

## Internal Shape

The usual flow is:

1. `designer.py` creates or repairs a spec,
2. `tool.py` materializes runnable artifacts,
3. `launcher.py` executes them,
4. `analyst.py` turns outputs into a review or patch,
5. `orchestrator.py` manages approval gates and long-running execution state.

## Boundary Rules

- This package owns simulation behavior, not full cross-unit orchestration.
- It should not depend on `literature_review` or `writer` internals.
- External callers should use contracts, facades, or assistant ports instead of reaching into internals.

## Change Here When

Edit this package when you are:

- changing spec or run contracts,
- improving the launcher or analysis logic,
- adding simulation-facing API endpoints,
- fixing the approval flow for simulation work.

If the change is really about how literature, coding, and writing are sequenced together, the change probably belongs in `research_platform`.

## Useful Tests

- `project/tests/test_contract.py`
- `project/tests/test_tool.py`
- `project/tests/test_orchestrator.py`
- `project/tests/test_api_server.py`

## Common Risks

- Mixing subsystem-local state transitions with professor-workflow orchestration.
- Letting non-JSON or malformed LLM outputs escape typed validation.
- Adding behavior in `research_directive.py` that should live in shared workflow wiring.
