# sim_tool Context-Overflow Patch

## Problem

When the simulation phase receives a long research brief (article text +
literature synthesis + procedure), the designer LLM returns prose instead
of JSON, causing:

    ValueError: [session_id] LLM returned non-JSON output

Root cause: `mediated_services._compose_description` concatenated full
artifact text verbatim, producing 8k–15k char descriptions that overflow
the LLM context window.

## Files in this patch

```
sim_tool/
  staged_designer.py        ← NEW: 4-phase staged code generation
  designer.py               ← REPLACE: adds staged fallback + description compression
  description_builder.py    ← NEW: precision-ordered description builder

research_platform/
  mediated_services_compose_diff.py  ← INSTRUCTIONS for the one-line method change
```

## Installation

### Step 1 — Copy the new sim_tool files

```bash
cp sim_tool/staged_designer.py      project/sim_tool/
cp sim_tool/designer.py             project/sim_tool/          # replaces original
cp sim_tool/description_builder.py  project/sim_tool/
```

### Step 2 — Patch mediated_services.py (one method, ~20 lines)

Open `project/research_platform/mediated_services.py` and find
`SimulationDesignerService._compose_description`. Replace the entire
method body with:

```python
def _compose_description(self, task: TaskEnvelope, sources: list[ArtifactRef]) -> str:
    from sim_tool.description_builder import build_designer_description
    return build_designer_description(
        task_instructions=task.instructions,
        sources=sources,
        task_metadata=dict(task.metadata or {}),
    )
```

The full diff is documented in `research_platform/mediated_services_compose_diff.py`.

## What changed and why

### description_builder.py (NEW)
Replaces the free-form concatenation with a precision-ordered builder:

| Priority | Content | Cap |
|---|---|---|
| 1 | PI instruction | unlimited |
| 2 | Selected simulation targets | unlimited |
| 3 | Model description (from brief metadata) | unlimited |
| 4 | Key parameters as JSON | unlimited |
| 5 | Procedure | 800 chars |
| 6 | Expected figures | 5 items |
| 7 | Literature synthesis | 600 chars |
| 8 | Other source summaries | 300 chars × 2 |
| 9 | Steering notes | unlimited |
| **Total** | | **6 000 chars hard cap** |

### staged_designer.py (NEW)
When the monolithic JSON call fails (returns prose), the designer
automatically falls back to a 4-phase staged generation:

```
Phase 1  Structural skeleton     — variables, state_fields, stop condition metadata
Phase 2  Code design             — function signatures, algorithm notes, invariants
Phase 3  Per-function impl       — one LLM call per code field, narrow prompt
Phase 4  Assembly + validation   — SpecValidator gate, targeted repair on failure
```

Each phase sends ≤ ~1 500 tokens, well within any context window.

### designer.py (REPLACED)
- Compresses descriptions > 4 000 chars before the initial monolithic call
- Catches non-JSON response and transparently falls back to `StagedSpecGenerator`
- Adds a staged-repair path in the auto-repair loop (targeted per-field repair
  rather than full regeneration)
- All external contracts (`start`, `answer`, `approve`, `reject`) unchanged

## No other files changed

`tool.py`, `orchestrator.py`, `contract.py`, `codegen.py`, `runner.py`,
`launcher.py`, `analyst.py`, and all `literature_review/` and `writer/`
files are untouched.
