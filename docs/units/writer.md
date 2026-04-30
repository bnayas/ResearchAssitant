# Unit Guide: `writer`

`writer` is the grounded drafting subsystem. It turns stored upstream artifacts into a traceable review or paper draft.

## What This Unit Owns

- planning the draft structure from available artifacts,
- drafting sections in dependency order,
- validating grounding claims,
- tracking cross-section promises,
- assembling the final paper or review output.

## Main Files

- `project/writer/contract.py` defines the writing tasks, claims, sections, and review-cycle contracts.
- `project/writer/academic_writer.py` implements the main writing agent.
- `project/writer/validator.py` validates grounding and promise consistency.
- `project/writer/tools.py` defines artifact-query and task-request abstractions.
- `project/writer/promise_registry.py` tracks cross-section promises during assembly.
- `project/research_platform/writer_runtime.py` is the live runtime bridge that connects this package to the shared artifact registry.

## Internal Shape

The usual writer flow is:

1. plan the table of contents from available artifacts,
2. draft sections in dependency order,
3. validate claims and promise resolution,
4. materialize the final combined output.

## Boundary Rules

- The writer should not assume which peer assistants exist.
- The writer should consume artifacts through a real registry or tool interface.
- The writer should not know the professor-workflow state machine directly.

## Change Here When

Edit this package when you are:

- changing draft structure or prompts,
- improving grounding validation,
- changing artifact lookup behavior,
- tightening how missing or weak upstream evidence is handled.

If the change is about where artifacts come from or how phases are sequenced, start in `research_platform`.

## Useful Tests

- `project/tests/test_writer.py`
- `project/tests/test_writer_api.py`

## Common Risks

- Reintroducing mock artifact stores into real runtime paths.
- Letting claims survive without concrete upstream evidence.
- Hard-coding knowledge of peer agents into writer logic.
