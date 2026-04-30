# Unit Guide: `research_platform`

`research_platform` is the shared coordination layer. If work crosses subsystem boundaries, it should usually pass through this package.

## What This Unit Owns

- shared assistant identities,
- task and artifact contracts,
- artifact persistence,
- local assistant adapters,
- the professor workflow runner,
- the writer runtime bridge over the artifact registry.

## Main Files

- `project/research_platform/contracts.py` defines assistant IDs, task envelopes, artifact references, mailbox messages, and assistant port protocols.
- `project/research_platform/article_lookup.py` provides the deterministic fallback for article-query shaping when the live LLM-prepared JSON lookup spec needs normalization or recovery.
- `project/research_platform/registry.py` implements the concrete artifact registry used by live paths and tests.
- `project/research_platform/workflow.py` composes the professor workflow over injected literature, coding, and writer assistants.
- `project/research_platform/assistants/literature.py` adapts the local literature subsystem to the shared literature port, including LLM-backed structured article-lookup preparation, clarification requests, and provider-cooldown handling.
- `project/research_platform/assistants/coding.py` adapts the simulation subsystem to the shared coding port.
- `project/research_platform/assistants/writer.py` adapts the writer subsystem to the shared writer port.
- `project/research_platform/assistants/backends.py` builds runtime-selected service backends and literature-review bridges.
- `project/research_platform/assistants/__init__.py` re-exports the assistant-layer public API.
- `project/research_platform/steering.py` defines the PI steering checkpoints and the control channel used by streamed directive runs.
- `project/research_platform/wiring.py` builds the real workflow dependencies for CLI and API entry points.
- `project/research_platform/writer_runtime.py` runs the writer against a concrete artifact registry rather than a mock toolbox.

## Boundary Rules

- This package is the only place that should know all assistants together.
- Cross-unit payloads should be expressed as shared contracts here.
- Peer subsystems should not reach into each other when this package can carry the handoff.

## How It Fits Into The Product

In the main directive flow:

1. the API or CLI creates a directive and mailbox,
2. `workflow.py` sequences the phases,
3. the `assistants/` package delegates each phase to the owning subsystem,
4. `registry.py` persists artifacts for downstream use,
5. mailbox messages are emitted at stable checkpoints, including clarification gates before ambiguous article searches.

## Change Here When

Edit this package when you are:

- adding a new cross-unit artifact kind,
- changing mailbox semantics,
- changing professor-workflow phase sequencing,
- wiring a subsystem behind a shared assistant port,
- fixing writer runtime integration over stored artifacts.

Do not edit this package just to add subsystem-local behavior that belongs in `sim_tool`, `literature_review`, `writer`, or `reviewer`.

## Useful Tests

- `project/tests/test_professor_workflow.py`
- `project/tests/test_writer_api.py`

## Common Risks

- Reintroducing direct peer-to-peer imports between subsystems.
- Letting raw subsystem-specific DTOs leak above the assistant layer.
- Adding runtime shortcuts here that diverge from the API and CLI paths.
- Retrying provider-backed article search after HTTP `429` instead of honoring backend cooldown state.
