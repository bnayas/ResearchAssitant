# Unit Guide: `comms`

`comms` is the professor-mailbox persistence layer. It stores human-facing workflow updates and attachment metadata.

## What This Unit Owns

- mailbox and email-like data structures,
- attachment serialization,
- local persistence of professor-facing updates,
- compatibility between newer shared mailbox contracts and older stored email shapes.

## Main Files

- `project/comms/pi_email.py` defines the mailbox, email, and attachment structures used by the CLI, API, and workflow runner.

## Boundary Rules

- This package should stay small and persistence-focused.
- Message sequencing belongs in `research_platform.workflow`, not here.
- Cross-unit artifact typing belongs in `research_platform.contracts`, not here.

## Change Here When

Edit this package when you are:

- changing attachment serialization,
- changing mailbox file persistence behavior,
- improving compatibility between stored emails and shared mailbox messages.

## Useful Callers To Read

- `project/research_platform/workflow.py`
- `project/sim_tool/api_server.py`
- `project/sim_tool/run_directive.py`

## Common Risks

- Mixing presentation text policy into persistence code.
- Letting mailbox file output diverge between CLI and API paths.
