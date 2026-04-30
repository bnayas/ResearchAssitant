# Unit Guide: `programs`

`programs` is the default runtime output area for directive runs and related generated artifacts.

## What This Unit Owns

- per-run output directories,
- saved article matches and article briefs,
- literature synthesis artifacts,
- professor mailbox messages and attachments,
- other generated run outputs created by live workflow execution.

## What You Will See Here

Typical contents include:

- `article/` directories with saved article match and brief files,
- `literature/` or synthesis files,
- `emails/` with persisted professor updates,
- run-specific folders such as `danino-repro-001/`.

## How To Use It

Use this directory when you need to:

- inspect what a live directive run produced,
- debug attachment persistence,
- compare workflow outputs across runs.

## Boundary Rules

- Treat this directory as generated output, not source code.
- Do not build product logic that depends on one specific generated run being present.

## Common Risks

- Hand-editing generated artifacts and forgetting that they are not canonical source.
- Checking in large or noisy run outputs without intent.
