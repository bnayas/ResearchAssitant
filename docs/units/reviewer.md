# Unit Guide: `reviewer`

`reviewer` is the manuscript-review subsystem. It is adjacent to the professor workflow rather than embedded inside it.

## What This Unit Owns

- parsing a manuscript into reviewable structure,
- scoring review dimensions,
- dispatching mathematical claims to specialized math tools,
- producing an accept, reject, or corrections-needed decision,
- generating annotated-diff style outputs when a revision is required.

## Main Files

- `project/reviewer/contract.py` defines the review contracts.
- `project/reviewer/journal_reviewer.py` implements the review pipeline.
- `project/reviewer/math_tools.py` routes math checks to the appropriate verifier.
- `project/reviewer/diff.py` builds annotated article and diff outputs.

## Boundary Rules

- This package should reason from manuscript text, not from simulation or literature internals.
- Math verification should go through the math-tool layer, not free-form guessing.
- It should remain decoupled from the writer runtime and directive orchestration paths.

## Change Here When

Edit this package when you are:

- adding a new review dimension,
- improving the decision logic,
- improving math-tool routing,
- improving annotated diff output.

## Useful Tests

- `project/tests/test_reviewer.py`

## Common Risks

- Pulling in assumptions from the professor workflow even though this package is conceptually separate.
- Letting unsupported math checks crash the review pipeline instead of returning a typed result.
