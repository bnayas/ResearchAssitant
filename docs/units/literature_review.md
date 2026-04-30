# Unit Guide: `literature_review`

`literature_review` owns target-paper lookup, article interrogation, related-paper search, scope validation, and literature synthesis.

## What This Unit Owns

- finding the primary paper from a topic hint or query,
- extracting a reproduction-oriented article brief,
- searching and filtering related literature,
- normalizing backend paper results into typed platform-facing artifacts,
- auditing the final literature artifact for obvious deterministic failures.

## Main Files

- `project/literature_review/contract.py` defines the typed literature contracts.
- `project/literature_review/article_parser.py` finds the target paper and extracts article-level answers.
- `project/literature_review/reviewer.py` runs multi-round related-literature search and synthesis.
- `project/literature_review/scope_validator.py` decides which papers are in scope.
- `project/literature_review/auditor.py` applies deterministic post-hoc checks to the result.
- `project/literature_review/director_bridge.py` preserves older call shapes while the package is used through the new assistant layer.
- `project/literature_review/search_backends/` contains the live search backends.

## Internal Shape

The target-paper path and the related-literature path are separate:

1. `article_parser.py` resolves the paper and builds the article brief.
2. `reviewer.py` expands keywords and gathers candidates from search backends.
3. `scope_validator.py` accepts or rejects candidates.
4. `reviewer.py` synthesizes accepted papers.
5. `auditor.py` flags deterministic issues in the final artifact.

## Boundary Rules

- This package should not import `sim_tool` or `writer` internals.
- Higher layers should receive normalized literature artifacts, not raw backend DTOs.
- This package should not know the full professor-workflow state machine.

## Change Here When

Edit this package when you are:

- improving article matching or article-brief extraction,
- changing scope acceptance rules,
- tuning related-literature search behavior,
- adding or updating search backends,
- tightening literature artifact normalization.

## Useful Tests

- `project/tests/test_literature_review.py`
- `project/tests/test_director_bridge.py`

## Common Risks

- Leaking backend-specific paper shapes above the assistant layer.
- Treating article-brief extraction and literature synthesis as if they were the same job.
- Regressing rejection-event semantics in `scope_validator.py`.
