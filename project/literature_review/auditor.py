"""
literature_review/auditor.py
─────────────────────────────
LiteratureAuditor — deterministic structural audit of a LiteratureReviewArtifact.

Design principles:
  - No LLM calls. Pure rule-based checks on the artifact structure.
  - The Director should NOT validate content; the auditor is separate.
  - Returns LiteratureAuditResult which the Director uses to decide:
      passed=True  → store artifact, continue pipeline
      passed=False → flag for re-run or human escalation
  - Scope re-validation here is a second deterministic pass (year bounds only).
    Semantic scope was already checked per-paper by the reviewer's scope_validator.
    This gives defence-in-depth without LLM cost duplication.

Check catalogue (8 checks):
  [D1] Year-min violations in accepted papers
  [D2] Year-max violations in accepted papers
  [D3] Accepted count exceeds scope.max_papers
  [D4] Papers with no abstract (quality floor)
  [D5] Papers with no URL (unreachable)
  [D6] Synthesis is empty
  [D7] Artifact status vs. is_sufficient consistency
  [D8] Search log is empty (reviewer produced no audit trail)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .contract import (
    LiteratureAuditResult,
    LiteratureReviewArtifact,
    LiteraturePaper,
    ScopeConstraint,
)

log = logging.getLogger(__name__)


@dataclass
class _CheckResult:
    check_id: str
    passed: bool
    detail: str


class LiteratureAuditor:
    """
    Audits a LiteratureReviewArtifact against the original ScopeConstraint.

    Usage:
        auditor = LiteratureAuditor()
        result = auditor.audit(artifact, task.scope)
    """

    def audit(
        self,
        artifact: LiteratureReviewArtifact,
        scope: ScopeConstraint,
    ) -> LiteratureAuditResult:
        checks: list[_CheckResult] = [
            self._check_year_min(artifact.papers, scope),
            self._check_year_max(artifact.papers, scope),
            self._check_max_papers(artifact.papers, scope),
            self._check_empty_abstracts(artifact.papers),
            self._check_no_url(artifact.papers),
            self._check_synthesis(artifact),
            self._check_status_consistency(artifact),
            self._check_search_log(artifact),
        ]

        # Scope violations = papers that failed year checks
        violations = list({
            p.title
            for p in artifact.papers
            if (scope.year_min and p.year is not None and p.year < scope.year_min)
            or (scope.year_max and p.year is not None and p.year > scope.year_max)
        })

        failed_checks = [c for c in checks if not c.passed]
        passed = len(failed_checks) == 0

        notes_parts = []
        for c in checks:
            status = "✓" if c.passed else "✗"
            notes_parts.append(f"[{c.check_id}] {status} {c.detail}")
        notes = "\n".join(notes_parts)

        if failed_checks:
            log.warning(
                "[%s] Audit FAILED. %d checks failed: %s",
                artifact.task_id,
                len(failed_checks),
                [c.check_id for c in failed_checks],
            )
        else:
            log.info("[%s] Audit PASSED. %d papers accepted.", artifact.task_id, artifact.accepted_count)

        return LiteratureAuditResult(
            task_id=artifact.task_id,
            branch_id=artifact.branch_id,
            passed=passed,
            scope_violations_found=violations,
            sufficiency_verdict=artifact.is_sufficient,
            sufficiency_reason=(
                artifact.insufficiency_reason or "Sufficient"
            ),
            papers_accepted=artifact.accepted_count,
            papers_removed=artifact.removed_count,
            auditor_notes=notes,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Individual checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_year_min(
        self, papers: list[LiteraturePaper], scope: ScopeConstraint
    ) -> _CheckResult:
        if not scope.year_min:
            return _CheckResult("D1", True, "No year_min constraint")
        offenders = [
            p.title for p in papers
            if p.year is not None and p.year < scope.year_min
        ]
        if offenders:
            return _CheckResult(
                "D1", False,
                f"{len(offenders)} papers violate year_min={scope.year_min}: "
                + ", ".join(f'"{t}"' for t in offenders[:3])
                + ("..." if len(offenders) > 3 else ""),
            )
        return _CheckResult("D1", True, f"All papers ≥ year_min={scope.year_min}")

    def _check_year_max(
        self, papers: list[LiteraturePaper], scope: ScopeConstraint
    ) -> _CheckResult:
        if not scope.year_max:
            return _CheckResult("D2", True, "No year_max constraint")
        offenders = [
            p.title for p in papers
            if p.year is not None and p.year > scope.year_max
        ]
        if offenders:
            return _CheckResult(
                "D2", False,
                f"{len(offenders)} papers violate year_max={scope.year_max}: "
                + ", ".join(f'"{t}"' for t in offenders[:3])
                + ("..." if len(offenders) > 3 else ""),
            )
        return _CheckResult("D2", True, f"All papers ≤ year_max={scope.year_max}")

    def _check_max_papers(
        self, papers: list[LiteraturePaper], scope: ScopeConstraint
    ) -> _CheckResult:
        count = len(papers)
        if count > scope.max_papers:
            return _CheckResult(
                "D3", False,
                f"Accepted count {count} exceeds max_papers={scope.max_papers}",
            )
        return _CheckResult("D3", True, f"Paper count {count} ≤ max_papers={scope.max_papers}")

    def _check_empty_abstracts(self, papers: list[LiteraturePaper]) -> _CheckResult:
        empty = [p.title for p in papers if not p.abstract.strip()]
        if empty:
            return _CheckResult(
                "D4", False,
                f"{len(empty)} accepted papers have empty abstracts: "
                + ", ".join(f'"{t}"' for t in empty[:3]),
            )
        return _CheckResult("D4", True, "All accepted papers have abstracts")

    def _check_no_url(self, papers: list[LiteraturePaper]) -> _CheckResult:
        no_url = [p.title for p in papers if not p.url.strip()]
        if no_url:
            return _CheckResult(
                "D5", False,
                f"{len(no_url)} accepted papers have no URL: "
                + ", ".join(f'"{t}"' for t in no_url[:3]),
            )
        return _CheckResult("D5", True, "All accepted papers have URLs")

    def _check_synthesis(self, artifact: LiteratureReviewArtifact) -> _CheckResult:
        if not artifact.synthesis.strip():
            return _CheckResult("D6", False, "Synthesis is empty")
        word_count = len(artifact.synthesis.split())
        return _CheckResult("D6", True, f"Synthesis present ({word_count} words)")

    def _check_status_consistency(self, artifact: LiteratureReviewArtifact) -> _CheckResult:
        """
        Cross-check: if is_sufficient=True but status="exhausted" that's a logic error.
        """
        if artifact.is_sufficient and artifact.status == "exhausted":
            return _CheckResult(
                "D7", False,
                "is_sufficient=True but status='exhausted' — logic inconsistency",
            )
        return _CheckResult(
            "D7", True,
            f"status={artifact.status!r} consistent with is_sufficient={artifact.is_sufficient}",
        )

    def _check_search_log(self, artifact: LiteratureReviewArtifact) -> _CheckResult:
        if not artifact.search_log:
            return _CheckResult("D8", False, "Search log is empty — no audit trail")
        return _CheckResult(
            "D8", True,
            f"Search log has {len(artifact.search_log)} round(s)",
        )
