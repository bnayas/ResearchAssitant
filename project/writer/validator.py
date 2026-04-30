"""
writer/validator.py
====================
GroundingValidator — the audit gate for the AcademicWriter.

Principle
---------
The validator's PRIMARY role is ensuring every factual claim in the paper
is grounded in a concrete artifact produced by one of the other agents.
It is NOT responsible for prose quality, citation style, or formatting.

Tier structure (mirrors sim_tool's SpecValidator pattern)
---------------------------------------------------------
Tier 1 – Artifact existence    G01–G04  (hard errors)
Tier 2 – Grounding density     G05–G10  (errors + warnings)
Tier 3 – Promise integrity     P01–P06  (errors + warnings)
Tier 4 – Paper-level sanity    L01–L06  (warnings)

The Director should treat any Tier 1/3 errors as a hard block.
Tier 2/4 issues may proceed to review with warnings attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from .contract import (
    FigureSpec,
    GroundingClaim,
    SectionDraft,
    WriterArtifact,
    Promise,
)


# ── ArtifactStore protocol ────────────────────────────────────────────────────


@runtime_checkable
class ArtifactStore(Protocol):
    """
    Minimal interface the validator needs from the broader system.

    Implement this in your Curator / memory layer and inject it.
    The validator never writes; it only reads.
    """

    def exists(self, artifact_id: str) -> bool:
        """Return True iff the artifact is in the store."""
        ...

    def get_agent(self, artifact_id: str) -> Optional[str]:
        """Return the agent that produced this artifact, or None if not found."""
        ...

    def get_summary(self, artifact_id: str) -> Optional[str]:
        """Return a short textual summary of the artifact, or None."""
        ...


# ── Validation result types ───────────────────────────────────────────────────


@dataclass
class ValidationIssue:
    check_id: str
    section_id: str
    severity: str        # "error" | "warning"
    message: str
    claim_id: Optional[str] = None
    promise_id: Optional[str] = None


@dataclass
class SectionValidationReport:
    section_id: str
    issues: List[ValidationIssue] = field(default_factory=list)
    passed: bool = True  # False if any error-severity issue exists

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False

    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


@dataclass
class PaperValidationReport:
    paper_id: str
    section_reports: Dict[str, SectionValidationReport] = field(default_factory=dict)
    promise_issues: List[ValidationIssue] = field(default_factory=list)
    paper_issues: List[ValidationIssue] = field(default_factory=list)
    overall_passed: bool = True

    @property
    def all_issues(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = list(self.promise_issues) + list(self.paper_issues)
        for r in self.section_reports.values():
            issues.extend(r.issues)
        return issues

    def error_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == "error")

    def warning_count(self) -> int:
        return sum(1 for i in self.all_issues if i.severity == "warning")

    def _fail_overall(self) -> None:
        self.overall_passed = False


# ── Grounding Validator ───────────────────────────────────────────────────────


class GroundingValidator:
    """
    Validates that every claim in a paper draft is backed by a real artifact.

    Configuration
    -------------
    MIN_CLAIMS_PER_SECTION      At least N grounding claims required.
    MIN_CLAIMS_PER_100_WORDS    Density floor; below this → warning.
    MIN_GROUNDING_RATIO         Fraction of claims with confidence ≥ 0.7.
    LOW_CONFIDENCE_FLOOR        Below this → hard error (not warning).
    """

    MIN_CLAIMS_PER_SECTION: int = 1
    MIN_CLAIMS_PER_100_WORDS: float = 0.5   # ≥ 1 claim per 200 words
    MIN_GROUNDING_RATIO: float = 0.80        # 80 % of claims must be high-confidence
    LOW_CONFIDENCE_FLOOR: float = 0.50       # < 0.50 → hard error

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_section(self, section: SectionDraft) -> SectionValidationReport:
        """Run Tier-1 and Tier-2 checks on a single section."""
        r = SectionValidationReport(section_id=section.section_id)
        self._t1_artifact_existence(section, r)
        self._t1_agent_consistency(section, r)
        self._t1_excerpt_non_empty(section, r)
        self._t1_confidence_floor(section, r)
        self._t2_claim_count(section, r)
        self._t2_claim_density(section, r)
        self._t2_grounding_ratio(section, r)
        self._t2_duplicate_claims(section, r)
        return r

    def validate_paper(
        self,
        artifact: WriterArtifact,
        promises: Dict[str, "Promise"],
    ) -> PaperValidationReport:
        """Run all four tiers across the full paper."""
        report = PaperValidationReport(paper_id=artifact.paper_id)

        # Per-section checks (Tier 1 + 2)
        for sid, section in artifact.sections.items():
            sec_report = self.validate_section(section)
            report.section_reports[sid] = sec_report
            if not sec_report.passed:
                report._fail_overall()

        # Tier 3: Promise integrity
        self._t3_promise_broken(promises, report)
        self._t3_promise_pending_in_complete(promises, artifact, report)
        self._t3_origin_section_exists(promises, artifact, report)

        # Tier 4: Paper-level sanity
        self._t4_all_sections_present(artifact, report)
        self._t4_figures_grounded(artifact, report)
        self._t4_abstract_present(artifact, report)

        return report

    # ── Tier 1: Artifact existence (hard errors) ──────────────────────────────

    def _t1_artifact_existence(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G01 – Every referenced artifact_id must exist in the store."""
        for claim in s.grounding_claims:
            if not self.store.exists(claim.artifact_id):
                r.add(ValidationIssue(
                    check_id="G01",
                    section_id=s.section_id,
                    severity="error",
                    message=f"Artifact '{claim.artifact_id}' not found in store",
                    claim_id=claim.claim_id,
                ))

    def _t1_agent_consistency(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G02 – Artifact exists AND was produced by the claimed agent."""
        for claim in s.grounding_claims:
            actual = self.store.get_agent(claim.artifact_id)
            if actual is not None and actual != claim.source_agent:
                r.add(ValidationIssue(
                    check_id="G02",
                    section_id=s.section_id,
                    severity="error",
                    message=(
                        f"Claim says source_agent='{claim.source_agent}' "
                        f"but artifact '{claim.artifact_id}' belongs to '{actual}'"
                    ),
                    claim_id=claim.claim_id,
                ))

    def _t1_excerpt_non_empty(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G03 – artifact_excerpt must be non-empty."""
        for claim in s.grounding_claims:
            if not claim.artifact_excerpt.strip():
                r.add(ValidationIssue(
                    check_id="G03",
                    section_id=s.section_id,
                    severity="error",
                    message=f"Claim '{claim.claim_id}' has empty artifact_excerpt",
                    claim_id=claim.claim_id,
                ))

    def _t1_confidence_floor(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G04 – confidence below LOW_CONFIDENCE_FLOOR is a hard error."""
        for claim in s.grounding_claims:
            if claim.confidence < self.LOW_CONFIDENCE_FLOOR:
                r.add(ValidationIssue(
                    check_id="G04",
                    section_id=s.section_id,
                    severity="error",
                    message=(
                        f"Claim '{claim.claim_id}' confidence {claim.confidence:.2f} "
                        f"< hard floor {self.LOW_CONFIDENCE_FLOOR}"
                    ),
                    claim_id=claim.claim_id,
                ))

    # ── Tier 2: Grounding density (errors + warnings) ─────────────────────────

    def _t2_claim_count(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G05 – Minimum absolute number of claims."""
        if len(s.grounding_claims) < self.MIN_CLAIMS_PER_SECTION:
            r.add(ValidationIssue(
                check_id="G05",
                section_id=s.section_id,
                severity="error",
                message=(
                    f"Section has {len(s.grounding_claims)} grounding claim(s); "
                    f"minimum is {self.MIN_CLAIMS_PER_SECTION}"
                ),
            ))

    def _t2_claim_density(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G06 – Claims per 100 words must be above threshold."""
        if s.word_count <= 0:
            return
        density = len(s.grounding_claims) / (s.word_count / 100)
        if density < self.MIN_CLAIMS_PER_100_WORDS:
            r.add(ValidationIssue(
                check_id="G06",
                section_id=s.section_id,
                severity="warning",
                message=(
                    f"Low claim density: {density:.2f} claims/100 words "
                    f"(threshold {self.MIN_CLAIMS_PER_100_WORDS})"
                ),
            ))

    def _t2_grounding_ratio(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G07 – Fraction of high-confidence claims must be ≥ MIN_GROUNDING_RATIO."""
        if s.grounding_ratio < self.MIN_GROUNDING_RATIO and s.grounding_claims:
            r.add(ValidationIssue(
                check_id="G07",
                section_id=s.section_id,
                severity="warning",
                message=(
                    f"Grounding ratio {s.grounding_ratio:.0%} < "
                    f"{self.MIN_GROUNDING_RATIO:.0%} threshold"
                ),
            ))

    def _t2_duplicate_claims(self, s: SectionDraft, r: SectionValidationReport) -> None:
        """G08 – Two claims pointing to the same artifact with identical text → warning."""
        seen: Dict[str, str] = {}  # artifact_id → claim_text
        for claim in s.grounding_claims:
            key = claim.artifact_id
            if key in seen and seen[key] == claim.claim_text:
                r.add(ValidationIssue(
                    check_id="G08",
                    section_id=s.section_id,
                    severity="warning",
                    message=(
                        f"Duplicate claim for artifact '{claim.artifact_id}': "
                        f"'{claim.claim_text[:60]}…'"
                    ),
                    claim_id=claim.claim_id,
                ))
            seen[key] = claim.claim_text

    # ── Tier 3: Promise integrity ─────────────────────────────────────────────

    def _t3_promise_broken(
        self,
        promises: Dict[str, Promise],
        report: PaperValidationReport,
    ) -> None:
        """P01 – Broken promises are hard errors."""
        for pid, p in promises.items():
            if p.status == "broken":
                report.promise_issues.append(ValidationIssue(
                    check_id="P01",
                    section_id=p.origin_section,
                    severity="error",
                    message=(
                        f"Promise '{pid}' from §{p.origin_section} "
                        f"→ §{p.target_section} is broken: {p.resolved_text}"
                    ),
                    promise_id=pid,
                ))
                report._fail_overall()

    def _t3_promise_pending_in_complete(
        self,
        promises: Dict[str, Promise],
        artifact: WriterArtifact,
        report: PaperValidationReport,
    ) -> None:
        """P02 – A promise is still pending even though the target section is accepted."""
        for pid, p in promises.items():
            if p.status != "pending":
                continue
            target = artifact.sections.get(p.target_section)
            if target and target.status in ("validated", "accepted"):
                report.promise_issues.append(ValidationIssue(
                    check_id="P02",
                    section_id=p.target_section,
                    severity="warning",
                    message=(
                        f"Promise '{pid}' still pending even though "
                        f"§{p.target_section} is '{target.status}'"
                    ),
                    promise_id=pid,
                ))

    def _t3_origin_section_exists(
        self,
        promises: Dict[str, Promise],
        artifact: WriterArtifact,
        report: PaperValidationReport,
    ) -> None:
        """P03 – Origin section of a promise must exist in the paper."""
        if not artifact.toc_plan:
            return
        section_ids = {s.section_id for s in artifact.toc_plan.sections}
        for pid, p in promises.items():
            if p.origin_section and p.origin_section not in section_ids:
                report.promise_issues.append(ValidationIssue(
                    check_id="P03",
                    section_id=p.origin_section,
                    severity="warning",
                    message=f"Promise '{pid}' origin §{p.origin_section} not in TOC",
                    promise_id=pid,
                ))

    # ── Tier 4: Paper-level sanity ────────────────────────────────────────────

    def _t4_all_sections_present(
        self, artifact: WriterArtifact, report: PaperValidationReport
    ) -> None:
        """L01 – Every planned section must have a draft."""
        if not artifact.toc_plan:
            return
        planned = {s.section_id for s in artifact.toc_plan.sections}
        drafted = set(artifact.sections.keys())
        for sid in planned - drafted:
            report.paper_issues.append(ValidationIssue(
                check_id="L01",
                section_id=sid,
                severity="warning",
                message=f"Planned section '{sid}' has no draft yet",
            ))

    def _t4_figures_grounded(
        self, artifact: WriterArtifact, report: PaperValidationReport
    ) -> None:
        """L02 – Every planned figure must reference an existing artifact."""
        if not artifact.toc_plan:
            return
        for fig in artifact.toc_plan.figures:
            if not self.store.exists(fig.artifact_id):
                report.paper_issues.append(ValidationIssue(
                    check_id="L02",
                    section_id=fig.section_placement,
                    severity="error",
                    message=(
                        f"Figure '{fig.figure_id}' references missing "
                        f"artifact '{fig.artifact_id}'"
                    ),
                ))
                report._fail_overall()

    def _t4_abstract_present(
        self, artifact: WriterArtifact, report: PaperValidationReport
    ) -> None:
        """L03 – At least one section should cover the abstract/introduction."""
        if not artifact.toc_plan:
            return
        has_intro = any(
            kw in s.title.lower()
            for s in artifact.toc_plan.sections
            for kw in ("abstract", "introduction", "intro")
        )
        if not has_intro:
            report.paper_issues.append(ValidationIssue(
                check_id="L03",
                section_id="toc",
                severity="warning",
                message="No abstract or introduction section found in TOC",
            ))
