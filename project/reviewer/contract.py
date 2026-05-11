"""
reviewer/contract.py
=====================
Typed schemas for the JournalReviewer agent.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class ArticleSection:
    section_id: str
    number: str
    title: str
    body: str
    start_line: int
    end_line: int
    has_math: bool = False
    has_claims: bool = False


@dataclass
class ArticleView:
    article_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[ArticleSection]
    raw_text: str
    total_lines: int
    parsed_at: float = field(default_factory=time.time)

    def section_by_id(self, section_id: str) -> Optional[ArticleSection]:
        return next((s for s in self.sections if s.section_id == section_id), None)

    def lines(self) -> List[str]:
        return self.raw_text.splitlines()


MathClaimKind = Literal[
    "algebraic_identity", "numerical_result", "statistical_claim",
    "proof_step", "bound_or_complexity", "differential_equation", "optimization_claim",
]
MathCheckStatus = Literal["verified", "refuted", "uncertain", "timeout", "unsupported"]


@dataclass
class MathCheckRequest:
    request_id: str
    section_id: str
    claim_kind: MathClaimKind
    claim_text: str
    context: str
    line_number: int
    timeout_seconds: float = 300.0

    @staticmethod
    def new(section_id: str, kind: MathClaimKind, claim_text: str,
            context: str, line_number: int) -> "MathCheckRequest":
        return MathCheckRequest(
            request_id=f"mc-{uuid.uuid4().hex[:8]}",
            section_id=section_id, claim_kind=kind,
            claim_text=claim_text, context=context, line_number=line_number,
        )


@dataclass
class MathCheckResult:
    request_id: str
    status: MathCheckStatus
    verdict: str
    counterexample: Optional[str] = None
    correction: Optional[str] = None
    confidence: float = 1.0
    checked_at: float = field(default_factory=time.time)

    @property
    def is_problem(self) -> bool:
        return self.status in ("refuted",) and self.confidence >= 0.7


ReviewDimension = Literal[
    "originality", "technical_correctness", "clarity",
    "experimental_validity", "related_work", "reproducibility", "ethical_considerations",
]
ScoreLevel = Literal["strong_accept", "accept", "borderline", "reject", "strong_reject"]
_SCORE_INT: Dict[str, int] = {
    "strong_accept": 5, "accept": 4, "borderline": 3, "reject": 2, "strong_reject": 1,
}


@dataclass
class DimensionScore:
    dimension: ReviewDimension
    score: ScoreLevel
    rationale: str
    evidence_quotes: List[str] = field(default_factory=list)

    @property
    def numeric(self) -> int:
        return _SCORE_INT[self.score]


@dataclass
class ReviewFindings:
    article_id: str
    dimension_scores: List[DimensionScore]
    math_check_results: List[MathCheckResult]
    summary_strengths: List[str]
    summary_weaknesses: List[str]
    confidence_in_assessment: float

    @property
    def mean_score(self) -> float:
        if not self.dimension_scores:
            return 0.0
        return sum(d.numeric for d in self.dimension_scores) / len(self.dimension_scores)

    @property
    def has_refuted_math(self) -> bool:
        return any(r.is_problem for r in self.math_check_results)

    def score_for(self, dim: ReviewDimension) -> Optional[DimensionScore]:
        return next((d for d in self.dimension_scores if d.dimension == dim), None)


AnnotationSeverity = Literal["critical", "major", "minor", "question"]
AnnotationKind = Literal[
    "math_error", "unsupported_claim", "missing_citation", "clarity_issue",
    "reproducibility_gap", "ethical_concern", "suggestion", "question",
]


@dataclass
class ReviewAnnotation:
    annotation_id: str
    section_id: str
    line_number: int
    target_text: str
    comment: str
    severity: AnnotationSeverity
    kind: AnnotationKind
    math_request_id: Optional[str] = None
    suggested_replacement: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(section_id: str, line_number: int, target_text: str, comment: str,
            severity: AnnotationSeverity, kind: AnnotationKind, **kwargs: Any) -> "ReviewAnnotation":
        return ReviewAnnotation(
            annotation_id=f"ann-{uuid.uuid4().hex[:8]}",
            section_id=section_id, line_number=line_number,
            target_text=target_text, comment=comment,
            severity=severity, kind=kind, **kwargs,
        )

    @property
    def marker(self) -> str:
        tag = f"[REVIEW-{self.annotation_id[:7].upper()} | {self.severity.upper()}]"
        correction = (f"\n    → Suggested: {self.suggested_replacement}"
                      if self.suggested_replacement else "")
        return f"{tag} {self.comment}{correction}"


DecisionKind = Literal["accepted", "rejected", "correction_needed"]


@dataclass
class AnnotatedArticle:
    article_id: str
    original_text: str
    annotated_text: str
    unified_diff: str
    annotations: List[ReviewAnnotation]
    annotation_count_by_severity: Dict[str, int] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return self.annotation_count_by_severity.get("critical", 0)

    @property
    def major_count(self) -> int:
        return self.annotation_count_by_severity.get("major", 0)


@dataclass
class ReviewDecision:
    decision_id: str
    article_id: str
    decision_kind: DecisionKind
    review_text: str
    findings: ReviewFindings
    annotated_article: Optional[AnnotatedArticle]
    confidence: float
    decided_at: float = field(default_factory=time.time)

    @staticmethod
    def accepted(article_id, review_text, findings, confidence) -> "ReviewDecision":
        return ReviewDecision(
            decision_id=f"rd-{uuid.uuid4().hex[:8]}", article_id=article_id,
            decision_kind="accepted", review_text=review_text,
            findings=findings, annotated_article=None, confidence=confidence,
        )

    @staticmethod
    def rejected(article_id, review_text, findings, confidence) -> "ReviewDecision":
        return ReviewDecision(
            decision_id=f"rd-{uuid.uuid4().hex[:8]}", article_id=article_id,
            decision_kind="rejected", review_text=review_text,
            findings=findings, annotated_article=None, confidence=confidence,
        )

    @staticmethod
    def correction_needed(article_id, review_text, findings,
                          annotated_article, confidence) -> "ReviewDecision":
        return ReviewDecision(
            decision_id=f"rd-{uuid.uuid4().hex[:8]}", article_id=article_id,
            decision_kind="correction_needed", review_text=review_text,
            findings=findings, annotated_article=annotated_article, confidence=confidence,
        )


@dataclass
class ReviewTask:
    task_id: str
    article_text: str
    venue: str
    review_criteria: Dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)

    @staticmethod
    def new(article_text: str, venue: str, **criteria: Any) -> "ReviewTask":
        return ReviewTask(
            task_id=f"rt-{uuid.uuid4().hex[:8]}",
            article_text=article_text, venue=venue, review_criteria=criteria,
        )


ReviewChunkKind = Literal[
    "reasoning", "text_delta", "dimension_score", "math_dispatch",
    "math_result", "annotation_added", "decision_made", "phase_change", "error",
]


@dataclass
class ReviewChunk:
    kind: ReviewChunkKind
    text: str = ""
    payload: Optional[Dict[str, Any]] = None
    ts: float = field(default_factory=time.time)
