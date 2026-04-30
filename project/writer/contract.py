"""
writer/contract.py
==================
Typed schemas for the AcademicWriter agent.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ─── Promise ──────────────────────────────────────────────────────────────────

PromiseStatus = Literal["pending", "resolved", "broken", "waived"]


@dataclass
class Promise:
    promise_id: str
    origin_section: str
    target_section: str
    description: str
    placeholder_text: str
    status: PromiseStatus = "pending"
    resolved_text: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(origin: str, target: str, description: str,
            placeholder: str) -> "Promise":
        return Promise(
            promise_id=f"prom-{uuid.uuid4().hex[:8]}",
            origin_section=origin,
            target_section=target,
            description=description,
            placeholder_text=placeholder,
        )

    @property
    def tag(self) -> str:
        return f"<<promise:{self.promise_id}>>"

    def resolve(self, text: str) -> None:
        self.status = "resolved"
        self.resolved_text = text

    def break_(self, reason: str) -> None:
        self.status = "broken"
        self.resolved_text = f"[BROKEN PROMISE: {reason}]"

    def waive(self) -> None:
        self.status = "waived"


# ─── Grounding ────────────────────────────────────────────────────────────────

@dataclass
class GroundingClaim:
    claim_id: str
    claim_text: str
    source_agent: str
    artifact_id: str
    artifact_excerpt: str
    confidence: float

    @staticmethod
    def new(claim_text: str, source_agent: str, artifact_id: str,
            artifact_excerpt: str, confidence: float) -> "GroundingClaim":
        return GroundingClaim(
            claim_id=f"c-{uuid.uuid4().hex[:8]}",
            claim_text=claim_text,
            source_agent=source_agent,
            artifact_id=artifact_id,
            artifact_excerpt=artifact_excerpt,
            confidence=confidence,
        )


# ─── Sections ─────────────────────────────────────────────────────────────────

@dataclass
class SectionSpec:
    section_id: str
    number: str
    title: str
    level: int
    estimated_words: int
    key_points: List[str]
    required_agents: List[str]
    depends_on: List[str]
    figures: List[str] = field(default_factory=list)


@dataclass
class SectionDraft:
    section_id: str
    title: str
    content: str
    grounding_claims: List[GroundingClaim]
    resolved_promise_ids: List[str]
    pending_promise_ids: List[str]
    summary: str
    word_count: int
    status: str = "draft"
    validation_issues: List[str] = field(default_factory=list)

    @property
    def grounding_ratio(self) -> float:
        if not self.grounding_claims:
            return 0.0
        high = sum(1 for c in self.grounding_claims if c.confidence >= 0.7)
        return high / len(self.grounding_claims)


# ─── Figures ──────────────────────────────────────────────────────────────────

@dataclass
class FigureSpec:
    figure_id: str
    number: int
    caption: str
    source_agent: str
    artifact_id: str
    section_placement: str
    figure_type: str


# ─── TOC / Plan ───────────────────────────────────────────────────────────────

@dataclass
class TOCPlan:
    paper_id: str
    title: str
    venue: str
    abstract_outline: str
    sections: List[SectionSpec]
    figures: List[FigureSpec]
    initial_promises: List[Promise]

    def section_by_id(self, section_id: str) -> Optional[SectionSpec]:
        return next((s for s in self.sections if s.section_id == section_id), None)


# ─── Artifact ─────────────────────────────────────────────────────────────────

@dataclass
class WriterArtifact:
    paper_id: str
    toc_plan: Optional[TOCPlan]
    sections: Dict[str, SectionDraft]
    promise_registry: Dict[str, Any]
    review_cycles: List["ReviewCycle"]
    status: str = "planning"
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


# ─── Writing task ─────────────────────────────────────────────────────────────

@dataclass
class WritingTask:
    task_id: str
    description: str
    venue: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(description: str, venue: str, **constraints: Any) -> "WritingTask":
        return WritingTask(
            task_id=f"wt-{uuid.uuid4().hex[:8]}",
            description=description,
            venue=venue,
            constraints=constraints,
        )


# ─── Review ───────────────────────────────────────────────────────────────────

@dataclass
class ReviewComment:
    comment_id: str
    section_id: Optional[str]
    text: str
    severity: str
    requires_new_agent_work: bool = False


@dataclass
class ReviewerDemand:
    reviewer_id: str
    comments: List[ReviewComment]
    meta_review: str


@dataclass
class RevisionResponse:
    comment_id: str
    action: str   # "addressed" | "declined" | "deferred"
    explanation: str
    changed_sections: List[str] = field(default_factory=list)


@dataclass
class ReviewCycle:
    cycle_id: str
    demand: ReviewerDemand
    responses: List[RevisionResponse] = field(default_factory=list)
    status: str = "pending"


# ─── Streaming ────────────────────────────────────────────────────────────────

StreamChunkKind = Literal[
    "reasoning", "text_delta", "phase_change", "tool_call", "tool_result",
    "promise_created", "promise_resolved", "grounding_claim",
    "validation_issue", "error",
]


@dataclass
class StreamChunk:
    kind: StreamChunkKind
    text: str = ""
    payload: Optional[Any] = None
    section_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
