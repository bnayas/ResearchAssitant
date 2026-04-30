"""
literature_review/contract.py
─────────────────────────────
All typed schemas used by the Literature Review pipeline.

Hierarchy:
  Director ──► LiteratureReviewTask ──► LiteratureReviewer
  LiteratureReviewer ──► LiteratureReviewArtifact ──► LiteratureAuditor
  LiteratureAuditor  ──► LiteratureAuditResult ──► Director
  LiteratureReviewer ──► StreamEvent (fanned out to GUI in real-time)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ScopeConstraint:
    include_topics: list[str]
    exclude_topics: list[str] = field(default_factory=list)
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    arxiv_categories: list[str] = field(default_factory=list)
    max_papers: int = 30


@dataclass
class SearchDepthConfig:
    max_rounds: int = 4
    max_term_variations: int = 2
    min_papers_threshold: int = 5
    papers_per_query: int = 10


@dataclass
class LiteratureReviewTask:
    task_id: str
    branch_id: str
    query: str
    scope: ScopeConstraint
    depth: SearchDepthConfig = field(default_factory=SearchDepthConfig)
    requestor_agent: str = "Director"


@dataclass
class LiteraturePaper:
    title: str
    authors: list[str]
    abstract: str
    url: str
    year: Optional[int]
    source: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    in_scope: bool = True
    scope_violation_reason: Optional[str] = None

    @property
    def short_id(self) -> str:
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        return self.url[:40]

    @property
    def citation_key(self) -> str:
        if not self.authors:
            return f"Unknown{self.year or '?'}"
        name = self.authors[0]
        if "," in name:
            surname = name.split(",")[0].strip()
        else:
            surname = name.split()[-1]
        return f"{surname}{self.year or '?'}"


@dataclass
class SearchRound:
    round_number: int
    keyword_sets: list[list[str]]
    sources_queried: list[str]
    papers_found_raw: int
    papers_in_scope: int
    is_variation: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class LiteratureReviewArtifact:
    task_id: str
    branch_id: str
    papers: list[LiteraturePaper]
    removed_papers: list[LiteraturePaper]
    synthesis: str
    search_log: list[SearchRound]
    status: Literal["complete", "partial", "exhausted"]
    is_sufficient: bool
    insufficiency_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def accepted_count(self) -> int:
        return len(self.papers)

    @property
    def removed_count(self) -> int:
        return len(self.removed_papers)

    def summary_line(self) -> str:
        return (
            f"[{self.status.upper()}] "
            f"{self.accepted_count} papers accepted, "
            f"{self.removed_count} removed, "
            f"{len(self.search_log)} search rounds"
        )


@dataclass
class StreamEvent:
    event_type: Literal[
        "search_round_start", "title_found", "scope_removed",
        "round_summary", "search_exhausted", "synthesis_start", "done",
    ]
    payload: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiteratureAuditResult:
    task_id: str
    branch_id: str
    passed: bool
    scope_violations_found: list[str]
    sufficiency_verdict: bool
    sufficiency_reason: str
    papers_accepted: int
    papers_removed: int
    auditor_notes: str = ""
