"""
tests/test_literature_review.py
────────────────────────────────
Test suite for the literature review agent.

Coverage:
  - Contract dataclasses (smoke tests)
  - ArXiv XML parser
  - Perplexity JSON parser
  - Scope validator (deterministic gates + LLM path via mock)
  - Search depth gate logic (mock backend that returns empty after N calls)
  - Deduplication
  - LiteratureAuditor (all 8 checks)
  - LiteratureReviewer end-to-end (mock backend + mock LLM)
  - StreamEvent emission sequence
  - Partial artifact when below threshold

Run with:
    pytest tests/test_literature_review.py -v
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make sure the package is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from literature_review.contract import (
    LiteratureAuditResult,
    LiteraturePaper,
    LiteratureReviewArtifact,
    LiteratureReviewTask,
    ScopeConstraint,
    SearchDepthConfig,
    SearchRound,
    StreamEvent,
)
from literature_review.auditor import LiteratureAuditor
from literature_review.search_backends.base import RawPaper, SearchBackend
from literature_review.scope_validator import check_paper_scope, raw_to_paper


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_scope(
    include=None, exclude=None, year_min=None, year_max=None, max_papers=20
) -> ScopeConstraint:
    return ScopeConstraint(
        include_topics=include or ["machine learning"],
        exclude_topics=exclude or [],
        year_min=year_min,
        year_max=year_max,
        max_papers=max_papers,
    )


def make_task(scope=None, depth=None) -> LiteratureReviewTask:
    return LiteratureReviewTask(
        task_id="t-001",
        branch_id="b-001",
        query="How do diffusion models compare to GANs for image synthesis?",
        scope=scope or make_scope(
            include=["diffusion models", "GANs", "image synthesis"],
            year_min=2018,
        ),
        depth=depth or SearchDepthConfig(
            max_rounds=2,
            max_term_variations=1,
            min_papers_threshold=2,
            papers_per_query=5,
        ),
    )


def make_raw_paper(
    title="Test Paper",
    abstract="This paper is about diffusion models for image synthesis.",
    year=2022,
    source="arxiv",
    arxiv_id="2201.99999",
) -> RawPaper:
    return RawPaper(
        title=title,
        authors=["Alice", "Bob"],
        abstract=abstract,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        year=year,
        source=source,
        arxiv_id=arxiv_id,
    )


def make_paper(title="Test Paper", year=2022, in_scope=True) -> LiteraturePaper:
    return LiteraturePaper(
        title=title,
        authors=["Alice"],
        abstract="Abstract text.",
        url="https://example.com/paper",
        year=year,
        source="arxiv",
        in_scope=in_scope,
        scope_violation_reason=None if in_scope else "Out of topic",
    )


class MockLLM:
    """Synchronous mock LLM that returns configurable responses."""

    def __init__(self, scope_result: bool = True):
        self.scope_result = scope_result
        self.calls: list[dict] = []

    async def complete_async(self, system: str, messages: list, max_tokens=1024, temperature=0.0) -> str:
        self.calls.append({"system": system[:40], "msg": messages[0]["content"][:40]})

        # Keyword extraction response
        if "keyword" in system.lower() or "query analyst" in system.lower():
            return '{"primary": [["diffusion models", "image synthesis"]], "variations": [["generative models"]]}'

        # Scope check response
        if "scope" in system.lower():
            verdict = "true" if self.scope_result else "false"
            reason = "" if self.scope_result else "Not relevant to scope topics"
            return f'{{"in_scope": {verdict}, "reason": "{reason}"}}'

        # Synthesis response
        if "synthesis" in system.lower():
            return "This is a synthesized literature review."

        return '{"in_scope": true, "reason": ""}'


class BrokenScopeLLM:
    async def complete_async(self, system: str, messages: list, max_tokens=1024, temperature=0.0) -> str:
        raise RuntimeError("llm unavailable")


class MockBackend(SearchBackend):
    """Returns a fixed list of papers; can be configured to return empty after N calls."""

    name = "mock"

    def __init__(self, papers: list[RawPaper], empty_after: int = 999):
        self._papers = papers
        self._empty_after = empty_after
        self._call_count = 0

    async def search(self, keywords, max_results, year_min=None, year_max=None, categories=None):
        self._call_count += 1
        if self._call_count > self._empty_after:
            return []
        return self._papers[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# Contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContracts:
    def test_scope_constraint_defaults(self):
        sc = ScopeConstraint(include_topics=["AI"])
        assert sc.exclude_topics == []
        assert sc.year_min is None
        assert sc.max_papers == 30

    def test_literature_paper_short_id_arxiv(self):
        p = make_paper()
        p.arxiv_id = "2201.99999"
        assert p.short_id == "arXiv:2201.99999"

    def test_literature_paper_citation_key(self):
        p = LiteraturePaper(
            title="T", authors=["Smith, John", "Doe"], abstract="A",
            url="u", year=2023, source="arxiv",
        )
        assert p.citation_key == "Smith2023"

    def test_artifact_summary_line(self):
        artifact = LiteratureReviewArtifact(
            task_id="t1", branch_id="b1",
            papers=[make_paper()],
            removed_papers=[make_paper("Removed", in_scope=False)],
            synthesis="Synthesis text",
            search_log=[SearchRound(1, [[]], ["arxiv"], 2, 1)],
            status="complete",
            is_sufficient=True,
        )
        summary = artifact.summary_line()
        assert "complete" in summary.lower()
        assert "1 papers accepted" in summary
        assert "1 removed" in summary

    def test_stream_event_payload(self):
        evt = StreamEvent(event_type="title_found", payload={"title": "X", "year": 2023})
        assert evt.event_type == "title_found"
        assert evt.payload["title"] == "X"


# ─────────────────────────────────────────────────────────────────────────────
# ArXiv backend XML parser
# ─────────────────────────────────────────────────────────────────────────────

class TestArXivParser:
    ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2201.00001v1</id>
    <title>Diffusion Models Beat GANs</title>
    <summary>We show that diffusion models achieve SOTA on image synthesis.</summary>
    <published>2022-01-01T00:00:00Z</published>
    <author><name>Alice Doe</name></author>
    <author><name>Bob Smith</name></author>
    <link href="https://arxiv.org/abs/2201.00001" type="text/html"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2019.00002v1</id>
    <title>Old GAN Paper</title>
    <summary>GANs were the original generative model.</summary>
    <published>2019-06-01T00:00:00Z</published>
    <author><name>Carol White</name></author>
  </entry>
</feed>"""

    def test_parses_titles_and_authors(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend
        backend = ArXivBackend()
        papers = backend._parse_atom(self.ATOM_XML, None, None)
        assert len(papers) == 2
        assert papers[0].title == "Diffusion Models Beat GANs"
        assert "Alice Doe" in papers[0].authors

    def test_extracts_arxiv_id(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend
        backend = ArXivBackend()
        papers = backend._parse_atom(self.ATOM_XML, None, None)
        assert papers[0].arxiv_id == "2201.00001v1"

    def test_year_min_filter(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend
        backend = ArXivBackend()
        papers = backend._parse_atom(self.ATOM_XML, year_min=2021, year_max=None)
        assert len(papers) == 1
        assert papers[0].year == 2022

    def test_year_max_filter(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend
        backend = ArXivBackend()
        papers = backend._parse_atom(self.ATOM_XML, year_min=None, year_max=2020)
        assert len(papers) == 1
        assert papers[0].year == 2019

    def test_malformed_xml_returns_empty(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend
        backend = ArXivBackend()
        papers = backend._parse_atom("<not valid xml>>>", None, None)
        assert papers == []

    def test_rate_limit_enters_cooldown_and_skips_followup_requests(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend

        cls = ArXivBackend
        old_state = (
            cls._cooldown_until,
            cls._temporary_unavailable_until,
            cls._last_skip_log_at,
            cls._last_unavailable_skip_log_at,
            cls._next_request_at,
        )
        try:
            cls._cooldown_until = 0.0
            cls._temporary_unavailable_until = 0.0
            cls._last_skip_log_at = 0.0
            cls._last_unavailable_skip_log_at = 0.0
            cls._next_request_at = 0.0
            backend = ArXivBackend()
            backend._fetch_with_retry = AsyncMock(side_effect=AssertionError("cooldown should skip fetching"))
            backend._mark_rate_limited(None)
            assert cls._cooldown_until - time.time() >= (10 * 60) - 5
            papers = asyncio.run(backend.search(["neutral model"], max_results=5))
            assert papers == []
            backend._fetch_with_retry.assert_not_awaited()
            assert backend._last_rate_limited is True
        finally:
            (
                cls._cooldown_until,
                cls._temporary_unavailable_until,
                cls._last_skip_log_at,
                cls._last_unavailable_skip_log_at,
                cls._next_request_at,
            ) = old_state

    def test_timeout_cooldown_skips_followup_requests(self):
        from literature_review.search_backends.arxiv_backend import ArXivBackend

        cls = ArXivBackend
        old_state = (
            cls._cooldown_until,
            cls._temporary_unavailable_until,
            cls._last_skip_log_at,
            cls._last_unavailable_skip_log_at,
            cls._next_request_at,
        )
        try:
            cls._cooldown_until = 0.0
            cls._temporary_unavailable_until = 0.0
            cls._last_skip_log_at = 0.0
            cls._last_unavailable_skip_log_at = 0.0
            cls._next_request_at = 0.0
            backend = ArXivBackend()
            backend._fetch_with_retry = AsyncMock(side_effect=AssertionError("temporary cooldown should skip fetching"))
            backend._mark_temporarily_unavailable(wait_seconds=60)
            papers = asyncio.run(backend.search(["neutral model"], max_results=5))
            assert papers == []
            backend._fetch_with_retry.assert_not_awaited()
            assert backend._last_temporarily_unavailable is True
            assert backend._last_rate_limited is False
        finally:
            (
                cls._cooldown_until,
                cls._temporary_unavailable_until,
                cls._last_skip_log_at,
                cls._last_unavailable_skip_log_at,
                cls._next_request_at,
            ) = old_state


class TestSemanticScholarBackend:
    def test_rate_limit_enters_cooldown_and_skips_followup_requests(self):
        from literature_review.search_backends.semantic_scholar_backend import SemanticScholarBackend

        cls = SemanticScholarBackend
        old_state = (cls._cooldown_until, cls._last_skip_log_at, cls._next_request_at)
        try:
            cls._cooldown_until = 0.0
            cls._last_skip_log_at = 0.0
            cls._next_request_at = 0.0
            backend = SemanticScholarBackend(api_key=None)
            backend._mark_rate_limited(None)
            assert cls._cooldown_until - time.time() >= (5 * 60) - 5
            papers = asyncio.run(backend.search(["neutral model"], max_results=5))
            assert papers == []
            assert backend._last_rate_limited is True
        finally:
            cls._cooldown_until, cls._last_skip_log_at, cls._next_request_at = old_state


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity backend JSON parser
# ─────────────────────────────────────────────────────────────────────────────

class TestPerplexityParser:
    def test_parses_clean_json(self):
        from literature_review.search_backends.perplexity_backend import PerplexityBackend

        class FakePerplexity(PerplexityBackend):
            def __init__(self):
                self.api_key = "fake"
                self.model = "test"

        backend = FakePerplexity()
        raw = '[{"title":"T1","authors":["A"],"abstract":"Ab","url":"http://x.com","year":2023}]'
        papers = backend._parse(raw, None, None)
        assert len(papers) == 1
        assert papers[0].title == "T1"
        assert papers[0].year == 2023

    def test_strips_markdown_fences(self):
        from literature_review.search_backends.perplexity_backend import PerplexityBackend

        class FakePerplexity(PerplexityBackend):
            def __init__(self):
                self.api_key = "fake"
                self.model = "test"

        backend = FakePerplexity()
        raw = '```json\n[{"title":"T2","authors":[],"abstract":"A","url":"u","year":2020}]\n```'
        papers = backend._parse(raw, None, None)
        assert len(papers) == 1

    def test_year_filter(self):
        from literature_review.search_backends.perplexity_backend import PerplexityBackend

        class FakePerplexity(PerplexityBackend):
            def __init__(self):
                self.api_key = "fake"
                self.model = "test"

        backend = FakePerplexity()
        raw = '[{"title":"Old","authors":[],"abstract":"A","url":"u","year":2010},{"title":"New","authors":[],"abstract":"B","url":"u2","year":2022}]'
        papers = backend._parse(raw, year_min=2020, year_max=None)
        assert len(papers) == 1
        assert papers[0].title == "New"

    def test_malformed_json_returns_empty(self):
        from literature_review.search_backends.perplexity_backend import PerplexityBackend

        class FakePerplexity(PerplexityBackend):
            def __init__(self):
                self.api_key = "fake"
                self.model = "test"

        backend = FakePerplexity()
        papers = backend._parse("not json at all!", None, None)
        assert papers == []


# ─────────────────────────────────────────────────────────────────────────────
# Scope validator
# ─────────────────────────────────────────────────────────────────────────────

class TestScopeValidator:
    def test_empty_title_always_rejected(self):
        raw = make_raw_paper(title="")
        scope = make_scope()
        result = asyncio.run(check_paper_scope(raw, scope, MockLLM()))
        assert result[0] is False
        assert "title" in result[1].lower()

    def test_year_min_violation_rejected_without_llm(self):
        raw = make_raw_paper(year=2010)
        scope = make_scope(year_min=2020)
        llm = MockLLM()
        result = asyncio.run(check_paper_scope(raw, scope, llm))
        assert result[0] is False
        assert "2010" in result[1]
        # No LLM calls should have been made
        assert len(llm.calls) == 0

    def test_year_max_violation_rejected_without_llm(self):
        raw = make_raw_paper(year=2025)
        scope = make_scope(year_max=2023)
        llm = MockLLM()
        result = asyncio.run(check_paper_scope(raw, scope, llm))
        assert result[0] is False
        assert len(llm.calls) == 0

    def test_in_scope_paper_accepted_via_llm(self):
        raw = make_raw_paper()
        scope = make_scope(year_min=2018)
        result = asyncio.run(check_paper_scope(raw, scope, MockLLM(scope_result=True)))
        assert result[0] is True

    def test_out_of_scope_paper_rejected_via_llm(self):
        raw = make_raw_paper(
            title="Survey of Database Indexing",
            abstract="B-trees are the primary data structure for database indexes.",
        )
        scope = make_scope(include=["diffusion models", "image synthesis"])
        result = asyncio.run(check_paper_scope(raw, scope, MockLLM(scope_result=False)))
        assert result[0] is False

    def test_llm_failure_uses_heuristic_scope_fallback(self):
        raw = make_raw_paper(
            title="Survey of Database Indexing",
            abstract="B-trees are the primary data structure for database indexes.",
        )
        scope = make_scope(include=["diffusion models", "image synthesis"])
        result = asyncio.run(check_paper_scope(raw, scope, BrokenScopeLLM()))
        assert result == (False, "no include topic match in title/abstract")

    def test_raw_to_paper_preserves_fields(self):
        raw = make_raw_paper(title="T", year=2021, arxiv_id="2101.12345")
        paper = raw_to_paper(raw, in_scope=True, reason="")
        assert paper.title == "T"
        assert paper.year == 2021
        assert paper.arxiv_id == "2101.12345"
        assert paper.in_scope is True
        assert paper.scope_violation_reason is None

    def test_raw_to_paper_stores_violation_reason(self):
        raw = make_raw_paper()
        paper = raw_to_paper(raw, in_scope=False, reason="Topic mismatch")
        assert paper.in_scope is False
        assert paper.scope_violation_reason == "Topic mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# LiteratureAuditor
# ─────────────────────────────────────────────────────────────────────────────

class TestLiteratureAuditor:
    def _make_artifact(self, **kwargs) -> LiteratureReviewArtifact:
        defaults = dict(
            task_id="t1",
            branch_id="b1",
            papers=[make_paper("P1", year=2022), make_paper("P2", year=2023)],
            removed_papers=[],
            synthesis="Good synthesis text here.",
            search_log=[SearchRound(1, [["kw"]], ["arxiv"], 5, 2)],
            status="complete",
            is_sufficient=True,
        )
        defaults.update(kwargs)
        return LiteratureReviewArtifact(**defaults)

    def test_clean_artifact_passes(self):
        artifact = self._make_artifact()
        scope = make_scope(year_min=2020, year_max=2024)
        result = LiteratureAuditor().audit(artifact, scope)
        assert result.passed is True
        assert result.scope_violations_found == []

    def test_year_min_violation_detected(self):
        papers = [make_paper("Old Paper", year=2015)]
        artifact = self._make_artifact(papers=papers)
        scope = make_scope(year_min=2020)
        result = LiteratureAuditor().audit(artifact, scope)
        assert result.passed is False
        assert "Old Paper" in result.scope_violations_found
        assert "D1" in result.auditor_notes

    def test_year_max_violation_detected(self):
        papers = [make_paper("Future Paper", year=2030)]
        artifact = self._make_artifact(papers=papers)
        scope = make_scope(year_max=2025)
        result = LiteratureAuditor().audit(artifact, scope)
        assert result.passed is False
        assert "Future Paper" in result.scope_violations_found

    def test_max_papers_exceeded(self):
        papers = [make_paper(f"P{i}", year=2022) for i in range(10)]
        artifact = self._make_artifact(papers=papers)
        scope = make_scope(max_papers=5)
        result = LiteratureAuditor().audit(artifact, scope)
        assert result.passed is False
        assert "D3" in result.auditor_notes

    def test_empty_abstract_detected(self):
        p = make_paper("No Abstract", year=2022)
        p.abstract = ""
        artifact = self._make_artifact(papers=[p])
        result = LiteratureAuditor().audit(artifact, make_scope())
        assert result.passed is False
        assert "D4" in result.auditor_notes

    def test_empty_synthesis_fails(self):
        artifact = self._make_artifact(synthesis="")
        result = LiteratureAuditor().audit(artifact, make_scope())
        assert result.passed is False
        assert "D6" in result.auditor_notes

    def test_status_inconsistency_detected(self):
        artifact = self._make_artifact(is_sufficient=True, status="exhausted")
        result = LiteratureAuditor().audit(artifact, make_scope())
        assert result.passed is False
        assert "D7" in result.auditor_notes

    def test_empty_search_log_fails(self):
        artifact = self._make_artifact(search_log=[])
        result = LiteratureAuditor().audit(artifact, make_scope())
        assert result.passed is False
        assert "D8" in result.auditor_notes

    def test_audit_result_counts(self):
        artifact = self._make_artifact(
            papers=[make_paper("P1"), make_paper("P2")],
            removed_papers=[make_paper("R1", in_scope=False)],
        )
        result = LiteratureAuditor().audit(artifact, make_scope())
        assert result.papers_accepted == 2
        assert result.papers_removed == 1


# ─────────────────────────────────────────────────────────────────────────────
# LiteratureReviewer end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestLiteratureReviewer:
    def _make_papers(self, n: int = 3) -> list[RawPaper]:
        return [
            make_raw_paper(
                title=f"Diffusion Paper {i}",
                abstract=f"Abstract about diffusion models and image synthesis #{i}.",
                year=2021 + i,
                arxiv_id=f"220{i}.{10000+i}",
            )
            for i in range(n)
        ]

    def test_happy_path_returns_artifact(self):
        from literature_review.reviewer import LiteratureReviewer

        backend = MockBackend(self._make_papers(3))
        llm = MockLLM(scope_result=True)
        events: list[StreamEvent] = []
        task = make_task()

        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=llm,
            stream_callback=events.append,
        )
        artifact = asyncio.run(reviewer.run(task))

        assert isinstance(artifact, LiteratureReviewArtifact)
        assert artifact.accepted_count > 0
        assert artifact.synthesis != ""
        assert len(artifact.search_log) >= 1

    def test_streams_title_found_events(self):
        from literature_review.reviewer import LiteratureReviewer

        backend = MockBackend(self._make_papers(3))
        events: list[StreamEvent] = []
        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=MockLLM(scope_result=True),
            stream_callback=events.append,
        )
        asyncio.run(reviewer.run(make_task()))

        found_events = [e for e in events if e.event_type == "title_found"]
        assert len(found_events) >= 1
        for e in found_events:
            assert "title" in e.payload
            assert "year" in e.payload
            assert "source" in e.payload

    def test_scope_removed_events_emitted(self):
        from literature_review.reviewer import LiteratureReviewer

        backend = MockBackend(self._make_papers(3))
        events: list[StreamEvent] = []
        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=MockLLM(scope_result=False),  # All papers rejected
            stream_callback=events.append,
        )
        asyncio.run(reviewer.run(make_task()))

        removed_events = [e for e in events if e.event_type == "scope_removed"]
        assert len(removed_events) >= 1
        for e in removed_events:
            assert "title" in e.payload
            assert "reason" in e.payload

    def test_deduplication_across_rounds(self):
        from literature_review.reviewer import LiteratureReviewer

        # Same papers returned every round
        backend = MockBackend(self._make_papers(2))
        events: list[StreamEvent] = []
        task = make_task(
            depth=SearchDepthConfig(max_rounds=3, max_term_variations=2, min_papers_threshold=10)
        )
        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=MockLLM(scope_result=True),
            stream_callback=events.append,
        )
        artifact = asyncio.run(reviewer.run(task))

        # Despite 3 rounds returning same papers, should only have 2 unique
        assert artifact.accepted_count == 2

    def test_empty_backend_produces_exhausted_status(self):
        from literature_review.reviewer import LiteratureReviewer

        backend = MockBackend([])  # Always empty
        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=MockLLM(),
            stream_callback=None,
        )
        artifact = asyncio.run(reviewer.run(make_task()))

        assert artifact.accepted_count == 0
        assert artifact.status in ("partial", "exhausted")
        assert artifact.is_sufficient is False
        assert artifact.insufficiency_reason is not None

    def test_backend_exception_does_not_crash_reviewer(self):
        from literature_review.reviewer import LiteratureReviewer

        class BrokenBackend(SearchBackend):
            name = "broken"
            async def search(self, **kwargs):
                raise ConnectionError("Network down")

        reviewer = LiteratureReviewer(
            backends=[BrokenBackend(), MockBackend(self._make_papers(2))],
            llm=MockLLM(scope_result=True),
        )
        # Should not raise; broken backend is skipped
        artifact = asyncio.run(reviewer.run(make_task()))
        assert artifact.accepted_count == 2

    def test_max_papers_ceiling_respected(self):
        from literature_review.reviewer import LiteratureReviewer

        backend = MockBackend(self._make_papers(10))
        task = make_task(
            scope=make_scope(max_papers=3),
            depth=SearchDepthConfig(max_rounds=4, min_papers_threshold=1),
        )
        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=MockLLM(scope_result=True),
        )
        artifact = asyncio.run(reviewer.run(task))
        assert artifact.accepted_count <= 3

    def test_done_event_always_emitted(self):
        from literature_review.reviewer import LiteratureReviewer

        events: list[StreamEvent] = []
        reviewer = LiteratureReviewer(
            backends=[MockBackend([])],
            llm=MockLLM(),
            stream_callback=events.append,
        )
        asyncio.run(reviewer.run(make_task()))

        assert any(e.event_type == "done" for e in events)

    def test_event_order(self):
        from literature_review.reviewer import LiteratureReviewer

        events: list[StreamEvent] = []
        backend = MockBackend(self._make_papers(2))
        reviewer = LiteratureReviewer(
            backends=[backend],
            llm=MockLLM(scope_result=True),
            stream_callback=events.append,
        )
        asyncio.run(reviewer.run(make_task()))

        types = [e.event_type for e in events]
        # Must start with search_round_start
        assert types[0] == "search_round_start"
        # Must end with done
        assert types[-1] == "done"
        # synthesis_start must precede done
        assert types.index("synthesis_start") < types.index("done")
        # search_exhausted must precede synthesis_start
        assert types.index("search_exhausted") < types.index("synthesis_start")


# ─────────────────────────────────────────────────────────────────────────────
# Keyword extractor
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordExtractor:
    def test_parses_valid_response(self):
        from literature_review.keyword_extractor import _parse_response
        resp = '{"primary": [["diffusion", "synthesis"]], "variations": [["generative models"]]}'
        primary, variations = _parse_response(resp)
        assert primary == [["diffusion", "synthesis"]]
        assert variations == [["generative models"]]

    def test_strips_markdown_fences(self):
        from literature_review.keyword_extractor import _parse_response
        resp = '```json\n{"primary": [["kw1"]], "variations": [["kw2"]]}\n```'
        primary, variations = _parse_response(resp)
        assert primary == [["kw1"]]

    def test_heuristic_fallback(self):
        from literature_review.keyword_extractor import _heuristic_fallback
        primary, variations = _heuristic_fallback(
            "How do diffusion models work?",
            ["diffusion models", "score matching"],
        )
        assert len(primary) >= 1
        assert len(variations) >= 1

    def test_invalid_json_raises(self):
        from literature_review.keyword_extractor import _parse_response
        with pytest.raises(Exception):
            _parse_response("not json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
