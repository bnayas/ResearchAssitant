"""
writer/tests/test_writer.py
============================
Test suite for the AcademicWriter package.

Run:  python -m pytest writer/tests/ -v
"""
from __future__ import annotations

import asyncio
import pytest
from typing import AsyncIterator, Dict, List, Optional, Any

# ── Minimal imports (no real LLM required) ────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from writer.contract import (
    GroundingClaim, Promise, ReviewComment, ReviewerDemand,
    SectionDraft, SectionSpec, StreamChunk, WritingTask,
)
from writer.promise_registry import PromiseRegistry
from writer.validator import (
    ArtifactStore, GroundingValidator,
    SectionValidationReport, PaperValidationReport,
)
from writer.tools import make_mock_toolbox
from writer.academic_writer import AcademicWriter, _json_safe, _split_grounding_block


# ── Fixtures ──────────────────────────────────────────────────────────────────

class FakeStore:
    """In-memory ArtifactStore for tests."""
    def __init__(self, artifacts: Dict[str, str]):
        self._db = artifacts  # artifact_id → agent_name

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._db

    def get_agent(self, artifact_id: str) -> Optional[str]:
        return self._db.get(artifact_id)

    def get_summary(self, artifact_id: str) -> Optional[str]:
        return f"summary of {artifact_id}"


FAKE_STORE = FakeStore({
    "art-001": "analyst",
    "art-002": "simulator",
    "art-003": "literature",
})


def make_claim(artifact_id="art-001", agent="analyst", confidence=0.9) -> GroundingClaim:
    return GroundingClaim.new(
        claim_text="The reproduction number R0=2.4.",
        source_agent=agent,
        artifact_id=artifact_id,
        artifact_excerpt="R0=2.4 (95% CI: 2.1–2.7)",
        confidence=confidence,
    )


def make_section(
    sid="s1",
    claims: Optional[List[GroundingClaim]] = None,
    word_count: int = 300,
    status="draft",
) -> SectionDraft:
    return SectionDraft(
        section_id=sid,
        title="Introduction",
        content="The epidemic peaked at day 14.",
        grounding_claims=[make_claim()] if claims is None else claims,
        resolved_promise_ids=[],
        pending_promise_ids=[],
        summary="SIR model analysis, R0=2.4, peak day 14.",
        word_count=word_count,
        status=status,
    )


# ── PromiseRegistry tests ─────────────────────────────────────────────────────

class TestPromiseRegistry:
    def test_create_and_pending(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        assert p.status == "pending"
        assert len(reg.pending_for_section("s4")) == 1

    def test_resolve(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        reg.resolve(p.promise_id, "peak at day 14")
        assert reg.get(p.promise_id).status == "resolved"
        assert len(reg.pending()) == 0

    def test_break_promise(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        reg.break_promise(p.promise_id, "section omitted")
        assert reg.get(p.promise_id).status == "broken"
        assert len(reg.broken()) == 1

    def test_materialize_resolved(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        reg.resolve(p.promise_id, "peak at day 14 (Figure 2)")
        text = f"As shown {p.tag} in our analysis."
        result = reg.materialize(text)
        assert "peak at day 14" in result
        assert "<<promise:" not in result

    def test_materialize_pending(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        text = f"As shown {p.tag}."
        result = reg.materialize(text)
        assert "[PENDING:" in result

    def test_materialize_broken(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        reg.break_promise(p.promise_id, "omitted")
        text = f"As shown {p.tag}."
        result = reg.materialize(text)
        assert "BROKEN PROMISE" in result

    def test_audit_section_completion_breaks(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        broken = reg.audit_section_completion("s4", "No mention of peak here.")
        assert p.promise_id in broken
        assert reg.get(p.promise_id).status == "broken"

    def test_serialisation_roundtrip(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        reg.resolve(p.promise_id, "peak day 14")
        d = reg.to_dict()
        reg2 = PromiseRegistry.from_dict(d)
        assert reg2.get(p.promise_id).status == "resolved"

    def test_embed_tag_replaces_placeholder(self):
        reg = PromiseRegistry()
        p = reg.create("s1", "s4", "show peak", "(see Results)")
        text = "We present these findings (see Results) in detail."
        embedded = reg.embed_tag(text, p)
        assert p.tag in embedded
        assert "(see Results)" not in embedded

    def test_status_summary(self):
        reg = PromiseRegistry()
        reg.create("s1", "s2", "d1", "p1")
        p2 = reg.create("s1", "s3", "d2", "p2")
        reg.resolve(p2.promise_id, "resolved text")
        summary = reg.status_summary()
        assert "pending" in summary
        assert "resolved" in summary


# ── GroundingValidator tests ──────────────────────────────────────────────────

class TestGroundingValidator:
    def setup_method(self):
        self.validator = GroundingValidator(FAKE_STORE)

    # Tier 1 checks
    def test_G01_missing_artifact(self):
        claim = make_claim(artifact_id="art-MISSING")
        section = make_section(claims=[claim])
        report = self.validator.validate_section(section)
        assert not report.passed
        assert any(i.check_id == "G01" for i in report.issues)

    def test_G01_passes_known_artifact(self):
        section = make_section(claims=[make_claim("art-001", "analyst")])
        report = self.validator.validate_section(section)
        errors = [i for i in report.issues if i.check_id == "G01"]
        assert not errors

    def test_G02_wrong_agent(self):
        claim = make_claim(artifact_id="art-001", agent="simulator")  # wrong agent
        section = make_section(claims=[claim])
        report = self.validator.validate_section(section)
        assert any(i.check_id == "G02" for i in report.issues)

    def test_G03_empty_excerpt(self):
        claim = GroundingClaim(
            claim_id="c1", claim_text="X", source_agent="analyst",
            artifact_id="art-001", artifact_excerpt="", confidence=0.9
        )
        section = make_section(claims=[claim])
        report = self.validator.validate_section(section)
        assert any(i.check_id == "G03" for i in report.issues)

    def test_G04_low_confidence_hard_error(self):
        claim = make_claim(confidence=0.3)  # below floor 0.5
        section = make_section(claims=[claim])
        report = self.validator.validate_section(section)
        assert not report.passed
        assert any(i.check_id == "G04" for i in report.issues)

    # Tier 2 checks
    def test_G05_no_claims(self):
        section = make_section(claims=[])
        report = self.validator.validate_section(section)
        assert not report.passed
        assert any(i.check_id == "G05" for i in report.issues)

    def test_G06_low_density(self):
        # 1 claim for 1000 words = 0.1 density, below 0.5 threshold
        section = make_section(claims=[make_claim()], word_count=1000)
        report = self.validator.validate_section(section)
        assert any(i.check_id == "G06" for i in report.issues)

    def test_G07_low_grounding_ratio(self):
        claims = [
            make_claim(confidence=0.9),  # high
            make_claim(confidence=0.9),
            make_claim(confidence=0.6),  # low (< 0.7)
            make_claim(confidence=0.6),
        ]
        section = make_section(claims=claims)
        report = self.validator.validate_section(section)
        warnings = [i for i in report.issues if i.check_id == "G07"]
        assert warnings  # 50% ratio < 80% threshold

    def test_passes_clean_section(self):
        claims = [make_claim() for _ in range(3)]
        section = make_section(claims=claims, word_count=200)
        report = self.validator.validate_section(section)
        errors = [i for i in report.issues if i.severity == "error"]
        assert not errors


# ── Utility function tests ────────────────────────────────────────────────────

class TestParseUtilities:
    def test_json_safe_clean(self):
        result = _json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_safe_with_fences(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = _json_safe(raw)
        assert result == {"key": "value"}

    def test_json_safe_invalid_returns_empty(self):
        result = _json_safe("not json at all")
        assert result == {}

    def test_split_grounding_block(self):
        raw = (
            "Section text here.\n\n"
            "---GROUNDING---\n"
            '{"grounding_claims": [{"claim_text": "X", "source_agent": "a",'
            ' "artifact_id": "art-001", "artifact_excerpt": "y", "confidence": 0.9}],'
            ' "resolved_promise_ids": [], "new_promises": [], "summary": "short"}\n'
            "---END---"
        )
        text, data = _split_grounding_block(raw)
        assert text == "Section text here."
        assert len(data["grounding_claims"]) == 1
        assert data["summary"] == "short"

    def test_split_grounding_no_block(self):
        raw = "Just section text."
        text, data = _split_grounding_block(raw)
        assert text == "Just section text."
        assert data == {}


# ── WritingTask factory ───────────────────────────────────────────────────────

class TestWritingTask:
    def test_new(self):
        task = WritingTask.new("Study SIR model", "NeurIPS", max_pages=8)
        assert task.task_id.startswith("wt-")
        assert task.venue == "NeurIPS"
        assert task.constraints["max_pages"] == 8


# ── Integration: mock LLM writer ─────────────────────────────────────────────

MOCK_PLAN_JSON = json_plan = """{
  "title": "SIR Epidemic Dynamics",
  "venue": "NeurIPS 2026",
  "abstract_outline": "We study an SIR model.",
  "sections": [
    {
      "section_id": "s1",
      "number": "1",
      "title": "Introduction",
      "level": 1,
      "estimated_words": 400,
      "key_points": ["Motivation", "Contributions"],
      "required_agents": ["literature"],
      "depends_on": [],
      "figures": []
    },
    {
      "section_id": "s2",
      "number": "2",
      "title": "Results",
      "level": 1,
      "estimated_words": 600,
      "key_points": ["Peak infection", "R0 estimate"],
      "required_agents": ["simulator", "analyst"],
      "depends_on": ["s1"],
      "figures": ["f1"]
    }
  ],
  "figures": [
    {
      "figure_id": "f1",
      "number": 1,
      "caption": "SIR dynamics over 60 days.",
      "source_agent": "simulator",
      "artifact_id": "art-002",
      "section_placement": "s2",
      "figure_type": "plot"
    }
  ],
  "initial_promises": [
    {
      "origin_section": "s1",
      "target_section": "s2",
      "description": "We will show peak at day 14.",
      "placeholder_text": "(see Results)"
    }
  ]
}"""

MOCK_SECTION_S1 = (
    "Epidemic modelling is fundamental to public health. "
    "We study SIR dynamics to understand disease spread. (see Results)\n\n"
    "---GROUNDING---\n"
    '{"grounding_claims": [{"claim_text": "SIR modelling is fundamental.",'
    ' "source_agent": "literature", "artifact_id": "art-003",'
    ' "artifact_excerpt": "SIR models have been widely used.", "confidence": 0.92}],'
    ' "resolved_promise_ids": [],'
    ' "new_promises": [],'
    ' "summary": "Motivation and contribution of SIR study."}\n'
    "---END---"
)

MOCK_SECTION_S2 = (
    "The epidemic peaked at day 14. <<promise:PLACEHOLDER>>\n\n"
    "---GROUNDING---\n"
    '{"grounding_claims": ['
    '{"claim_text": "Peak at day 14.", "source_agent": "simulator",'
    ' "artifact_id": "art-002", "artifact_excerpt": "peak_day=14", "confidence": 0.95},'
    '{"claim_text": "R0=2.4.", "source_agent": "analyst",'
    ' "artifact_id": "art-001", "artifact_excerpt": "R0=2.4", "confidence": 0.90}'
    '],'
    ' "resolved_promise_ids": [],'
    ' "new_promises": [],'
    ' "summary": "SIR peak day 14, R0=2.4."}\n'
    "---END---"
)


class MockLLM:
    """Returns pre-written responses; cycles through them in order."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._idx = 0

    async def stream(self, system: str, prompt: str):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        # Yield in small chunks to simulate streaming
        chunk_size = 50
        for i in range(0, len(resp), chunk_size):
            yield resp[i:i + chunk_size]


class TestAcademicWriterIntegration:
    @pytest.mark.asyncio
    async def test_plan_produces_toc(self):
        llm = MockLLM([MOCK_PLAN_JSON])
        toolbox = make_mock_toolbox()
        writer = AcademicWriter(llm=llm, toolbox=toolbox, store=FAKE_STORE)

        task = WritingTask.new("SIR model study", "NeurIPS 2026")
        chunks = []
        async for chunk in writer.plan(task):
            chunks.append(chunk)

        assert writer.artifact is not None
        assert writer.artifact.toc_plan is not None
        toc = writer.artifact.toc_plan
        assert toc.title == "SIR Epidemic Dynamics"
        assert len(toc.sections) == 2
        assert len(toc.figures) == 1
        assert len(toc.initial_promises) == 1

        # Initial promise was registered
        assert len(writer._registry) == 1

        # Phase change chunks were emitted
        phase_chunks = [c for c in chunks if c.kind == "phase_change"]
        assert len(phase_chunks) >= 2

    @pytest.mark.asyncio
    async def test_draft_produces_sections(self):
        # LLM returns: plan, then section text for s1, s2
        llm = MockLLM([MOCK_PLAN_JSON, MOCK_SECTION_S1, MOCK_SECTION_S2])
        toolbox = make_mock_toolbox(agents={
            "literature": [{"artifact_id": "art-003", "agent": "literature",
                            "summary": "SIR widely used"}],
            "simulator":  [{"artifact_id": "art-002", "agent": "simulator",
                            "summary": "peak_day=14"}],
            "analyst":    [{"artifact_id": "art-001", "agent": "analyst",
                            "summary": "R0=2.4"}],
        })
        writer = AcademicWriter(llm=llm, toolbox=toolbox, store=FAKE_STORE)

        task = WritingTask.new("SIR model study", "NeurIPS 2026")
        async for _ in writer.plan(task):
            pass

        draft_chunks = []
        async for chunk in writer.draft():
            draft_chunks.append(chunk)

        art = writer.artifact
        assert "s1" in art.sections
        assert "s2" in art.sections
        assert art.sections["s1"].word_count > 0
        assert art.sections["s2"].word_count > 0

    @pytest.mark.asyncio
    async def test_stream_chunk_kinds(self):
        llm = MockLLM([MOCK_PLAN_JSON])
        toolbox = make_mock_toolbox()
        writer = AcademicWriter(llm=llm, toolbox=toolbox, store=FAKE_STORE)

        task = WritingTask.new("SIR", "NeurIPS")
        kinds = set()
        async for chunk in writer.plan(task):
            kinds.add(chunk.kind)

        # Must have at least these kinds
        assert "phase_change" in kinds
        assert "tool_call" in kinds
        assert "tool_result" in kinds

    @pytest.mark.asyncio
    async def test_reviewer_demand(self):
        llm = MockLLM([
            MOCK_PLAN_JSON,
            MOCK_SECTION_S1,
            MOCK_SECTION_S2,
            # Revised section for reviewer
            MOCK_SECTION_S1 + "\n---RESPONSES---\n"
            '{"responses": [{"comment_id": "rc-1", "action": "addressed",'
            ' "explanation": "Added context."}]}\n---END---',
        ])
        toolbox = make_mock_toolbox()
        writer = AcademicWriter(llm=llm, toolbox=toolbox, store=FAKE_STORE)

        task = WritingTask.new("SIR", "NeurIPS")
        async for _ in writer.plan(task):
            pass
        async for _ in writer.draft():
            pass

        demand = ReviewerDemand(
            reviewer_id="R1",
            comments=[ReviewComment(
                comment_id="rc-1",
                section_id="s1",
                text="Please add more motivation.",
                severity="major",
                requires_new_agent_work=False,
            )],
            meta_review="Minor revisions needed.",
        )

        chunks = []
        async for chunk in writer.respond_to_review(demand):
            chunks.append(chunk)

        assert writer.artifact.review_cycles
        cycle = writer.artifact.review_cycles[0]
        assert cycle.status == "addressed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
