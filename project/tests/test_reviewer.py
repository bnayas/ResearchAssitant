"""
reviewer/tests/test_reviewer.py
================================
Test suite for the JournalReviewer package.

Run:  python -m pytest reviewer/tests/ -v
"""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reviewer.contract import (
    AnnotatedArticle,
    MathCheckRequest,
    MathCheckResult,
    ReviewAnnotation,
    ReviewDecision,
    ReviewTask,
)
from reviewer.diff import (
    ArticleAnnotator,
    extract_annotations_from_diff,
    locate_line,
    parse_article,
    ANNOTATION_PREFIX,
)
from reviewer.math_tools import (
    MathAgentRegistry,
    make_stub_registry,
    CLAIM_KIND_TO_AGENT,
)
from reviewer.journal_reviewer import JournalReviewer, _json_safe, _format_math_results


# ── Sample article fixture ────────────────────────────────────────────────────

SAMPLE_ARTICLE = """\
# SIR Epidemic Dynamics: A Mathematical Analysis

**Authors:** Alice Smith, Bob Jones

## 1. Introduction

Epidemic modelling is central to public health. The SIR model partitions the
population into Susceptible (S), Infected (I), and Recovered (R) compartments.
We show that for β=0.3 and γ=0.05, the basic reproduction number satisfies R0=6.0.

## 2. Methods

The governing ODEs are:
  dS/dt = -β·S·I/N
  dI/dt = β·S·I/N - γ·I
  dR/dt = γ·I

The integral ∫₀^∞ I(t) dt evaluates to N·(1 - 1/R0) ≈ 0.833·N.

## 3. Results

Simulations with N=10,000 show the epidemic peaks at day 12 with I_max = 3,420.
We observe that p < 0.001 for the comparison between model and data (t-test, n=50).
By induction, the discrete-time version P(n) holds for all n ≥ 1.

## 4. Discussion

The results demonstrate the effectiveness of interventions.
Further work is needed.
""".strip()


# ── parse_article tests ───────────────────────────────────────────────────────

class TestParseArticle:
    def test_section_count(self):
        view = parse_article(SAMPLE_ARTICLE)
        # h1 title + 4 numbered sections = 5
        assert len(view.sections) == 5

    def test_title_extracted(self):
        view = parse_article(SAMPLE_ARTICLE)
        assert "SIR" in view.title

    def test_section_ids_unique(self):
        view = parse_article(SAMPLE_ARTICLE)
        ids = [s.section_id for s in view.sections]
        assert len(ids) == len(set(ids))

    def test_math_sections_detected(self):
        view = parse_article(SAMPLE_ARTICLE)
        math_secs = [s for s in view.sections if s.has_math]
        assert len(math_secs) >= 1  # Methods section has equations

    def test_claims_detected(self):
        view = parse_article(SAMPLE_ARTICLE)
        claim_secs = [s for s in view.sections if s.has_claims]
        assert len(claim_secs) >= 1

    def test_line_numbers_ordered(self):
        view = parse_article(SAMPLE_ARTICLE)
        for i in range(len(view.sections) - 1):
            assert view.sections[i].start_line <= view.sections[i + 1].start_line

    def test_no_headers_fallback(self):
        plain = "This is a plain article with no headings whatsoever."
        view = parse_article(plain)
        assert len(view.sections) == 1
        assert view.sections[0].body == plain

    def test_article_id_provided(self):
        view = parse_article(SAMPLE_ARTICLE, article_id="art-test")
        assert view.article_id == "art-test"

    def test_raw_text_preserved(self):
        view = parse_article(SAMPLE_ARTICLE)
        assert view.raw_text == SAMPLE_ARTICLE


# ── locate_line tests ─────────────────────────────────────────────────────────

class TestLocateLine:
    def setup_method(self):
        self.view = parse_article(SAMPLE_ARTICLE)

    def test_finds_existing_text(self):
        line = locate_line(self.view, "basic reproduction number")
        assert line >= 0

    def test_returns_minus_one_for_missing(self):
        line = locate_line(self.view, "this phrase does not exist in the article xyz")
        assert line == -1

    def test_line_is_zero_based(self):
        line = locate_line(self.view, "SIR Epidemic")  # first line
        assert line == 0


# ── ArticleAnnotator / diff tests ─────────────────────────────────────────────

class TestArticleAnnotator:
    def setup_method(self):
        self.view = parse_article(SAMPLE_ARTICLE)
        self.annotator = ArticleAnnotator(self.view)

    def _make_annotation(self, line=5, severity="major", kind="unsupported_claim"):
        return ReviewAnnotation.new(
            section_id="s1",
            line_number=line,
            target_text="",
            comment="This claim needs a citation.",
            severity=severity,
            kind=kind,
        )

    def test_annotated_text_contains_prefix(self):
        ann = self._make_annotation(line=5)
        result = self.annotator.build([ann])
        assert ANNOTATION_PREFIX in result.annotated_text

    def test_original_text_preserved_in_annotated(self):
        ann = self._make_annotation(line=5)
        result = self.annotator.build([ann])
        # Every original line must appear in the annotated text
        for line in self.view.raw_text.splitlines()[:10]:
            assert line in result.annotated_text

    def test_unified_diff_is_nonempty_when_annotations_added(self):
        ann = self._make_annotation(line=3)
        result = self.annotator.build([ann])
        assert result.unified_diff != ""
        assert "@@" in result.unified_diff

    def test_unified_diff_empty_for_no_annotations(self):
        result = self.annotator.build([])
        # No diff when nothing added
        assert result.unified_diff == "" or "@@" not in result.unified_diff

    def test_severity_histogram(self):
        anns = [
            self._make_annotation(line=2, severity="critical"),
            self._make_annotation(line=4, severity="critical"),
            self._make_annotation(line=6, severity="minor"),
        ]
        result = self.annotator.build(anns)
        assert result.annotation_count_by_severity.get("critical") == 2
        assert result.annotation_count_by_severity.get("minor") == 1

    def test_critical_before_minor_same_line(self):
        anns = [
            self._make_annotation(line=5, severity="minor"),
            self._make_annotation(line=5, severity="critical"),
        ]
        result = self.annotator.build(anns)
        lines = result.annotated_text.splitlines()
        # Find the two annotation lines after line 5
        ann_lines = [l for l in lines if ANNOTATION_PREFIX in l]
        assert len(ann_lines) == 2
        # critical appears first (check presence of severity tag)
        assert "CRITICAL" in ann_lines[0]
        assert "MINOR" in ann_lines[1]

    def test_annotation_in_diff_parseable(self):
        ann = self._make_annotation(line=3, severity="major", kind="unsupported_claim")
        result = self.annotator.build([ann])
        extracted = extract_annotations_from_diff(result.unified_diff)
        assert len(extracted) == 1
        assert extracted[0]["severity"] == "major"
        assert extracted[0]["kind"] == "unsupported_claim"

    def test_original_text_in_result(self):
        result = self.annotator.build([])
        assert result.original_text == SAMPLE_ARTICLE

    def test_article_id_carried_through(self):
        view = parse_article(SAMPLE_ARTICLE, article_id="art-xyz")
        annotator = ArticleAnnotator(view)
        result = annotator.build([])
        assert result.article_id == "art-xyz"


# ── MathAgentRegistry tests ───────────────────────────────────────────────────

class TestMathAgentRegistry:
    @pytest.mark.asyncio
    async def test_unsupported_when_no_agent(self):
        registry = MathAgentRegistry()  # empty
        req = MathCheckRequest.new("s1", "algebraic_identity", "A=B", "ctx", 10)
        result = await registry.dispatch(req)
        assert result.status == "unsupported"

    @pytest.mark.asyncio
    async def test_stub_registry_returns_uncertain(self):
        registry = make_stub_registry()
        req = MathCheckRequest.new("s1", "algebraic_identity", "A=B", "ctx", 10)
        result = await registry.dispatch(req)
        assert result.status == "uncertain"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        registry = MathAgentRegistry()

        async def slow_agent(req):
            await asyncio.sleep(999)
            return MathCheckResult(request_id=req.request_id, status="verified", verdict="ok")

        registry.register("AlgebraVerifier", slow_agent)
        req = MathCheckRequest.new("s1", "algebraic_identity", "A=B", "ctx", 10)
        req.timeout_seconds = 0.05  # force timeout
        result = await registry.dispatch(req)
        assert result.status == "timeout"

    @pytest.mark.asyncio
    async def test_dispatch_all_concurrent(self):
        registry = make_stub_registry()
        requests = [
            MathCheckRequest.new("s1", "algebraic_identity", f"x={i}", "ctx", i)
            for i in range(5)
        ]
        results = await registry.dispatch_all(requests)
        assert len(results) == 5

    def test_routing_table_complete(self):
        # All claim kinds must have a routing entry
        from reviewer.contract import MathClaimKind
        all_kinds = [
            "algebraic_identity", "numerical_result", "statistical_claim",
            "proof_step", "bound_or_complexity", "differential_equation",
            "optimization_claim",
        ]
        for kind in all_kinds:
            assert kind in CLAIM_KIND_TO_AGENT, f"No routing for {kind}"

    @pytest.mark.asyncio
    async def test_exception_in_agent_returns_uncertain(self):
        registry = MathAgentRegistry()

        async def broken_agent(req):
            raise RuntimeError("agent crashed")

        registry.register("AlgebraVerifier", broken_agent)
        req = MathCheckRequest.new("s1", "algebraic_identity", "A=B", "ctx", 0)
        result = await registry.dispatch(req)
        assert result.status == "uncertain"
        assert "crashed" in result.verdict

    def test_register_and_lookup(self):
        registry = MathAgentRegistry()

        async def dummy(req): ...

        registry.register("AlgebraVerifier", dummy)
        assert registry.is_registered("AlgebraVerifier")
        assert not registry.is_registered("NonExistent")


# ── ReviewDecision factory tests ──────────────────────────────────────────────

class TestReviewDecision:
    def _findings(self):
        return ReviewDecision  # just to get type hints

    def test_accepted_has_no_diff(self):
        from reviewer.contract import ReviewFindings
        findings = ReviewFindings(
            article_id="art-1",
            dimension_scores=[],
            math_check_results=[],
            summary_strengths=[],
            summary_weaknesses=[],
            confidence_in_assessment=0.9,
        )
        d = ReviewDecision.accepted("art-1", "Great paper!", findings, 0.9)
        assert d.decision_kind == "accepted"
        assert d.annotated_article is None

    def test_rejected_has_no_diff(self):
        from reviewer.contract import ReviewFindings
        findings = ReviewFindings(
            article_id="art-1",
            dimension_scores=[],
            math_check_results=[],
            summary_strengths=[],
            summary_weaknesses=[],
            confidence_in_assessment=0.9,
        )
        d = ReviewDecision.rejected("art-1", "Not accepted.", findings, 0.7)
        assert d.decision_kind == "rejected"
        assert d.annotated_article is None

    def test_correction_needed_has_diff(self):
        from reviewer.contract import ReviewFindings
        findings = ReviewFindings(
            article_id="art-1",
            dimension_scores=[],
            math_check_results=[],
            summary_strengths=[],
            summary_weaknesses=[],
            confidence_in_assessment=0.9,
        )
        view = parse_article(SAMPLE_ARTICLE, article_id="art-1")
        annotator = ArticleAnnotator(view)
        ann = ReviewAnnotation.new("s1", 3, "foo", "Fix this.", "major", "clarity_issue")
        annotated = annotator.build([ann])
        d = ReviewDecision.correction_needed("art-1", "Revise.", findings, annotated, 0.8)
        assert d.decision_kind == "correction_needed"
        assert d.annotated_article is not None
        assert d.annotated_article.unified_diff != ""

    def test_decision_id_unique(self):
        from reviewer.contract import ReviewFindings
        findings = ReviewFindings("a", [], [], [], [], 0.8)
        d1 = ReviewDecision.accepted("a", "", findings, 0.8)
        d2 = ReviewDecision.accepted("a", "", findings, 0.8)
        assert d1.decision_id != d2.decision_id


# ── Integration: mock LLM reviewer ────────────────────────────────────────────

MOCK_ANALYSE_JSON = """{
  "dimension_scores": [
    {
      "dimension": "originality",
      "score": "accept",
      "rationale": "Presents a clear SIR analysis.",
      "evidence_quotes": ["We show that for β=0.3"]
    },
    {
      "dimension": "technical_correctness",
      "score": "borderline",
      "rationale": "Some numerical claims need verification.",
      "evidence_quotes": ["integral evaluates to N·(1 - 1/R0)"]
    },
    {
      "dimension": "clarity",
      "score": "accept",
      "rationale": "Well structured.",
      "evidence_quotes": []
    }
  ],
  "math_claims": [
    {
      "section_id": "s2",
      "claim_kind": "numerical_result",
      "claim_text": "the integral evaluates to N·(1 - 1/R0) ≈ 0.833·N",
      "context": "The governing ODEs are...",
      "line_hint": "integral evaluates to"
    },
    {
      "section_id": "s1",
      "claim_kind": "algebraic_identity",
      "claim_text": "R0=6.0",
      "context": "β=0.3 and γ=0.05",
      "line_hint": "basic reproduction number satisfies"
    }
  ],
  "summary_strengths": ["Clear ODE formulation", "Reasonable parameter choice"],
  "summary_weaknesses": ["Statistical test details sparse"],
  "confidence": 0.78
}"""

MOCK_DECIDE_JSON = """{
  "decision_kind": "correction_needed",
  "review_letter": "Dear Authors,\\n\\nThank you for your submission. The paper presents an interesting analysis of SIR dynamics. However, some corrections are required.\\n\\nPlease verify the integral result and provide more detail on the statistical test.\\n\\nSincerely, Reviewer 1",
  "annotations": [
    {
      "section_id": "s2",
      "line_hint": "integral evaluates to",
      "comment": "Please verify this integral result analytically and provide a derivation.",
      "severity": "major",
      "kind": "unsupported_claim",
      "suggested_replacement": null
    },
    {
      "section_id": "s3",
      "line_hint": "p < 0.001",
      "comment": "Please report the test statistic and degrees of freedom.",
      "severity": "minor",
      "kind": "reproducibility_gap",
      "suggested_replacement": null
    }
  ],
  "confidence": 0.80
}"""


class MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def stream(self, system, prompt):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        chunk_size = 80
        for i in range(0, len(resp), chunk_size):
            yield resp[i:i + chunk_size]


class TestJournalReviewerIntegration:
    @pytest.mark.asyncio
    async def test_full_review_cycle_produces_decision(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        chunks = []
        async for chunk in reviewer.review(task):
            chunks.append(chunk)

        assert reviewer.decision is not None
        assert reviewer.decision.decision_kind == "correction_needed"

    @pytest.mark.asyncio
    async def test_correction_needed_has_annotated_article(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        async for _ in reviewer.review(task):
            pass

        decision = reviewer.decision
        assert decision.annotated_article is not None
        assert ANNOTATION_PREFIX in decision.annotated_article.annotated_text
        assert "@@" in decision.annotated_article.unified_diff

    @pytest.mark.asyncio
    async def test_math_dispatches_emitted(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        math_dispatches = []
        async for chunk in reviewer.review(task):
            if chunk.kind == "math_dispatch":
                math_dispatches.append(chunk)

        # Two math claims in MOCK_ANALYSE_JSON
        assert len(math_dispatches) == 2

    @pytest.mark.asyncio
    async def test_math_results_emitted(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        math_results = []
        async for chunk in reviewer.review(task):
            if chunk.kind == "math_result":
                math_results.append(chunk)

        assert len(math_results) == 2

    @pytest.mark.asyncio
    async def test_decision_chunk_emitted(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        decision_chunks = []
        async for chunk in reviewer.review(task):
            if chunk.kind == "decision_made":
                decision_chunks.append(chunk)

        assert len(decision_chunks) == 1
        assert decision_chunks[0].payload["decision_kind"] == "correction_needed"

    @pytest.mark.asyncio
    async def test_annotation_chunks_emitted(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        ann_chunks = []
        async for chunk in reviewer.review(task):
            if chunk.kind == "annotation_added":
                ann_chunks.append(chunk)

        assert len(ann_chunks) >= 2  # two annotations in MOCK_DECIDE_JSON

    @pytest.mark.asyncio
    async def test_review_letter_in_decision(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        async for _ in reviewer.review(task):
            pass

        assert "Dear Authors" in reviewer.decision.review_text

    @pytest.mark.asyncio
    async def test_diff_parseable_back_to_annotations(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        async for _ in reviewer.review(task):
            pass

        diff = reviewer.decision.annotated_article.unified_diff
        extracted = extract_annotations_from_diff(diff)
        assert len(extracted) >= 2
        severities = {e["severity"] for e in extracted}
        assert "major" in severities or "minor" in severities

    @pytest.mark.asyncio
    async def test_phase_change_chunks_cover_all_phases(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        phase_texts = []
        async for chunk in reviewer.review(task):
            if chunk.kind == "phase_change":
                phase_texts.append(chunk.text)

        combined = " ".join(phase_texts)
        assert "PARSE" in combined
        assert "ANALYSE" in combined
        assert "VERIFY" in combined
        assert "DECIDE" in combined

    @pytest.mark.asyncio
    async def test_findings_stored_on_agent(self):
        llm = MockLLM([MOCK_ANALYSE_JSON, MOCK_DECIDE_JSON])
        registry = make_stub_registry()
        reviewer = JournalReviewer(llm=llm, math_registry=registry)

        task = ReviewTask.new(SAMPLE_ARTICLE, "NeurIPS 2026")
        async for _ in reviewer.review(task):
            pass

        assert reviewer._findings is not None
        assert len(reviewer._findings.dimension_scores) == 3
        assert reviewer._findings.mean_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
