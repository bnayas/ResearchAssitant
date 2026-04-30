"""
reviewer/journal_reviewer.py
=============================
JournalReviewer agent — judges articles solely on their written content.

Design constraints (enforced architecturally)
----------------------------------------------
1. The reviewer has NO access to the upstream agent artifacts that the
   article may cite (no ArtifactStore reference).
2. The reviewer's ONLY external tool is the MathAgentRegistry (for
   verifying mathematical claims *as written in the article*).
3. The decision is a typed sealed union: Accepted | Rejected | CorrectionNeeded.
4. CorrectionNeeded carries an AnnotatedArticle with a unified diff.

Phases
------
1. PARSE       split article into ArticleView; identify math/claim-bearing sections.
2. ANALYSE     score each review dimension; extract all math claims.
3. VERIFY      dispatch MathCheckRequests concurrently; integrate results.
4. DECIDE      synthesise FindingsReport; emit ReviewDecision + optional diff.

Streaming
---------
All phases yield ReviewChunk objects for live UI consumption.
Reasoning ("thinking") chunks are emitted before each major step.
The UI can abort at any yield point.
"""

from __future__ import annotations

import re
import json
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from .contract import (
    AnnotatedArticle,
    AnnotationKind,
    AnnotationSeverity,
    ArticleSection,
    ArticleView,
    DecisionKind,
    DimensionScore,
    MathCheckRequest,
    MathCheckResult,
    ReviewAnnotation,
    ReviewChunk,
    ReviewDecision,
    ReviewDimension,
    ReviewFindings,
    ReviewTask,
    ScoreLevel,
)
from .diff import ArticleAnnotator, locate_line, parse_article
from .math_tools import MathAgentRegistry

# ── System prompts ─────────────────────────────────────────────────────────────

_ANALYSE_SYSTEM = """
You are a rigorous academic journal reviewer.
You judge ONLY by what is written in the article.
You have no access to external databases, the internet, or upstream simulation artifacts.
Mathematical claims you cannot verify mentally will be delegated to automated checkers.

Read the article and produce a structured analysis as a single JSON object:

{
  "dimension_scores": [
    {
      "dimension": "originality",
      "score": "accept",
      "rationale": "...",
      "evidence_quotes": ["verbatim line from article"]
    }
  ],
  "math_claims": [
    {
      "section_id": "s2",
      "claim_kind": "numerical_result",
      "claim_text": "the integral evaluates to 3.14",
      "context": "surrounding paragraph text",
      "line_hint": "the integral evaluates to"
    }
  ],
  "summary_strengths": ["..."],
  "summary_weaknesses": ["..."],
  "confidence": 0.85
}

dimension must be one of:
  originality, technical_correctness, clarity, experimental_validity,
  related_work, reproducibility, ethical_considerations

score must be one of:
  strong_accept, accept, borderline, reject, strong_reject

claim_kind must be one of:
  algebraic_identity, numerical_result, statistical_claim, proof_step,
  bound_or_complexity, differential_equation, optimization_claim

Reply ONLY with valid JSON. No prose, no markdown fences.
""".strip()

_DECIDE_SYSTEM = """
You are a rigorous academic journal reviewer writing your final review letter.

Given the dimension scores and math verification results, produce:
1. A complete review letter addressed to the authors.
2. A list of inline annotations for specific locations in the article.
3. A final decision.

Output a single JSON object:

{
  "decision_kind": "correction_needed",
  "review_letter": "Dear Authors,\\n\\n...",
  "annotations": [
    {
      "section_id": "s2",
      "line_hint": "exact phrase from the article line to annotate",
      "comment": "reviewer comment",
      "severity": "major",
      "kind": "math_error",
      "suggested_replacement": "corrected version or null"
    }
  ],
  "confidence": 0.82
}

decision_kind must be exactly one of: accepted, rejected, correction_needed.
severity must be one of: critical, major, minor, question.
kind must be one of: math_error, unsupported_claim, missing_citation,
  clarity_issue, reproducibility_gap, ethical_concern, suggestion, question.
suggested_replacement is a string or null.

Rules:
- Judge ONLY by the article text and the math verification results provided.
- The review letter must reference specific passages from the article.
- accepted → no annotations required (may still include minor suggestions).
- rejected → annotations optional; letter must explain decisive reasons.
- correction_needed → all critical/major issues must have an annotation.
- Do NOT reference external knowledge about the authors or their prior work.
- Reply ONLY with valid JSON. No prose, no markdown fences.
""".strip()


# ── Scoring thresholds → decision ────────────────────────────────────────────

def _mean_score_to_decision(mean: float, has_refuted_math: bool) -> DecisionKind:
    """
    Deterministic mapping from aggregate score to decision kind.
    Math errors can veto an acceptance regardless of prose quality.
    """
    if has_refuted_math and mean < 4.5:
        return "correction_needed"
    if mean >= 4.0:
        return "accepted"
    if mean >= 3.0:
        return "correction_needed"
    return "rejected"


# ── Main agent ────────────────────────────────────────────────────────────────


class JournalReviewer:
    """
    Multi-phase journal review agent.

    Parameters
    ----------
    llm          LLM backend (stream_with_thinking | stream | complete)
    math_registry  MathAgentRegistry populated by the orchestrator
    """

    def __init__(self, llm, math_registry: MathAgentRegistry) -> None:
        self._llm = llm
        self._math = math_registry
        self._view: Optional[ArticleView] = None
        self._findings: Optional[ReviewFindings] = None
        self._decision: Optional[ReviewDecision] = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def review(
        self, task: ReviewTask
    ) -> AsyncGenerator[ReviewChunk, None]:
        """
        Run all four phases and yield ReviewChunks throughout.
        The final chunk has kind="decision_made" and payload=ReviewDecision.
        """
        # Phase 1: Parse
        async for chunk in self._phase_parse(task):
            yield chunk

        # Phase 2: Analyse
        math_claims: List[MathCheckRequest] = []
        async for chunk in self._phase_analyse(task):
            yield chunk
            # Intercept math_dispatch chunks to collect requests
            if chunk.kind == "math_dispatch" and chunk.payload:
                math_claims.append(chunk.payload["request"])

        # Phase 3: Verify
        math_results: List[MathCheckResult] = []
        async for chunk in self._phase_verify(math_claims):
            yield chunk
            if chunk.kind == "math_result" and chunk.payload:
                math_results.append(chunk.payload["result"])

        # Phase 4: Decide
        async for chunk in self._phase_decide(task, math_results):
            yield chunk

    @property
    def decision(self) -> Optional[ReviewDecision]:
        return self._decision

    @property
    def article_view(self) -> Optional[ArticleView]:
        return self._view

    # ── Phase 1: Parse ────────────────────────────────────────────────────────

    async def _phase_parse(self, task: ReviewTask) -> AsyncGenerator[ReviewChunk, None]:
        yield ReviewChunk(kind="phase_change", text="[PARSE] Splitting article into sections")
        self._view = parse_article(task.article_text)

        math_sections = [s for s in self._view.sections if s.has_math]
        claim_sections = [s for s in self._view.sections if s.has_claims]

        yield ReviewChunk(
            kind="phase_change",
            text=(
                f"[PARSE] Done — {len(self._view.sections)} sections, "
                f"{len(math_sections)} with math, {len(claim_sections)} with claims"
            ),
            payload={
                "article_id": self._view.article_id,
                "title": self._view.title,
                "sections": [
                    {"id": s.section_id, "title": s.title, "has_math": s.has_math}
                    for s in self._view.sections
                ],
            },
        )

    # ── Phase 2: Analyse ─────────────────────────────────────────────────────

    async def _phase_analyse(
        self, task: ReviewTask
    ) -> AsyncGenerator[ReviewChunk, None]:
        yield ReviewChunk(kind="phase_change", text="[ANALYSE] Reading article and scoring dimensions")

        # Build article summary for LLM (full text, or compressed for long articles)
        article_block = _compress_if_needed(self._view.raw_text, max_tokens=12_000)

        prompt = (
            f"## Article to review\n\n{article_block}\n\n"
            f"## Venue\n{task.venue}\n\n"
            "Produce the structured analysis JSON."
        )

        raw_json = ""
        async for chunk in self._stream_llm(_ANALYSE_SYSTEM, prompt):
            yield chunk
            if chunk.kind == "text_delta":
                raw_json += chunk.text

        data = _json_safe(raw_json)

        # Build dimension scores
        dim_scores: List[DimensionScore] = []
        for d in data.get("dimension_scores", []):
            try:
                dim_scores.append(DimensionScore(
                    dimension=d["dimension"],
                    score=d["score"],
                    rationale=d.get("rationale", ""),
                    evidence_quotes=d.get("evidence_quotes", []),
                ))
                yield ReviewChunk(
                    kind="dimension_score",
                    text=f"[{d['dimension']}] {d['score']}: {d.get('rationale', '')[:80]}",
                    payload=d,
                )
            except (KeyError, TypeError):
                pass

        # Build math claim requests
        for claim_data in data.get("math_claims", []):
            section_id = claim_data.get("section_id", "s0")
            section = self._view.section_by_id(section_id)
            line_hint = claim_data.get("line_hint", claim_data.get("claim_text", ""))
            line_number = locate_line(self._view, line_hint)
            if line_number < 0:
                line_number = section.start_line if section else 0

            req = MathCheckRequest.new(
                section_id=section_id,
                kind=claim_data.get("claim_kind", "algebraic_identity"),
                claim_text=claim_data.get("claim_text", ""),
                context=claim_data.get("context", ""),
                line_number=line_number,
            )
            yield ReviewChunk(
                kind="math_dispatch",
                text=f"[MATH] Dispatch [{req.claim_kind}]: {req.claim_text[:80]}",
                payload={"request": req},
            )

        # Store intermediate findings (math_results filled in phase 3)
        self._findings = ReviewFindings(
            article_id=self._view.article_id,
            dimension_scores=dim_scores,
            math_check_results=[],
            summary_strengths=data.get("summary_strengths", []),
            summary_weaknesses=data.get("summary_weaknesses", []),
            confidence_in_assessment=float(data.get("confidence", 0.7)),
        )

    # ── Phase 3: Verify ───────────────────────────────────────────────────────

    async def _phase_verify(
        self, requests: List[MathCheckRequest]
    ) -> AsyncGenerator[ReviewChunk, None]:
        if not requests:
            yield ReviewChunk(
                kind="phase_change",
                text="[VERIFY] No mathematical claims to verify",
            )
            return

        yield ReviewChunk(
            kind="phase_change",
            text=f"[VERIFY] Dispatching {len(requests)} math claim(s) concurrently",
        )

        results = await self._math.dispatch_all(requests)
        self._findings.math_check_results = results

        for req, res in zip(requests, results):
            status_tag = "✓" if res.status == "verified" else ("✗" if res.status == "refuted" else "?")
            yield ReviewChunk(
                kind="math_result",
                text=(
                    f"[MATH {status_tag}] [{res.status.upper()}] "
                    f"{req.claim_text[:60]}: {res.verdict[:80]}"
                ),
                payload={"result": res, "request_id": req.request_id},
            )

        refuted = [r for r in results if r.is_problem]
        yield ReviewChunk(
            kind="phase_change",
            text=(
                f"[VERIFY] Done — {len(results)} checked, "
                f"{sum(1 for r in results if r.status == 'verified')} verified, "
                f"{len(refuted)} refuted"
            ),
        )

    # ── Phase 4: Decide ───────────────────────────────────────────────────────

    async def _phase_decide(
        self, task: ReviewTask, math_results: List[MathCheckResult]
    ) -> AsyncGenerator[ReviewChunk, None]:
        yield ReviewChunk(kind="phase_change", text="[DECIDE] Synthesising final review")

        findings = self._findings

        # Heuristic pre-decision so the LLM can see the suggested outcome
        heuristic_decision = _mean_score_to_decision(
            findings.mean_score, findings.has_refuted_math
        )

        # Build context for the LLM
        dim_summary = "\n".join(
            f"  {d.dimension}: {d.score} (score={d.numeric}) — {d.rationale[:100]}"
            for d in findings.dimension_scores
        )
        math_summary = _format_math_results(math_results)
        strengths = "\n".join(f"  + {s}" for s in findings.summary_strengths)
        weaknesses = "\n".join(f"  - {w}" for w in findings.summary_weaknesses)

        prompt = (
            f"## Article\n\n{_compress_if_needed(self._view.raw_text, 8_000)}\n\n"
            f"## Dimension scores\n{dim_summary}\n\n"
            f"## Mathematical verification results\n{math_summary}\n\n"
            f"## Strengths\n{strengths}\n\n"
            f"## Weaknesses\n{weaknesses}\n\n"
            f"## Mean score: {findings.mean_score:.2f}/5.0\n"
            f"## Heuristic decision: {heuristic_decision}\n\n"
            "Produce the final review JSON."
        )

        raw_json = ""
        async for chunk in self._stream_llm(_DECIDE_SYSTEM, prompt):
            yield chunk
            if chunk.kind == "text_delta":
                raw_json += chunk.text

        data = _json_safe(raw_json)

        decision_kind: DecisionKind = data.get("decision_kind", heuristic_decision)
        review_letter: str = data.get("review_letter", "(review letter missing)")
        confidence: float = float(data.get("confidence", findings.confidence_in_assessment))

        # Build annotations
        annotations = self._build_annotations(
            data.get("annotations", []), math_results
        )
        for ann in annotations:
            yield ReviewChunk(
                kind="annotation_added",
                text=f"[ANN] §{ann.section_id} L{ann.line_number} [{ann.severity}] {ann.comment[:80]}",
                payload={
                    "annotation_id": ann.annotation_id,
                    "severity": ann.severity,
                    "kind": ann.kind,
                },
            )

        # Build final decision object
        if decision_kind == "correction_needed":
            annotator = ArticleAnnotator(self._view)
            annotated = annotator.build(annotations)
            self._decision = ReviewDecision.correction_needed(
                article_id=self._view.article_id,
                review_text=review_letter,
                findings=findings,
                annotated_article=annotated,
                confidence=confidence,
            )
        elif decision_kind == "accepted":
            self._decision = ReviewDecision.accepted(
                article_id=self._view.article_id,
                review_text=review_letter,
                findings=findings,
                confidence=confidence,
            )
        else:
            self._decision = ReviewDecision.rejected(
                article_id=self._view.article_id,
                review_text=review_letter,
                findings=findings,
                confidence=confidence,
            )

        yield ReviewChunk(
            kind="decision_made",
            text=f"[DECISION] {decision_kind.upper()} (confidence={confidence:.2f})",
            payload={
                "decision_id": self._decision.decision_id,
                "decision_kind": decision_kind,
                "article_id": self._view.article_id,
                "annotation_count": len(annotations),
                "has_diff": self._decision.annotated_article is not None,
            },
        )

    # ── Internal: annotation builder ──────────────────────────────────────────

    def _build_annotations(
        self,
        ann_data: List[dict],
        math_results: List[MathCheckResult],
    ) -> List[ReviewAnnotation]:
        """
        Build ReviewAnnotation objects from LLM output + math results.

        Math refutations always generate a critical annotation regardless
        of whether the LLM listed them (belt-and-suspenders).
        """
        annotations: List[ReviewAnnotation] = []
        seen_line_comments: set = set()

        for a in ann_data:
            line_hint = a.get("line_hint", "")
            line_number = locate_line(self._view, line_hint) if line_hint else 0
            if line_number < 0:
                line_number = 0

            key = (line_number, a.get("comment", "")[:40])
            if key in seen_line_comments:
                continue
            seen_line_comments.add(key)

            try:
                ann = ReviewAnnotation.new(
                    section_id=a.get("section_id", "s0"),
                    line_number=line_number,
                    target_text=line_hint,
                    comment=a.get("comment", ""),
                    severity=a.get("severity", "minor"),
                    kind=a.get("kind", "suggestion"),
                    suggested_replacement=a.get("suggested_replacement") or None,
                )
                annotations.append(ann)
            except (KeyError, ValueError):
                pass

        # Inject annotations for refuted math claims
        for res in math_results:
            if not res.is_problem:
                continue
            # Find the matching request by request_id in findings
            req = next(
                (r for r in self._findings.math_check_results
                 if r.request_id == res.request_id),
                None,
            )
            line_number = 0
            section_id = "s0"
            claim_text = ""
            if req is not None:
                # We need the original request; find it via findings
                pass  # line_number already 0 as fallback

            # Look up original request from the phase-2 dispatches
            # (stored in findings.math_check_results contains results only;
            #  we re-locate by matching verdict context)
            comment = (
                f"Mathematical claim could not be verified: {res.verdict}"
                + (f" Counterexample: {res.counterexample}" if res.counterexample else "")
            )
            key = (line_number, comment[:40])
            if key not in seen_line_comments:
                seen_line_comments.add(key)
                annotations.append(ReviewAnnotation.new(
                    section_id=section_id,
                    line_number=line_number,
                    target_text="",
                    comment=comment,
                    severity="critical",
                    kind="math_error",
                    math_request_id=res.request_id,
                    suggested_replacement=res.correction,
                ))

        return annotations

    # ── Internal: LLM streaming (same pattern as writer) ──────────────────────

    async def _stream_llm(
        self, system: str, prompt: str
    ) -> AsyncGenerator[ReviewChunk, None]:
        try:
            if hasattr(self._llm, "stream_with_thinking"):
                async for kind, text in self._llm.stream_with_thinking(system, prompt):
                    yield ReviewChunk(
                        kind="reasoning" if kind == "thinking" else "text_delta",
                        text=text,
                    )
            elif hasattr(self._llm, "stream"):
                async for text in self._llm.stream(system, prompt):
                    yield ReviewChunk(kind="text_delta", text=text)
            else:
                text = self._llm.complete(system, prompt)
                yield ReviewChunk(kind="text_delta", text=text)
        except Exception as exc:
            yield ReviewChunk(kind="error", text=str(exc))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _json_safe(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def _compress_if_needed(text: str, max_tokens: int) -> str:
    """
    Rough token-aware truncation (4 chars ≈ 1 token).
    Preserves the beginning and end of the text.
    """
    budget = max_tokens * 4
    if len(text) <= budget:
        return text
    half = budget // 2
    return (
        text[:half]
        + "\n\n... [article truncated for context window] ...\n\n"
        + text[-half:]
    )


def _format_math_results(results: List[MathCheckResult]) -> str:
    if not results:
        return "(no mathematical claims were verified)"
    lines = []
    for r in results:
        status = r.status.upper()
        conf = f"conf={r.confidence:.2f}"
        lines.append(f"  [{status}] ({conf}) {r.verdict[:120]}")
        if r.counterexample:
            lines.append(f"    counterexample: {r.counterexample}")
        if r.correction:
            lines.append(f"    correction: {r.correction}")
    return "\n".join(lines)
