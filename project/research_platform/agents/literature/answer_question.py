"""
literature.answer_question
──────────────────────────
Tools: Extract answers from a known text.

Both tools operate on TEXT ONLY — they do not search arXiv, fetch URLs,
or call any external service.  The caller is responsible for providing
the text.  This makes the tools deterministic and testable in isolation.

Two tools are exposed:

  answer_from_article  — extracts from a primary_article artifact's stored
                         text (abstract + any full text in the registry).
  answer_from_review   — extracts from a literature_review artifact's
                         synthesis text.

In both cases the answer is produced via a structured evidence extraction
prompt: locate evidence → synthesise → hedge.
"""
from __future__ import annotations

from typing import Any

from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_has_content, is_non_empty_string

# ---------------------------------------------------------------------------
# Shared extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are a precise academic text analyst.  Your job is to extract a specific
answer from a provided text.  You must NOT use external knowledge — only
what appears in the provided text.

Procedure:
1. EVIDENCE: identify every sentence or passage in the text that is relevant
   to the question.  List them verbatim (quote each, max 60 words each).
2. SYNTHESISE: combine the located passages into a direct answer.
3. CONFIDENCE: rate your confidence 0.0–1.0 based on how explicitly the
   text supports the answer.
4. HEDGE: if the text does not contain a clear answer, say so explicitly
   rather than guessing.

Return ONLY valid JSON:
{
  "answer": "The synthesised answer in 1-4 sentences.",
  "evidence": [
    {"passage": "verbatim excerpt from text", "relevance": "why this matters"},
    ...
  ],
  "confidence": 0.0-1.0,
  "not_found": true | false,
  "not_found_reason": "Explanation if not_found is true, else empty string."
}

If the answer is genuinely not in the text, set not_found=true and
answer="NOT_FOUND".  Do NOT fabricate.
"""

_MULTI_EXTRACTION_SYSTEM = """\
You are a precise academic text analyst.  Extract answers to MULTIPLE
questions from a provided text.  For each question extract evidence,
synthesise, rate confidence, and hedge.

Return ONLY valid JSON — a list in the same order as the questions:
[
  {
    "question": "the question",
    "answer": "synthesised answer",
    "evidence": [{"passage": "...", "relevance": "..."}],
    "confidence": 0.0-1.0,
    "not_found": false,
    "not_found_reason": ""
  },
  ...
]
"""

# ---------------------------------------------------------------------------
# Tool 1: answer_from_article
# ---------------------------------------------------------------------------

ARTICLE_QUESTION_TOOL = ToolDescriptor(
    name="answer_from_article",
    display_name="Answer From Article Text",
    description=(
        "Extract an answer to a specific question from a primary_article "
        "artifact's stored text (abstract and any full text in the registry). "
        "Does NOT perform any network calls.  Returns the answer, supporting "
        "evidence passages, and a confidence score."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="article",
            description="The primary_article ArtifactRef whose text to query.",
            type="ArtifactRef",
        ),
        ToolRequirement(
            name="question",
            description="The question to answer from the article text.",
            type="str",
            validator=is_non_empty_string,
            validator_description="Must be a non-empty question string",
        ),
        ToolRequirement(
            name="questions",
            description=(
                "Additional questions to answer in the same call "
                "(answered alongside the primary 'question')."
            ),
            type="list[str]",
            required=False,
            default=[],
        ),
    ],
    produces=[],
    tags=["literature", "extraction", "qa"],
    idempotent=True,
    estimated_seconds=12.0,
)


def execute_article_question(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_backend: Any = None,
) -> ToolResult:
    """Extract answer(s) from an article artifact's stored text."""
    if llm_backend is None:
        return ToolResult(status="failed", message="No LLM backend available for extraction")

    article = context.artifacts["article"]
    primary_question = str(context.inputs["question"]).strip()
    extra_questions = [
        str(q).strip()
        for q in (context.inputs.get("questions") or [])
        if str(q).strip()
    ]
    all_questions = [primary_question] + extra_questions

    text = _read_artifact_text(article, registry)
    if not text.strip():
        return ToolResult(
            status="failed",
            message="Article has no readable text.  Ensure the artifact stores content or abstract.",
        )

    results = _extract_answers(llm_backend, text, all_questions)
    primary = results[0] if results else {}

    answer = str(primary.get("answer") or "NOT_FOUND").strip()
    confidence = float(primary.get("confidence") or 0.0)
    not_found = bool(primary.get("not_found", answer == "NOT_FOUND"))

    return ToolResult(
        status="completed",
        message=answer,
        metadata={
            "question": primary_question,
            "answer": answer,
            "confidence": confidence,
            "not_found": not_found,
            "not_found_reason": str(primary.get("not_found_reason") or ""),
            "evidence": primary.get("evidence") or [],
            "all_answers": results,
        },
    )


# ---------------------------------------------------------------------------
# Tool 2: answer_from_review
# ---------------------------------------------------------------------------

REVIEW_QUESTION_TOOL = ToolDescriptor(
    name="answer_from_review",
    display_name="Answer From Literature Review",
    description=(
        "Extract an answer to a specific question from a literature_review "
        "artifact's synthesis text.  Useful for questions about the state of "
        "the field, consensus, gaps, or contradictions.  Does NOT perform "
        "any network calls or searches.  Returns the answer, supporting "
        "evidence passages, and a confidence score."
    ),
    agent_id="literature",
    requirements=[
        ToolRequirement(
            name="review",
            description="The literature_review ArtifactRef whose text to query.",
            type="ArtifactRef",
            validator=artifact_has_content,
            validator_description="Review must have readable content or a valid path",
        ),
        ToolRequirement(
            name="question",
            description="The question to answer from the review synthesis.",
            type="str",
            validator=is_non_empty_string,
            validator_description="Must be a non-empty question string",
        ),
        ToolRequirement(
            name="questions",
            description="Additional questions to answer in the same call.",
            type="list[str]",
            required=False,
            default=[],
        ),
        ToolRequirement(
            name="scope",
            description=(
                "Which parts of the review to draw from: "
                "'synthesis' (default), 'papers', or 'all'."
            ),
            type="str",
            required=False,
            default="synthesis",
        ),
    ],
    produces=[],
    tags=["literature", "extraction", "qa", "synthesis"],
    idempotent=True,
    estimated_seconds=10.0,
)


def execute_review_question(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_backend: Any = None,
) -> ToolResult:
    """Extract answer(s) from a literature review artifact's text."""
    if llm_backend is None:
        return ToolResult(status="failed", message="No LLM backend available for extraction")

    review = context.artifacts["review"]
    primary_question = str(context.inputs["question"]).strip()
    extra_questions = [
        str(q).strip()
        for q in (context.inputs.get("questions") or [])
        if str(q).strip()
    ]
    all_questions = [primary_question] + extra_questions
    scope = str(context.inputs.get("scope") or "synthesis").lower()

    text = _read_review_text(review, registry, scope)
    if not text.strip():
        return ToolResult(
            status="failed",
            message="Review has no readable text.  Ensure synthesis was produced.",
        )

    results = _extract_answers(llm_backend, text, all_questions)
    primary = results[0] if results else {}

    answer = str(primary.get("answer") or "NOT_FOUND").strip()
    confidence = float(primary.get("confidence") or 0.0)
    not_found = bool(primary.get("not_found", answer == "NOT_FOUND"))

    return ToolResult(
        status="completed",
        message=answer,
        metadata={
            "question": primary_question,
            "answer": answer,
            "confidence": confidence,
            "not_found": not_found,
            "not_found_reason": str(primary.get("not_found_reason") or ""),
            "evidence": primary.get("evidence") or [],
            "all_answers": results,
        },
    )


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def _extract_answers(
    backend: Any,
    text: str,
    questions: list[str],
) -> list[dict[str, Any]]:
    """Run extraction for one or more questions against text.

    Uses a single LLM call for multiple questions to minimise latency.
    Falls back to one-by-one if the batch response cannot be parsed.
    """
    if not questions:
        return []

    # Truncate text to stay within context limits (~8 000 words ≈ 48 000 chars)
    text_trimmed = _trim_text(text, max_chars=48_000)

    if len(questions) == 1:
        return [_extract_single(backend, text_trimmed, questions[0])]

    # Batch path
    try:
        questions_block = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(questions)
        )
        user_msg = f"Text:\n{text_trimmed}\n\nQuestions:\n{questions_block}"
        raw = backend.complete(
            system=_MULTI_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        payload = _parse_json(raw)
        if isinstance(payload, list) and len(payload) == len(questions):
            return [_normalise_answer(r, q) for r, q in zip(payload, questions)]
    except Exception:
        pass

    # Fallback: one call per question
    return [_extract_single(backend, text_trimmed, q) for q in questions]


def _extract_single(
    backend: Any,
    text: str,
    question: str,
) -> dict[str, Any]:
    """Single-question extraction with the evidence prompt."""
    user_msg = f"Text:\n{text}\n\nQuestion: {question}"
    try:
        raw = backend.complete(
            system=_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        payload = _parse_json(raw)
        if isinstance(payload, dict):
            return _normalise_answer(payload, question)
    except Exception:
        pass
    return {
        "question": question,
        "answer": "NOT_FOUND",
        "evidence": [],
        "confidence": 0.0,
        "not_found": True,
        "not_found_reason": "Extraction failed (LLM error or parse failure)",
    }


def _normalise_answer(raw: dict[str, Any], question: str) -> dict[str, Any]:
    answer = str(raw.get("answer") or "NOT_FOUND").strip()
    evidence = raw.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []
    return {
        "question": str(raw.get("question") or question),
        "answer": answer,
        "evidence": [
            {
                "passage": str(e.get("passage") or ""),
                "relevance": str(e.get("relevance") or ""),
            }
            for e in evidence
            if isinstance(e, dict)
        ],
        "confidence": max(0.0, min(1.0, float(raw.get("confidence") or 0.0))),
        "not_found": bool(raw.get("not_found", answer == "NOT_FOUND")),
        "not_found_reason": str(raw.get("not_found_reason") or ""),
    }


# ---------------------------------------------------------------------------
# Text readers
# ---------------------------------------------------------------------------


def _read_artifact_text(artifact: Any, registry: ArtifactRegistry) -> str:
    """Read text from an article artifact: registry → content field → abstract."""
    # 1. Registry (full text if previously stored)
    if hasattr(artifact, "artifact_id"):
        stored = registry.read_artifact_text(artifact.artifact_id)
        if stored and len(stored.strip()) > 200:
            return stored

    # 2. Content field
    content = str(getattr(artifact, "content", "") or "").strip()
    if len(content) > 100:
        return content

    # 3. Abstract from metadata
    abstract = str((getattr(artifact, "metadata", {}) or {}).get("abstract") or "").strip()
    if abstract:
        return abstract

    # 4. Summary fallback
    return str(getattr(artifact, "summary", "") or "")


def _read_review_text(
    artifact: Any,
    registry: ArtifactRegistry,
    scope: str,
) -> str:
    """Read text from a review artifact, respecting scope."""
    meta = getattr(artifact, "metadata", {}) or {}

    # Synthesis text
    synthesis_obj = meta.get("synthesis") or {}
    prose = ""
    if isinstance(synthesis_obj, dict):
        prose = str(synthesis_obj.get("prose_summary") or "").strip()
        if not prose:
            # Flatten structured synthesis
            parts: list[str] = []
            for key in ("agreements", "contradictions", "gaps_addressed", "open_questions"):
                vals = synthesis_obj.get(key) or []
                if vals:
                    parts.append(f"{key.replace('_', ' ').title()}:\n" + "\n".join(f"- {v}" for v in vals))
            prose = "\n\n".join(parts)
    elif isinstance(synthesis_obj, str):
        prose = synthesis_obj.strip()

    if scope == "synthesis":
        return prose or str(getattr(artifact, "summary", "") or "")

    # Papers text
    papers_text = ""
    if scope in ("papers", "all"):
        papers = meta.get("accepted_papers") or []
        papers_text = "\n\n".join(
            f"Title: {p.get('title','')}\n"
            f"Authors: {', '.join(p.get('authors',[]))}\n"
            f"Year: {p.get('year','?')}\n"
            f"Abstract: {str(p.get('abstract',''))[:300]}"
            for p in papers
        )

    if scope == "all":
        return "\n\n---\n\n".join(filter(None, [prose, papers_text]))
    return papers_text or str(getattr(artifact, "summary", "") or "")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _trim_text(text: str, max_chars: int) -> str:
    """Trim text to max_chars, keeping start and end (most info-dense regions)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... text trimmed ...]\n\n" + text[-half:]


def _parse_json(raw: Any) -> Any:
    """Parse JSON from LLM output, stripping markdown fences."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    import json
    return json.loads(text.strip())
