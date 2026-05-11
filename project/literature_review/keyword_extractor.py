"""
literature_review/keyword_extractor.py
───────────────────────────────────────
Breaks a natural-language research question into structured keyword sets
for database search queries.

Returns two tiers:
  primary    – 2–3 focused sets for round 1
  variations – 2 broader/alternative sets for fallback rounds

Each set is a list of 2–4 terms to be AND-ed together in a search query.
Multiple sets within a tier are OR-ed (tried in parallel).

Example output for "How do diffusion models compare to GANs for image synthesis?":
  primary:    [["diffusion models", "image synthesis"],
               ["score-based generative models", "image generation"]]
  variations: [["generative models", "image synthesis evaluation"],
               ["denoising diffusion", "GAN comparison"]]
"""
from __future__ import annotations

import json
import logging
import re

from .llm_interface import AsyncLLMBackend

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a research query analyst for academic literature search.
Given a research question and a list of in-scope topics, generate keyword sets
for searching academic databases (ArXiv, Semantic Scholar, etc.).

Return ONLY a JSON object with exactly these two keys:
{
  "primary": [["term1", "term2"], ["term3", "term4"]],
  "variations": [["broader1", "broader2"], ["alt1", "alt2"]]
}

Rules:
- primary: 2–3 keyword sets, each with 2–4 specific terms (AND-ed together)
- variations: exactly 2 sets using broader or synonym-based terms
- Each set must be different — do not repeat terms across sets
- Use standard academic terminology, not colloquial phrases
- Do NOT include stop words (a, the, of, in, ...) as standalone terms
- Return ONLY valid JSON, no markdown, no explanation\
"""

_RETRY_SYSTEM = """\
Return compact academic database keyword sets as valid JSON only.
Do not explain. Do not include hidden reasoning. Do not think step by step.
Output exactly:
{"primary":[["term","term"]],"variations":[["term"],["term"]]}
"""

_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "because", "between",
    "could", "during", "each", "from", "have", "into", "just", "model", "models",
    "paper", "papers", "read", "reproduce", "result", "results", "should", "that",
    "their", "there", "these", "this", "through", "using", "with", "without",
}


async def extract_keyword_sets(
    query: str,
    include_topics: list[str],
    llm: AsyncLLMBackend,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Returns (primary_sets, variation_sets).

    primary_sets    → used in search round 1
    variation_sets  → used in rounds 2..N (fallback if insufficient results)

    On parse failure, falls back to a heuristic extraction.
    """
    user_msg = (
        f"Research question: {query}\n"
        f"In-scope topics: {', '.join(include_topics)}\n\n"
        "Generate keyword sets for academic database search."
    )

    try:
        response = await llm.complete_async(
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        primary, variations = _parse_response(response)
    except Exception as exc:
        log.warning("Keyword extraction LLM call failed (%s); retrying with constrained JSON prompt", exc)
        primary, variations = await _retry_or_fallback(query, include_topics, llm)

    primary, variations = await _revise_keywords_with_llm(
        query,
        include_topics,
        primary,
        variations,
        llm,
    )

    # Safety: always return at least something
    if not primary:
        primary, variations = _heuristic_fallback(query, include_topics)

    log.info("Extracted literature keywords primary=%s variations=%s", primary, variations)
    return primary, variations


async def _retry_or_fallback(
    query: str,
    include_topics: list[str],
    llm: AsyncLLMBackend,
) -> tuple[list[list[str]], list[list[str]]]:
    compact_context = "; ".join(_compact_terms([query, *include_topics])[:12])
    try:
        response = await llm.complete_async(
            system=_RETRY_SYSTEM,
            messages=[{"role": "user", "content": f"Topic terms: {compact_context}"}],
            temperature=0.0,
        )
        primary, variations = _parse_response(response)
        if primary:
            return primary, variations
    except Exception as retry_exc:
        log.warning("Keyword extraction retry failed (%s); using deterministic fallback", retry_exc)
    primary, variations = _heuristic_fallback(query, include_topics)
    log.info("Deterministic keyword fallback primary=%s variations=%s", primary, variations)
    return primary, variations


def _parse_response(
    text: str,
) -> tuple[list[list[str]], list[list[str]]]:
    """Parse LLM JSON response; raises on failure."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    if not text:
        raise ValueError("empty keyword JSON response")

    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    obj = json.loads(text)

    primary: list[list[str]] = obj.get("primary") or []
    variations: list[list[str]] = obj.get("variations") or []

    # Validate structure: must be list-of-list-of-str
    def _validate(sets: list) -> list[list[str]]:
        result = []
        for s in sets:
            if isinstance(s, list) and all(isinstance(t, str) for t in s):
                clean = [t.strip() for t in s if t.strip()]
                if clean:
                    result.append(clean)
        return result

    parsed_primary = _validate(primary)
    parsed_variations = _validate(variations)
    if not parsed_primary:
        raise ValueError("keyword JSON did not contain usable primary sets")
    return parsed_primary, parsed_variations


def _heuristic_fallback(
    query: str,
    include_topics: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Fallback when LLM extraction fails.
    Uses compact phrases and high-signal terms rather than full paragraphs.
    """
    terms = _compact_terms([query, *include_topics])
    if not terms:
        terms = ["research topic"]

    primary: list[list[str]] = []
    if len(terms) >= 2:
        primary.append(terms[:2])
    for term in terms:
        if len(primary) >= 3:
            break
        used = {item for group in primary for item in group}
        if term not in used:
            primary.append([term])

    variation_terms = [term for term in terms if term not in {item for group in primary for item in group}]
    variations = [[term] for term in variation_terms[:2]]
    while len(variations) < 2:
        variations.append([terms[min(len(terms) - 1, len(variations))]])
    return primary[:3], variations[:2]


async def _revise_keywords_with_llm(
    query: str,
    include_topics: list[str],
    primary: list[list[str]],
    variations: list[list[str]],
    llm: AsyncLLMBackend,
) -> tuple[list[list[str]], list[list[str]]]:
    prompts = [(
        "Review and, if needed, revise these academic database keyword sets.\n"
        "Use the research question and in-scope topics as the only source of intent.\n"
        "Remove or replace terms that are outside the request's domain, too broad, "
        "or likely to retrieve irrelevant literature. Preserve explicit named authors, "
        "named models, article titles, and definite years if present.\n"
        "Prefer several precise keyword sets over one broad query.\n\n"
        f"Research question: {query}\n"
        f"In-scope topics: {include_topics}\n"
        f"Candidate keyword JSON: {json.dumps({'primary': primary, 'variations': variations}, ensure_ascii=True)}\n\n"
        "Return ONLY JSON with keys primary and variations."
    ), (
        "Rewrite the keyword sets for this academic search. Return JSON only.\n"
        f"Question: {query}\n"
        f"In-scope topics: {include_topics}\n"
        f"Current JSON: {json.dumps({'primary': primary, 'variations': variations}, ensure_ascii=True)}\n"
        "Output shape: {\"primary\":[[\"term\"]],\"variations\":[[\"term\"],[\"term\"]]}"
    )]
    last_error: Exception | None = None
    for prompt in prompts:
        try:
            response = await llm.complete_async(
                system=(
                    "Return valid JSON only. No markdown. No explanation. "
                    "No step-by-step reasoning."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            revised_primary, revised_variations = _parse_response(response)
            if revised_primary:
                return revised_primary, revised_variations
        except Exception as exc:
            last_error = exc
            log.warning("Keyword revision LLM call failed (%s)", exc)
    if last_error is not None:
        log.warning("Keeping existing keywords after revision retry failure")
    return primary, variations


def _compact_terms(values: list[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        text = str(value or "")
        for phrase in re.findall(r'"([^"]{3,80})"', text):
            _append_unique(found, _normalize_term(phrase.lower()))
        for part in re.split(r"[,;\n]+", text):
            candidate = _normalize_term(part.lower())
            if 3 <= len(candidate) <= 80 and len(candidate.split()) <= 6:
                _append_unique(found, candidate)
    text = " ".join(str(value or "") for value in values).lower()
    words = [
        word
        for word in re.findall(r"[a-z][a-z-]{3,}", text)
        if word not in _STOPWORDS and not word.isdigit()
    ]
    for size in (3, 2, 1):
        for idx in range(0, max(0, len(words) - size + 1)):
            if len(found) >= 12:
                return found
            _append_unique(found, " ".join(words[idx:idx + size]))
    return found


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" .:-\t"))


def _append_unique(values: list[str], value: str) -> None:
    if value and value.lower() not in {item.lower() for item in values}:
        values.append(value)
