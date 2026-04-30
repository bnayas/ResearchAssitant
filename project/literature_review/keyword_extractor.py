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
            max_tokens=512,
            temperature=0.0,
        )
        primary, variations = _parse_response(response)
    except Exception as exc:
        log.warning("Keyword extraction LLM call failed (%s); using heuristic fallback", exc)
        primary, variations = _heuristic_fallback(query, include_topics)

    # Safety: always return at least something
    if not primary:
        primary, variations = _heuristic_fallback(query, include_topics)

    log.debug("Extracted primary=%s variations=%s", primary, variations)
    return primary, variations


def _parse_response(
    text: str,
) -> tuple[list[list[str]], list[list[str]]]:
    """Parse LLM JSON response; raises on failure."""
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

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

    return _validate(primary), _validate(variations)


def _heuristic_fallback(
    query: str,
    include_topics: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Fallback when LLM extraction fails.
    Pairs specific include_topics as AND-ed multi-word queries.
    """
    # Use pairs of include_topics as AND-queries (keeps phrase specificity)
    primary: list[list[str]] = []
    for i in range(0, min(len(include_topics), 6), 2):
        pair = include_topics[i:i+2]
        if pair:
            primary.append(pair)
    if not primary:
        primary = [[query[:60]]]

    # Variation: first 4 significant words from the query
    words = [w for w in query.split() if len(w) > 4][:4]
    variations = [words] if len(words) >= 2 else [include_topics[:2]]

    return primary, variations
