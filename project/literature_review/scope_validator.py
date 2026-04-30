"""
literature_review/scope_validator.py
──────────────────────────────────────
Per-paper scope validation using LLM + deterministic year gates.
"""
from __future__ import annotations
from typing import Optional, Tuple
from .contract import ScopeConstraint, LiteraturePaper
from .search_backends.base import RawPaper


def _heuristic_scope_check(
    raw: RawPaper,
    scope: ScopeConstraint,
) -> Tuple[bool, str]:
    text = f"{raw.title}\n{raw.abstract}".lower()
    for topic in scope.exclude_topics:
        candidate = topic.strip().lower()
        if candidate and candidate in text:
            return False, f"matched excluded topic: {topic}"

    include_topics = [topic.strip().lower() for topic in scope.include_topics if topic.strip()]
    if not include_topics:
        return True, ""

    # Match any significant word (>3 chars) from any include topic.
    # This handles plural/morphological variants (e.g. "neutral models" vs "neutral model")
    for topic in include_topics:
        words = [w for w in topic.split() if len(w) > 3]
        if words and all(w in text for w in words):
            return True, ""
        # Also try the topic as a whole
        if topic in text:
            return True, ""

    return False, "no include topic match in title/abstract"


async def check_paper_scope(
    raw: RawPaper,
    scope: ScopeConstraint,
    llm,
) -> Tuple[bool, str]:
    """
    Returns (in_scope, reason).
    Applies deterministic year gates first, then LLM topic check.
    """
    if not raw.title.strip():
        return False, "empty title"
    if scope.year_min and raw.year is not None and raw.year < scope.year_min:
        return False, f"year {raw.year} < year_min {scope.year_min}"
    if scope.year_max and raw.year is not None and raw.year > scope.year_max:
        return False, f"year {raw.year} > year_max {scope.year_max}"

    # LLM scope check
    content = f"Title: {raw.title}\nAbstract: {raw.abstract}"
    topics_str = ", ".join(scope.include_topics)
    system = "scope validator: return JSON {\"in_scope\": bool, \"reason\": str}"
    msg = (
        f"In-scope topics: {topics_str}\n"
        f"Exclude topics: {', '.join(scope.exclude_topics)}\n\n"
        f"Paper:\n{content}\n\nIs this paper in scope?"
    )
    try:
        import json
        raw_resp = await llm.complete_async(
            system=system,
            messages=[{"role": "user", "content": msg}],
            max_tokens=128,
            temperature=0.0,
        )
        data = json.loads(raw_resp.strip().strip("`").strip())
        llm_in_scope = bool(data.get("in_scope", True))
        llm_reason = data.get("reason", "")

        heuristic_ok, heuristic_reason = _heuristic_scope_check(raw, scope)
        if not llm_in_scope:
            return False, llm_reason or heuristic_reason or "LLM scope rejection"
        if heuristic_ok:
            return True, ""
        return True, ""
    except Exception:
        return _heuristic_scope_check(raw, scope)


def raw_to_paper(raw: RawPaper, in_scope: bool, reason: str) -> LiteraturePaper:
    return LiteraturePaper(
        title=raw.title,
        authors=raw.authors,
        abstract=raw.abstract,
        url=raw.url,
        year=raw.year,
        source=raw.source,
        arxiv_id=raw.arxiv_id,
        doi=raw.doi,
        in_scope=in_scope,
        scope_violation_reason=reason if not in_scope else None,
    )
