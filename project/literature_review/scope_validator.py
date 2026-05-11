"""
literature_review/scope_validator.py
──────────────────────────────────────
Per-paper scope validation using LLM + deterministic year gates.
"""
from __future__ import annotations
import re
from typing import Optional, Tuple
from .contract import ScopeConstraint, LiteraturePaper
from .search_backends.base import RawPaper

_SCOPE_STOPWORDS = {
    "about", "after", "again", "also", "between", "could", "during", "each",
    "from", "have", "into", "model", "models", "paper", "papers", "result",
    "results", "that", "their", "there", "these", "this", "using", "with",
}

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

    author_hits = _author_overlap(raw.authors, scope.required_authors)
    required_author_count = len(_author_surnames(scope.required_authors))
    if required_author_count and author_hits < required_author_count:
        missing = ", ".join(_missing_required_authors(raw.authors, scope.required_authors))
        return False, f"missing required author(s): {missing}"
    topic_hits = _topic_hit_count(text, include_topics)
    if author_hits >= 2 and topic_hits >= 1:
        return True, ""
    if author_hits >= 1 and topic_hits >= 2:
        return True, ""
    if topic_hits >= 2:
        return True, ""

    for topic in include_topics:
        if len(topic) <= 80 and topic in text:
            return True, ""
        words = _topic_words(topic)
        if 1 <= len(words) <= 3 and all(_word_in_text(word, text) for word in words):
            return True, ""

    if author_hits:
        return False, "author match but no topic signal in title/abstract"
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

    heuristic_ok, heuristic_reason = _heuristic_scope_check(raw, scope)
    if _author_overlap(raw.authors, scope.required_authors) > 0 and _topic_hit_count(
        f"{raw.title}\n{raw.abstract}".lower(),
        [topic.strip().lower() for topic in scope.include_topics if topic.strip()],
    ) >= 1:
        return True, ""

    # LLM scope check
    content = f"Title: {raw.title}\nAbstract: {raw.abstract}"
    topics_str = ", ".join(scope.include_topics)
    system = (
        "scope validator: return minimal JSON "
        "{\"in_scope\": bool, \"reason\": str}"
    )
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

        if not llm_in_scope:
            return False, llm_reason or heuristic_reason or "LLM scope rejection"
        return True, ""
    except Exception:
        return heuristic_ok, heuristic_reason


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


def _author_overlap(raw_authors: list[str], required_authors: list[str]) -> int:
    if not required_authors:
        return 0
    raw_text = " ".join(raw_authors).lower()
    hits = 0
    for author in required_authors:
        surname = str(author or "").strip().lower().split()[-1]
        if surname and re.search(rf"\b{re.escape(surname)}\b", raw_text):
            hits += 1
    return hits


def _author_surnames(authors: list[str]) -> list[str]:
    surnames: list[str] = []
    for author in authors:
        text = str(author or "").strip()
        if not text:
            continue
        surname = text.split(",", 1)[0].strip() if "," in text else text.split()[-1].strip()
        if surname and surname.lower() not in {item.lower() for item in surnames}:
            surnames.append(surname)
    return surnames


def _missing_required_authors(raw_authors: list[str], required_authors: list[str]) -> list[str]:
    raw_text = " ".join(raw_authors).lower()
    missing: list[str] = []
    for surname in _author_surnames(required_authors):
        if not re.search(rf"\b{re.escape(surname.lower())}\b", raw_text):
            missing.append(surname)
    return missing


def _topic_hit_count(text: str, include_topics: list[str]) -> int:
    tokens: set[str] = set()
    for topic in include_topics:
        tokens.update(_topic_words(topic))
    return sum(1 for token in tokens if _word_in_text(token, text))


def _topic_words(topic: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z-]{3,}", topic.lower())
        if token not in _SCOPE_STOPWORDS and len(token) >= 5
    ][:8]


def _word_in_text(word: str, text: str) -> bool:
    if word.endswith("y"):
        pattern = rf"\b{re.escape(word[:-1])}(y|ies)\b"
    elif word.endswith("s"):
        pattern = rf"\b{re.escape(word[:-1])}s?\b"
    else:
        pattern = rf"\b{re.escape(word)}s?\b"
    return re.search(pattern, text) is not None
