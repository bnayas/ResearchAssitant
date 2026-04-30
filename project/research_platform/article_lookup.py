"""
research_platform.article_lookup
────────────────────────────────
Generic fallback helpers for turning a PI directive into a targeted
article-lookup query when the live LLM-prepared query needs recovery.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'`-]*|(?:19|20)\d{2}")

_LOWERCASE_STOPWORDS = {
    "a",
    "an",
    "about",
    "add",
    "adding",
    "analyse",
    "analyze",
    "and",
    "apply",
    "around",
    "as",
    "at",
    "build",
    "by",
    "called",
    "compare",
    "compute",
    "concerning",
    "correct",
    "derive",
    "describe",
    "discuss",
    "explain",
    "find",
    "for",
    "from",
    "get",
    "how",
    "implement",
    "in",
    "into",
    "investigate",
    "launch",
    "locate",
    "of",
    "on",
    "paper",
    "papers",
    "prepare",
    "read",
    "regarding",
    "reproduce",
    "results",
    "result",
    "review",
    "run",
    "simulate",
    "study",
    "summarize",
    "target",
    "the",
    "this",
    "that",
    "to",
    "titled",
    "use",
    "using",
    "what",
    "with",
    "work",
    "write",
}

_CAPITALIZED_STOPWORDS = {word.title() for word in _LOWERCASE_STOPWORDS}
_GENERIC_TOPIC_TAILS = {"paper", "papers", "article", "articles", "study", "studies", "work", "works", "results", "result"}


def build_article_lookup_query(topic_hint: str, instruction: str) -> str:
    """Return a concise comma-separated query for target-paper lookup."""
    hint = _normalize_space(topic_hint)
    instruction_terms = _derive_terms(instruction)
    if hint:
        terms = _split_terms(hint)
        derived = _derive_terms(hint)
        if terms and not _looks_like_keyword_list(hint):
            terms = derived + instruction_terms + terms
        else:
            terms = terms + derived + instruction_terms
        terms = _dedupe_terms(terms)
        if terms:
            return ", ".join(terms[:8])
        return hint

    if instruction_terms:
        return ", ".join(instruction_terms[:8])
    return _normalize_space(instruction)


def article_query_candidates(*values: str) -> list[str]:
    candidates: list[str] = []
    for value in values:
        normalized = _normalize_space(value)
        if not normalized:
            continue
        if normalized.lower() not in {item.lower() for item in candidates}:
            candidates.append(normalized)
    return candidates


def _derive_terms(text: str) -> list[str]:
    source = _normalize_space(text)
    if not source:
        return []

    terms: list[str] = []
    for match in re.finditer(r"\b([A-Z][A-Za-z'`.-]{2,})\s+(?:and|&)\s+([A-Z][A-Za-z'`.-]{2,})\b", source):
        terms.extend([match.group(1), match.group(2)])

    for year in re.findall(r"\b(19\d{2}|20\d{2})\b", source):
        terms.append(year)

    terms.extend(_extract_topic_phrases(source))

    for token in re.findall(r"\b[A-Z][A-Za-z'`.-]{2,}\b", source):
        if token in _CAPITALIZED_STOPWORDS:
            continue
        token_lower = token.lower()
        if any(token_lower in item.lower().split() for item in terms):
            continue
        terms.append(token)

    return _dedupe_terms(terms)


def _extract_topic_phrases(source: str) -> list[str]:
    tokens = _TOKEN_RE.findall(source)
    if not tokens:
        return []

    phrases: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            phrases.extend(_phrases_from_segment(current))
            current = []

    for token in tokens:
        lowered = token.lower()
        if re.fullmatch(r"(19|20)\d{2}", token):
            flush()
            continue
        if lowered in _LOWERCASE_STOPWORDS:
            flush()
            continue
        current.append(token)
    flush()
    return _dedupe_terms(phrases)


def _phrases_from_segment(tokens: list[str]) -> list[str]:
    segment = list(tokens)
    while segment and segment[-1].lower() in _GENERIC_TOPIC_TAILS:
        segment.pop()
    while len(segment) >= 3 and _looks_like_name_token(segment[0]) and _looks_like_topic_token(segment[1]):
        segment.pop(0)

    if len(segment) < 2:
        return []
    if len(segment) <= 4:
        return [" ".join(segment)]
    return [" ".join(segment[:4])]


def _looks_like_name_token(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z'`.-]{2,}", token))


def _looks_like_topic_token(token: str) -> bool:
    return bool(re.search(r"[-a-z]", token))


def _split_terms(text: str) -> list[str]:
    if not text:
        return []
    if _looks_like_keyword_list(text):
        return [part.strip() for part in re.split(r"[,\n;]+", text) if part.strip()]
    return []


def _looks_like_keyword_list(text: str) -> bool:
    if "," in text or ";" in text or "\n" in text:
        return True
    return len(text.split()) <= 6


def _dedupe_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_space(value)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
