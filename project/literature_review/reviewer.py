"""
literature_review/reviewer.py
──────────────────────────────
LiteratureReviewer — multi-round search, scope filtering, synthesis.
"""
from __future__ import annotations
import asyncio
import inspect
import json
import logging
import re
import uuid
from typing import Callable, Optional
from .contract import (
    LiteratureReviewArtifact, LiteratureReviewTask, LiteraturePaper,
    SearchRound, StreamEvent,
)
from .scope_validator import check_paper_scope, raw_to_paper
from .keyword_extractor import extract_keyword_sets
from .search_backends.base import RawPaper, SearchBackend

log = logging.getLogger(__name__)


class LiteratureReviewer:
    def __init__(self, backends: list[SearchBackend], llm,
                 stream_callback: Optional[Callable] = None):
        self._backends = backends
        self._llm = llm
        self._cb = stream_callback or (lambda e: None)

    def _emit(self, event_type: str, payload: dict) -> None:
        self._cb(StreamEvent(event_type=event_type, payload=payload))

    async def run(self, task: LiteratureReviewTask) -> LiteratureReviewArtifact:
        scope = task.scope
        depth = task.depth
        seen_urls: set[str] = set()
        accepted: list[LiteraturePaper] = []
        removed: list[LiteraturePaper] = []
        search_log: list[SearchRound] = []

        await _infer_definite_scope_constraints(task, self._llm)

        primary_sets, variation_sets = await extract_keyword_sets(
            task.query, scope.include_topics, self._llm
        )
        all_kw_rounds = [primary_sets] + [
            [vs] for vs in variation_sets[:depth.max_term_variations]
        ]

        round_num = 0
        refinement_attempts = 0
        while len(accepted) < depth.min_papers_threshold and round_num < depth.max_rounds:
            if round_num >= len(all_kw_rounds):
                refined = await self._refine_keywords(
                    task=task,
                    search_log=search_log,
                    accepted=accepted,
                    removed=removed,
                    previous_rounds=all_kw_rounds,
                )
                refinement_attempts += 1
                if refined:
                    all_kw_rounds.append(refined)
                    self._emit("keyword_refined", {"round": round_num + 1, "keywords": refined})
                elif refinement_attempts >= 2:
                    break
                else:
                    continue
            kw_sets = all_kw_rounds[round_num]
            round_num += 1
            round_log = await self._run_keyword_round(
                round_num=round_num,
                kw_sets=kw_sets,
                scope=scope,
                depth=depth,
                seen_urls=seen_urls,
                accepted=accepted,
                removed=removed,
            )
            search_log.append(round_log)
            if len(accepted) >= depth.min_papers_threshold:
                break

        if len(accepted) < depth.min_papers_threshold:
            relaxed_rounds = self._relaxed_rounds(task.query, scope)
            for offset, (kw_sets, authors, year_min, year_max) in enumerate(relaxed_rounds, 1):
                if len(accepted) >= depth.min_papers_threshold:
                    break
                round_num = len(search_log) + 1
                round_log = await self._run_keyword_round(
                    round_num=round_num,
                    kw_sets=kw_sets,
                    scope=scope,
                    depth=depth,
                    seen_urls=seen_urls,
                    accepted=accepted,
                    removed=removed,
                    authors=authors,
                    year_min=year_min,
                    year_max=year_max,
                    is_variation=True,
                )
                search_log.append(round_log)

        self._emit("search_exhausted", {"total_accepted": len(accepted)})
        synthesis = await self._synthesize(task.query, accepted)
        self._emit("synthesis_start", {})

        is_sufficient = len(accepted) >= depth.min_papers_threshold
        status = "complete" if is_sufficient else (
            "exhausted" if len(accepted) == 0 else "partial"
        )
        artifact = LiteratureReviewArtifact(
            task_id=task.task_id, branch_id=task.branch_id,
            papers=accepted[:scope.max_papers], removed_papers=removed,
            synthesis=synthesis, search_log=search_log,
            status=status, is_sufficient=is_sufficient,
            insufficiency_reason=None if is_sufficient else (
                f"Only {len(accepted)} papers found; minimum is {depth.min_papers_threshold}."
            ),
        )
        self._emit("done", {"status": status, "accepted": len(accepted)})
        return artifact

    async def _refine_keywords(
        self,
        *,
        task: LiteratureReviewTask,
        search_log: list[SearchRound],
        accepted: list[LiteraturePaper],
        removed: list[LiteraturePaper],
        previous_rounds: list[list[list[str]]],
    ) -> list[list[str]]:
        tried = [kw for round_sets in previous_rounds for kw in round_sets]
        rejected = [
            {
                "title": paper.title,
                "year": paper.year,
                "reason": paper.scope_violation_reason,
            }
            for paper in removed[-8:]
        ]
        prompt = (
            "Choose the next academic database keyword sets. "
            "Prefer terms that reduce noisy candidates before paper validation. "
            "Use compact scientific terms, 1-3 terms per set. "
            "Return only JSON: {\"primary\": [[\"term\", \"term\"]]}.\n\n"
            f"Query: {task.query}\n"
            f"In-scope topics: {task.scope.include_topics}\n"
            f"Required authors: {task.scope.required_authors}\n"
            f"Anchor year: {task.scope.anchor_year}\n"
            f"Already accepted: {[paper.title for paper in accepted[-5:]]}\n"
            f"Recent rejected/noisy candidates: {json.dumps(rejected, ensure_ascii=True)}\n"
            f"Previous keyword sets: {tried}\n"
            f"Round summaries: {[{'raw': r.papers_found_raw, 'in_scope': r.papers_in_scope, 'keywords': r.keyword_sets} for r in search_log]}"
        )
        try:
            raw = await self._llm.complete_async(
                system=(
                    "You refine academic search keywords. Return valid JSON only. "
                    "Do not explain and do not reason step by step."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            parsed = _parse_keyword_refinement(raw)
            if parsed:
                return parsed
        except Exception as exc:
            log.warning("Keyword refinement failed: %s", exc)
        return _next_deterministic_keywords(task.query, task.scope, tried)

    async def _run_keyword_round(
        self,
        *,
        round_num: int,
        kw_sets: list[list[str]],
        scope,
        depth,
        seen_urls: set[str],
        accepted: list[LiteraturePaper],
        removed: list[LiteraturePaper],
        authors: Optional[list[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        is_variation: bool = False,
    ) -> SearchRound:
        author_filter = authors if authors is not None else _author_surnames(scope.required_authors)
        self._emit(
            "search_round_start",
            {
                "round": round_num,
                "keywords": kw_sets,
                "authors": author_filter or [],
                "year_min": year_min if year_min is not None else scope.year_min,
                "year_max": year_max if year_max is not None else scope.year_max,
                "relaxed": is_variation,
            },
        )
        log.info(
            "Literature search round %s keywords=%s authors=%s year=%s-%s relaxed=%s",
            round_num,
            kw_sets,
            author_filter or [],
            year_min if year_min is not None else scope.year_min,
            year_max if year_max is not None else scope.year_max,
            is_variation,
        )
        round_raw = 0
        round_in_scope = 0
        sources: list[str] = []
        candidates: list[RawPaper] = []

        for kw_set in kw_sets:
            for backend in self._backends:
                try:
                    papers = await _backend_search(
                        backend,
                        keywords=kw_set,
                        max_results=depth.papers_per_query,
                        year_min=year_min if year_min is not None else scope.year_min,
                        year_max=year_max if year_max is not None else scope.year_max,
                        authors=author_filter,
                    )
                except Exception as exc:
                    log.warning("Literature backend %s failed for %s: %s", backend.name, kw_set, exc)
                    continue
                if backend.name not in sources:
                    sources.append(backend.name)
                for raw in papers:
                    if not raw.is_usable() or raw.url in seen_urls:
                        continue
                    seen_urls.add(raw.url)
                    round_raw += 1
                    candidates.append(raw)

        ranked = _rank_candidates(candidates, scope)
        validation_budget = max(depth.min_papers_threshold * 3, scope.max_papers)
        validated = 0
        skipped = max(0, len(ranked) - min(len(ranked), validation_budget))
        required_author_surnames = _author_surnames(scope.required_authors)
        for raw, score in ranked[:validation_budget]:
            missing_authors = _missing_required_authors(raw.authors, required_author_surnames)
            if missing_authors:
                skipped += 1
                reason = "missing required author(s): " + ", ".join(missing_authors)
                paper = raw_to_paper(raw, False, reason)
                removed.append(paper)
                self._emit("scope_removed", {
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "authors": paper.authors,
                    "url": paper.url,
                    "arxiv_id": paper.arxiv_id,
                    "doi": paper.doi,
                    "abstract": paper.abstract,
                    "reason": reason,
                    "score": score,
                })
                continue
            if score < _minimum_candidate_score(scope):
                skipped += 1
                continue
            validated += 1
            in_scope, reason = await check_paper_scope(raw, scope, self._llm)
            paper = raw_to_paper(raw, in_scope, reason)
            if in_scope:
                accepted.append(paper)
                round_in_scope += 1
                self._emit("title_found", {
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "authors": paper.authors,
                    "url": paper.url,
                    "arxiv_id": paper.arxiv_id,
                    "doi": paper.doi,
                    "score": score,
                })
                if len(accepted) >= scope.max_papers:
                    break
            else:
                removed.append(paper)
                self._emit("scope_removed", {
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "authors": paper.authors,
                    "url": paper.url,
                    "arxiv_id": paper.arxiv_id,
                    "doi": paper.doi,
                    "abstract": paper.abstract,
                    "reason": reason,
                    "score": score,
                })

        self._emit(
            "round_summary",
            {
                "round": round_num,
                "raw": round_raw,
                "in_scope": round_in_scope,
                "validated": validated,
                "skipped_prefilter": skipped,
                "sources": sources,
            },
        )
        return SearchRound(
            round_number=round_num,
            keyword_sets=kw_sets,
            sources_queried=sources,
            papers_found_raw=round_raw,
            papers_in_scope=round_in_scope,
            is_variation=is_variation,
        )

    def _relaxed_rounds(self, query: str, scope) -> list[tuple[list[list[str]], list[str], Optional[int], Optional[int]]]:
        authors = _author_surnames(scope.required_authors)
        year_min = scope.year_min
        year_max = scope.year_max
        if scope.anchor_year and year_min is None and year_max is None:
            year_min = max(1900, int(scope.anchor_year) - 12)
            year_max = int(scope.anchor_year) + 12
        compact_terms = _relaxed_terms(query, scope.include_topics)
        kw_sets = [[term] for term in compact_terms[:4]]
        if not kw_sets:
            fallback = _normalize_term(query)[:80] or "research topic"
            kw_sets = [[fallback]]
        rounds: list[tuple[list[list[str]], list[str], Optional[int], Optional[int]]] = []
        if authors:
            if len(authors) >= 2:
                rounds.append((kw_sets, authors[:2], year_min, year_max))
            for author in authors[:3]:
                rounds.append((kw_sets, [author], year_min, year_max))
        else:
            rounds.append((kw_sets, [], year_min, year_max))
        return rounds

    async def _synthesize(self, query: str, papers: list[LiteraturePaper]) -> str:
        if not papers:
            return "No in-scope papers were found for this query."
        paper_list = "\n".join(
            (
                f"- Title: {p.title}\n"
                f"  Authors: {', '.join(p.authors)}\n"
                f"  Year: {p.year or '?'}\n"
                f"  Abstract: {p.abstract[:1000]}"
            )
            for p in papers[:15]
        )
        prompts = [
            (
                f"Query: {query}\n\nPapers:\n{paper_list}\n\n"
                "Write the final synthesis only. Do not include reasoning traces, "
                "planning notes, self-corrections, or references to snippets. "
                "Prioritize papers whose authors/title match the query. If the query asks "
                "about simulations, summarize every simulation-relevant detail present in "
                "the titles/abstracts and clearly mark what still requires full text."
            ),
            (
                f"Query: {query}\n\nPapers:\n{paper_list}\n\n"
                "Final answer only: concise scientific synthesis, no meta commentary, "
                "no hidden reasoning, no step-by-step analysis."
            ),
        ]
        last_response = ""
        for prompt in prompts:
            try:
                response = await self._llm.complete_async(
                    system=(
                        "You are a scientific literature synthesizer. Return only the final "
                        "reader-facing synthesis."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                last_response = str(response or "").strip()
                if last_response and not _looks_like_reasoning_trace(last_response):
                    return last_response
            except Exception:
                continue
        return last_response or f"Synthesis of {len(papers)} papers on: {query}"


async def _backend_search(
    backend: SearchBackend,
    *,
    keywords: list[str],
    max_results: int,
    year_min: Optional[int],
    year_max: Optional[int],
    authors: Optional[list[str]],
) -> list[RawPaper]:
    signature = inspect.signature(backend.search)
    if "authors" in signature.parameters:
        return await backend.search(
            keywords=keywords,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            authors=authors,
        )
    return await backend.search(
        keywords=keywords,
        max_results=max_results,
        year_min=year_min,
        year_max=year_max,
    )


async def _infer_definite_scope_constraints(task: LiteratureReviewTask, llm) -> None:
    if task.scope.required_authors and task.scope.anchor_year is not None:
        return
    source_text = "\n".join([task.query, *task.scope.include_topics])
    prompts = [(
        "Extract only definite literature search constraints from the request. "
        "Do not guess. If a person is named as an author of the requested paper/article, "
        "put the surname in required_authors. If a publication year is explicitly stated, "
        "put it in anchor_year. If not definite, return an empty list/null.\n\n"
        f"Request:\n{source_text}\n\n"
        "Return ONLY JSON like "
        '{"required_authors":["Surname"],"anchor_year":2017,"notes":"short reason"}'
    ), (
        "Text:\n"
        f"{source_text}\n\n"
        "Return this exact JSON shape and nothing else:\n"
        '{"required_authors":[],"anchor_year":null}\n'
        "Fill required_authors only with surnames of people the text identifies "
        "as authors of the requested paper/article. Fill anchor_year only when "
        "the text gives a definite publication year."
    )]
    payload: dict = {}
    last_error: Optional[Exception] = None
    for prompt in prompts:
        try:
            raw = await llm.complete_async(
                system=(
                    "Return only valid JSON. No markdown. No explanation. "
                    "No step-by-step reasoning."
                ),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            payload = _parse_json_object(raw)
            if payload:
                break
        except Exception as exc:
            last_error = exc
            log.warning("Literature constraint extraction attempt failed: %s", exc)
    if not payload:
        if last_error is not None:
            log.warning("Literature constraint extraction failed after retry: %s", last_error)
        return

    if not task.scope.required_authors:
        authors = _coerce_string_list(
            payload.get("required_authors") or payload.get("authors") or []
        )
        grounded_authors = [
            author
            for author in authors
            if _author_constraint_is_grounded(author, source_text)
        ]
        task.scope.required_authors = _dedupe_strings(grounded_authors)[:4]
    if task.scope.anchor_year is None:
        year = _coerce_year(payload.get("anchor_year") or payload.get("year"))
        if year is not None and re.search(rf"\b{year}\b", source_text):
            task.scope.anchor_year = year


def _parse_json_object(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", str(raw or "")).strip().strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def _coerce_string_list(value) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _coerce_year(value) -> Optional[int]:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1800 <= year <= 2200 else None


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() not in {item.lower() for item in deduped}:
            deduped.append(text)
    return deduped


def _author_constraint_is_grounded(author: str, source_text: str) -> bool:
    surname = _author_surnames([author])
    if not surname:
        return False
    return re.search(rf"\b{re.escape(surname[0])}\b", source_text, re.IGNORECASE) is not None


def _author_surnames(authors: list[str]) -> list[str]:
    surnames: list[str] = []
    for author in authors:
        text = str(author or "").strip()
        if not text:
            continue
        if "," in text:
            surname = text.split(",", 1)[0].strip()
        else:
            surname = text.split()[-1].strip()
        if surname and surname.lower() not in {item.lower() for item in surnames}:
            surnames.append(surname)
    return surnames


def _missing_required_authors(raw_authors: list[str], required_author_surnames: list[str]) -> list[str]:
    if not required_author_surnames:
        return []
    raw_text = " ".join(raw_authors).lower()
    missing: list[str] = []
    for surname in required_author_surnames:
        if not re.search(rf"\b{re.escape(surname.lower())}\b", raw_text):
            missing.append(surname)
    return missing


def _relaxed_terms(query: str, include_topics: list[str]) -> list[str]:
    return _compact_search_terms([query, *include_topics])[:6]


def _parse_keyword_refinement(raw: str) -> list[list[str]]:
    text = re.sub(r"```(?:json)?", "", str(raw or "")).strip().strip("`").strip()
    if not text:
        return []
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    payload = json.loads(text)
    if isinstance(payload, dict):
        candidates = payload.get("primary") or payload.get("keywords") or []
    else:
        candidates = payload
    parsed: list[list[str]] = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, list):
                terms = [str(term).strip() for term in item if str(term).strip()]
            else:
                terms = [str(item).strip()] if str(item).strip() else []
            if terms:
                parsed.append(terms[:3])
    return parsed[:3]


def _looks_like_reasoning_trace(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "thinking process",
        "self-correction",
        "execution plan",
        "analyze the request",
        "the user is asking",
        "scan snippets",
        "final output generation",
    ]
    return any(marker in lowered for marker in markers)


_GENERIC_TERM_STOPWORDS = {
    "about", "after", "also", "article", "based", "before", "between", "compare",
    "describe", "does", "from", "have", "into", "paper", "papers", "read",
    "result", "results", "review", "show", "study", "that", "their", "there",
    "these", "this", "through", "using", "what", "when", "where", "which",
    "with", "without",
}


def _next_deterministic_keywords(query: str, scope, tried: list[list[str]]) -> list[list[str]]:
    tried_norm = {
        " | ".join(term.lower() for term in terms)
        for terms in tried
    }
    compact_terms = _compact_search_terms([query, *scope.include_topics])
    candidates: list[list[str]] = [[term] for term in compact_terms]
    if len(compact_terms) >= 2:
        candidates.insert(0, compact_terms[:2])
    if len(compact_terms) >= 4:
        candidates.insert(1, compact_terms[2:4])
    for topic in scope.include_topics:
        text = str(topic or "").strip()
        if text and len(text) <= 60:
            candidates.append([text])
    selected: list[list[str]] = []
    for terms in candidates:
        key = " | ".join(term.lower() for term in terms)
        if key not in tried_norm:
            selected.append(terms)
        if len(selected) >= 3:
            break
    return selected


def _compact_search_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for phrase in re.findall(r'"([^"]{3,80})"', text):
            _append_unique(terms, _normalize_term(phrase))
        for part in re.split(r"[,;\n]+", text):
            candidate = _normalize_term(part)
            if 3 <= len(candidate) <= 80 and len(candidate.split()) <= 6:
                _append_unique(terms, candidate)
    words = [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", " ".join(values).lower())
        if word not in _GENERIC_TERM_STOPWORDS
    ]
    for size in (3, 2, 1):
        for idx in range(0, max(0, len(words) - size + 1)):
            term = " ".join(words[idx:idx + size])
            _append_unique(terms, term)
            if len(terms) >= 12:
                return terms
    return terms


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" .:-\t"))


def _append_unique(values: list[str], value: str) -> None:
    if value and value.lower() not in {item.lower() for item in values}:
        values.append(value)


def _rank_candidates(candidates: list[RawPaper], scope) -> list[tuple[RawPaper, int]]:
    ranked = [(raw, _candidate_score(raw, scope)) for raw in candidates]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _candidate_score(raw: RawPaper, scope) -> int:
    text = f"{raw.title}\n{raw.abstract}".lower()
    score = 0
    author_hits = _author_hit_count(raw.authors, scope.required_authors)
    score += author_hits * 4
    if scope.anchor_year and raw.year is not None:
        delta = abs(int(raw.year) - int(scope.anchor_year))
        if delta == 0:
            score += 3
        elif delta <= 3:
            score += 2
        elif delta <= 12:
            score += 1
    topic_hits = _topic_hit_count(text, scope.include_topics)
    score += min(topic_hits, 6)
    for topic in scope.include_topics:
        phrase = str(topic or "").strip().lower()
        if phrase and len(phrase) <= 80 and phrase in text:
            score += 2
    for topic in scope.exclude_topics:
        phrase = str(topic or "").strip().lower()
        if phrase and phrase in text:
            score -= 6
    return score


def _minimum_candidate_score(scope) -> int:
    if scope.required_authors or scope.anchor_year:
        return 2
    return 1


def _author_hit_count(raw_authors: list[str], required_authors: list[str]) -> int:
    raw_text = " ".join(raw_authors).lower()
    hits = 0
    for author in required_authors:
        surname = str(author or "").strip().lower().split()[-1]
        if surname and re.search(rf"\b{re.escape(surname)}\b", raw_text):
            hits += 1
    return hits


def _topic_hit_count(text: str, include_topics: list[str]) -> int:
    tokens: set[str] = set()
    for topic in include_topics:
        tokens.update(
            token
            for token in re.findall(r"[a-z][a-z-]{3,}", str(topic or "").lower())
            if token not in {
                "about", "after", "also", "during", "from", "model", "models",
                "paper", "papers", "result", "results", "that", "their", "this",
                "using", "with",
            }
        )
    hits = 0
    for token in tokens:
        if re.search(rf"\b{re.escape(token.rstrip('s'))}s?\b", text):
            hits += 1
    return hits
