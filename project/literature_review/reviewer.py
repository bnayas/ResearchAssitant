"""
literature_review/reviewer.py
──────────────────────────────
LiteratureReviewer — multi-round search, scope filtering, synthesis.
"""
from __future__ import annotations
import asyncio
import uuid
from typing import Callable, Optional
from .contract import (
    LiteratureReviewArtifact, LiteratureReviewTask, LiteraturePaper,
    SearchRound, StreamEvent,
)
from .scope_validator import check_paper_scope, raw_to_paper
from .keyword_extractor import extract_keyword_sets
from .search_backends.base import RawPaper, SearchBackend


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

        primary_sets, variation_sets = await extract_keyword_sets(
            task.query, scope.include_topics, self._llm
        )
        all_kw_rounds = [primary_sets] + [
            [vs] for vs in variation_sets[:depth.max_term_variations]
        ]

        for round_num, kw_sets in enumerate(all_kw_rounds[:depth.max_rounds], 1):
            self._emit("search_round_start", {"round": round_num, "keywords": kw_sets})
            round_raw = 0
            round_in_scope = 0
            sources: list[str] = []

            for kw_set in kw_sets:
                for backend in self._backends:
                    try:
                        papers = await backend.search(
                            keywords=kw_set,
                            max_results=depth.papers_per_query,
                            year_min=scope.year_min,
                            year_max=scope.year_max,
                        )
                    except Exception:
                        continue
                    if backend.name not in sources:
                        sources.append(backend.name)
                    for raw in papers:
                        if not raw.is_usable() or raw.url in seen_urls:
                            continue
                        seen_urls.add(raw.url)
                        round_raw += 1
                        in_scope, reason = await check_paper_scope(raw, scope, self._llm)
                        paper = raw_to_paper(raw, in_scope, reason)
                        if in_scope:
                            accepted.append(paper)
                            round_in_scope += 1
                            self._emit("title_found", {"title": paper.title,
                                                        "year": paper.year,
                                                        "source": paper.source})
                            if len(accepted) >= scope.max_papers:
                                break
                        else:
                            removed.append(paper)
                            self._emit("scope_removed", {"title": paper.title,
                                                          "reason": reason})
                    if len(accepted) >= scope.max_papers:
                        break
                if len(accepted) >= scope.max_papers:
                    break

            search_log.append(SearchRound(
                round_number=round_num, keyword_sets=kw_sets,
                sources_queried=sources, papers_found_raw=round_raw,
                papers_in_scope=round_in_scope,
            ))
            if len(accepted) >= depth.min_papers_threshold:
                break

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

    async def _synthesize(self, query: str, papers: list[LiteraturePaper]) -> str:
        if not papers:
            return "No in-scope papers were found for this query."
        paper_list = "\n".join(
            f"- {p.title} ({p.year}): {p.abstract[:200]}" for p in papers[:15]
        )
        system = "You are a scientific literature synthesizer. Write a concise synthesis."
        msg = f"Query: {query}\n\nPapers:\n{paper_list}\n\nProvide a synthesis."
        try:
            return await self._llm.complete_async(
                system=system,
                messages=[{"role": "user", "content": msg}],
                max_tokens=512,
                temperature=0.0,
            )
        except Exception:
            return f"Synthesis of {len(papers)} papers on: {query}"
