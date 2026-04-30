"""
literature_review.article_parser
────────────────────────────────
Helpers for target-article lookup and article-question answering.

This module does two narrow jobs:
1. find one candidate paper from a prepared article-lookup query,
2. answer a fixed list of article-reading questions from the paper text.

It does not build a `SimulationBlueprint` object. The caller is responsible
for turning the returned answers into whatever higher-level artifact it needs.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Optional, Protocol
from bs4 import BeautifulSoup

from .contract import LiteraturePaper
from .scope_validator import raw_to_paper
from .search_backends.arxiv_backend import ArXivBackend
from .search_backends.base import SearchBackend

log = logging.getLogger(__name__)


class SyncLLMBackend(Protocol):
    def complete(self, system: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str: ...


class ArticleParser:
    def __init__(
        self,
        primary_backend: SearchBackend,
        fallback_backends: Optional[list[SearchBackend]] = None,
    ):
        self.primary_backend = primary_backend
        self.fallback_backends = list(fallback_backends or [])
        self.last_search_diagnostics: dict[str, object] = {}

    def find_article(
        self,
        topic_hint: str,
        *,
        lookup_spec: Optional[dict[str, object]] = None,
    ) -> Optional[LiteraturePaper]:
        """
        Search the configured primary backend and then fallbacks for one target
        article and normalize the top hit.

        `topic_hint` here is expected to be a prepared lookup query such as
        `"environmental stochasticity, neutral model, Danino, Shnerb"`, not an
        arbitrary paragraph of PI instructions.
        """
        import asyncio
        normalized_spec = self._normalize_lookup_spec(topic_hint, lookup_spec or {})
        keywords = normalized_spec["query_terms"]
        if not keywords:
            keywords = [topic_hint]
        year_min = normalized_spec.get("year_min")
        year_max = normalized_spec.get("year_max")

        async def _search_all_backends():
            combined: list[object] = []
            rate_limited_backends: list[str] = []
            attempted_backends: list[str] = []
            for backend in [self.primary_backend, *self.fallback_backends]:
                backend_name = getattr(backend, "name", backend.__class__.__name__)
                attempted_backends.append(backend_name)
                results = await backend.search(
                    keywords,
                    max_results=5,
                    year_min=year_min,
                    year_max=year_max,
                )
                if getattr(backend, "_last_rate_limited", False):
                    rate_limited_backends.append(backend_name)
                if results:
                    log.info(
                        "Article lookup received %d result(s) via %s for query %r",
                        len(results),
                        backend_name,
                        topic_hint,
                    )
                    combined.extend(results)
            self.last_search_diagnostics = {
                "attempted_backends": attempted_backends,
                "rate_limited_backends": rate_limited_backends,
                "all_rate_limited": bool(attempted_backends) and len(rate_limited_backends) == len(attempted_backends),
            }
            return combined

        results = asyncio.run(_search_all_backends())
        if not results:
            return None
        results = self._dedupe_results(results)
        results = self._apply_hard_constraints(results, normalized_spec)
        if not results:
            return None
        # Re-rank the returned set so explicit author/year/title hints can
        # overcome broad ArXiv relevance when the PI asks for a specific paper.
        ranked = sorted(
            results,
            key=lambda result: self._match_score(result, normalized_spec),
            reverse=True,
        )
        return raw_to_paper(ranked[0], in_scope=True, reason="")

    @staticmethod
    def _match_score(result: object, lookup_spec: dict[str, object]) -> tuple[int, int, int, int, int, int]:
        keywords = [str(item).strip().lower() for item in (lookup_spec.get("query_terms") or []) if str(item).strip()]
        required_authors = [str(item).strip().lower() for item in (lookup_spec.get("required_authors") or []) if str(item).strip()]
        title_phrases = [str(item).strip().lower() for item in (lookup_spec.get("title_phrases") or []) if str(item).strip()]
        preferred_year = lookup_spec.get("preferred_year")
        title = str(getattr(result, "title", "") or "").lower()
        abstract = str(getattr(result, "abstract", "") or "").lower()
        authors = " ".join(getattr(result, "authors", []) or []).lower()
        year = getattr(result, "year", None)
        score = 0
        exact_phrase_hits = 0
        author_hits = sum(1 for author in required_authors if re.search(rf"\b{re.escape(author)}\b", authors))
        title_hits = sum(1 for phrase in title_phrases if phrase and phrase in title)
        abstract_title_hits = sum(1 for phrase in title_phrases if phrase and phrase in abstract)
        year_match = int(preferred_year is not None and year is not None and str(year) == str(preferred_year))
        for keyword in keywords:
            token = keyword.strip().lower()
            if not token:
                continue
            if re.fullmatch(r"(19|20)\d{2}", token):
                if year is not None and str(year) == token:
                    score += 7
                    exact_phrase_hits += 1
                continue
            if " " in token:
                if token in title:
                    score += 9
                    exact_phrase_hits += 1
                elif token in abstract:
                    score += 3
            else:
                if re.search(rf"\b{re.escape(token)}\b", authors):
                    score += 10
                    exact_phrase_hits += 1
                elif re.search(rf"\b{re.escape(token)}\b", title):
                    score += 6
                    exact_phrase_hits += 1
                elif re.search(rf"\b{re.escape(token)}\b", abstract):
                    score += 2
        all_required_authors = int(bool(required_authors) and author_hits == len(required_authors))
        return (
            all_required_authors,
            author_hits,
            year_match,
            title_hits,
            exact_phrase_hits + abstract_title_hits,
            score,
        )

    @staticmethod
    def _normalize_lookup_spec(topic_hint: str, lookup_spec: dict[str, object]) -> dict[str, object]:
        query_string = str(lookup_spec.get("query_string") or topic_hint or "").strip()
        query_terms = [
            str(item).strip()
            for item in (lookup_spec.get("query_terms") or [])
            if str(item).strip()
        ]
        if not query_terms:
            query_terms = [part.strip() for part in query_string.split(",") if part.strip()]
        required_authors = [
            str(item).strip()
            for item in (lookup_spec.get("required_authors") or [])
            if str(item).strip()
        ]
        title_phrases = [
            str(item).strip()
            for item in (lookup_spec.get("title_phrases") or [])
            if str(item).strip()
        ]
        preferred_year = lookup_spec.get("preferred_year")
        year_min = lookup_spec.get("year_min")
        year_max = lookup_spec.get("year_max")
        try:
            preferred_year = int(preferred_year) if preferred_year is not None else None
        except (TypeError, ValueError):
            preferred_year = None
        try:
            year_min = int(year_min) if year_min is not None else None
        except (TypeError, ValueError):
            year_min = None
        try:
            year_max = int(year_max) if year_max is not None else None
        except (TypeError, ValueError):
            year_max = None
        return {
            "query_string": query_string,
            "query_terms": query_terms,
            "required_authors": required_authors,
            "preferred_year": preferred_year,
            "year_min": year_min,
            "year_max": year_max,
            "title_phrases": title_phrases,
        }

    @staticmethod
    def _dedupe_results(results: list[object]) -> list[object]:
        deduped: list[object] = []
        seen: set[str] = set()
        for result in results:
            key = "|".join(
                [
                    str(getattr(result, "arxiv_id", "") or ""),
                    str(getattr(result, "doi", "") or ""),
                    str(getattr(result, "title", "") or "").strip().lower(),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    @staticmethod
    def _apply_hard_constraints(results: list[object], lookup_spec: dict[str, object]) -> list[object]:
        required_authors = [str(item).strip().lower() for item in (lookup_spec.get("required_authors") or []) if str(item).strip()]
        if required_authors:
            author_filtered = [
                result
                for result in results
                if all(
                    re.search(
                        rf"\b{re.escape(author)}\b",
                        " ".join(getattr(result, "authors", []) or []).lower(),
                    )
                    for author in required_authors
                )
            ]
            if author_filtered:
                results = author_filtered
            else:
                return []

        preferred_year = lookup_spec.get("preferred_year")
        if preferred_year is not None:
            year_filtered = [
                result for result in results
                if getattr(result, "year", None) is not None and str(getattr(result, "year", None)) == str(preferred_year)
            ]
            if year_filtered:
                results = year_filtered
        else:
            year_min = lookup_spec.get("year_min")
            year_max = lookup_spec.get("year_max")
            if year_min is not None or year_max is not None:
                range_filtered = []
                for result in results:
                    year = getattr(result, "year", None)
                    if year is None:
                        continue
                    if year_min is not None and int(year) < int(year_min):
                        continue
                    if year_max is not None and int(year) > int(year_max):
                        continue
                    range_filtered.append(result)
                if range_filtered:
                    results = range_filtered

        title_phrases = [str(item).strip().lower() for item in (lookup_spec.get("title_phrases") or []) if str(item).strip()]
        if title_phrases:
            title_filtered = [
                result
                for result in results
                if any(
                    phrase in str(getattr(result, "title", "") or "").lower()
                    for phrase in title_phrases
                )
            ]
            if title_filtered:
                results = title_filtered

        return results

    def fetch_full_text(self, arxiv_id: str, fallback_abstract: str, paper_url: str = "") -> str:
        """Fetch ar5iv HTML text, then paper URL text, or fallback to the abstract."""
        if not arxiv_id:
            text = self._fetch_paper_url_text(paper_url)
            return text or fallback_abstract

        ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        try:
            text = self._fetch_html_text(ar5iv_url)
            if text:
                log.info("Successfully fetched ar5iv full text for %s", arxiv_id)
                return text
        except Exception as e:
            log.warning("Failed to fetch ar5iv text for %s: %s", arxiv_id, e)

        text = self._fetch_paper_url_text(paper_url)
        if text:
            return text
        return fallback_abstract

    def _fetch_paper_url_text(self, paper_url: str) -> str:
        url = str(paper_url or "").strip()
        if not url:
            return ""
        try:
            text = self._fetch_html_text(url)
            if text:
                log.info("Fetched paper text from URL fallback: %s", url)
                return text
        except Exception as exc:
            log.warning("Failed to fetch paper text from URL %s: %s", url, exc)
        return ""

    @staticmethod
    def _fetch_html_text(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchPlatform/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read()
        html = raw.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text if len(text) > 1000 else ""

    def extract_answers(
        self,
        text: str,
        questions: list[str],
        llm: SyncLLMBackend,
        *,
        guidance: str = "",
    ) -> dict[str, str]:
        """
        Return a structured answer map for a fixed list of questions.

        Why JSON:
        - the caller asks multiple questions in one LLM request,
        - the caller needs stable question-to-answer matching,
        - later normalization logic expects a dict keyed by the original
          question text.

        Expected JSON shape:
        {
          "<question 1>": "<answer 1>",
          "<question 2>": "<answer 2>"
        }

        The exact question strings are used as the keys so the caller can
        safely look up answers without relying on positional parsing.
        """
        qs_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        shape_example = "{\n" + ",\n".join(
            f'  {json.dumps(question)}: "..."' for question in questions
        ) + "\n}"
        guidance_text = guidance.strip()
        guidance_block = (
            "PI guidance that must take precedence when interpreting ambiguous sections:\n"
            f"{guidance_text}\n\n"
            if guidance_text
            else ""
        )
        
        system_prompt = (
            "You are an expert academic reader. "
            "Your task is to answer the following questions based ONLY on the provided text.\n"
            "Return your answers as a raw JSON object with NO markdown formatting, NO markdown codeblocks, and NO backticks.\n"
            "The JSON must have the exact string of the question as the key, and your answer as the value.\n"
            f"{guidance_block}"
            "Expected JSON shape example:\n"
            f"{shape_example}\n"
            "If a question cannot be answered from the text, reply with 'NOT_FOUND'.\n\n"
            f"Questions to answer:\n{qs_formatted}"
        )
        
        # truncate text if it's too huge
        truncated_text = text[:10000]
        
        response = llm.complete(
            system=system_prompt,
            messages=[{"role": "user", "content": f"Text:\n\n{truncated_text}"}],
            temperature=0.0
        )
        
        try:
            # Clean up the response in case it still added markdown
            clean_resp = response.strip()
            if clean_resp.startswith("```json"):
                clean_resp = clean_resp[7:]
            elif clean_resp.startswith("```"):
                clean_resp = clean_resp[3:]
            if clean_resp.endswith("```"):
                clean_resp = clean_resp[:-3]
            clean_resp = clean_resp.strip()
            
            data = json.loads(clean_resp)
            return data
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse answers JSON: {e}\nResponse was:\n{response}")
            return {}
