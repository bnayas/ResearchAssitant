"""
literature_review/search_backends/semantic_scholar_backend.py
─────────────────────────────────────────────────────────────
Semantic Scholar Academic Graph API backend.

Endpoint: https://api.semanticscholar.org/graph/v1/paper/search
Docs:     https://api.semanticscholar.org/api-docs/

Rate limits (unauthenticated): shared pool, treat as roughly 100 req/5min
Rate limits (with API key):    about 1 req/s   (set SS_API_KEY in env or pass directly)

The API natively supports year range filtering via the `year` param ("2018-2024").
Returns metadata including ArXiv ID and DOI for cross-referencing dedup.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

from .base import RawPaper, SearchBackend

log = logging.getLogger(__name__)

SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# All fields we want from the API
_FIELDS = ",".join([
    "title",
    "abstract",
    "authors",
    "year",
    "externalIds",
    "url",
    "openAccessPdf",
])

class SemanticScholarBackend(SearchBackend):
    name = "semantic_scholar"
    _cooldown_until: float = 0.0
    _last_skip_log_at: float = 0.0
    _next_request_at: float = 0.0
    _schedule_lock = threading.Lock()

    def __init__(self, api_key: Optional[str] = None):
        """
        api_key: Optional Semantic Scholar API key.
                 Falls back to SS_API_KEY env var if not given.
        """
        key = api_key or os.getenv("SS_API_KEY")
        self._headers = {
            "User-Agent": "ResearchAssistant/1.0 (mailto:admin@example.com)"
        }
        if key:
            self._headers["x-api-key"] = key
        self._last_rate_limited = False
        # Be conservative in the unauthenticated shared pool and stay below the
        # nominal 100-requests-per-5-minutes ceiling.
        self._min_interval_seconds = 3.2 if not key else 1.2

    async def search(
        self,
        keywords: list[str],
        max_results: int,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        categories: Optional[list[str]] = None,   # Not supported; ignored
        authors: Optional[list[str]] = None,
    ) -> list[RawPaper]:
        self._last_rate_limited = False
        if self._in_cooldown():
            self._last_rate_limited = True
            self._log_cooldown_skip()
            return []
        try:
            import aiohttp
        except ImportError:
            log.warning("SemanticScholarBackend requires aiohttp for live search requests")
            return []

        query_parts = list(keywords)
        if authors:
            query_parts.extend(authors)
        query = " ".join(query_parts)
        params: dict = {
            "query": query,
            "fields": _FIELDS,
            "limit": min(max_results, 100),        # API cap is 100
        }
        if year_min or year_max:
            lo = year_min or 1900
            hi = year_max or 2100
            params["year"] = f"{lo}-{hi}"

        try:
            await self._respect_pacing()
            async with aiohttp.ClientSession(
                headers=self._headers, timeout=aiohttp.ClientTimeout(total=20)
            ) as session:
                async with session.get(SS_API, params=params, allow_redirects=True) as resp:
                    if resp.status == 429:
                        self._mark_rate_limited(resp.headers.get("Retry-After"))
                        return []
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as exc:
            log.warning("SemanticScholar request failed: %s", exc)
            return []

        return self._parse(data)

    def _mark_rate_limited(self, retry_after: Optional[str]) -> None:
        self._last_rate_limited = True
        wait_seconds = self._parse_retry_after(retry_after) or (5 * 60)
        until = time.time() + max(wait_seconds, 2 * 60)
        cls = type(self)
        cls._cooldown_until = max(cls._cooldown_until, until)
        log.warning(
            "SemanticScholar rate-limited; entering cooldown for %ds",
            int(max(cls._cooldown_until - time.time(), 0)),
        )

    @classmethod
    def _in_cooldown(cls) -> bool:
        return time.time() < cls._cooldown_until

    @classmethod
    def _log_cooldown_skip(cls) -> None:
        now = time.time()
        if now - cls._last_skip_log_at < 5:
            return
        cls._last_skip_log_at = now
        remaining = int(max(cls._cooldown_until - now, 0))
        log.warning("SemanticScholar cooldown active; skipping request for %ds", remaining)

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        try:
            return int(str(value).strip())
        except ValueError:
            return None

    async def _respect_pacing(self) -> None:
        wait_seconds = self._reserve_request_slot()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

    def _reserve_request_slot(self) -> float:
        cls = type(self)
        with cls._schedule_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, cls._next_request_at - now)
            base = max(now, cls._next_request_at)
            cls._next_request_at = base + self._min_interval_seconds
            return wait_seconds

    def _parse(self, data: dict) -> list[RawPaper]:
        papers: list[RawPaper] = []
        for item in data.get("data", []):
            eids = item.get("externalIds") or {}
            arxiv_id = eids.get("ArXiv")
            doi = eids.get("DOI")

            # Prefer open-access PDF link; fall back to SS canonical URL
            oa = item.get("openAccessPdf") or {}
            url = oa.get("url") or item.get("url") or (
                f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"
            )

            abstract = item.get("abstract") or ""
            year_raw = item.get("year")
            year: Optional[int] = int(year_raw) if year_raw is not None else None

            authors = [
                a.get("name", "") for a in (item.get("authors") or [])
            ]

            papers.append(RawPaper(
                title=item.get("title") or "",
                authors=authors,
                abstract=abstract,
                url=url,
                year=year,
                source=self.name,
                arxiv_id=arxiv_id,
                doi=doi,
            ))
        return papers
