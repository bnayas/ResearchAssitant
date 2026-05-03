"""
literature_review/search_backends/arxiv_backend.py
───────────────────────────────────────────────────
ArXiv backend using the public export API (no key required).

Endpoint: http://export.arxiv.org/api/query
Returns:  Atom XML, parsed into RawPaper list.

Category filter is applied at query time using cat: syntax.
Year filter is applied post-parse (ArXiv API does not support date range
filtering natively in free-text queries).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional

from .base import RawPaper, SearchBackend

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

# XML namespace map for Atom feed
NS = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

class ArXivBackend(SearchBackend):
    name = "arxiv"
    _cooldown_until: float = 0.0
    _last_skip_log_at: float = 0.0
    _next_request_at: float = 0.0
    _schedule_lock = threading.Lock()

    def __init__(self, sort_by: str = "relevance"):
        """
        sort_by: "relevance" | "lastUpdatedDate" | "submittedDate"
        """
        self.sort_by = sort_by
        self._last_rate_limited = False
        self._min_interval_seconds = 3.1

    async def search(
        self,
        keywords: list[str],
        max_results: int,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        categories: Optional[list[str]] = None,
        authors: Optional[list[str]] = None,
    ) -> list[RawPaper]:
        self._last_rate_limited = False
        if self._in_cooldown():
            self._last_rate_limited = True
            self._log_cooldown_skip()
            return []
        # Try strict AND query first, fall back to OR if it returns nothing
        for query in [
            self._build_query(keywords, categories, authors),              # strict: all AND
            self._build_or_query(keywords, categories, authors),          # loose:  any OR
        ]:
            result = await self._fetch_with_retry(query, max_results)
            if self._last_rate_limited:
                return []
            papers = self._parse_atom(result, year_min, year_max)
            if papers:
                return papers
        return []

    async def _fetch_with_retry(self, query: str, max_results: int) -> str:
        """Fetch ArXiv query string; stop quickly on 429 and enter cooldown."""
        params = {"search_query": query, "max_results": max_results, "sortBy": self.sort_by}

        # --- try aiohttp with 2 retries ---
        try:
            import aiohttp
            for attempt in range(2):
                try:
                    await self._respect_pacing()
                    log.info("ArXiv request URL: %s, params: %s", ARXIV_API, params)
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=25),
                        headers={"User-Agent": "ResearchAssistant/1.0 (mailto:admin@example.com)"}
                    ) as session:
                        async with session.get(ARXIV_API, params=params, allow_redirects=True) as resp:
                            if resp.status == 429:
                                self._mark_rate_limited(resp.headers.get("Retry-After"))
                                return ""
                            if resp.status >= 400:
                                text = await resp.text()
                                log.warning("ArXiv HTTP %d error: %s", resp.status, text)
                                resp.raise_for_status()
                            return await resp.text()
                except Exception as exc:
                    log.warning("ArXiv aiohttp attempt %d failed: %r", attempt + 1, exc, exc_info=True)
                    if attempt < 1:
                        await asyncio.sleep(3)
        except ImportError:
            pass

        # --- urllib fallback (no external dep) ---
        import urllib.request, urllib.parse
        url = ARXIV_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchAssistant/1.0 (mailto:admin@example.com)"}
        )
        try:
            await self._respect_pacing()
            with urllib.request.urlopen(req, timeout=25) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if getattr(exc, "code", None) == 429:
                self._mark_rate_limited(exc.headers.get("Retry-After") if exc.headers else None)
                return ""
            log.warning("ArXiv urllib fallback failed: %s", exc)
            return ""
        except Exception as exc:
            log.warning("ArXiv urllib fallback failed: %s", exc)
            return ""

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _build_query(
        self, keywords: list[str], categories: Optional[list[str]], authors: Optional[list[str]] = None
    ) -> str:
        """
        Strict AND query: all terms must appear.
        Example: all:"time-averaged neutral" AND all:"environmental stochasticity"
        """
        kw_parts = [f'all:"{kw}"' for kw in keywords]
        if authors:
            kw_parts.extend([f'au:"{author}"' for author in authors])
        query = " AND ".join(kw_parts)
        if categories:
            cat_part = " OR ".join(f"cat:{c}" for c in categories)
            query = f"({query}) AND ({cat_part})"
        return query

    def _build_or_query(
        self, keywords: list[str], categories: Optional[list[str]], authors: Optional[list[str]] = None
    ) -> str:
        """
        Loose OR query: any term matches.
        Used as fallback when strict AND returns nothing.
        """
        kw_parts = [f'all:"{kw}"' for kw in keywords]
        query = " OR ".join(kw_parts)
        if authors:
            # We still want to strictly require the authors even in an OR query fallback for keywords
            author_part = " AND ".join([f'au:"{author}"' for author in authors])
            query = f"({query}) AND ({author_part})"
        if categories:
            cat_part = " OR ".join(f"cat:{c}" for c in categories)
            query = f"({query}) AND ({cat_part})"
        return query

    def _parse_atom(
        self,
        xml_text: str,
        year_min: Optional[int],
        year_max: Optional[int],
    ) -> list[RawPaper]:
        if not xml_text.strip():
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            log.warning("ArXiv XML parse error: %s", exc)
            return []

        papers: list[RawPaper] = []
        for entry in root.findall("atom:entry", NS):
            title = (
                entry.findtext("atom:title", "", NS) or ""
            ).strip().replace("\n", " ")
            abstract = (entry.findtext("atom:summary", "", NS) or "").strip()
            id_url = entry.findtext("atom:id", "", NS) or ""

            # Canonical URL: replace /abs/ link with itself (it's already canonical)
            arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else None

            published = entry.findtext("atom:published", "", NS) or ""
            year: Optional[int] = None
            if len(published) >= 4:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass

            # Year gate (deterministic, avoids wasting scope-validator LLM calls)
            if year_min and year is not None and year < year_min:
                continue
            if year_max and year is not None and year > year_max:
                continue

            authors = [
                (a.findtext("atom:name", "", NS) or "").strip()
                for a in entry.findall("atom:author", NS)
            ]

            # Prefer PDF link; fall back to abs link
            url = id_url
            for link in entry.findall("atom:link", NS):
                if link.attrib.get("type") == "text/html":
                    url = link.attrib.get("href", url)
                    break

            papers.append(RawPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                year=year,
                source=self.name,
                arxiv_id=arxiv_id,
            ))

        return papers

    def _mark_rate_limited(self, retry_after: Optional[str]) -> None:
        self._last_rate_limited = True
        wait_seconds = self._parse_retry_after(retry_after) or (20 * 60)
        until = time.time() + max(wait_seconds, 10 * 60)
        cls = type(self)
        cls._cooldown_until = max(cls._cooldown_until, until)
        log.warning("ArXiv rate-limited; entering cooldown for %ds", int(max(cls._cooldown_until - time.time(), 0)))

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
        log.warning("ArXiv cooldown active; skipping request for %ds", remaining)

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
