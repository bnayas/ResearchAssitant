"""
literature_review/search_backends/perplexity_backend.py
────────────────────────────────────────────────────────
Perplexity API backend using a search-enabled chat model.

Why Perplexity here (vs just ArXiv + Semantic Scholar):
  - Catches papers not on ArXiv (conference-only, pre-prints on other repos)
  - Handles queries that need natural-language reformulation
  - Acts as a complementary signal, not a replacement for structured APIs

Model: llama-3.1-sonar-large-128k-online (default)
       llama-3.1-sonar-small-128k-online (faster, cheaper, good for varied queries)

The model is prompted to return *only* a JSON array of papers.
We parse the JSON, then hand off to the common scope validator.

Note: Perplexity may hallucinate papers — the scope validator + auditor
downstream catch this via empty abstracts and URL validation.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .base import RawPaper, SearchBackend

log = logging.getLogger(__name__)

PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"

_SYSTEM = """\
You are a scientific literature search assistant.
Given a research query, return a JSON array of real, published academic papers.
Each object must have exactly these keys:
  title    (string)
  authors  (array of strings)
  abstract (string, at least 1 sentence)
  url      (string, direct paper link)
  year     (integer or null)

Return ONLY the JSON array. No markdown, no preamble, no explanation.\
"""

class PerplexityBackend(SearchBackend):
    name = "perplexity"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-sonar-large-128k-online",
    ):
        key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not key:
            raise ValueError(
                "PerplexityBackend requires an API key via api_key= or PERPLEXITY_API_KEY env var"
            )
        self.api_key = key
        self.model = model

    async def search(
        self,
        keywords: list[str],
        max_results: int,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        categories: Optional[list[str]] = None,   # Folded into prompt
    ) -> list[RawPaper]:
        try:
            import aiohttp
        except ImportError:
            log.warning("PerplexityBackend requires aiohttp for live search requests")
            return []

        user_msg = self._build_prompt(keywords, max_results, year_min, year_max, categories)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
                async with session.post(
                    PERPLEXITY_API, json=payload, headers=headers
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("Perplexity %s: %s", resp.status, body[:200])
                        return []
                    data = await resp.json()
        except Exception as exc:
            log.warning("Perplexity request failed: %s", exc)
            return []

        raw_text = data["choices"][0]["message"]["content"]
        return self._parse(raw_text, year_min, year_max)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _build_prompt(
        self,
        keywords: list[str],
        max_results: int,
        year_min: Optional[int],
        year_max: Optional[int],
        categories: Optional[list[str]],
    ) -> str:
        parts = [f"Find up to {max_results} academic papers about: {', '.join(keywords)}."]
        if year_min or year_max:
            lo = year_min or "any year"
            hi = year_max or "present"
            parts.append(f"Published between {lo} and {hi}.")
        if categories:
            parts.append(f"Prefer papers in fields: {', '.join(categories)}.")
        parts.append("Return a JSON array as instructed.")
        return " ".join(parts)

    def _parse(
        self,
        text: str,
        year_min: Optional[int],
        year_max: Optional[int],
    ) -> list[RawPaper]:
        # Strip markdown code fences Perplexity sometimes adds
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

        # Find the first [...] block in case there's surrounding text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            items = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("Perplexity JSON parse failed (%s); raw=%r", exc, text[:200])
            return []

        if not isinstance(items, list):
            return []

        papers: list[RawPaper] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            year: Optional[int] = None
            try:
                raw_year = item.get("year")
                if raw_year is not None:
                    year = int(raw_year)
            except (ValueError, TypeError):
                pass

            # Local year gate (saves scope-validator LLM calls)
            if year_min and year is not None and year < year_min:
                continue
            if year_max and year is not None and year > year_max:
                continue

            authors = item.get("authors") or []
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",")]

            papers.append(RawPaper(
                title=(item.get("title") or "").strip(),
                authors=authors,
                abstract=(item.get("abstract") or "").strip(),
                url=(item.get("url") or "").strip(),
                year=year,
                source=self.name,
            ))

        return papers
