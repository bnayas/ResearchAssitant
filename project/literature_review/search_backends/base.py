"""
literature_review/search_backends/base.py
─────────────────────────────────────────
Abstract base and RawPaper DTO.

All backends return list[RawPaper].  Scope checking happens upstream in the
reviewer so that backends stay thin and composable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class RawPaper:
    """
    Unvalidated paper as returned directly from an external API.
    Fields may be empty strings / None if the API did not supply them.
    """
    title: str
    authors: list[str]
    abstract: str
    url: str
    year: Optional[int]
    source: str                       # Set by the backend; matches SearchBackend.name
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

    def is_usable(self) -> bool:
        """Minimum bar: must have a title and something to evaluate scope on."""
        return bool(self.title.strip()) and (
            bool(self.abstract.strip()) or bool(self.url.strip())
        )


class SearchBackend(ABC):
    """
    Minimal interface every search backend must implement.

    Implementations should be stateless and async-safe.
    Network errors should be caught internally and return [].
    """
    name: str  # Must be set by subclass; used for provenance logging

    @abstractmethod
    async def search(
        self,
        keywords: list[str],
        max_results: int,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        categories: Optional[list[str]] = None,
    ) -> list[RawPaper]:
        """
        Parameters
        ----------
        keywords    : Terms to AND together in the query
        max_results : Backend-level ceiling; reviewer may apply an additional one
        year_min    : Inclusive lower year bound (backend applies if supported)
        year_max    : Inclusive upper year bound (backend applies if supported)
        categories  : Domain/category filter (e.g. ArXiv categories); ignored by
                      backends that don't support it
        """
        ...
