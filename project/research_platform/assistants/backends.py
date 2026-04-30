from __future__ import annotations

import os
from typing import Any, Optional

from literature_review.director_bridge import LiteratureReviewBridge
from literature_review.llm_interface import ThreadedAsyncAdapter
from literature_review.search_backends.arxiv_backend import ArXivBackend
from literature_review.search_backends.base import SearchBackend
from literature_review.search_backends.perplexity_backend import PerplexityBackend
from literature_review.search_backends.semantic_scholar_backend import SemanticScholarBackend
import sim_tool.llm as sim_llm


def make_service_backend(service: str, config: Optional[Any]) -> Any:
    if config is not None and not getattr(config, "enabled", True):
        raise ValueError(f"{service} backend is disabled")
    provider = (getattr(config, "provider", None) or "").strip().lower() if config else ""
    if provider == "mock":
        from sim_tool.llm_mock import MockBackend

        return MockBackend()

    kwargs: dict[str, Any] = {"service": service}
    if config is not None:
        for field_name in ("provider", "model", "base_url", "api_key", "timeout_seconds"):
            value = getattr(config, field_name, None)
            if value is not None:
                kwargs[field_name] = value
    return sim_llm.make_backend(**kwargs)


def build_literature_review_bridge_from_env(
    *,
    llm_sync_backend: Any,
    gui_event_sink: Optional[Any] = None,
) -> LiteratureReviewBridge:
    perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
    semantic_scholar_api_key = os.environ.get("SS_API_KEY")

    enable_arxiv = _parse_bool(
        os.environ.get("LITERATURE_REVIEW_ENABLE_ARXIV"),
        default=True,
    )
    enable_semantic = _parse_bool(
        os.environ.get("LITERATURE_REVIEW_ENABLE_SEMANTIC_SCHOLAR"),
        default=True,
    )
    enable_perplexity = _parse_bool(
        os.environ.get("LITERATURE_REVIEW_ENABLE_PERPLEXITY"),
        default=bool(perplexity_api_key),
    )

    backends: list[SearchBackend] = []
    if enable_arxiv:
        backends.append(
            ArXivBackend(
                sort_by=os.environ.get("LITERATURE_REVIEW_ARXIV_SORT_BY", "relevance")
            )
        )
    if enable_semantic:
        backends.append(SemanticScholarBackend(api_key=semantic_scholar_api_key))
    if enable_perplexity and perplexity_api_key:
        backends.append(
            PerplexityBackend(
                api_key=perplexity_api_key,
                model=os.environ.get(
                    "LITERATURE_REVIEW_PERPLEXITY_MODEL",
                    "llama-3.1-sonar-large-128k-online",
                ),
            )
        )

    if not backends:
        raise ValueError("No literature review search backends enabled")

    return LiteratureReviewBridge(
        backends=backends,
        llm=ThreadedAsyncAdapter(llm_sync_backend),
        gui_event_sink=gui_event_sink,
    )


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
