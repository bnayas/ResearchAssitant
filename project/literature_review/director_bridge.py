"""
literature_review/director_bridge.py
──────────────────────────────────────
Demonstrates how the Director integrates the LiteratureReviewer.

This module is the seam between the existing orchestration layer and the
literature review agent. It handles:
  1. Building LiteratureReviewTask from a Director task packet
  2. Setting up the streaming queue that fans events to the GUI
  3. Running the reviewer in an asyncio context from synchronous Director code
  4. Running the LiteratureAuditor after the artifact is received
  5. Returning the full result (artifact + audit) back to the Director

The Director retains orchestration authority but delegates:
  - content search & filtering → LiteratureReviewer
  - scope re-validation        → LiteratureAuditor
  - streaming to GUI           → _StreamBridge (queue-based)

Usage from synchronous Director code:
    bridge = LiteratureReviewBridge(backends, llm, gui_queue)
    result = bridge.run_sync(task_packet)
    # result.artifact  → store in Curator
    # result.audit     → check result.audit.passed
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .auditor import LiteratureAuditor
from .contract import (
    LiteratureAuditResult,
    LiteratureReviewArtifact,
    LiteratureReviewTask,
    StreamEvent,
)
from .llm_interface import AsyncLLMBackend
from .reviewer import LiteratureReviewer
from .search_backends.base import SearchBackend

log = logging.getLogger(__name__)


@dataclass
class LiteratureReviewResult:
    """Everything the Director needs after a review completes."""
    artifact: LiteratureReviewArtifact
    audit: LiteratureAuditResult


class LiteratureReviewBridge:
    """
    Thin orchestration adapter between the synchronous Director and the
    async LiteratureReviewer.

    Parameters
    ----------
    backends        : Search backend instances to use
    llm             : AsyncLLMBackend (wrap sync backends with ThreadedAsyncAdapter)
    gui_event_sink  : Callable or queue.Queue that receives StreamEvent objects.
                      If a Queue, events are put() on it from the reviewer thread.
                      If a Callable, it is called directly (must be thread-safe).
    """

    def __init__(
        self,
        backends: list[SearchBackend],
        llm: AsyncLLMBackend,
        gui_event_sink: Optional[Any] = None,  # queue.Queue | Callable | None
    ):
        self.backends = backends
        self.llm = llm
        self._sink = gui_event_sink
        self._auditor = LiteratureAuditor()

    # ──────────────────────────────────────────────────────────────────────────
    # Synchronous entry point (for Director's threaded model)
    # ──────────────────────────────────────────────────────────────────────────

    def run_sync(self, task: LiteratureReviewTask) -> LiteratureReviewResult:
        """
        Run the full review pipeline synchronously.
        Internally runs an asyncio event loop in the current thread.
        Safe to call from Director's background thread.
        """
        return asyncio.run(self.run_async(task))

    # ──────────────────────────────────────────────────────────────────────────
    # Async entry point (for Director's async model)
    # ──────────────────────────────────────────────────────────────────────────

    async def run_async(self, task: LiteratureReviewTask) -> LiteratureReviewResult:
        """Async version — use this if the Director already runs an event loop."""
        reviewer = LiteratureReviewer(
            backends=self.backends,
            llm=self.llm,
            stream_callback=self._make_stream_callback(),
        )

        log.info("[Director] Starting literature review task_id=%s", task.task_id)
        artifact = await reviewer.run(task)

        log.info("[Director] Running audit for task_id=%s", task.task_id)
        audit = self._auditor.audit(artifact, task.scope)

        if not audit.passed:
            log.warning(
                "[Director] Audit failed for task_id=%s. violations=%s",
                task.task_id, audit.scope_violations_found,
            )
        else:
            log.info("[Director] Audit passed for task_id=%s", task.task_id)

        return LiteratureReviewResult(artifact=artifact, audit=audit)

    # ──────────────────────────────────────────────────────────────────────────
    # Stream routing
    # ──────────────────────────────────────────────────────────────────────────

    def _make_stream_callback(self) -> Callable[[StreamEvent], None]:
        """
        Returns a thread-safe callback that routes StreamEvents to the GUI sink.

        The callback is called from the reviewer's async context. If the sink is
        a queue, we put() onto it. If it's a callable, we call it directly.
        If no sink is configured, events are only logged at DEBUG level.
        """
        sink = self._sink

        if sink is None:
            def _noop(event: StreamEvent) -> None:
                log.debug("[Stream] %s %s", event.event_type, event.payload)
            return _noop

        if isinstance(sink, queue.Queue):
            def _queue_emit(event: StreamEvent) -> None:
                log.debug("[Stream→GUI] %s", event.event_type)
                try:
                    sink.put_nowait(event)
                except queue.Full:
                    log.warning("GUI event queue full; dropping %s event", event.event_type)
            return _queue_emit

        # Assume callable (websocket sender, SSE emitter, etc.)
        def _callable_emit(event: StreamEvent) -> None:
            log.debug("[Stream→GUI] %s", event.event_type)
            try:
                sink(event)
            except Exception as exc:
                log.warning("GUI sink raised on %s: %s", event.event_type, exc)

        return _callable_emit


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def build_default_bridge(
    perplexity_api_key: Optional[str] = None,
    semantic_scholar_api_key: Optional[str] = None,
    llm_sync_backend: Optional[Any] = None,       # Existing framework LLMBackend
    gui_event_sink: Optional[Any] = None,
) -> LiteratureReviewBridge:
    """
    Build a bridge with all three backends (ArXiv + Semantic Scholar + Perplexity).
    Omits PerplexityBackend if no API key is provided.

    llm_sync_backend: pass your existing AnthropicBackend / OpenAICompatibleBackend.
                      It will be wrapped in ThreadedAsyncAdapter automatically.
    """
    from .llm_interface import ThreadedAsyncAdapter
    from .search_backends.arxiv_backend import ArXivBackend
    from .search_backends.perplexity_backend import PerplexityBackend
    from .search_backends.semantic_scholar_backend import SemanticScholarBackend

    backends: list[SearchBackend] = [
        ArXivBackend(),
        SemanticScholarBackend(api_key=semantic_scholar_api_key),
    ]
    if perplexity_api_key:
        backends.append(PerplexityBackend(api_key=perplexity_api_key))

    if llm_sync_backend is None:
        raise ValueError(
            "llm_sync_backend is required "
            "(pass a configured sync backend, e.g. LM Studio, Anthropic, or OpenAI-compatible)"
        )

    async_llm = ThreadedAsyncAdapter(llm_sync_backend)

    return LiteratureReviewBridge(
        backends=backends,
        llm=async_llm,
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


def build_bridge_from_env(
    llm_sync_backend: Optional[Any] = None,
    gui_event_sink: Optional[Any] = None,
) -> LiteratureReviewBridge:
    from research_platform.wiring import build_literature_bridge_from_env

    return build_literature_bridge_from_env(
        llm_sync_backend=llm_sync_backend,
        gui_event_sink=gui_event_sink,
    )
