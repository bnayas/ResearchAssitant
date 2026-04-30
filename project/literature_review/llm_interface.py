"""
literature_review/llm_interface.py
───────────────────────────────────
Async LLM interface used by the literature review pipeline.

The existing framework's LLMBackend has a synchronous `complete()`.
This module provides:
  1. AsyncLLMBackend – ABC for natively async backends
  2. ThreadedAsyncAdapter – wraps any sync backend in asyncio.to_thread()

The reviewer, keyword extractor, and scope validator all depend only on
AsyncLLMBackend, so the same code works with Anthropic, OpenAI-compatible
local models, or the mock backend.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class AsyncLLMBackend(ABC):
    """Minimal async interface the literature review pipeline depends on."""

    @abstractmethod
    async def complete_async(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Returns the assistant text response as a plain string.
        Raises on API errors (caller decides how to handle).
        """
        ...


class ThreadedAsyncAdapter(AsyncLLMBackend):
    """
    Wraps a synchronous LLMBackend (the existing framework style) so it can
    be used in async contexts via asyncio.to_thread.

    Usage:
        from llm import make_backend
        llm = ThreadedAsyncAdapter(make_backend(service="literature_review"))
    """

    def __init__(self, sync_backend: Any):
        """
        sync_backend must have a `complete(system, messages, max_tokens, temperature) -> str`
        method matching the existing framework contract.
        """
        self._backend = sync_backend

    async def complete_async(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        return await asyncio.to_thread(
            self._backend.complete,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class DirectAsyncBackend(AsyncLLMBackend):
    """
    For backends that already support async natively (e.g., Anthropic SDK v3+).
    Subclass this and implement complete_async directly.
    """

    async def complete_async(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError
