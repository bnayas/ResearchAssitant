from __future__ import annotations

from unittest.mock import patch

import pytest

from literature_review.director_bridge import build_bridge_from_env


class DummySyncLLM:
    def complete(self, system: str, messages: list, max_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        return "ok"


def _backend_names(bridge) -> list[str]:
    return [backend.name for backend in bridge.backends]


def test_build_bridge_from_env_defaults(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("LITERATURE_REVIEW_ENABLE_ARXIV", raising=False)
    monkeypatch.delenv("LITERATURE_REVIEW_ENABLE_SEMANTIC_SCHOLAR", raising=False)
    monkeypatch.delenv("LITERATURE_REVIEW_ENABLE_PERPLEXITY", raising=False)

    bridge = build_bridge_from_env(llm_sync_backend=DummySyncLLM())

    assert _backend_names(bridge) == ["arxiv", "semantic_scholar"]


def test_build_bridge_from_env_enables_perplexity_when_key_present(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    monkeypatch.delenv("LITERATURE_REVIEW_ENABLE_PERPLEXITY", raising=False)

    bridge = build_bridge_from_env(llm_sync_backend=DummySyncLLM())

    assert _backend_names(bridge) == ["arxiv", "semantic_scholar", "perplexity"]


def test_build_bridge_from_env_can_disable_all_backends(monkeypatch):
    monkeypatch.setenv("LITERATURE_REVIEW_ENABLE_ARXIV", "0")
    monkeypatch.setenv("LITERATURE_REVIEW_ENABLE_SEMANTIC_SCHOLAR", "0")
    monkeypatch.setenv("LITERATURE_REVIEW_ENABLE_PERPLEXITY", "0")

    with pytest.raises(ValueError, match="No literature review search backends enabled"):
        build_bridge_from_env(llm_sync_backend=DummySyncLLM())


def test_build_bridge_from_env_uses_literature_review_service(monkeypatch):
    monkeypatch.setenv("LITERATURE_REVIEW_ENABLE_PERPLEXITY", "0")
    dummy = DummySyncLLM()

    with patch("sim_tool.llm.make_backend", return_value=dummy) as mocked:
        bridge = build_bridge_from_env()

    mocked.assert_called_once_with(service="literature_review")
    assert _backend_names(bridge) == ["arxiv", "semantic_scholar"]
