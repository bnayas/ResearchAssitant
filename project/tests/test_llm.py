"""
test_llm.py
───────────
Unit tests for per-service LLM backend resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sim_tool.llm import (
    AnthropicBackend,
    OpenAICompatibleBackend,
    make_backend,
    resolve_service_config,
)


def test_default_backend_prefers_lmstudio(monkeypatch):
    monkeypatch.delenv("SIM_TOOL_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SIM_TOOL_DESIGNER_LLM_PROVIDER", raising=False)
    monkeypatch.setattr("sim_tool.llm._discover_openai_compatible_model", lambda _: "qwen/qwen2.5-coder-14b")
    backend = make_backend()
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.model_name == "qwen/qwen2.5-coder-14b"
    assert backend._base_url == "http://localhost:1234/v1"


def test_lmstudio_falls_back_to_auto_when_discovery_fails(monkeypatch):
    monkeypatch.setattr(
        "sim_tool.llm._discover_openai_compatible_model",
        lambda _: (_ for _ in ()).throw(RuntimeError("down")),
    )
    cfg = resolve_service_config("designer", provider="lmstudio")
    assert cfg.model == "auto"


def test_ollama_backend_uses_ollama_url():
    backend = make_backend("ollama")
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend._base_url == "http://localhost:11434/v1"


def test_anthropic_backend_still_supported():
    backend = make_backend("anthropic", model="claude-test")
    assert isinstance(backend, AnthropicBackend)
    assert backend.model_name == "claude-test"


def test_service_specific_provider_overrides_global(monkeypatch):
    monkeypatch.setenv("SIM_TOOL_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("SIM_TOOL_ANALYST_LLM_PROVIDER", "lmstudio")
    cfg = resolve_service_config("analyst")
    assert cfg.provider == "lmstudio"


def test_service_specific_base_url_and_model(monkeypatch):
    monkeypatch.setenv("SIM_TOOL_SYMBOLIC_LLM_PROVIDER", "custom")
    monkeypatch.setenv("SIM_TOOL_SYMBOLIC_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("SIM_TOOL_SYMBOLIC_LLM_MODEL", "custom-model")
    cfg = resolve_service_config("symbolic")
    assert cfg.provider == "custom"
    assert cfg.base_url == "https://llm.example/v1"
    assert cfg.model == "custom-model"


def test_service_specific_timeout(monkeypatch):
    monkeypatch.setenv("SIM_TOOL_LITERATURE_REVIEW_LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr("sim_tool.llm._discover_openai_compatible_model", lambda _: "local-model")
    cfg = resolve_service_config("literature_review", provider="lmstudio")
    assert cfg.timeout_seconds == 12.5

    backend = make_backend(service="literature_review", provider="lmstudio", model="auto")
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend._timeout_seconds == 12.5
