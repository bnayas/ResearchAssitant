"""
sim_tool.llm
─────────────
LLM backend abstraction. Supports Anthropic and OpenAI-compatible endpoints.
Each service can resolve its own default backend independently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import os
import urllib.request
from typing import Optional


class LLMError(Exception):
    pass


log = logging.getLogger("sim_tool.llm")


_DEFAULT_PROVIDER = "lmstudio"

_OPENAI_COMPAT_PRESETS: dict[str, dict[str, str]] = {
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "auto",
        "api_key_env": "",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "api_key_env": "",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai-compat": {
        "base_url": "http://localhost:1234/v1",
        "model": "auto",
        "api_key_env": "SIM_TOOL_LLM_API_KEY",
    },
    "custom": {
        "base_url": "http://localhost:8000/v1",
        "model": "",
        "api_key_env": "SIM_TOOL_LLM_API_KEY",
    },
}

_ANTHROPIC_PRESET = {
    "model": "claude-sonnet-4-20250514",
    "api_key_env": "ANTHROPIC_API_KEY",
}

_PROVIDER_ALIASES = {
    "auto": _DEFAULT_PROVIDER,
    "default": _DEFAULT_PROVIDER,
    "local": "lmstudio",
}

_AUTO_MODEL_NAMES = {"", "auto", "local-model"}


@dataclass(frozen=True)
class LLMServiceConfig:
    service: str
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 30.0

    @property
    def is_openai_compatible(self) -> bool:
        return self.provider in _OPENAI_COMPAT_PRESETS


def _service_key(service: Optional[str]) -> str:
    service_name = (service or "default").strip().lower().replace("-", "_")
    return service_name or "default"


def _service_env_key(service: Optional[str], suffix: str) -> str:
    return f"SIM_TOOL_{_service_key(service).upper()}_LLM_{suffix}"


def _resolve_provider(service: Optional[str], provider: Optional[str]) -> str:
    if provider:
        raw = provider
    else:
        raw = (
            os.environ.get(_service_env_key(service, "PROVIDER"))
            or os.environ.get("SIM_TOOL_LLM_PROVIDER")
            or _DEFAULT_PROVIDER
        )
    return _PROVIDER_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def _resolve_api_key(
    service: Optional[str],
    provider: str,
    api_key: Optional[str],
) -> str:
    if api_key is not None:
        return api_key
    for env_name in (
        _service_env_key(service, "API_KEY"),
        "SIM_TOOL_LLM_API_KEY",
    ):
        value = os.environ.get(env_name)
        if value:
            return value
    if provider == "anthropic":
        return os.environ.get(_ANTHROPIC_PRESET["api_key_env"], "")
    preset = _OPENAI_COMPAT_PRESETS.get(provider)
    if preset:
        env_name = preset.get("api_key_env", "")
        if env_name:
            return os.environ.get(env_name, "")
    return ""


def _resolve_timeout_seconds(
    service: Optional[str],
    timeout_seconds: Optional[float],
) -> float:
    if timeout_seconds is not None:
        return float(timeout_seconds)
    raw_value = (
        os.environ.get(_service_env_key(service, "TIMEOUT_SECONDS"))
        or os.environ.get("SIM_TOOL_LLM_TIMEOUT_SECONDS")
        or "30"
    )
    try:
        resolved = float(raw_value)
    except ValueError:
        return 30.0
    return resolved if resolved > 0 else 30.0


def _discover_openai_compatible_model(base_url: str) -> str:
    models_url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(models_url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        payload = json.loads(response.read())
    data = payload.get("data") or []
    for item in data:
        model_id = (item.get("id") or "").strip()
        if model_id:
            return model_id
    raise LLMError(f"No models advertised by {models_url}")


def _resolve_model_name(
    provider_name: str,
    base_url: str,
    model_name: str,
) -> str:
    normalized = model_name.strip()
    if provider_name not in {"lmstudio", "openai-compat"}:
        return normalized
    if normalized not in _AUTO_MODEL_NAMES:
        return normalized
    try:
        discovered = _discover_openai_compatible_model(base_url)
        log.info("Resolved %s model %r from %s", provider_name, discovered, base_url)
        return discovered
    except Exception as exc:
        log.warning(
            "Could not auto-resolve model for %s at %s: %s",
            provider_name,
            base_url,
            exc,
        )
        return normalized or "auto"


def resolve_service_config(
    service: str = "default",
    provider: Optional[str] = None,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> LLMServiceConfig:
    provider_name = _resolve_provider(service, provider)
    resolved_timeout = _resolve_timeout_seconds(service, timeout_seconds)
    if provider_name == "anthropic":
        resolved_model = (
            model
            or os.environ.get(_service_env_key(service, "MODEL"))
            or os.environ.get("SIM_TOOL_LLM_MODEL")
            or _ANTHROPIC_PRESET["model"]
        )
        return LLMServiceConfig(
            service=_service_key(service),
            provider=provider_name,
            model=resolved_model,
            api_key=_resolve_api_key(service, provider_name, api_key),
            timeout_seconds=resolved_timeout,
        )

    preset = _OPENAI_COMPAT_PRESETS.get(provider_name)
    if preset is None:
        if base_url is None:
            raise ValueError(
                f"Unknown provider {provider_name!r}. "
                "Pass base_url/model explicitly for custom OpenAI-compatible services."
            )
        preset = {"base_url": base_url, "model": model or "", "api_key_env": ""}
        provider_name = "custom"

    resolved_base_url = (
        base_url
        or os.environ.get(_service_env_key(service, "BASE_URL"))
        or os.environ.get("SIM_TOOL_LLM_BASE_URL")
        or preset["base_url"]
    )
    resolved_model = (
        model
        or os.environ.get(_service_env_key(service, "MODEL"))
        or os.environ.get("SIM_TOOL_LLM_MODEL")
        or preset["model"]
    )
    resolved_model = _resolve_model_name(provider_name, resolved_base_url, resolved_model)
    return LLMServiceConfig(
        service=_service_key(service),
        provider=provider_name,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=_resolve_api_key(service, provider_name, api_key),
        timeout_seconds=resolved_timeout,
    )


class LLMBackend(ABC):
    model_name: str = "base"

    @abstractmethod
    def complete(self, system: str, messages: list, max_tokens: int = 4096,
                 temperature: float = 0.0) -> str: ...


class AnthropicBackend(LLMBackend):
    model_name = "claude-sonnet-4-20250514"

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: Optional[str] = None, timeout_seconds: float = 30.0):
        self.model_name = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._timeout_seconds = timeout_seconds

    def complete(self, system: str, messages: list, max_tokens: int = 4096,
                 temperature: float = 0.0) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return msg.content[0].text


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model: str = "auto", api_key: str = "",
                 timeout_seconds: float = 30.0):
        self.model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def complete(self, system: str, messages: list, max_tokens: int = 4096,
                 temperature: float = 0.0) -> str:
        import urllib.request, json
        payload = json.dumps({
            "model": self.model_name, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}] + messages,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._base_url.rstrip('/')}/chat/completions",
            data=payload, headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self._timeout_seconds) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


def make_backend(provider: Optional[str] = None, **kwargs) -> LLMBackend:
    service = kwargs.pop("service", "default")
    config = resolve_service_config(
        service=service,
        provider=provider,
        model=kwargs.pop("model", None),
        base_url=kwargs.pop("base_url", None),
        api_key=kwargs.pop("api_key", None),
        timeout_seconds=kwargs.pop("timeout_seconds", None),
    )
    if config.provider == "anthropic":
        return AnthropicBackend(
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )
    return OpenAICompatibleBackend(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )
