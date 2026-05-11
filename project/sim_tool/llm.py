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
import shutil
import subprocess
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
    timeout_seconds: float = 300.0

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
        or "300"
    )
    try:
        resolved = float(raw_value)
    except ValueError:
        return 300.0
    return resolved if resolved > 0 else 300.0


def _discover_openai_compatible_model(base_url: str) -> str:
    models_url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(models_url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as response:
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
    def complete(self, system: str, messages: list, max_tokens: Optional[int] = None,
                 temperature: float = 0.0) -> str: ...


class AnthropicBackend(LLMBackend):
    model_name = "claude-sonnet-4-20250514"

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: Optional[str] = None, timeout_seconds: float = 30.0):
        self.model_name = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._timeout_seconds = timeout_seconds

    def complete(self, system: str, messages: list, max_tokens: Optional[int] = None,
                 temperature: float = 0.0) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        log.info(f"LLM Agent Request to {self.model_name}:\nSystem:\n{system}\nMessages:\n{json.dumps(messages, indent=2)}")
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens or int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
            system=system,
            messages=messages,
        )
        content = msg.content[0].text
        log.info(f"LLM Agent Response from {self.model_name}:\n{content}")
        return content


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model: str = "auto", api_key: str = "",
                 timeout_seconds: float = 300.0):
        self.model_name = model
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def complete(self, system: str, messages: list, max_tokens: Optional[int] = None,
                 temperature: float = 0.0) -> str:
        payload_obj = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}] + messages,
        }
        if max_tokens is not None:
            payload_obj["max_tokens"] = max_tokens
        if not self._api_key and self._is_local_endpoint():
            payload_obj["reasoning"] = {"effort": "none"}
            payload_obj["reasoning_effort"] = "none"
        payload = json.dumps(payload_obj).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        log.info(f"LLM Agent Request to {url} (model={self.model_name}):\nSystem:\n{system}\nMessages:\n{json.dumps(messages, indent=2)}")
        if shutil.which("curl") and not self._api_key:
            content = self._complete_with_curl(url, payload, headers)
        else:
            content = self._complete_with_urllib(url, payload, headers)
        log.info(f"LLM Agent Response from {url} (model={self.model_name}):\n{content}")
        return content

    def _complete_with_curl(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
    ) -> str:
        timeout_seconds = max(float(self._timeout_seconds), 1.0)
        cmd = [
            "curl",
            "-sS",
            "--fail-with-body",
            "--connect-timeout",
            str(min(10.0, timeout_seconds)),
            "--max-time",
            str(timeout_seconds),
        ]
        for key, value in headers.items():
            cmd.extend(["-H", f"{key}: {value}"])
        cmd.extend(["--data-binary", "@-", url])
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                timeout=timeout_seconds + 2.0,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"LLM request to {url} exceeded {timeout_seconds:.1f}s"
            ) from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            body = proc.stdout.decode("utf-8", errors="replace").strip()
            detail = body or stderr or f"curl exited with {proc.returncode}"
            if proc.returncode == 28:
                raise TimeoutError(
                    f"LLM request to {url} exceeded {timeout_seconds:.1f}s"
                )
            raise LLMError(f"LLM request failed: {detail[:500]}")
        data = json.loads(proc.stdout)
        return self._extract_message_content(data, url)

    def _complete_with_urllib(
        self,
        url: str,
        payload: bytes,
        headers: dict[str, str],
    ) -> str:
        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self._timeout_seconds) as r:
            data = json.loads(r.read())
        return self._extract_message_content(data, url)

    def _extract_message_content(self, data: dict, url: str) -> str:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        if content.strip():
            return content
        reasoning = str(message.get("reasoning_content") or "")
        finish_reason = str(choice.get("finish_reason") or "")
        if reasoning.strip():
            raise LLMError(
                "LLM returned empty content with reasoning_content "
                f"(finish_reason={finish_reason or 'unknown'}) from {url}. "
                "Retry with a constrained no-reasoning JSON prompt or increase max_tokens."
            )
        raise LLMError(f"LLM returned empty content from {url}")

    def _is_local_endpoint(self) -> bool:
        normalized = self._base_url.lower()
        return "localhost" in normalized or "127.0.0.1" in normalized or "[::1]" in normalized


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
