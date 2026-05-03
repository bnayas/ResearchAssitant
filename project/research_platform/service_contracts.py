"""
research_platform.service_contracts
──────────────────────────────────
Service-layer contracts for the mediator-first refactor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import time
import uuid
from typing import Any, Optional


@dataclass
class AgentRuntimeProfile:
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: Optional[float] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    tool_allowlist: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    plugin_config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_backend_kwargs(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name in ("provider", "model", "base_url", "api_key", "timeout_seconds"):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload


@dataclass
class ServiceRuntimeConfig:
    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)
    mcp_servers: list[str] = field(default_factory=list)


@dataclass
class MixedServiceRuntimeConfig(ServiceRuntimeConfig):
    nl_profile: Optional[AgentRuntimeProfile] = None


@dataclass
class OrchestrationRequestEnvelope:
    request_id: str
    resume_token: str
    request_kind: str
    question: str
    context_refs: list[str] = field(default_factory=list)
    expected_schema: dict[str, Any] = field(default_factory=dict)
    capability_hint: Optional[str] = None
    priority: str = "normal"
    blocking: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(
        *,
        resume_token: str,
        request_kind: str,
        question: str,
        context_refs: Optional[list[str]] = None,
        expected_schema: Optional[dict[str, Any]] = None,
        capability_hint: Optional[str] = None,
        priority: str = "normal",
        blocking: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "OrchestrationRequestEnvelope":
        return OrchestrationRequestEnvelope(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            resume_token=resume_token,
            request_kind=request_kind,
            question=question,
            context_refs=list(context_refs or []),
            expected_schema=dict(expected_schema or {}),
            capability_hint=capability_hint,
            priority=priority,
            blocking=blocking,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextInjection:
    target_session_id: str
    mode: str
    payload: dict[str, Any]
    reason: str = ""
    injected_by: str = "system"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeDiagnostics:
    status: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failure_class: str = ""
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    result_ref: Optional[str] = None
    diagnostics_ref: Optional[str] = None
    raw_log_ref: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class MediatedResponse:
    status: str
    session_id: str = ""
    payload: Any = None
    artifact_ids: list[str] = field(default_factory=list)
    request: Optional[OrchestrationRequestEnvelope] = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "payload": _jsonify(self.payload),
            "artifact_ids": list(self.artifact_ids),
            "request": self.request.to_dict() if self.request else None,
            "error": self.error,
        }


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if is_dataclass(value):
        return {
            field_name: _jsonify(getattr(value, field_name))
            for field_name in value.__dataclass_fields__.keys()
        }
    return str(value)
