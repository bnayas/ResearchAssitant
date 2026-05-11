"""
research_platform.agents.base
─────────────────────────────
Foundation types for the tool-first agent architecture.

Design principles:
  • ToolDescriptor is *data* — serialisable to JSON for the LLM planner,
    introspectable in tests, composable programmatically.
  • ToolRequirement carries both a human-readable description (guides the
    LLM) and a machine-callable validator (deterministic gate).
  • BaseAgent exposes `tools()` for discovery and `invoke()` for execution.
  • ToolContext / ToolResult are generic envelopes — no agent-specific
    types leak across the boundary.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from ..contracts import ArtifactRef
from ..service_contracts import OrchestrationRequestEnvelope


RequirementSource = Literal["auto", "input", "artifact", "metadata"]


# ---------------------------------------------------------------------------
# ToolRequirement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolRequirement:
    """A typed, machine-validated precondition for enabling a tool.

    Serves two audiences simultaneously:
      • **LLM planner** — reads `name`, `description`, and `type` to decide
        how to wire inputs when building an execution plan.
      • **Deterministic validator** — the orchestrator calls `validate(value)`
        before dispatching, producing a clear error message on failure.

    Parameters
    ----------
    name:
        Machine-readable identifier used as a key in ``ToolContext.inputs``
        or ``ToolContext.artifacts``.  E.g. ``"article"``, ``"instruction"``.
    description:
        Short, informative sentence for the LLM.
        E.g. ``"An ArtifactRef pointing to the primary article with an abstract in metadata"``.
    type:
        One of the supported type tags:
        ``"str"``, ``"int"``, ``"float"``, ``"bool"``, ``"dict"``,
        ``"list[str]"``, ``"list[float]"``,
        ``"ArtifactRef"``, ``"list[ArtifactRef]"``.
    required:
        If ``False``, the input may be absent; ``default`` is used instead.
    default:
        Fallback value when ``required=False`` and the input is absent.
    validator:
        Optional callable ``(value) -> bool``.  Called after type-checking.
        Returning ``False`` blocks invocation with a clear error.
    validator_description:
        Human-readable explanation of what ``validator`` checks.
        Included in error messages and in the serialised tool catalog.
    source:
        Where the orchestrator should look for this requirement at enablement
        time. ``"auto"`` keeps backward-compatible behaviour:
        ``ArtifactRef``/``list[ArtifactRef]`` are read from artifacts, all
        other types from inputs.
    """

    name: str
    description: str
    type: str  # "str" | "int" | "float" | "bool" | "dict" | "ArtifactRef" | "list[ArtifactRef]" | ...
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = field(default=None, repr=False, compare=False)
    validator_description: str = ""
    source: RequirementSource = "auto"

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation for the LLM planner."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
            "default": self.default,
            "validator_description": self.validator_description,
            "source": self.source,
        }

    # -- validation ----------------------------------------------------------

    def validate(self, value: Any) -> "RequirementValidationResult":
        """Run type check + optional validator.  Returns a result object."""
        if self.type not in _TYPE_MAP:
            return RequirementValidationResult(
                ok=False,
                requirement=self,
                message=f"Requirement '{self.name}' uses unsupported type tag '{self.type}'.",
            )
        if self.source not in _SOURCE_VALUES:
            return RequirementValidationResult(
                ok=False,
                requirement=self,
                message=f"Requirement '{self.name}' uses unsupported source '{self.source}'.",
            )
        if value is None and self.required:
            return RequirementValidationResult(
                ok=False,
                requirement=self,
                message=f"Required input '{self.name}' is missing.",
            )
        if value is None and not self.required:
            return RequirementValidationResult(ok=True, requirement=self)

        type_ok, type_msg = _check_type(value, self.type)
        if not type_ok:
            return RequirementValidationResult(
                ok=False,
                requirement=self,
                message=(
                    f"Input '{self.name}' has wrong type: expected {self.type}, "
                    f"got {type(value).__name__}. {type_msg}"
                ),
            )

        if self.validator is not None:
            try:
                passed = self.validator(value)
            except Exception as exc:
                return RequirementValidationResult(
                    ok=False,
                    requirement=self,
                    message=(
                        f"Validator for '{self.name}' raised an exception: {exc}. "
                        f"Validator description: {self.validator_description}"
                    ),
                )
            if not passed:
                return RequirementValidationResult(
                    ok=False,
                    requirement=self,
                    message=(
                        f"Input '{self.name}' failed validation: {self.validator_description}"
                    ),
                )

        return RequirementValidationResult(ok=True, requirement=self)

    def resolve_value(
        self,
        inputs: dict[str, Any],
        artifacts: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Resolve this requirement from generic tool invocation maps."""
        source = self.source
        if source == "auto":
            source = "artifact" if self.type in ("ArtifactRef", "list[ArtifactRef]") else "input"
        if source == "input":
            return inputs.get(self.name)
        if source == "artifact":
            return artifacts.get(self.name)
        if source == "metadata":
            return (metadata or {}).get(self.name)
        return None


@dataclass(frozen=True)
class RequirementValidationResult:
    """Outcome of validating a single requirement."""

    ok: bool
    requirement: Optional[ToolRequirement] = None
    message: str = ""


# ---------------------------------------------------------------------------
# ToolDescriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDescriptor:
    """A capability exposed by an agent.

    This is the *only* thing the orchestrator needs to know about an agent's
    abilities.  It is pure data — no behaviour, no imports from the agent's
    internals.

    Parameters
    ----------
    name:
        Machine name, e.g. ``"find_primary_article"``.  Unique across all
        agents in a single orchestration session.
    display_name:
        Human-readable name, e.g. ``"Find Primary Article"``.
    description:
        1-2 sentence summary for the LLM planner explaining *what* this
        tool does and *when* to use it.
    agent_id:
        Identifier of the owning agent (matches ``BaseAgent.agent_id``).
    requirements:
        Typed inputs.  Each ``ToolRequirement`` guides the LLM planner and
        gates invocation deterministically.
    produces:
        Artifact kinds this tool can produce, e.g. ``["primary_article"]``.
        The planner uses this to resolve downstream dependencies.
    tags:
        Free-form tags for filtering, e.g. ``["literature", "search"]``.
    idempotent:
        ``True`` if re-running with the same inputs is safe / yields the
        same result.  Helps the orchestrator decide whether to cache.
    estimated_seconds:
        Rough wall-clock hint for the planner.
    """

    name: str
    display_name: str
    description: str
    agent_id: str
    requirements: list[ToolRequirement]
    produces: list[str]
    tags: list[str] = field(default_factory=list)
    idempotent: bool = False
    estimated_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation for the LLM planner."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "agent_id": self.agent_id,
            "requirements": [req.to_dict() for req in self.requirements],
            "produces": list(self.produces),
            "tags": list(self.tags),
            "idempotent": self.idempotent,
            "estimated_seconds": self.estimated_seconds,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def validate_inputs(
        self,
        inputs: dict[str, Any],
        artifacts: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[RequirementValidationResult]:
        """Validate all requirements against provided inputs/artifacts.

        Returns a list of *all* results (including passing ones).
        The caller should check ``all(r.ok for r in results)`` to decide
        whether to proceed.
        """
        results: list[RequirementValidationResult] = []
        for req in self.requirements:
            value = req.resolve_value(inputs, artifacts, metadata)
            if value is None and not req.required:
                value = req.default
            results.append(req.validate(value))
        return results

    def requirement_by_name(self, name: str) -> Optional[ToolRequirement]:
        """Return the named requirement if this tool declares it."""
        for req in self.requirements:
            if req.name == name:
                return req
        return None

    def contract_errors(self, *, expected_agent_id: Optional[str] = None) -> list[str]:
        """Return deterministic contract errors for this descriptor.

        This is intentionally stricter than invocation-time validation.  It
        catches bad tool catalogs as agents are wired, before the LLM planner
        can see or select invalid capabilities.
        """
        errors: list[str] = []
        if not self.name.strip():
            errors.append("tool name is empty")
        if not self.display_name.strip():
            errors.append(f"{self.name}: display_name is empty")
        if not self.description.strip():
            errors.append(f"{self.name}: description is empty")
        if expected_agent_id is not None and self.agent_id != expected_agent_id:
            errors.append(
                f"{self.name}: descriptor agent_id '{self.agent_id}' does not match owner '{expected_agent_id}'"
            )
        seen_requirements: set[str] = set()
        for req in self.requirements:
            if not req.name.strip():
                errors.append(f"{self.name}: requirement name is empty")
            if not req.description.strip():
                errors.append(f"{self.name}.{req.name}: description is empty")
            if req.name in seen_requirements:
                errors.append(f"{self.name}: duplicate requirement '{req.name}'")
            seen_requirements.add(req.name)
            if req.type not in _TYPE_MAP:
                errors.append(f"{self.name}.{req.name}: unsupported type tag '{req.type}'")
            if req.source not in _SOURCE_VALUES:
                errors.append(f"{self.name}.{req.name}: unsupported source '{req.source}'")
        return errors


# ---------------------------------------------------------------------------
# ToolContext & ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Everything a tool invocation receives.

    No agent-specific types leak here — this is a generic envelope.
    """

    tool_name: str
    directive_id: str
    instruction: str
    inputs: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("."))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a tool invocation returns — uniform across all agents."""

    status: str  # "completed" | "needs_input" | "failed"
    artifacts: list[ArtifactRef] = field(default_factory=list)
    message: str = ""
    request: Optional[OrchestrationRequestEnvelope] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    follow_up_steps: list[dict[str, Any]] = field(default_factory=list)
    inquiries: list[OrchestrationRequestEnvelope] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @property
    def needs_input(self) -> bool:
        return self.status == "needs_input"

    @property
    def failed(self) -> bool:
        return self.status == "failed"


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for every agent in the platform.

    Subclasses must implement three methods:
      • ``agent_id`` — unique identifier string
      • ``tools``    — list of ``ToolDescriptor`` (the agent's public API)
      • ``invoke``   — execute a named tool given a ``ToolContext``
    """

    @abstractmethod
    def agent_id(self) -> str:
        """Unique string identifying this agent, e.g. ``'literature'``."""
        ...

    @abstractmethod
    def tools(self) -> list[ToolDescriptor]:
        """Return all tool descriptors this agent exposes."""
        ...

    @abstractmethod
    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        """Execute the named tool.

        Raises ``KeyError`` if ``tool_name`` is not in ``self.tools()``.
        """
        ...

    # -- convenience ---------------------------------------------------------

    def tool_by_name(self, name: str) -> Optional[ToolDescriptor]:
        """Look up a single tool descriptor by name."""
        for tool in self.tools():
            if tool.name == name:
                return tool
        return None

    def tool_catalog_json(self, *, indent: int = 2) -> str:
        """Serialise all tool descriptors to JSON (for the LLM planner)."""
        return json.dumps(
            [tool.to_dict() for tool in self.tools()],
            indent=indent,
        )

    def catalog_errors(self) -> list[str]:
        """Return deterministic errors in this agent's public tool catalog."""
        errors: list[str] = []
        agent_id = self.agent_id()
        if not agent_id.strip():
            errors.append("agent_id is empty")
        seen_tools: set[str] = set()
        for tool in self.tools():
            if tool.name in seen_tools:
                errors.append(f"duplicate tool '{tool.name}' in agent '{agent_id}'")
            seen_tools.add(tool.name)
            errors.extend(tool.contract_errors(expected_agent_id=agent_id))
        return errors

    def validate_catalog(self) -> None:
        """Raise ValueError if this agent exposes an invalid tool catalog."""
        errors = self.catalog_errors()
        if errors:
            raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# Type-checking helper
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "dict": dict,
    "list": list,
    "list[str]": list,
    "list[float]": list,
    "list[int]": list,
    "list[dict]": list,
    "ArtifactRef": ArtifactRef,
    "list[ArtifactRef]": list,
}

_SOURCE_VALUES = {"auto", "input", "artifact", "metadata"}


def _check_type(value: Any, expected: str) -> tuple[bool, str]:
    """Basic runtime type check.  Returns (ok, message)."""
    python_type = _TYPE_MAP.get(expected)
    if python_type is None:
        return False, f"Unsupported type tag: {expected}"
    if expected == "int" and isinstance(value, bool):
        return False, "bool is not accepted as int"
    if expected == "float" and isinstance(value, bool):
        return False, "bool is not accepted as float"
    if not isinstance(value, python_type):
        return False, f"Expected {expected}, got {type(value).__name__}"

    # For typed lists, check element types
    if expected.startswith("list[") and isinstance(value, list) and value:
        inner = expected[5:-1]  # e.g. "str" from "list[str]"
        inner_type = _TYPE_MAP.get(inner)
        if inner_type is not None:
            for idx, item in enumerate(value):
                if inner in ("int", "float") and isinstance(item, bool):
                    return False, (
                        f"Element [{idx}] has wrong type: expected {inner}, got bool"
                    )
                if not isinstance(item, inner_type):
                    return False, (
                        f"Element [{idx}] has wrong type: expected {inner}, "
                        f"got {type(item).__name__}"
                    )
    return True, ""
