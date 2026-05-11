"""
research_platform.agents.runtime
────────────────────────────────
Runtime agent — wraps RuntimeAssembleService as typed tools.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaseAgent, ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_kind_is, dict_non_empty
from ...registry import ArtifactRegistry
from ...runtime_assemble import RuntimeAssembleService

AGENT_ID = "runtime"

MATERIALIZE_TOOL = ToolDescriptor(
    name="materialize_runtime",
    display_name="Materialize Runtime",
    description="Generate executable script and notebook from a simulation spec.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="spec", description="Simulation spec ArtifactRef", type="ArtifactRef",
                        validator=artifact_kind_is("simulation_spec"),
                        validator_description="Must be simulation_spec"),
    ],
    produces=["script", "notebook"],
    tags=["runtime", "codegen"], idempotent=True, estimated_seconds=5.0,
)

EXECUTE_RUN_TOOL = ToolDescriptor(
    name="execute_run",
    display_name="Execute Run",
    description="Execute a simulation run with given config and timeout.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="bundle", description="Runtime bundle dict from materialize_runtime",
                        type="dict", validator=dict_non_empty, validator_description="Non-empty bundle"),
        ToolRequirement(name="spec_payload", description="Spec payload dict", type="dict",
                        validator=dict_non_empty, validator_description="Non-empty spec"),
        ToolRequirement(name="sample_steps", description="Number of steps", type="int",
                        required=False, default=200),
    ],
    produces=["run_results"],
    tags=["runtime", "execution"], idempotent=False, estimated_seconds=60.0,
)


class RuntimeAgent(BaseAgent):
    """Wraps RuntimeAssembleService as typed tools."""

    def __init__(self, registry: ArtifactRegistry, *, service: Optional[RuntimeAssembleService] = None) -> None:
        self._registry = registry
        self._service = service or RuntimeAssembleService(registry)

    def agent_id(self) -> str:
        return AGENT_ID

    def tools(self) -> list[ToolDescriptor]:
        return [MATERIALIZE_TOOL, EXECUTE_RUN_TOOL]

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        if tool_name == "materialize_runtime":
            return self._materialize(context)
        if tool_name == "execute_run":
            return self._execute_run(context)
        raise KeyError(f"Unknown runtime tool: {tool_name}")

    def _materialize(self, ctx: ToolContext) -> ToolResult:
        import json
        spec_art = ctx.artifacts["spec"]
        text = self._registry.read_artifact_text(spec_art.artifact_id) or ""
        try:
            spec_payload = json.loads(text)
        except Exception as exc:
            return ToolResult(status="failed", message=f"Invalid spec: {exc}")
        try:
            bundle = self._service.materialize_runtime(
                session_id=ctx.directive_id, spec_payload=spec_payload,
                output_dir=str(ctx.output_dir))
        except Exception as exc:
            return ToolResult(status="failed", message=f"Materialisation failed: {exc}")
        artifacts = []
        for key in ("script_artifact_id", "notebook_artifact_id", "spec_artifact_id"):
            ref = self._registry.get(bundle.get(key, ""))
            if ref:
                artifacts.append(ref)
        return ToolResult(status="completed", artifacts=artifacts,
                          message="Runtime materialised", metadata={"bundle": bundle})

    def _execute_run(self, ctx: ToolContext) -> ToolResult:
        bundle = ctx.inputs["bundle"]
        spec_payload = ctx.inputs["spec_payload"]
        steps = ctx.inputs.get("sample_steps", 200)
        try:
            diag = self._service.run_sample(
                session_id=ctx.directive_id, spec_payload=spec_payload,
                bundle=bundle, sample_steps=steps)
        except Exception as exc:
            return ToolResult(status="failed", message=f"Execution failed: {exc}")
        status = "completed" if diag.status == "success" else "failed"
        return ToolResult(status=status, message=f"Run {diag.status}",
                          metadata={"diagnostics": diag.to_dict()})
