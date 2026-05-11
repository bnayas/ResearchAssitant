"""
coding.run_sweep
────────────────
Tool: Execute the full parameter sweep.
"""
from __future__ import annotations

from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_kind_is

TOOL = ToolDescriptor(
    name="run_full_sweep",
    display_name="Run Full Sweep",
    description=(
        "Execute the full parameter sweep of the simulation.  Runs all "
        "configurations defined in the spec and produces consolidated results."
    ),
    agent_id="coding",
    requirements=[
        ToolRequirement(
            name="spec",
            description="The simulation spec ArtifactRef",
            type="ArtifactRef",
            validator=artifact_kind_is("simulation_spec"),
            validator_description="Must be a 'simulation_spec' artifact",
        ),
        ToolRequirement(
            name="bundle",
            description="Runtime bundle dict from a previous materialize_runtime call",
            type="dict",
            required=False,
            default={},
        ),
    ],
    produces=["sweep_results"],
    tags=["coding", "execution", "sweep"],
    idempotent=False,
    estimated_seconds=300.0,
)


def execute(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_config: Any = None,
) -> ToolResult:
    """Delegate to RuntimeAssembleService for full sweep."""
    from ...runtime_assemble import RuntimeAssembleService
    import json

    spec_artifact = context.artifacts["spec"]
    bundle = context.inputs.get("bundle", {})

    spec_text = registry.read_artifact_text(spec_artifact.artifact_id)
    if not spec_text:
        return ToolResult(status="failed", message="Cannot read spec artifact")

    try:
        spec_payload = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        return ToolResult(status="failed", message=f"Invalid spec JSON: {exc}")

    runtime = RuntimeAssembleService(registry, output_root=context.output_dir)
    session_id = context.directive_id

    if not bundle:
        try:
            bundle = runtime.materialize_runtime(session_id=session_id, spec_payload=spec_payload)
        except Exception as exc:
            return ToolResult(status="failed", message=f"Materialisation failed: {exc}")

    try:
        diagnostics = runtime.run_sweep(
            session_id=session_id,
            spec_payload=spec_payload,
            bundle=bundle,
        )
    except Exception as exc:
        return ToolResult(status="failed", message=f"Sweep failed: {exc}")

    artifacts = []
    for ref_id in [diagnostics.result_ref, diagnostics.diagnostics_ref, diagnostics.raw_log_ref]:
        if ref_id:
            ref = registry.get(ref_id)
            if ref:
                artifacts.append(ref)

    status = "completed" if diagnostics.status == "success" else "failed"
    return ToolResult(
        status=status,
        artifacts=artifacts,
        message=f"Sweep {diagnostics.status}: {diagnostics.metrics_summary}",
        metadata={"diagnostics": diagnostics.to_dict(), "bundle": bundle},
    )
