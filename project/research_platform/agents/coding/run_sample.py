"""
coding.run_sample
─────────────────
Tool: Execute a sample run of the simulation to validate correctness.
"""
from __future__ import annotations

from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_kind_is, is_positive_number

TOOL = ToolDescriptor(
    name="run_sample",
    display_name="Run Sample",
    description=(
        "Execute a short sample run of the simulation to validate correctness "
        "before committing to a full sweep.  Returns diagnostics with metrics, "
        "warnings, and errors."
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
            name="sample_steps",
            description="Number of steps for the sample run",
            type="int",
            required=False,
            default=200,
            validator=is_positive_number,
            validator_description="Must be a positive integer",
        ),
    ],
    produces=["sample_results"],
    tags=["coding", "execution", "validation"],
    idempotent=False,
    estimated_seconds=60.0,
)


def execute(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_config: Any = None,
) -> ToolResult:
    """Delegate to RuntimeAssembleService for sample execution."""
    from ...runtime_assemble import RuntimeAssembleService
    import json

    spec_artifact = context.artifacts["spec"]
    sample_steps = context.inputs.get("sample_steps", 200)

    spec_text = registry.read_artifact_text(spec_artifact.artifact_id)
    if not spec_text:
        return ToolResult(status="failed", message="Cannot read spec artifact content")

    try:
        spec_payload = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        return ToolResult(status="failed", message=f"Invalid spec JSON: {exc}")

    runtime = RuntimeAssembleService(registry, output_root=context.output_dir)
    session_id = context.directive_id

    try:
        bundle = runtime.materialize_runtime(session_id=session_id, spec_payload=spec_payload)
    except Exception as exc:
        return ToolResult(status="failed", message=f"Runtime materialisation failed: {exc}")

    try:
        diagnostics = runtime.run_sample(
            session_id=session_id,
            spec_payload=spec_payload,
            bundle=bundle,
            sample_steps=sample_steps,
        )
    except Exception as exc:
        return ToolResult(status="failed", message=f"Sample run failed: {exc}")

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
        message=(
            f"Sample run {diagnostics.status}: "
            f"{diagnostics.metrics_summary.get('n_success', 0)} success, "
            f"{diagnostics.metrics_summary.get('n_failed', 0)} failed"
        ),
        metadata={
            "diagnostics": diagnostics.to_dict(),
            "bundle": bundle,
        },
    )
