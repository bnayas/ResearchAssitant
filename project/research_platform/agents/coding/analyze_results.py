"""
coding.analyze_results
──────────────────────
Tool: Analyse simulation output and propose patches if needed.
"""
from __future__ import annotations

from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import artifact_kind_is, dict_non_empty

TOOL = ToolDescriptor(
    name="analyze_results",
    display_name="Analyse Results",
    description=(
        "Analyse simulation run output using the Analyst LLM.  Evaluates "
        "data quality, convergence, and physical plausibility.  May propose "
        "spec patches if results indicate minor or major issues."
    ),
    agent_id="coding",
    requirements=[
        ToolRequirement(
            name="spec", description="The simulation spec ArtifactRef",
            type="ArtifactRef", validator=artifact_kind_is("simulation_spec"),
            validator_description="Must be a 'simulation_spec' artifact",
        ),
        ToolRequirement(
            name="diagnostics",
            description="Runtime diagnostics dict from a run_sample or run_sweep execution",
            type="dict", validator=dict_non_empty,
            validator_description="Must be a non-empty diagnostics dict",
        ),
    ],
    produces=["analysis"],
    tags=["coding", "analysis", "validation"],
    idempotent=True,
    estimated_seconds=120.0,
)


def execute(
    context: ToolContext,
    *,
    registry: ArtifactRegistry,
    llm_config: Any = None,
) -> ToolResult:
    """Delegate to RunAnalyst for result analysis."""
    from pathlib import Path
    from sim_tool.analyst import RunAnalyst
    from ...assistants.backends import make_service_backend

    diagnostics = context.inputs["diagnostics"]
    sweep_dir = str((diagnostics.get("metadata") or {}).get("sweep_dir", "")).strip()
    if not sweep_dir:
        return ToolResult(status="failed", message="Diagnostics missing sweep_dir")

    backend = make_service_backend("analyst", llm_config) if llm_config else None
    if backend is None:
        return ToolResult(status="failed", message="No LLM backend for analysis")

    analyst = RunAnalyst(backend=backend)
    try:
        analysis = analyst.analyse_sweep(Path(sweep_dir), session_id=context.directive_id)
    except Exception as exc:
        return ToolResult(status="failed", message=f"Analysis failed: {exc}")

    ref = registry.save_json(
        assistant=AssistantId.CODING_AGENT.value,
        kind="simulation_analysis", title="Simulation Analysis",
        filename="simulation/analysis.json",
        payload={"verdict": analysis.verdict.value, "llm_reasoning": analysis.llm_reasoning},
        summary=f"Verdict: {analysis.verdict.value}",
        metadata={"session_id": context.directive_id},
        artifact_id="simulation-analysis",
    )
    return ToolResult(
        status="completed", artifacts=[ref],
        message=f"Analysis verdict: {analysis.verdict.value}",
        metadata={"verdict": analysis.verdict.value, "has_patch": analysis.patch is not None},
    )
