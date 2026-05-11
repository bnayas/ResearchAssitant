"""
writer.plan_review
──────────────────
Tool: Plan the section structure for a review article.
"""
from __future__ import annotations

from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import is_non_empty_string, list_non_empty

TOOL = ToolDescriptor(
    name="plan_review_structure",
    display_name="Plan Review Structure",
    description=(
        "Plan the section structure for a review article based on available "
        "source artifacts.  Returns a section plan with titles, descriptions, "
        "and which sources each section should draw from."
    ),
    agent_id="writer",
    requirements=[
        ToolRequirement(name="instruction", description="The PI directive", type="str",
                        validator=is_non_empty_string, validator_description="Non-empty"),
        ToolRequirement(name="sources", description="Source artifacts for the review",
                        type="list[ArtifactRef]", validator=list_non_empty,
                        validator_description="Must have at least one source"),
        ToolRequirement(name="venue", description="Target venue", type="str",
                        required=False, default="arXiv preprint"),
    ],
    produces=["review_plan"],
    tags=["writing", "planning"],
    idempotent=True,
    estimated_seconds=15.0,
)


def execute(context: ToolContext, *, registry: ArtifactRegistry, llm_backend: Any = None) -> ToolResult:
    """Plan section structure using LLM."""
    if llm_backend is None:
        return ToolResult(status="failed", message="No LLM backend for planning")
    sources = context.artifacts.get("sources") or list(context.artifacts.values())
    source_summaries = "\n".join(
        f"- {s.title} ({s.kind}): {s.summary}" for s in sources
    )
    try:
        import json
        raw = llm_backend.complete(
            system="Plan review article sections. Return JSON array of {title, description, source_kinds}.",
            messages=[{"role": "user", "content": f"Instruction: {context.instruction}\nVenue: {context.inputs.get('venue','arXiv')}\nSources:\n{source_summaries}"}],
            temperature=0.0,
        )
        plan = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        return ToolResult(status="failed", message=f"Planning failed: {exc}")
    plan_ref = registry.save_json(
        assistant=AssistantId.WRITER.value, kind="review_plan",
        title="Review Plan", filename="review/plan.json",
        payload={"sections": plan}, summary=f"{len(plan)} section(s)",
        metadata={}, artifact_id="review-plan",
    )
    return ToolResult(status="completed", artifacts=[plan_ref],
                      message=f"Planned {len(plan)} sections", metadata={"plan": plan})
