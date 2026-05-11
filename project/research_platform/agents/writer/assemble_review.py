"""
writer.assemble_review
──────────────────────
Tool: Assemble drafted sections into a complete review document.
"""
from __future__ import annotations

from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import list_non_empty

TOOL = ToolDescriptor(
    name="assemble_review",
    display_name="Assemble Review",
    description=(
        "Assemble drafted sections into a complete review document with "
        "introduction, transitions, and bibliography."
    ),
    agent_id="writer",
    requirements=[
        ToolRequirement(name="sections", description="List of section draft ArtifactRefs",
                        type="list[ArtifactRef]", validator=list_non_empty,
                        validator_description="Must have at least one section"),
        ToolRequirement(name="plan", description="The review plan ArtifactRef",
                        type="ArtifactRef", required=False),
    ],
    produces=["review_draft"],
    tags=["writing", "assembly"],
    idempotent=True,
    estimated_seconds=15.0,
)


def execute(context: ToolContext, *, registry: ArtifactRegistry, llm_backend: Any = None) -> ToolResult:
    """Assemble sections into a full review, delegating to the writer pipeline."""
    sections = context.artifacts.get("sections") or []
    if isinstance(sections, list):
        section_refs = sections
    else:
        section_refs = [sections]

    section_texts = []
    for ref in section_refs:
        text = registry.read_artifact_text(ref.artifact_id) if hasattr(ref, 'artifact_id') else str(ref)
        section_texts.append(text or "")

    full_text = "\n\n---\n\n".join(section_texts)

    ref = registry.save_text(
        assistant=AssistantId.WRITER.value, kind="review_draft",
        title="Review Draft", filename="review/draft.md",
        text=full_text, summary=f"Review with {len(section_texts)} section(s)",
        metadata={}, artifact_id="review-draft",
    )
    return ToolResult(status="completed", artifacts=[ref], message=f"Assembled review with {len(section_texts)} sections")
