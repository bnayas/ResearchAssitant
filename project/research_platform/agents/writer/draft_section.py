"""
writer.draft_section
────────────────────
Tool: Draft a single section of the review.
"""
from __future__ import annotations

from typing import Any

from ...contracts import AssistantId
from ...registry import ArtifactRegistry
from ..base import ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import dict_has_keys, is_non_empty_string

TOOL = ToolDescriptor(
    name="draft_section",
    display_name="Draft Section",
    description=(
        "Draft a single section of the review, grounded in source artifacts. "
        "Each section is one LLM call with focused context."
    ),
    agent_id="writer",
    requirements=[
        ToolRequirement(name="section_spec", description="Section spec dict with title and description",
                        type="dict", validator=dict_has_keys("title"),
                        validator_description="Must have 'title' key"),
        ToolRequirement(name="sources", description="Source artifacts for grounding",
                        type="list[ArtifactRef]", required=False, default=[]),
        ToolRequirement(name="previous_sections", description="Already drafted section texts for continuity",
                        type="list[str]", required=False, default=[]),
    ],
    produces=["section_draft"],
    tags=["writing", "drafting"],
    idempotent=True,
    estimated_seconds=20.0,
)


def execute(context: ToolContext, *, registry: ArtifactRegistry, llm_backend: Any = None) -> ToolResult:
    """Draft one section using LLM."""
    if llm_backend is None:
        return ToolResult(status="failed", message="No LLM backend for drafting")
    section_spec = context.inputs["section_spec"]
    title = section_spec["title"]
    try:
        raw = llm_backend.complete(
            system="Draft an academic review section. Be precise, cite sources, use formal tone.",
            messages=[{"role": "user", "content": f"Section: {title}\nDescription: {section_spec.get('description','')}\nContext: {context.instruction}"}],
            temperature=0.3,
        )
        text = str(raw).strip()
    except Exception as exc:
        return ToolResult(status="failed", message=f"Drafting failed: {exc}")
    ref = registry.save_text(
        assistant=AssistantId.WRITER.value, kind="section_draft",
        title=f"Section: {title}", filename=f"review/sections/{title.replace(' ','_').lower()}.md",
        text=text, summary=title, metadata={"section_spec": section_spec},
        artifact_id=f"section-{title.replace(' ','-').lower()}",
    )
    return ToolResult(status="completed", artifacts=[ref], message=f"Drafted: {title}", metadata={"text": text})
