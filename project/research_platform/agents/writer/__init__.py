"""
research_platform.agents.writer
───────────────────────────────
Writer agent — exposes review planning, section drafting, and assembly.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaseAgent, ToolContext, ToolDescriptor, ToolResult
from ...registry import ArtifactRegistry

from .plan_review import TOOL as PLAN_REVIEW_TOOL, execute as execute_plan_review
from .draft_section import TOOL as DRAFT_SECTION_TOOL, execute as execute_draft_section
from .assemble_review import TOOL as ASSEMBLE_REVIEW_TOOL, execute as execute_assemble_review

AGENT_ID = "writer"


class WriterAgent(BaseAgent):
    """Wraps writing capabilities as typed tools."""

    def __init__(self, registry: ArtifactRegistry, *, llm_backend: Any = None) -> None:
        self._registry = registry
        self._llm_backend = llm_backend

    def agent_id(self) -> str:
        return AGENT_ID

    def tools(self) -> list[ToolDescriptor]:
        return [PLAN_REVIEW_TOOL, DRAFT_SECTION_TOOL, ASSEMBLE_REVIEW_TOOL]

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        dispatch = {
            PLAN_REVIEW_TOOL.name: execute_plan_review,
            DRAFT_SECTION_TOOL.name: execute_draft_section,
            ASSEMBLE_REVIEW_TOOL.name: execute_assemble_review,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            raise KeyError(f"Unknown writer tool: {tool_name}")
        return handler(context, registry=self._registry, llm_backend=self._llm_backend)
