"""
research_platform.agents.literature
────────────────────────────────────
Literature agent — exposes academic search, article parsing, and
literature review capabilities as typed tools.

Each tool lives in its own file.  This module assembles them into a
single ``LiteratureAgent(BaseAgent)`` that the orchestrator discovers.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaseAgent, ToolContext, ToolDescriptor, ToolResult
from ...registry import ArtifactRegistry

from .prepare_lookup import TOOL as PREPARE_LOOKUP_TOOL, execute as execute_prepare_lookup
from .find_article import TOOL as FIND_ARTICLE_TOOL, execute as execute_find_article
from .build_brief import (
    TOOL as BUILD_BRIEF_TOOL,
    PARSE_ARTICLE_WITH_INTENTION_TOOL,
    execute as execute_build_brief,
    execute_parse_article_with_intention,
)
from .review_literature import TOOL as REVIEW_LITERATURE_TOOL, execute as execute_review_literature
from .answer_question import (
    ARTICLE_QUESTION_TOOL,
    REVIEW_QUESTION_TOOL,
    execute_article_question,
    execute_review_question,
)

AGENT_ID = "literature"


class LiteratureAgent(BaseAgent):
    """Wraps existing literature assistant capabilities as typed tools.

    Phase 4: each tool drives search backends and LLM directly.
    No dependency on ``LocalLiteratureAssistant``.
    """

    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        llm_backend: Any = None,
        plugin_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._registry = registry
        self._llm_backend = llm_backend
        self._plugin_config = plugin_config or {}

    def agent_id(self) -> str:
        return AGENT_ID

    def tools(self) -> list[ToolDescriptor]:
        return [
            PREPARE_LOOKUP_TOOL,
            FIND_ARTICLE_TOOL,
            PARSE_ARTICLE_WITH_INTENTION_TOOL,
            BUILD_BRIEF_TOOL,
            REVIEW_LITERATURE_TOOL,
            ARTICLE_QUESTION_TOOL,
            REVIEW_QUESTION_TOOL,
        ]

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        dispatch = {
            PREPARE_LOOKUP_TOOL.name: execute_prepare_lookup,
            FIND_ARTICLE_TOOL.name: execute_find_article,
            PARSE_ARTICLE_WITH_INTENTION_TOOL.name: execute_parse_article_with_intention,
            BUILD_BRIEF_TOOL.name: execute_build_brief,
            REVIEW_LITERATURE_TOOL.name: execute_review_literature,
            ARTICLE_QUESTION_TOOL.name: execute_article_question,
            REVIEW_QUESTION_TOOL.name: execute_review_question,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            raise KeyError(f"Unknown literature tool: {tool_name}")
        return handler(context, registry=self._registry, llm_backend=self._llm_backend)
