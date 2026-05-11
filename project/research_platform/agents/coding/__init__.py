"""
research_platform.agents.coding
───────────────────────────────
Coding agent — exposes simulation design, code generation, execution,
and analysis capabilities as typed tools.

Each tool lives in its own file.  This module assembles them into a
single ``CodingAgent(BaseAgent)`` that the orchestrator discovers.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaseAgent, ToolContext, ToolDescriptor, ToolResult
from ...registry import ArtifactRegistry

from .plan_functions import TOOL as PLAN_FUNCTIONS_TOOL, execute as execute_plan_functions
from .generate_function import TOOL as GENERATE_FUNCTION_TOOL, execute as execute_generate_function
from .design_spec import TOOL as DESIGN_SPEC_TOOL, execute as execute_design_spec
from .run_sample import TOOL as RUN_SAMPLE_TOOL, execute as execute_run_sample
from .analyze_results import TOOL as ANALYZE_RESULTS_TOOL, execute as execute_analyze_results
from .run_sweep import TOOL as RUN_SWEEP_TOOL, execute as execute_run_sweep

AGENT_ID = "coding"


class CodingAgent(BaseAgent):
    """Wraps simulation lifecycle capabilities as typed tools.

    During Phase 2 this delegates to ``LocalCodingAssistant``,
    ``SimulationDesigner``, and ``RuntimeAssembleService``.
    Phase 4 will refactor internals for recursive function planning.
    """

    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        llm_config: Any = None,
    ) -> None:
        self._registry = registry
        self._llm_config = llm_config

    def agent_id(self) -> str:
        return AGENT_ID

    def tools(self) -> list[ToolDescriptor]:
        return [
            PLAN_FUNCTIONS_TOOL,
            GENERATE_FUNCTION_TOOL,
            DESIGN_SPEC_TOOL,
            RUN_SAMPLE_TOOL,
            ANALYZE_RESULTS_TOOL,
            RUN_SWEEP_TOOL,
        ]

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        dispatch = {
            PLAN_FUNCTIONS_TOOL.name: execute_plan_functions,
            GENERATE_FUNCTION_TOOL.name: execute_generate_function,
            DESIGN_SPEC_TOOL.name: execute_design_spec,
            RUN_SAMPLE_TOOL.name: execute_run_sample,
            ANALYZE_RESULTS_TOOL.name: execute_analyze_results,
            RUN_SWEEP_TOOL.name: execute_run_sweep,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            raise KeyError(f"Unknown coding tool: {tool_name}")
        return handler(context, registry=self._registry, llm_config=self._llm_config)
