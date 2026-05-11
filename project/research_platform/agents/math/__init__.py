"""
research_platform.agents.math
─────────────────────────────
Math agent — wraps MathAgentService as typed tools.
"""
from __future__ import annotations

from typing import Any, Optional

from ..base import BaseAgent, ToolContext, ToolDescriptor, ToolRequirement, ToolResult
from ..validators import is_non_empty_string, is_positive_number, list_non_empty, number_in_range
from ...math_agent import MathAgentService

AGENT_ID = "math"

# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------

EVALUATE_TOOL = ToolDescriptor(
    name="evaluate_expression", display_name="Evaluate Expression",
    description="Safely evaluate a mathematical expression with optional variables.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="expression", description="Math expression string", type="str",
                        validator=is_non_empty_string, validator_description="Non-empty"),
        ToolRequirement(name="variables", description="Variable bindings dict", type="dict", required=False, default={}),
    ],
    produces=[], tags=["math", "evaluate"], idempotent=True, estimated_seconds=1.0,
)

SOLVE_EQUATION_TOOL = ToolDescriptor(
    name="solve_equation", display_name="Solve Equation",
    description="Find roots of an algebraic equation using scipy root_scalar.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="equation", description="Equation string with '='", type="str",
                        validator=is_non_empty_string, validator_description="Non-empty"),
        ToolRequirement(name="variable", description="Variable to solve for", type="str", required=False, default="x"),
    ],
    produces=[], tags=["math", "algebra"], idempotent=True, estimated_seconds=2.0,
)

SOLVE_ODE_TOOL = ToolDescriptor(
    name="solve_ode", display_name="Solve ODE",
    description="Numerically solve an ODE system via scipy solve_ivp.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="derivative", description="Derivative expression string", type="str",
                        validator=is_non_empty_string, validator_description="Non-empty"),
        ToolRequirement(name="t_span", description="[t_start, t_end]", type="list[float]",
                        validator=list_non_empty, validator_description="Non-empty t_span"),
        ToolRequirement(name="y0", description="Initial state values", type="list[float]",
                        validator=list_non_empty, validator_description="Non-empty y0"),
    ],
    produces=[], tags=["math", "ode"], idempotent=True, estimated_seconds=5.0,
)

VERIFY_NUMERIC_TOOL = ToolDescriptor(
    name="verify_numeric_claim", display_name="Verify Numeric Claim",
    description="Check if an expression equals a claimed value within tolerance.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="expression", description="Math expression", type="str",
                        validator=is_non_empty_string, validator_description="Non-empty"),
        ToolRequirement(name="claimed_value", description="The claimed numeric result", type="float"),
        ToolRequirement(name="tolerance", description="Acceptable error", type="float",
                        required=False, default=1e-4),
    ],
    produces=[], tags=["math", "verification"], idempotent=True, estimated_seconds=1.0,
)

VERIFY_STAT_TOOL = ToolDescriptor(
    name="verify_statistical_claim", display_name="Verify Statistical Claim",
    description="Recompute and verify a statistical test result.",
    agent_id=AGENT_ID,
    requirements=[
        ToolRequirement(name="test", description="Test name (ttest_ind, ttest_1samp)", type="str",
                        validator=is_non_empty_string, validator_description="Non-empty"),
        ToolRequirement(name="sample_a", description="First sample", type="list[float]",
                        validator=list_non_empty, validator_description="Non-empty sample"),
        ToolRequirement(name="sample_b", description="Second sample", type="list[float]", required=False),
        ToolRequirement(name="claimed_p_value", description="Claimed p-value", type="float", required=False),
    ],
    produces=[], tags=["math", "statistics", "verification"], idempotent=True, estimated_seconds=2.0,
)

ALL_TOOLS = [EVALUATE_TOOL, SOLVE_EQUATION_TOOL, SOLVE_ODE_TOOL, VERIFY_NUMERIC_TOOL, VERIFY_STAT_TOOL]


class MathAgent(BaseAgent):
    """Wraps MathAgentService methods as typed tools."""

    def __init__(self, service: Optional[MathAgentService] = None) -> None:
        self._service = service or MathAgentService()

    def agent_id(self) -> str:
        return AGENT_ID

    def tools(self) -> list[ToolDescriptor]:
        return list(ALL_TOOLS)

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        dispatch = {
            "evaluate_expression": self._evaluate,
            "solve_equation": self._solve_equation,
            "solve_ode": self._solve_ode,
            "verify_numeric_claim": self._verify_numeric,
            "verify_statistical_claim": self._verify_stat,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            raise KeyError(f"Unknown math tool: {tool_name}")
        return handler(context)

    def _evaluate(self, ctx: ToolContext) -> ToolResult:
        result = self._service.evaluate_expression(
            ctx.inputs["expression"], variables=ctx.inputs.get("variables"))
        return self._to_result(result)

    def _solve_equation(self, ctx: ToolContext) -> ToolResult:
        result = self._service.solve_equation(
            ctx.inputs["equation"], variable=ctx.inputs.get("variable", "x"))
        return self._to_result(result)

    def _solve_ode(self, ctx: ToolContext) -> ToolResult:
        result = self._service.solve_ode(
            ctx.inputs["derivative"], t_span=ctx.inputs["t_span"], y0=ctx.inputs["y0"])
        return self._to_result(result)

    def _verify_numeric(self, ctx: ToolContext) -> ToolResult:
        result = self._service.verify_numeric_claim(
            expression=ctx.inputs["expression"],
            claimed_value=ctx.inputs["claimed_value"],
            tolerance=ctx.inputs.get("tolerance", 1e-4))
        return self._to_result(result)

    def _verify_stat(self, ctx: ToolContext) -> ToolResult:
        result = self._service.verify_statistical_claim(
            test=ctx.inputs["test"],
            sample_a=ctx.inputs["sample_a"],
            sample_b=ctx.inputs.get("sample_b"),
            claimed_p_value=ctx.inputs.get("claimed_p_value"))
        return self._to_result(result)

    @staticmethod
    def _to_result(payload: dict) -> ToolResult:
        status = payload.get("status", "uncertain")
        ok = status in ("ok", "verified")
        return ToolResult(
            status="completed" if ok else "failed",
            message=payload.get("reason", str(payload)),
            metadata=payload,
        )
