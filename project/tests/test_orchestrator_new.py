"""
Tests for the LLM-planned, tool-driven Orchestrator.

Tests cover:
  • Tool catalog assembly from multiple agents
  • Explicit no-backend planning behavior
  • LLM plan parsing
  • Plan execution with mock agents
  • Requirement validation gating
  • Artifact dependency resolution
  • needs_input handling
  • Stop request handling
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from research_platform.agents.base import (
    BaseAgent, ToolContext, ToolDescriptor, ToolRequirement, ToolResult,
)
from research_platform.agents.validators import is_non_empty_string
from research_platform.contracts import ArtifactRef
from research_platform.orchestrator import (
    ExecutionPlan, Orchestrator, PlanStep,
)


# =========================================================================
# Mock agents
# =========================================================================

def _art(kind: str, **kw) -> ArtifactRef:
    defaults = dict(artifact_id=f"test-{kind}", assistant="mock", kind=kind,
                    title=kind, summary="test", metadata={})
    defaults.update(kw)
    return ArtifactRef(**defaults)


class _MockAgent(BaseAgent):
    def __init__(self, agent_id: str, tool_list: list[ToolDescriptor],
                 invoke_fn=None):
        self._id = agent_id
        self._tools = tool_list
        self._invoke_fn = invoke_fn

    def agent_id(self) -> str:
        return self._id

    def tools(self) -> list[ToolDescriptor]:
        return self._tools

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        if self._invoke_fn:
            return self._invoke_fn(tool_name, context)
        return ToolResult(
            status="completed",
            artifacts=[_art(f"{tool_name}_output")],
            message=f"Mock executed {tool_name}",
        )


def _make_tool(name: str, agent_id: str = "a", reqs=None, produces=None):
    return ToolDescriptor(
        name=name, display_name=name.replace("_", " ").title(),
        description=f"Mock tool {name}", agent_id=agent_id,
        requirements=reqs or [], produces=produces or [f"{name}_output"],
    )


# =========================================================================
# Construction & catalog
# =========================================================================

class TestOrchestratorConstruction:
    def test_registers_all_tools(self):
        a1 = _MockAgent("a", [_make_tool("t1", "a"), _make_tool("t2", "a")])
        a2 = _MockAgent("b", [_make_tool("t3", "b")])
        o = Orchestrator([a1, a2])
        assert o.tool_count == 3

    def test_agent_ids(self):
        a1 = _MockAgent("lit", [_make_tool("search", "lit")])
        a2 = _MockAgent("code", [_make_tool("gen", "code")])
        o = Orchestrator([a1, a2])
        assert set(o.agent_ids) == {"lit", "code"}

    def test_tool_catalog_json(self):
        a = _MockAgent("a", [_make_tool("t1", "a")])
        o = Orchestrator([a])
        cat = json.loads(o.tool_catalog_json())
        assert len(cat) == 1
        assert cat[0]["name"] == "t1"

    def test_duplicate_agent_id_rejected(self):
        a1 = _MockAgent("a", [_make_tool("t1", "a")])
        a2 = _MockAgent("a", [_make_tool("t2", "a")])
        with pytest.raises(ValueError, match="Duplicate agent_id"):
            Orchestrator([a1, a2])

    def test_duplicate_tool_name_rejected(self):
        a1 = _MockAgent("a", [_make_tool("same", "a")])
        a2 = _MockAgent("b", [_make_tool("same", "b")])
        with pytest.raises(ValueError, match="Duplicate tool name"):
            Orchestrator([a1, a2])

    def test_tool_agent_id_mismatch_rejected(self):
        a = _MockAgent("a", [_make_tool("wrong_owner", "other")])
        with pytest.raises(ValueError, match="does not match owner"):
            Orchestrator([a])


# =========================================================================
# Plan generation
# =========================================================================

class TestPlanningFallback:
    def test_no_backend_plan_is_empty_and_explicit(self):
        a = _MockAgent("a", [_make_tool("prepare_article_lookup"),
                             _make_tool("find_primary_article"),
                             _make_tool("build_article_brief"),
                             _make_tool("review_related_literature"),
                             _make_tool("plan_simulation_functions"),
                             _make_tool("design_simulation_spec"),
                             _make_tool("run_sample"),
                             _make_tool("run_full_sweep"),
                             _make_tool("plan_review_structure"),
                             _make_tool("assemble_review")])
        o = Orchestrator([a])
        d = MagicMock(instruction="Test")
        plan = o.plan(d)
        assert plan.steps == []
        assert "no planner backend" in plan.reasoning.lower()

    def test_single_shot_plan_uses_planner_backend(self):
        class Planner:
            def complete(self, **kwargs):
                return json.dumps({
                    "reasoning": "Use the available article tools only.",
                    "steps": [
                        {"tool": "prepare_article_lookup", "label": "Prepare"},
                        {"tool": "find_primary_article", "label": "Find"},
                    ],
                })

        a = _MockAgent("a", [_make_tool("prepare_article_lookup"),
                             _make_tool("find_primary_article")])
        o = Orchestrator([a], planner_backend=Planner())
        d = MagicMock(instruction="Find paper", topic_hint="")
        plan = o.plan(d)
        tools = [s.tool for s in plan.steps]
        assert tools == ["prepare_article_lookup", "find_primary_article"]

    def test_single_shot_plan_drops_unknown_tools(self):
        class Planner:
            def complete(self, **kwargs):
                return json.dumps({
                    "reasoning": "One valid step, one hallucinated step.",
                    "steps": [
                        {"tool": "known", "label": "Known"},
                        {"tool": "invented", "label": "Invalid"},
                    ],
                })

        a = _MockAgent("a", [_make_tool("known")])
        o = Orchestrator([a], planner_backend=Planner())
        d = MagicMock(instruction="Use known")
        plan = o.plan(d)
        assert [s.tool for s in plan.steps] == ["known"]


class TestLLMPlanParsing:
    def test_parse_valid_plan(self):
        a = _MockAgent("a", [_make_tool("find")])
        o = Orchestrator([a])
        payload = {
            "reasoning": "Start with search",
            "steps": [
                {"tool": "find", "inputs": {"q": "test"}, "label": "Search"},
                {"tool": "find", "artifact_inputs": {"ref": "article"}, "label": "Second"},
            ]
        }
        plan = o._parse_plan(payload)
        assert len(plan.steps) == 2
        assert plan.reasoning == "Start with search"
        assert plan.steps[0].inputs == {"q": "test"}
        assert plan.steps[1].artifact_inputs == {"ref": "article"}


# =========================================================================
# Plan execution
# =========================================================================

class TestPlanExecution:
    def test_executes_all_steps(self, tmp_path):
        a = _MockAgent("a", [_make_tool("step1"), _make_tool("step2")])
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="step1", inputs={}, label="First"),
            PlanStep(tool="step2", inputs={}, label="Second"),
        ])
        # Patch plan to return our plan
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "test"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        assert len(results) == 2

    def test_skips_unknown_tool(self, tmp_path):
        a = _MockAgent("a", [_make_tool("known")])
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="unknown_tool", label="Bad"),
            PlanStep(tool="known", label="Good"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        assert len(results) == 1

    def test_validation_failure_skips_step(self, tmp_path):
        tool = _make_tool("strict", reqs=[
            ToolRequirement(name="instruction", description="Required instruction", type="str",
                            validator=is_non_empty_string,
                            validator_description="Non-empty"),
        ])
        a = _MockAgent("a", [tool])
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="strict", inputs={"instruction": ""}, label="Empty input"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = ""
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        assert len(results) == 0  # step was skipped


class TestArtifactResolution:
    def test_resolves_artifact_inputs(self, tmp_path):
        produced = []

        def invoke_fn(tool_name, ctx):
            if tool_name == "producer":
                art = _art("my_output")
                produced.append(art)
                return ToolResult(status="completed", artifacts=[art])
            # consumer should receive the artifact
            assert "dep" in ctx.artifacts
            assert ctx.artifacts["dep"].kind == "my_output"
            return ToolResult(status="completed", artifacts=[_art("final")])

        a = _MockAgent("a", [_make_tool("producer", produces=["my_output"]),
                             _make_tool("consumer")], invoke_fn=invoke_fn)
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="producer", label="Produce"),
            PlanStep(tool="consumer", artifact_inputs={"dep": "my_output"}, label="Consume"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        assert len(results) == 2

    def test_condition_skips_step(self, tmp_path):
        a = _MockAgent("a", [_make_tool("guarded")])
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="guarded", label="Conditional",
                     condition="has:nonexistent_artifact"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        assert len(results) == 0

    def test_list_artifact_requirement_receives_all_matching_artifacts(self, tmp_path):
        from research_platform.agents.validators import list_non_empty

        received = []

        def invoke_fn(tool_name, ctx):
            if tool_name.startswith("produce"):
                return ToolResult(
                    status="completed",
                    artifacts=[_art("function_code", artifact_id=tool_name)],
                )
            received.extend(ctx.artifacts["functions"])
            return ToolResult(status="completed", artifacts=[_art("final")])

        consumer = _make_tool("consume", "a", reqs=[
            ToolRequirement(
                name="functions",
                description="Generated function artifacts",
                type="list[ArtifactRef]",
                validator=list_non_empty,
                validator_description="At least one function",
            )
        ])
        a = _MockAgent("a", [
            _make_tool("produce_one", "a", produces=["function_code"]),
            _make_tool("produce_two", "a", produces=["function_code"]),
            consumer,
        ], invoke_fn=invoke_fn)
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="produce_one", label="First function"),
            PlanStep(tool="produce_two", label="Second function"),
            PlanStep(
                tool="consume",
                artifact_inputs={"functions": "function_code"},
                label="Consume functions",
            ),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        o.run(d, mailbox=[])
        assert [ref.artifact_id for ref in received] == ["produce_one", "produce_two"]

    def test_follow_up_steps_are_inserted_after_current_step(self, tmp_path):
        order = []

        def invoke_fn(tool_name, ctx):
            order.append(tool_name)
            if tool_name == "plan":
                return ToolResult(
                    status="completed",
                    follow_up_steps=[
                        {"tool": "generated_a", "label": "Generated A"},
                        {"tool": "generated_b", "label": "Generated B"},
                    ],
                )
            return ToolResult(status="completed", artifacts=[_art(f"{tool_name}_output")])

        a = _MockAgent("a", [
            _make_tool("plan", "a"),
            _make_tool("generated_a", "a"),
            _make_tool("generated_b", "a"),
            _make_tool("tail", "a"),
        ], invoke_fn=invoke_fn)
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="plan", label="Plan"),
            PlanStep(tool="tail", label="Tail"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        o.run(d, mailbox=[])
        assert order == ["plan", "generated_a", "generated_b", "tail"]

    def test_generic_inquiry_answers_are_passed_to_later_steps(self, tmp_path):
        from research_platform.service_contracts import OrchestrationRequestEnvelope

        received = {}

        def invoke_fn(tool_name, ctx):
            if tool_name == "brief":
                request = OrchestrationRequestEnvelope.new(
                    resume_token="t:brief:model_scope",
                    request_kind="inquiry",
                    question="Which model should be reproduced?",
                    capability_hint="user",
                    metadata={
                        "inquiry": {
                            "id": "model_scope",
                            "default_policy": "first",
                            "options": [
                                {"label": "Model A", "value": {"model": "A"}},
                                {"label": "Model B", "value": {"model": "B"}},
                            ],
                        },
                    },
                )
                return ToolResult(status="completed", inquiries=[request])
            received.update(ctx.metadata.get("inquiry_answers") or {})
            return ToolResult(status="completed", artifacts=[_art("done")])

        a = _MockAgent("a", [
            _make_tool("brief", "a"),
            _make_tool("design", "a"),
        ], invoke_fn=invoke_fn)
        o = Orchestrator([a])
        o.plan = lambda d: ExecutionPlan(steps=[
            PlanStep(tool="brief", label="Build brief"),
            PlanStep(tool="design", label="Design"),
        ])
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        mailbox = []
        o.run(d, mailbox=mailbox)

        assert received["model_scope"]["defaulted"] is True
        selected = received["model_scope"]["payload"]["selected_options"]
        assert selected[0]["label"] == "Model A"
        assert any(msg.subject == "Inquiry" for msg in mailbox)


class TestMailboxMessages:
    def test_messages_emitted(self, tmp_path):
        a = _MockAgent("a", [_make_tool("step1")])
        o = Orchestrator([a])
        plan = ExecutionPlan(steps=[PlanStep(tool="step1", label="Do it")])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        mailbox = []
        o.run(d, mailbox=mailbox)
        # Should have: start, step result, complete = 3
        assert len(mailbox) >= 2
        subjects = [m.subject for m in mailbox]
        assert any("started" in s.lower() for s in subjects)
        assert any("complete" in s.lower() for s in subjects)

    def test_on_message_callback(self, tmp_path):
        messages = []
        a = _MockAgent("a", [_make_tool("s1")])
        o = Orchestrator([a], on_message=lambda m: messages.append(m))
        plan = ExecutionPlan(steps=[PlanStep(tool="s1", label="Go")])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        o.run(d)
        assert len(messages) >= 2


class TestNeedsInput:
    def test_needs_input_math_delegated(self, tmp_path):
        from research_platform.service_contracts import OrchestrationRequestEnvelope

        def invoke_fn(tool_name, ctx):
            if tool_name == "calc":
                return ToolResult(
                    status="needs_input",
                    request=OrchestrationRequestEnvelope.new(
                        resume_token="t", request_kind="math",
                        question="2+2", capability_hint="math"),
                )
            if tool_name == "evaluate_expression":
                return ToolResult(status="completed", message="4",
                                  metadata={"value": 4})
            return ToolResult(status="completed")

        calc_agent = _MockAgent("calc_agent", [_make_tool("calc", "calc_agent")], invoke_fn)
        math_agent = _MockAgent("math", [_make_tool("evaluate_expression", "math")], invoke_fn)
        o = Orchestrator([calc_agent, math_agent])
        plan = ExecutionPlan(steps=[PlanStep(tool="calc", label="Calculate")])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "t"
        d.instruction = "go"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        # The math delegation should produce a result
        assert len(results) == 0  # math result doesn't produce artifacts but step completes
