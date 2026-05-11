"""
Integration tests: wire real agent instances into the Orchestrator
and run with mock LLM backends.

These tests verify the full wiring — agent construction, tool catalog
assembly, requirement validation, and dispatch — without real APIs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from research_platform.agents.base import ToolContext, ToolResult
from research_platform.agents.coding import CodingAgent
from research_platform.agents.literature import LiteratureAgent
from research_platform.agents.math import MathAgent
from research_platform.agents.runtime import RuntimeAgent
from research_platform.agents.writer import WriterAgent
from research_platform.contracts import ArtifactRef
from research_platform.orchestrator import ExecutionPlan, Orchestrator, PlanStep
from research_platform.registry import ArtifactRegistry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(root_dir=tmp_path)


@pytest.fixture
def all_agents(registry: ArtifactRegistry) -> list:
    mock_backend = MagicMock()
    return [
        LiteratureAgent(registry, llm_backend=mock_backend),
        CodingAgent(registry, llm_config=mock_backend),
        WriterAgent(registry, llm_backend=mock_backend),
        MathAgent(),
        RuntimeAgent(registry),
    ]


@pytest.fixture
def orchestrator(all_agents: list) -> Orchestrator:
    return Orchestrator(all_agents)


# =========================================================================
# Full catalog wiring
# =========================================================================

class TestFullCatalogWiring:
    def test_all_23_tools_registered(self, orchestrator: Orchestrator):
        assert orchestrator.tool_count == 23

    def test_all_5_agents_registered(self, orchestrator: Orchestrator):
        assert set(orchestrator.agent_ids) == {"literature", "coding", "writer", "math", "runtime"}

    def test_catalog_json_valid_and_complete(self, orchestrator: Orchestrator):
        catalog = json.loads(orchestrator.tool_catalog_json())
        assert len(catalog) == 23
        names = {t["name"] for t in catalog}
        # Spot-check key tools
        assert "find_primary_article" in names
        assert "parse_article_with_intention" in names
        assert "generate_function" in names
        assert "evaluate_expression" in names
        assert "materialize_runtime" in names
        assert "draft_section" in names

    def test_every_tool_has_requirements_and_produces(self, orchestrator: Orchestrator):
        catalog = json.loads(orchestrator.tool_catalog_json())
        for tool in catalog:
            assert "requirements" in tool, f"{tool['name']} missing requirements"
            assert isinstance(tool["requirements"], list)
            assert "produces" in tool, f"{tool['name']} missing produces"

    def test_no_duplicate_tool_names(self, orchestrator: Orchestrator):
        catalog = json.loads(orchestrator.tool_catalog_json())
        names = [t["name"] for t in catalog]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


# =========================================================================
# Math agent live invocation
# =========================================================================

class TestMathAgentLiveInvocation:
    """Math agent doesn't need LLM — test real execution through orchestrator."""

    def test_evaluate_through_orchestrator(self, orchestrator: Orchestrator, tmp_path):
        plan = ExecutionPlan(steps=[
            PlanStep(
                tool="evaluate_expression",
                inputs={"expression": "2 ** 10"},
                label="Compute 2^10",
            ),
        ])
        orchestrator.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "math-test"
        d.instruction = "compute"
        d.output_dir = str(tmp_path)
        mailbox = []
        results = orchestrator.run(d, mailbox=mailbox)
        # Math doesn't produce artifacts but step completes
        assert any("complete" in m.subject.lower() for m in mailbox)

    def test_verify_numeric_through_orchestrator(self, orchestrator: Orchestrator, tmp_path):
        plan = ExecutionPlan(steps=[
            PlanStep(
                tool="verify_numeric_claim",
                inputs={"expression": "3 * 7", "claimed_value": 21.0},
                label="Verify 3*7=21",
            ),
        ])
        orchestrator.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "verify-test"
        d.instruction = "verify"
        d.output_dir = str(tmp_path)
        mailbox = []
        orchestrator.run(d, mailbox=mailbox)
        step_msgs = [m for m in mailbox if "verify" in m.subject.lower() or "✓" in m.subject]
        assert len(step_msgs) >= 1


# =========================================================================
# Planner-generated plan with real agents
# =========================================================================

class TestPlannerPlanWithRealAgents:
    def test_planner_plan_lit_only(self, all_agents: list):
        class Planner:
            def complete(self, **kwargs):
                return json.dumps({
                    "reasoning": "Search then parse the requested article.",
                    "steps": [
                        {"tool": "prepare_article_lookup", "label": "Prepare lookup"},
                        {"tool": "find_primary_article", "label": "Find article"},
                        {"tool": "parse_article_with_intention", "label": "Parse article"},
                    ],
                })

        orchestrator = Orchestrator(all_agents, planner_backend=Planner())
        d = MagicMock()
        d.instruction = "Find the Ising model paper"
        d.topic_hint = "Ising"
        d.phases = ["find_article", "parse_article"]
        plan = orchestrator.plan(d)
        tools_in_plan = [s.tool for s in plan.steps]
        assert "prepare_article_lookup" in tools_in_plan
        assert "find_primary_article" in tools_in_plan
        assert "parse_article_with_intention" in tools_in_plan
        # Should NOT include simulation tools
        assert "run_sample" not in tools_in_plan

    def test_planner_plan_full_pipeline(self, all_agents: list):
        class Planner:
            def complete(self, **kwargs):
                return json.dumps({
                    "reasoning": "Use the full available pipeline.",
                    "steps": [
                        {"tool": "prepare_article_lookup", "label": "Prepare lookup"},
                        {"tool": "find_primary_article", "label": "Find article"},
                        {"tool": "parse_article_with_intention", "label": "Parse article"},
                        {"tool": "review_related_literature", "label": "Review literature"},
                        {"tool": "plan_simulation_functions", "label": "Plan functions"},
                        {"tool": "generate_function", "label": "Generate function"},
                        {"tool": "design_simulation_spec", "label": "Design spec"},
                        {"tool": "run_sample", "label": "Run sample"},
                        {"tool": "plan_review_structure", "label": "Plan writing"},
                        {"tool": "assemble_review", "label": "Assemble"},
                    ],
                })

        orchestrator = Orchestrator(all_agents, planner_backend=Planner())
        d = MagicMock()
        d.instruction = "Full pipeline"
        d.topic_hint = ""
        d.phases = ["find_article", "parse_article", "literature", "simulation", "write"]
        plan = orchestrator.plan(d)
        tools_in_plan = [s.tool for s in plan.steps]
        assert "prepare_article_lookup" in tools_in_plan
        assert "design_simulation_spec" in tools_in_plan
        assert "plan_review_structure" in tools_in_plan
        assert len(plan.steps) >= 8


# =========================================================================
# Multi-step with artifact chaining
# =========================================================================

class TestArtifactChainingIntegration:
    def test_chained_steps_resolve_artifacts(self, registry, tmp_path):
        """Two mock agents where step 2 depends on step 1's output."""
        from research_platform.agents.base import BaseAgent, ToolDescriptor

        invocations = []

        class ProducerAgent(BaseAgent):
            def agent_id(self): return "producer"
            def tools(self):
                return [ToolDescriptor(
                    name="produce", display_name="Produce", description="Produce data",
                    agent_id="producer", requirements=[], produces=["data_output"],
                )]
            def invoke(self, tool_name, ctx):
                invocations.append(("produce", ctx))
                return ToolResult(
                    status="completed",
                    artifacts=[ArtifactRef(
                        artifact_id="data-1", assistant="producer",
                        kind="data_output", title="Data", metadata={"value": 42},
                    )],
                )

        class ConsumerAgent(BaseAgent):
            def agent_id(self): return "consumer"
            def tools(self):
                return [ToolDescriptor(
                    name="consume", display_name="Consume", description="Consume data",
                    agent_id="consumer",
                    requirements=[],
                    produces=["final_output"],
                )]
            def invoke(self, tool_name, ctx):
                invocations.append(("consume", ctx))
                data = ctx.artifacts.get("data")
                return ToolResult(
                    status="completed",
                    artifacts=[ArtifactRef(
                        artifact_id="final-1", assistant="consumer",
                        kind="final_output", title="Final",
                        metadata={"received": data.metadata["value"] if data else None},
                    )],
                )

        o = Orchestrator([ProducerAgent(), ConsumerAgent()])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="produce", label="Generate data"),
            PlanStep(tool="consume", artifact_inputs={"data": "data_output"}, label="Process data"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "chain"
        d.instruction = "chain"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])

        assert len(results) == 2
        assert invocations[0][0] == "produce"
        assert invocations[1][0] == "consume"
        # Consumer should have received the artifact
        consumer_ctx = invocations[1][1]
        assert "data" in consumer_ctx.artifacts
        assert consumer_ctx.artifacts["data"].metadata["value"] == 42


# =========================================================================
# Error resilience
# =========================================================================

class TestErrorResilience:
    def test_agent_exception_doesnt_crash_orchestrator(self, tmp_path):
        from research_platform.agents.base import BaseAgent, ToolDescriptor

        class FailAgent(BaseAgent):
            def agent_id(self): return "fail"
            def tools(self):
                return [ToolDescriptor(
                    name="boom", display_name="Boom", description="Always fails",
                    agent_id="fail", requirements=[], produces=[],
                )]
            def invoke(self, tool_name, ctx):
                raise RuntimeError("Agent crashed!")

        class OkAgent(BaseAgent):
            def agent_id(self): return "ok"
            def tools(self):
                return [ToolDescriptor(
                    name="safe", display_name="Safe", description="Always works",
                    agent_id="ok", requirements=[], produces=["ok_output"],
                )]
            def invoke(self, tool_name, ctx):
                return ToolResult(status="completed", artifacts=[
                    ArtifactRef(artifact_id="ok-1", assistant="ok",
                                kind="ok_output", title="OK", metadata={}),
                ])

        o = Orchestrator([FailAgent(), OkAgent()])
        plan = ExecutionPlan(steps=[
            PlanStep(tool="boom", label="This will fail"),
            PlanStep(tool="safe", label="This should still run"),
        ])
        o.plan = lambda d: plan
        d = MagicMock()
        d.directive_id = "resilience"
        d.instruction = "test"
        d.output_dir = str(tmp_path)
        results = o.run(d, mailbox=[])
        # Only the safe step produces artifacts
        assert len(results) == 1
        assert results[0].kind == "ok_output"
