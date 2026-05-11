from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research_platform.agents.base import (
    BaseAgent,
    ToolContext,
    ToolDescriptor,
    ToolResult,
)
from research_platform.contracts import ArtifactRef
from research_platform.flow_management import FlowManagementService
from research_platform.orchestrator import Orchestrator
from research_platform.planning import DeepPlanner, SteeringRouter


def _tool(name: str) -> dict:
    return {
        "name": name,
        "display_name": name,
        "description": f"Tool {name}",
        "agent_id": "mock",
        "requirements": [],
        "produces": [f"{name}_output"],
        "tags": [],
        "idempotent": True,
        "estimated_seconds": 1.0,
    }


class StructuredPlannerBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["system"]
        if "produce a structured list of questions" in system:
            return (
                "planner notes\n"
                + json.dumps({
                    "directive_summary": "Run a simple mock workflow.",
                    "tool_catalog_digest": "known",
                    "steps": [
                        {
                            "step_id": "q1",
                            "question": "Which tool should run first?",
                            "rationale": "Selects the executable step.",
                            "context_keys": ["known", "invented"],
                            "priority": 0,
                        }
                    ],
                })
            )
        if "resolving one planning question" in system:
            return json.dumps({
                "step_id": "q1",
                "answer": "Use the known tool.",
                "confidence": 0.95,
                "assumptions": [],
                "constraints": ["use known"],
                "flags": [],
            })
        return json.dumps({
            "reasoning": "Known is the only catalog tool.",
            "steps": [
                {"tool": "known", "label": "Known step"},
                {"tool": "invented", "label": "Drop this"},
            ],
        })


def test_deep_planner_filters_context_and_unknown_plan_tools():
    backend = StructuredPlannerBackend()
    directive = SimpleNamespace(instruction="Do the mock workflow", topic_hint="")
    result = DeepPlanner(backend=backend).plan(directive, [_tool("known")])

    assert [step.tool for step in result.plan.steps] == ["known"]
    assert result.agenda.steps[0].context_keys == ["known"]
    assert result.aggregated_constraints == ["use known"]
    assert all("max_tokens" not in call for call in backend.calls)


def test_steering_router_skip_shortcut_targets_plan_step():
    plan = SimpleNamespace(steps=[
        SimpleNamespace(tool="prepare_article_lookup", label="Prepare"),
        SimpleNamespace(tool="design_simulation_spec", label="Design simulation"),
    ])
    match = SteeringRouter().route(
        "skip to simulation",
        plan,
        current_step_idx=0,
        completed_artifacts={},
    )

    assert match is not None
    assert match.is_actionable
    assert match.steps_to_skip_until == 1


def test_steering_router_returns_none_for_unmatched_text():
    plan = SimpleNamespace(steps=[SimpleNamespace(tool="known", label="Known")])
    assert SteeringRouter().route(
        "please make the output more concise",
        plan,
        current_step_idx=0,
        completed_artifacts={},
    ) is None


class _Agent(BaseAgent):
    def agent_id(self) -> str:
        return "mock"

    def tools(self) -> list[ToolDescriptor]:
        return [
            ToolDescriptor(
                name="known",
                display_name="Known",
                description="Run the known mock step.",
                agent_id="mock",
                requirements=[],
                produces=["known_output"],
            )
        ]

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        return ToolResult(
            status="completed",
            artifacts=[
                ArtifactRef(
                    artifact_id="known-1",
                    assistant="mock",
                    kind="known_output",
                    title="Known",
                    metadata={"step": context.metadata.get("step_index")},
                )
            ],
            message="done",
        )


def test_orchestrator_deep_planning_runs_and_emits_trace(tmp_path: Path):
    backend = StructuredPlannerBackend()
    orchestrator = Orchestrator([_Agent()], planner_backend=backend)
    directive = SimpleNamespace(
        directive_id="deep",
        instruction="Run known",
        topic_hint="",
        output_dir=str(tmp_path),
    )
    mailbox = []

    results = orchestrator.run(directive, mailbox=mailbox)

    assert [ref.kind for ref in results] == ["known_output"]
    assert any(msg.subject == "Planning complete" for msg in mailbox)
    assert any(msg.metadata.get("kind") == "deep_plan_trace" for msg in mailbox)


def test_orchestrator_registers_flow_session_before_updates(tmp_path: Path):
    class Planner:
        def complete(self, **kwargs):
            return json.dumps({
                "reasoning": "Run one step.",
                "steps": [{"tool": "known", "label": "Known"}],
            })

    flow = FlowManagementService()
    orchestrator = Orchestrator([_Agent()], planner_backend=Planner(), flow_management=flow)
    directive = SimpleNamespace(
        directive_id="flow",
        instruction="Run known",
        topic_hint="",
        output_dir=str(tmp_path),
    )

    results = orchestrator.run(directive, mailbox=[], deep=False)
    snapshot = flow.get_workflow_snapshot("flow")

    assert [ref.kind for ref in results] == ["known_output"]
    assert any(session["session_id"] == "flow:orchestrator" for session in snapshot["sessions"])
