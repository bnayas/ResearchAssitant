from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from comms.pi_email import PIMailbox
from research_platform.flow_management import FlowManagementService
from research_platform.math_agent import MathAgentService
from research_platform.mediated_workflow import MediatorWorkflowRunner
from research_platform.mediated_services import SimulationDesignerService
from research_platform.registry import ArtifactRegistry
from research_platform.runtime_assemble import RuntimeAssembleService
from research_platform.service_contracts import AgentRuntimeProfile, MediatedResponse, OrchestrationRequestEnvelope
from research_platform.contracts import AssistantId, TaskEnvelope


def test_flow_management_injects_context_immediately():
    flow = FlowManagementService()
    snapshot = flow.open_workflow("wf-1")
    assert snapshot["workflow_id"] == "wf-1"

    flow.register_session("wf-1", "literature_agent", "sess-1")

    received = {}

    def _waiter():
        injection = flow.wait_for_injection("sess-1", timeout=2.0)
        received["mode"] = injection.mode
        received["payload"] = injection.payload

    worker = threading.Thread(target=_waiter)
    worker.start()
    flow.inject_context("sess-1", payload={"feedback": "Refine the search."}, mode="hard")
    worker.join(timeout=3.0)

    assert received["mode"] == "hard"
    assert received["payload"]["feedback"] == "Refine the search."


def test_math_agent_supports_deterministic_and_nl_entrypoints():
    math = MathAgentService()

    deterministic = math.evaluate_expression("2 + 3 * 4")
    assert deterministic["status"] == "ok"
    assert deterministic["value"] == 14.0

    verified = math.verify_numeric_claim(expression="2 + 2", claimed_value=4)
    assert verified["status"] == "verified"

    nl = math.verify_from_text("2 + 2 = 4")
    assert nl["status"] == "verified"


def test_simulation_designer_service_runs_through_mediated_runtime(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    designer = SimulationDesignerService(
        registry,
        profile=AgentRuntimeProfile(provider="mock"),
    )
    runtime = RuntimeAssembleService(registry, output_root=tmp_path / "simulation_runtime")
    task = TaskEnvelope(
        task_id="sim-task-1",
        directive_id="dir-1",
        assistant=AssistantId.CODING_AGENT.value,
        subject="Design a simulation",
        instructions="Study the 2D Ising model with a short offline mock setup.",
        output_dir=str(tmp_path),
        metadata={"sample_steps": 80},
    )

    response = designer.start_simulation(task, sources=[])
    assert response.request is not None
    assert response.request.metadata["action"] == "materialize_runtime"

    bundle = runtime.materialize_runtime(
        session_id=response.session_id,
        spec_payload=response.request.metadata["spec_payload"],
        output_dir=response.request.metadata["output_dir"],
    )
    response = designer.resume_simulation(
        response.session_id,
        {"kind": "runtime_response", "action": "materialize_runtime", "bundle": bundle},
    )
    assert response.request is not None
    assert response.request.metadata["action"] == "run_sample"

    sample = runtime.run_sample(
        session_id=response.session_id,
        spec_payload=response.request.metadata["spec_payload"],
        bundle=response.request.metadata["bundle"],
        sample_steps=80,
    )
    response = designer.resume_simulation(
        response.session_id,
        {"kind": "runtime_response", "action": "run_sample", "diagnostics": sample.to_dict()},
    )
    assert response.request is not None
    assert response.request.metadata["action"] == "run_sweep"

    sweep = runtime.run_sweep(
        session_id=response.session_id,
        spec_payload=response.request.metadata["spec_payload"],
        bundle=response.request.metadata["bundle"],
    )
    response = designer.resume_simulation(
        response.session_id,
        {"kind": "runtime_response", "action": "run_sweep", "diagnostics": sweep.to_dict()},
    )
    assert response.status == "completed"
    assert response.artifact_ids


def test_mediated_workflow_routes_agent_question_answers_back_to_same_session(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    flow = FlowManagementService()

    class _ClarifyingLiterature:
        def __init__(self):
            self.resume_payload = None
            self.find_task = None

        def prepare_article_lookup_spec(self, task):
            return MediatedResponse(
                status="needs_request",
                session_id="lookup-session",
                request=OrchestrationRequestEnvelope.new(
                    resume_token="lookup-session",
                    request_kind="clarification",
                    question="Which publication window should I use?",
                    capability_hint="user",
                    expected_schema={"answers": {"clarification": "string"}},
                ),
            )

        def resume_literature_task(self, session_id, payload):
            self.resume_payload = payload
            return MediatedResponse(
                status="completed",
                session_id=session_id,
                payload={
                    "query_string": "refined query",
                    "query_terms": ["refined query"],
                    "required_authors": [],
                    "preferred_year": None,
                    "year_min": 2020,
                    "year_max": None,
                    "title_phrases": [],
                    "needs_clarification": False,
                    "clarification_question": "",
                },
            )

        def find_primary_article(self, task):
            self.find_task = task
            article = registry.save_json(
                assistant=AssistantId.LITERATURE_REVIEWER.value,
                kind="primary_article",
                title="Primary Article Match",
                filename="article/article_match.json",
                payload={"title": "Example Article", "authors": ["A. Researcher"], "year": 2021},
                summary="Example Article",
                metadata={"title": "Example Article", "authors": ["A. Researcher"], "year": 2021},
                artifact_id="article-match",
            )
            return MediatedResponse(status="completed", session_id=task.task_id, payload={"artifact_id": article.artifact_id})

    literature = _ClarifyingLiterature()
    directive = SimpleNamespace(
        directive_id="dir-question-routing",
        pi_name="Professor",
        instruction="Find the target paper and prepare for reproduction.",
        topic_hint="",
        phases=["find_article"],
        output_dir=str(tmp_path),
    )
    runner = MediatorWorkflowRunner(
        directive,
        mailbox,
        artifact_registry=registry,
        flow_management=flow,
        literature_agent=literature,
        simulation_designer=object(),
        runtime_assemble=object(),
        math_agent=object(),
        writer_agent=object(),
    )

    worker = threading.Thread(target=runner.run_all_phases, daemon=True)
    worker.start()
    pending = _wait_for_pending_steering(flow, directive.directive_id)
    assert pending["kind"] == "agent_request"
    assert pending["session_id"] == "lookup-session"
    assert pending["expected_schema"]["answers"]["clarification"] == "string"

    flow.inject_context(
        "lookup-session",
        payload={"answers": {"clarification": "Use articles from 2020 onward."}},
        injected_by="test",
    )
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert literature.resume_payload == {
        "kind": "clarification_answers",
        "answers": {"clarification": "Use articles from 2020 onward."},
    }
    assert literature.find_task.metadata["article_lookup_query"] == "refined query"


def test_mediated_workflow_asks_for_generic_simulation_target_selection(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    flow = FlowManagementService()

    class _TargetLiterature:
        def prepare_article_lookup_spec(self, task):
            return MediatedResponse(status="completed", session_id=task.task_id, payload={"query_string": "example", "query_terms": ["example"]})

        def resume_literature_task(self, session_id, payload):
            return MediatedResponse(status="failed", session_id=session_id, error="unexpected resume")

        def find_primary_article(self, task):
            article = registry.save_json(
                assistant=AssistantId.LITERATURE_REVIEWER.value,
                kind="primary_article",
                title="Primary Article Match",
                filename="article/article_match.json",
                payload={"title": "Example Article", "authors": ["A. Researcher"], "year": 2021},
                summary="Example Article",
                metadata={"title": "Example Article", "authors": ["A. Researcher"], "year": 2021},
                artifact_id="article-match",
            )
            return MediatedResponse(status="completed", session_id=task.task_id, payload={"artifact_id": article.artifact_id})

        def build_article_brief(self, task, article):
            brief = registry.save_json(
                assistant=AssistantId.LITERATURE_REVIEWER.value,
                kind="article_brief",
                title="Article Brief",
                filename="article/article_brief.json",
                payload={
                    "model_description": "Two reproducible variants are described.",
                    "simulation_targets": [
                        {"name": "First target", "description": "Baseline variant."},
                        {"name": "Second target", "description": "Extended variant."},
                    ],
                    "key_parameters": {},
                    "procedure": "Use the reported procedure.",
                    "expected_figures": [],
                },
                summary="Brief",
                metadata={
                    "model_description": "Two reproducible variants are described.",
                    "simulation_targets": [
                        {"name": "First target", "description": "Baseline variant."},
                        {"name": "Second target", "description": "Extended variant."},
                    ],
                },
                artifact_id="article-brief",
            )
            return MediatedResponse(status="completed", session_id=task.task_id, payload={"artifact_id": brief.artifact_id})

        def review_related_literature(self, task, article, brief):
            review = registry.save_json(
                assistant=AssistantId.LITERATURE_REVIEWER.value,
                kind="literature_review",
                title="Related Literature Review",
                filename="literature/literature_review.json",
                payload={},
                summary="No related review needed for this test.",
                metadata={"accepted_count": 0},
                artifact_id="literature-review",
            )
            return MediatedResponse(status="completed", session_id=task.task_id, payload={"artifact_id": review.artifact_id})

    class _RecordingSimulation:
        def __init__(self):
            self.task = None

        def start_simulation(self, task, *, sources):
            self.task = task
            artifact = registry.save_json(
                assistant=AssistantId.CODING_AGENT.value,
                kind="simulation_analysis",
                title="Simulation Analysis",
                filename="simulation/analysis.json",
                payload={"status": "ok"},
                summary="Simulation completed.",
                artifact_id="simulation-analysis",
            )
            return MediatedResponse(
                status="completed",
                session_id=task.task_id,
                payload={},
                artifact_ids=[artifact.artifact_id],
            )

        def resume_simulation(self, session_id, payload):
            return MediatedResponse(status="failed", session_id=session_id, error="unexpected resume")

    simulation = _RecordingSimulation()
    directive = SimpleNamespace(
        directive_id="dir-target-selection",
        pi_name="Professor",
        instruction="Reproduce the selected paper results.",
        topic_hint="",
        phases=["simulation"],
        output_dir=str(tmp_path),
    )
    runner = MediatorWorkflowRunner(
        directive,
        mailbox,
        artifact_registry=registry,
        flow_management=flow,
        literature_agent=_TargetLiterature(),
        simulation_designer=simulation,
        runtime_assemble=object(),
        math_agent=object(),
        writer_agent=object(),
    )

    worker = threading.Thread(target=runner.run_all_phases, daemon=True)
    worker.start()
    pending = _wait_for_pending_steering(flow, directive.directive_id)
    assert pending["request_kind"] == "selection"
    assert pending["session_id"] == "dir-target-selection-simulation-scope"

    flow.inject_context(
        pending["session_id"],
        payload={"answers": {"selection": "2"}},
        injected_by="test",
    )
    worker.join(timeout=5)
    assert not worker.is_alive()
    selected = simulation.task.metadata["selected_simulation_targets"]
    assert len(selected) == 1
    assert selected[0]["name"] == "Second target"


def _wait_for_pending_steering(flow: FlowManagementService, workflow_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            snapshot = flow.get_workflow_snapshot(workflow_id)
        except KeyError:
            time.sleep(0.02)
            continue
        for session in reversed(snapshot["sessions"]):
            metadata = session.get("metadata") or {}
            steering = metadata.get("steering")
            if session.get("state") == "waiting" and isinstance(steering, dict):
                return steering
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for pending steering")
