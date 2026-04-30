from __future__ import annotations

import threading

from research_platform.flow_management import FlowManagementService
from research_platform.math_agent import MathAgentService
from research_platform.mediated_services import SimulationDesignerService
from research_platform.registry import ArtifactRegistry
from research_platform.runtime_assemble import RuntimeAssembleService
from research_platform.service_contracts import AgentRuntimeProfile
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
