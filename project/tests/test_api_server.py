from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sim_tool.api_server import _http_error, _jsonify, create_app
from sim_tool.contract import ClarificationQuestion, ClarificationRequest, QuestionTopic
from sim_tool.orchestrator import OrchestratorState, OrchestratorUpdate


class DummySimulationManager:
    def start(self, request):
        return OrchestratorUpdate(
            session_id="sess-123",
            state=OrchestratorState.CLARIFYING,
            message="Designer needs clarification.",
            next_action="answer questions",
            clarification=ClarificationRequest(
                session_id="sess-123",
                iteration=1,
                questions=[
                    ClarificationQuestion(
                        index=1,
                        text="What temperature range should be swept?",
                        topic=QuestionTopic.SWEEP_TARGET,
                    )
                ],
            ),
        )


class DummyLiteratureService:
    def run(self, request):
        return {"artifact": {"status": "complete"}, "audit": {"passed": True}}


class DummyDirectiveService:
    def run_stream(self, request):
        async def _gen():
            yield '{"type":"status","phase":"complete","message":"ok"}\n'
        return _gen()

    def submit_steering(self, directive_id, request):
        return {
            "directive_id": directive_id,
            "checkpoint_id": request.checkpoint_id,
            "action": request.action,
            "feedback": request.feedback,
        }

    def stop(self, directive_id, reason=""):
        return {
            "directive_id": directive_id,
            "status": "stop_requested",
            "reason": reason,
        }

    def snapshot(self, directive_id):
        return {
            "directive_id": directive_id,
            "done": False,
            "error": None,
            "emails": [{"subject": "Instruction accepted — workflow initiated", "timestamp": 1}],
            "pending_steering": None,
            "stop_requested": False,
            "stop_reason": "",
        }


def test_jsonify_serializes_paths_and_enums():
    payload = _jsonify(
        OrchestratorUpdate(
            session_id="sess-1",
            state=OrchestratorState.READY_TO_SAMPLE,
            message="ok",
            next_action="run sample",
        )
    )
    assert payload["state"] == "ready_to_sample"
    assert payload["rendered"]

    path_payload = _jsonify({"script_path": Path("/tmp/example.py")})
    assert path_payload["script_path"] == "/tmp/example.py"


def test_health_endpoint():
    app = create_app(
        simulation_manager=DummySimulationManager(),
        literature_service=DummyLiteratureService(),
    )
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simulation_start_endpoint_returns_serialized_update():
    app = create_app(
        simulation_manager=DummySimulationManager(),
        literature_service=DummyLiteratureService(),
    )
    client = TestClient(app)
    response = client.post(
        "/api/simulations/start",
        json={"researchGoal": "Study the 2D Ising model"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "sess-123"
    assert payload["state"] == "clarifying"
    assert payload["clarification"]["questions"][0]["topic"] == "sweep_target"


def test_literature_run_endpoint_uses_service_result():
    app = create_app(
        simulation_manager=DummySimulationManager(),
        literature_service=DummyLiteratureService(),
    )
    client = TestClient(app)
    response = client.post(
        "/api/literature/reviews/run",
        json={"query": "diffusion models"},
    )
    assert response.status_code == 200
    assert response.json()["audit"]["passed"] is True


def test_http_error_maps_timeout_to_504():
    error = _http_error(TimeoutError("timed out"))
    assert error.status_code == 504


def test_directive_steering_endpoint_uses_service_result():
    app = create_app(
        simulation_manager=DummySimulationManager(),
        literature_service=DummyLiteratureService(),
        directive_service=DummyDirectiveService(),
    )
    client = TestClient(app)
    response = client.post(
        "/api/directives/dir-1/steering",
        json={
            "checkpointId": "dir-1:find_article",
            "action": "revise",
            "feedback": "Wrong paper.",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["directive_id"] == "dir-1"
    assert payload["action"] == "revise"


def test_directive_stop_endpoint_uses_service_result():
    app = create_app(
        simulation_manager=DummySimulationManager(),
        literature_service=DummyLiteratureService(),
        directive_service=DummyDirectiveService(),
    )
    client = TestClient(app)
    response = client.post(
        "/api/directives/dir-1/stop",
        json={"reason": "Stop this workflow."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["directive_id"] == "dir-1"
    assert payload["status"] == "stop_requested"


def test_directive_snapshot_endpoint_uses_service_result():
    app = create_app(
        simulation_manager=DummySimulationManager(),
        literature_service=DummyLiteratureService(),
        directive_service=DummyDirectiveService(),
    )
    client = TestClient(app)
    response = client.get("/api/directives/dir-1/snapshot")
    assert response.status_code == 200
    payload = response.json()
    assert payload["directive_id"] == "dir-1"
    assert payload["emails"][0]["subject"] == "Instruction accepted — workflow initiated"
