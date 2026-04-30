from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from sim_tool.api_server import create_app, WriterService


MOCK_PLAN_JSON = """{
  "title": "Neutral Dynamics Review",
  "venue": "arXiv preprint",
  "abstract_outline": "We summarize the reproduction.",
  "sections": [
    {
      "section_id": "s1",
      "number": "1",
      "title": "Introduction",
      "level": 1,
      "estimated_words": 250,
      "key_points": ["Context"],
      "required_agents": ["literature_reviewer"],
      "depends_on": [],
      "figures": []
    }
  ],
  "figures": [],
  "initial_promises": []
}"""

MOCK_SECTION = (
    "The review is grounded in the literature synthesis.\n\n"
    "---GROUNDING---\n"
    "{\"grounding_claims\": [{\"claim_text\": \"The review uses the literature synthesis.\", "
    "\"source_agent\": \"literature_reviewer\", \"artifact_id\": \"lit-synthesis\", "
    "\"artifact_excerpt\": \"Related neutral theory papers.\", \"confidence\": 0.95}], "
    "\"resolved_promise_ids\": [], \"new_promises\": [], \"summary\": \"Grounded introduction.\"}\n"
    "---END---"
)


class FakeWriterBackend:
    def __init__(self):
        self._responses = [MOCK_PLAN_JSON, MOCK_SECTION]
        self.model_name = "fake-writer"

    def complete(self, system, messages, max_tokens=4096, temperature=0.0):
        return self._responses.pop(0)


def test_writer_api_uses_real_registry_and_streams_results():
    app = create_app(writer_service=WriterService())
    client = TestClient(app)

    with patch("sim_tool.api_server._make_llm_backend", return_value=FakeWriterBackend()):
        response = client.post(
            "/api/writer/start",
            json={
                "description": "Write a grounded review of the neutral model reproduction.",
                "venue": "arXiv preprint",
                "artifactContext": "## Literature Review\nRelated neutral theory papers.",
                "artifactCatalog": [
                    {
                        "artifactId": "lit-synthesis",
                        "assistant": "literature_reviewer",
                        "kind": "literature_synthesis",
                        "title": "Literature Synthesis",
                        "summary": "Related neutral theory papers.",
                        "content": "Related neutral theory papers."
                    }
                ],
                "llm": {"provider": "mock"},
            },
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        with client.stream("GET", f"/api/writer/{job_id}/stream") as stream_response:
            stream_lines = [line for line in stream_response.iter_lines() if line]
        assert any("phase_change" in line for line in stream_lines)

        deadline = time.time() + 5
        result_response = None
        while time.time() < deadline:
            result_response = client.get(f"/api/writer/{job_id}/result")
            if result_response.status_code == 200:
                break
            time.sleep(0.05)

    assert result_response is not None
    assert result_response.status_code == 200
    payload = result_response.json()
    assert "assembled_paper" in payload
    assert payload["draft_artifact"]["path"].endswith("mini_review.md")
    assistants = {artifact["assistant"] for artifact in payload["artifacts"]}
    assert "writer" in assistants
    assert "literature_reviewer" in assistants
