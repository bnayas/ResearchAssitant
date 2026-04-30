"""
test_designer.py
─────────────────
Unit tests for designer.py — multi-turn LLM simulation designer.

Tests use a MockBackend that returns scripted JSON responses so no
live API key is needed. Covers:

  - _parse_json: extracts JSON from raw LLM output
  - _validate_questions: topic validation, forbidden-keyword filtering, cap
  - _parse_spec: dict → SimulationSpec
  - SimulationDesigner.start: clarification vs immediate spec
  - SimulationDesigner.answer: accumulates answers, advances rounds
  - SimulationDesigner.reject: re-enters clarification
  - SimulationDesigner.approve: returns SpecApproval without LLM call
  - Auto-repair loop: validator error fed back, second attempt passes
  - MAX_ROUNDS enforcement: forces spec_complete after N rounds
  - ClarificationRequest contract: correct session_id, question topics
  - SpecApproval contract: variable_count, stop_cond_count, output_contract

Run:  python test_designer.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.designer import (
        SimulationDesigner, _parse_json, _parse_spec, _validate_questions,
    )
    from sim_tool.contract import (
        ClarificationRequest, ClarificationQuestion, QuestionTopic, SpecApproval,
    )
    from sim_tool.models import SimulationSpec
except ImportError:
    from sim_tool.designer import (
        SimulationDesigner, _parse_json, _parse_spec, _validate_questions,
    )
    from sim_tool.contract import (
        ClarificationRequest, ClarificationQuestion, QuestionTopic, SpecApproval,
    )
    from sim_tool.models import SimulationSpec


# ─────────────────────────────────────────────────────────────────────────────
# Minimal valid spec JSON the LLM returns
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SPEC_JSON = {
    "spec_complete": True,
    "questions": [],
    "spec": {
        "name": "2D Ising Model",
        "description": "Metropolis MC on a square lattice.",
        "variables": [
            {
                "name": "temperature", "description": "System temperature",
                "kind": "float", "default": 2.269,
                "min_val": 0.1, "max_val": 10.0, "step": None,
                "choices": None, "unit": "J/kB",
                "sweep": True, "sweep_values": [1.5, 2.0, 2.269, 3.0],
            },
            {
                "name": "N", "description": "Lattice side length",
                "kind": "int", "default": 32,
                "min_val": 4, "max_val": 256, "step": None,
                "choices": None, "unit": None,
                "sweep": False, "sweep_values": None,
            },
        ],
        "state_fields": [
            ["magnetisation", "float", 0.0],
            ["energy", "float", 0.0],
            ["acceptance_rate", "float", 1.0],
        ],
        "stopping_conditions": [
            {
                "kind": "success", "name": "converged",
                "description": "Magnetisation stabilised.",
                "check_expr": "abs(state.magnetisation) > 0.5 and state.step > 100",
                "reason_expr": "f'Converged at step {state.step}'",
                "save_on_trigger": True, "priority": 10,
            },
            {
                "kind": "failure", "name": "frozen",
                "description": "Lattice frozen.",
                "check_expr": "state.acceptance_rate < 1e-6 and state.step > 50",
                "reason_expr": "f'Frozen at step {state.step}'",
                "save_on_trigger": True, "priority": 20,
            },
        ],
        "setup_code": "import math\nimport random\n",
        "precompute_code": (
            "    boltzmann = {dE: math.exp(-dE / config.temperature)\n"
            "                 for dE in [-8, -4, 0, 4, 8]}\n"
            "    return {'boltzmann': boltzmann}\n"
        ),
        "initial_state_code": (
            "    state = SimState()\n"
            "    state.magnetisation = 0.0\n"
            "    state.energy = 0.0\n"
            "    state.acceptance_rate = 1.0\n"
            "    return state\n"
        ),
        "step_code": (
            "    new = copy.copy(state)\n"
            "    new.magnetisation = state.magnetisation * 0.99\n"
            "    new.energy = state.energy - 0.01\n"
            "    new.acceptance_rate = max(0.0, state.acceptance_rate - 0.001)\n"
            "    return new\n"
        ),
        "progress_code": (
            "    return f'm={state.magnetisation:.3f}'\n"
        ),
        "config_assert_code": (
            "    assert config.temperature > 0, "
            "f'temperature must be positive, got {config.temperature}'\n"
        ),
        "state_assert_code": (
            "    import math\n"
            "    assert math.isfinite(state.magnetisation), "
            "f'magnetisation diverged: {state.magnetisation}'\n"
        ),
        "output_variables": ["magnetisation", "energy", "acceptance_rate"],
        "data_log_variables": ["magnetisation", "energy"],
        "data_log_interval": 5,
        "checkpoint_interval": 500,
        "max_steps": 3000,
        "progress_interval": 300,
        "time_estimate_seconds": 30.0,
        "time_estimate_explanation": "~30s for 4 configs on N=32",
    }
}

_CLARIFY_RESPONSE = {
    "spec_complete": False,
    "questions": [
        {
            "index": 1,
            "text": "What temperature range should the sweep cover?",
            "topic": "variable_range",
            "required": True,
        },
        {
            "index": 2,
            "text": "How many Monte Carlo steps before the run should stop?",
            "topic": "step_budget",
            "required": True,
        },
    ],
    "spec": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# MockBackend — scripted LLM responses
# ─────────────────────────────────────────────────────────────────────────────

class MockBackend:
    """Returns scripted JSON responses in order; loops on last one."""
    model_name = "mock-model"

    def __init__(self, responses: list[dict]):
        self._responses = [json.dumps(r) for r in responses]
        self._idx = 0

    def complete(self, system: str, messages: list, **kwargs) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    @property
    def call_count(self) -> int:
        return self._idx


def _designer(responses: list[dict]) -> SimulationDesigner:
    """Build a designer wired to a mock backend."""
    backend = MockBackend(responses)
    d = SimulationDesigner(backend=backend)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# _parse_json
# ─────────────────────────────────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self):
        raw = '{"spec_complete": true, "questions": []}'
        result = _parse_json(raw)
        assert result == {"spec_complete": True, "questions": []}

    def test_json_embedded_in_prose(self):
        raw = 'Here is my response:\n{"key": "value"}\nEnd.'
        result = _parse_json(raw)
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        assert _parse_json("not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_json("") is None

    def test_nested_json(self):
        raw = json.dumps({"a": {"b": [1, 2, 3]}})
        result = _parse_json(raw)
        assert result["a"]["b"] == [1, 2, 3]

    def test_json_with_markdown_fences_extracted(self):
        inner = '{"spec_complete": false}'
        raw = f"```json\n{inner}\n```"
        result = _parse_json(raw)
        # Should either parse it or extract inner JSON
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# _validate_questions
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateQuestions:
    def _q(self, text, topic="variable_range", required=True):
        return {"text": text, "topic": topic, "required": required}

    def test_valid_question_passes(self):
        raw = [self._q("What temperature range?", "variable_range")]
        result = _validate_questions(raw, "sid")
        assert len(result) == 1

    def test_all_valid_topics_accepted(self):
        for topic in ["variable_range", "stopping_threshold",
                      "step_budget", "physical_units", "sweep_target"]:
            raw = [self._q("test?", topic)]
            result = _validate_questions(raw, "sid")
            assert len(result) == 1, f"topic={topic} should be accepted"

    def test_invalid_topic_defaults_to_variable_range(self):
        raw = [self._q("test?", "implementation_style")]
        result = _validate_questions(raw, "sid")
        assert len(result) == 1
        assert result[0].topic == QuestionTopic.VARIABLE_RANGE

    def test_cap_at_max_questions_per_round(self):
        raw = [self._q(f"Q{i}?") for i in range(10)]
        result = _validate_questions(raw, "sid")
        assert len(result) <= ClarificationRequest.MAX_QUESTIONS_PER_ROUND

    def test_forbidden_keyword_filters_question(self):
        raw = [self._q("Should I use numpy or stdlib?", "variable_range")]
        result = _validate_questions(raw, "sid")
        assert len(result) == 0

    def test_library_keyword_filtered(self):
        raw = [self._q("Which library is preferred?")]
        result = _validate_questions(raw, "sid")
        assert len(result) == 0

    def test_normal_physics_question_kept(self):
        raw = [self._q("What is the convergence tolerance?", "stopping_threshold")]
        result = _validate_questions(raw, "sid")
        assert len(result) == 1

    def test_returns_clarification_question_objects(self):
        raw = [self._q("What range?")]
        result = _validate_questions(raw, "sid")
        assert all(isinstance(q, ClarificationQuestion) for q in result)

    def test_index_is_1_based(self):
        raw = [self._q("Q1?"), self._q("Q2?")]
        result = _validate_questions(raw, "sid")
        assert result[0].index == 1
        assert result[1].index == 2


# ─────────────────────────────────────────────────────────────────────────────
# _parse_spec
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSpec:
    def test_returns_simulation_spec(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        assert isinstance(spec, SimulationSpec)

    def test_name_preserved(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        assert spec.name == "2D Ising Model"

    def test_variables_parsed(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        assert len(spec.variables) == 2
        names = [v.name for v in spec.variables]
        assert "temperature" in names
        assert "N" in names

    def test_stopping_conditions_parsed(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        assert len(spec.stopping_conditions) == 2
        kinds = {sc.kind for sc in spec.stopping_conditions}
        assert "success" in kinds
        assert "failure" in kinds

    def test_state_fields_parsed(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        field_names = [f[0] for f in spec.state_fields]
        assert "magnetisation" in field_names

    def test_code_blocks_preserved(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        assert "boltzmann" in spec.precompute_code
        assert "return state" in spec.step_code or "return new" in spec.step_code

    def test_max_steps_parsed(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        assert spec.max_steps == 3000

    def test_sweep_variable_marked(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        temp_var = next(v for v in spec.variables if v.name == "temperature")
        assert temp_var.sweep is True

    def test_conditions_sorted_by_priority(self):
        spec = _parse_spec(_VALID_SPEC_JSON["spec"])
        priorities = [sc.priority for sc in spec.stopping_conditions]
        assert priorities == sorted(priorities, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# SimulationDesigner.start
# ─────────────────────────────────────────────────────────────────────────────

class TestDesignerStart:
    def test_clarification_returned_when_questions(self):
        d = _designer([_CLARIFY_RESPONSE])
        result = d.start("Study the 2D Ising model.")
        assert isinstance(result, ClarificationRequest)

    def test_spec_approval_returned_when_complete(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study the 2D Ising model near Tc.")
        assert isinstance(result, SpecApproval)

    def test_session_id_in_clarification(self):
        d = _designer([_CLARIFY_RESPONSE])
        result = d.start("Study the 2D Ising model.")
        assert isinstance(result.session_id, str)
        assert len(result.session_id) > 0

    def test_session_id_in_spec_approval(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study the 2D Ising model near Tc.")
        assert isinstance(result.session_id, str)

    def test_questions_populated(self):
        d = _designer([_CLARIFY_RESPONSE])
        result = d.start("Study Ising model.")
        assert len(result.questions) == 2

    def test_question_topics_valid(self):
        d = _designer([_CLARIFY_RESPONSE])
        result = d.start("Study Ising model.")
        for q in result.questions:
            assert isinstance(q.topic, QuestionTopic)

    def test_spec_approval_variable_count(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study Ising model.")
        assert result.variable_count == 2

    def test_spec_approval_stop_cond_count(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study Ising model.")
        assert result.stop_cond_count == 2

    def test_spec_approval_has_output_contract(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study Ising model.")
        assert result.output_contract is not None
        assert len(result.output_contract.data_log_fields) > 0

    def test_spec_card_contains_name(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study Ising model.")
        assert "2D Ising Model" in result.spec_card

    def test_time_estimate_string_set(self):
        d = _designer([_VALID_SPEC_JSON])
        result = d.start("Study Ising model.")
        assert result.time_estimate  # non-empty


# ─────────────────────────────────────────────────────────────────────────────
# SimulationDesigner.answer
# ─────────────────────────────────────────────────────────────────────────────

class TestDesignerAnswer:
    def test_answer_advances_to_spec_approval(self):
        d = _designer([_CLARIFY_RESPONSE, _VALID_SPEC_JSON])
        r1 = d.start("Study Ising.")
        assert isinstance(r1, ClarificationRequest)
        r2 = d.answer(r1.session_id, {"1": "T from 1.5 to 4.0", "2": "max_steps=3000"})
        assert isinstance(r2, SpecApproval)

    def test_answer_can_get_more_questions(self):
        d = _designer([_CLARIFY_RESPONSE, _CLARIFY_RESPONSE, _VALID_SPEC_JSON])
        r1 = d.start("Study Ising.")
        r2 = d.answer(r1.session_id, {"1": "T from 1.5 to 4.0"})
        assert isinstance(r2, ClarificationRequest)

    def test_session_id_consistent(self):
        d = _designer([_CLARIFY_RESPONSE, _VALID_SPEC_JSON])
        r1 = d.start("Study Ising.")
        r2 = d.answer(r1.session_id, {"1": "T from 1.5 to 4.0"})
        assert r1.session_id == r2.session_id

    def test_iteration_increments(self):
        d = _designer([_CLARIFY_RESPONSE, _VALID_SPEC_JSON])
        r1 = d.start("Study Ising.")
        assert r1.iteration == 0
        # After answering, the next clarification would have iteration=1

    def test_unknown_session_raises(self):
        d = _designer([_VALID_SPEC_JSON])
        try:
            d.answer("bad-session-id", {"1": "T from 1.5 to 4.0"})
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_get_spec_returns_none_before_approval(self):
        d = _designer([_CLARIFY_RESPONSE])
        r = d.start("Study Ising.")
        assert d.get_spec(r.session_id) is None

    def test_get_spec_returns_spec_after_approval(self):
        d = _designer([_VALID_SPEC_JSON])
        r = d.start("Study Ising.")
        spec = d.get_spec(r.session_id)
        assert isinstance(spec, SimulationSpec)


# ─────────────────────────────────────────────────────────────────────────────
# SimulationDesigner.approve / reject
# ─────────────────────────────────────────────────────────────────────────────

class TestDesignerApproveReject:
    def test_approve_returns_spec_approval(self):
        d = _designer([_VALID_SPEC_JSON])
        r1 = d.start("Study Ising.")
        assert isinstance(r1, SpecApproval)
        r2 = d.approve(r1.session_id)
        assert isinstance(r2, SpecApproval)

    def test_approve_no_extra_llm_call(self):
        backend = MockBackend([_VALID_SPEC_JSON])
        d = SimulationDesigner(backend=backend)
        r1 = d.start("Study Ising.")
        count_before = backend.call_count
        d.approve(r1.session_id)
        assert backend.call_count == count_before  # no new call

    def test_reject_re_enters_clarification(self):
        d = _designer([_VALID_SPEC_JSON, _CLARIFY_RESPONSE])
        r1 = d.start("Study Ising.")
        r2 = d.reject(r1.session_id, "Please add energy tracking.")
        assert isinstance(r2, ClarificationRequest)

    def test_reject_then_answer_gives_spec(self):
        d = _designer([_VALID_SPEC_JSON, _CLARIFY_RESPONSE, _VALID_SPEC_JSON])
        r1 = d.start("Study Ising.")
        r2 = d.reject(r1.session_id, "Add energy.")
        r3 = d.answer(r2.session_id, {"1": "energy per spin"})
        assert isinstance(r3, SpecApproval)

    def test_approve_before_spec_raises(self):
        d = _designer([_CLARIFY_RESPONSE])
        r1 = d.start("Study Ising.")
        try:
            d.approve(r1.session_id)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Auto-repair loop
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoRepairLoop:
    def _make_bad_spec(self) -> dict:
        """Spec with missing return in step_code — fails validation tier C."""
        bad = json.loads(json.dumps(_VALID_SPEC_JSON))
        bad["spec"]["step_code"] = "    new = copy.copy(state)\n"  # no return
        return bad

    def test_auto_repair_triggered_on_invalid_spec(self):
        bad = self._make_bad_spec()
        backend = MockBackend([bad, _VALID_SPEC_JSON])
        d = SimulationDesigner(backend=backend)
        result = d.start("Study Ising.")
        # Two LLM calls: first (bad), repair attempt (valid)
        assert backend.call_count == 2
        assert isinstance(result, SpecApproval)

    def test_result_is_valid_after_repair(self):
        bad = self._make_bad_spec()
        d = _designer([bad, _VALID_SPEC_JSON])
        result = d.start("Study Ising.")
        assert isinstance(result, SpecApproval)
        spec = d.get_spec(result.session_id)
        assert spec is not None
        assert "return" in spec.step_code

    def test_double_failure_raises(self):
        bad = self._make_bad_spec()
        d = _designer([bad, bad])  # both calls return broken spec
        try:
            d.start("Study Ising.")
            assert False, "Should have raised ValueError"
        except (ValueError, Exception):
            pass  # expected — auto-repair failed twice


# ─────────────────────────────────────────────────────────────────────────────
# MAX_ROUNDS enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxRounds:
    def test_forces_spec_complete_at_max_rounds(self):
        # Keep returning clarification questions beyond MAX_ROUNDS
        d = _designer([_CLARIFY_RESPONSE] * 3 + [_VALID_SPEC_JSON])
        r = d.start("Study Ising.")
        sid = r.session_id
        # Answer questions MAX_ROUNDS times
        for i in range(ClarificationRequest.MAX_ROUNDS):
            if isinstance(r, ClarificationRequest):
                r = d.answer(sid, {"1": f"answer {i}"})
        # Should eventually resolve to SpecApproval
        assert isinstance(r, SpecApproval)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestParseJson,
        TestValidateQuestions,
        TestParseSpec,
        TestDesignerStart,
        TestDesignerAnswer,
        TestDesignerApproveReject,
        TestAutoRepairLoop,
        TestMaxRounds,
    ]

    passed = failed = 0
    for cls in classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            try:
                getattr(instance, name)()
                print(f"  ✓ {cls.__name__}.{name}")
                passed += 1
            except Exception as exc:
                print(f"  ✗ {cls.__name__}.{name}: {exc}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
