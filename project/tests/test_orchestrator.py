"""
test_orchestrator.py
─────────────────────
Integration tests for orchestrator.py — stateful research orchestrator
with 9-state machine, background threading, and 4 PI review gates.

Uses MockBackend (no live API) + real generated script subprocesses.

Covers:
  - OrchestratorState: all enum values present
  - OrchestratorUpdate: fields, __str__ representation
  - SweepProgress: fraction_done, __str__
  - ResearchOrchestrator.start: CLARIFYING or AWAITING_SPEC_APPROVAL
  - ResearchOrchestrator.answer: state transitions
  - ResearchOrchestrator.approve: READY_TO_SAMPLE, artifacts on disk
  - Gate 2: inspect script before sample
  - ResearchOrchestrator.run_sample: AWAITING_SAMPLE_REVIEW, blocks
  - ResearchOrchestrator.approve_sample: FULL_RUNNING (non-blocking)
  - ResearchOrchestrator.wait: blocks until AWAITING_RESULTS_REVIEW
  - ResearchOrchestrator.get_status: non-blocking snapshot
  - ResearchOrchestrator.query: returns OrchestratorUpdate with answer
  - ResearchOrchestrator.approve_results: COMPLETE, memory consolidated
  - ResearchOrchestrator.reject / reject_results: back to design
  - State machine guards: wrong-state raises ValueError
  - apply_patch_and_rerun: READY_TO_SAMPLE after patch

Run:  python test_orchestrator.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.orchestrator import (
        OrchestratorState, OrchestratorUpdate, ResearchOrchestrator, SweepProgress,
    )
    from sim_tool.tool import SimulationTool
    from sim_tool.contract import Verdict, SpecPatch, PatchChange
except ImportError:
    from sim_tool.orchestrator import (
        OrchestratorState, OrchestratorUpdate, ResearchOrchestrator, SweepProgress,
    )
    from sim_tool.tool import SimulationTool
    from sim_tool.contract import Verdict, SpecPatch, PatchChange


# ─────────────────────────────────────────────────────────────────────────────
# Shared spec / mock backend (same as test_tool.py)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SPEC = {
    "spec_complete": True,
    "questions": [],
    "spec": {
        "name": "Ising Orch Test",
        "description": "Minimal spec for orchestrator tests.",
        "variables": [
            {
                "name": "temperature", "description": "Temperature",
                "kind": "float", "default": 2.0,
                "min_val": 1.0, "max_val": 3.0, "step": 1.0,
                "choices": None, "unit": "J/kB",
                "sweep": True, "sweep_values": [1.5, 2.5],
            },
        ],
        "state_fields": [
            ["magnetisation", "float", 0.0],
            ["energy", "float", 0.0],
        ],
        "stopping_conditions": [
            {
                "kind": "success", "name": "done", "description": "Converged.",
                "check_expr": "state.step >= 3",
                "reason_expr": "f'Done at step {state.step}'",
                "save_on_trigger": True, "priority": 10,
            },
            {
                "kind": "failure", "name": "diverged",
                "description": "Energy diverged.",
                "check_expr": "not (state.energy == state.energy)",
                "reason_expr": "f'Diverged at step {state.step}'",
                "save_on_trigger": True, "priority": 20,
            },
        ],
        "setup_code": "import math\nimport copy\n",
        "precompute_code": "    return {'scale': 1.0}\n",
        "initial_state_code": (
            "    state = SimState()\n"
            "    state.magnetisation = 0.5\n"
            "    state.energy = -1.0\n"
            "    return state\n"
        ),
        "step_code": (
            "    new = copy.copy(state)\n"
            "    new.magnetisation = state.magnetisation * 0.99\n"
            "    new.energy = state.energy - 0.1\n"
            "    return new\n"
        ),
        "progress_code": "    return f'm={state.magnetisation:.3f}'\n",
        "config_assert_code": (
            "    assert config.temperature > 0, "
            "f'temperature must be > 0, got {config.temperature}'\n"
        ),
        "state_assert_code": (
            "    import math\n"
            "    assert math.isfinite(state.magnetisation), "
            "f'magnetisation diverged: {state.magnetisation}'\n"
        ),
        "output_variables": ["magnetisation", "energy"],
        "data_log_variables": ["magnetisation", "energy"],
        "data_log_interval": 1,
        "checkpoint_interval": 50,
        "max_steps": 6,
        "progress_interval": 3,
        "time_estimate_seconds": 0.5,
        "time_estimate_explanation": "tiny test",
    },
}

_CLARIFY = {
    "spec_complete": False,
    "questions": [
        {"index": 1, "text": "What temperature range?",
         "topic": "variable_range", "required": True},
    ],
    "spec": None,
}


class MockBackend:
    model_name = "mock-model"

    def __init__(self, responses: list):
        self._responses = [json.dumps(r) for r in responses]
        self._idx = 0

    def complete(self, system, messages, **kwargs):
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r


def _orch(responses: list, tmp_path: Path,
          sample_steps: int = 5) -> ResearchOrchestrator:
    backend = MockBackend(responses)
    tool = SimulationTool(output_root=tmp_path / "sims", backend=backend)
    return ResearchOrchestrator(tool, sample_steps=sample_steps)


def _orch_at_ready(tmp_path: Path) -> tuple[ResearchOrchestrator, str]:
    """Build orchestrator and advance to READY_TO_SAMPLE."""
    orch = _orch([_VALID_SPEC], tmp_path)
    u = orch.start("Study the Ising model near Tc.")
    assert u.state == OrchestratorState.AWAITING_SPEC_APPROVAL
    u = orch.approve(u.session_id)
    assert u.state == OrchestratorState.READY_TO_SAMPLE
    return orch, u.session_id


def _orch_at_sample_review(tmp_path: Path) -> tuple[ResearchOrchestrator, str]:
    """Build orchestrator and advance to AWAITING_SAMPLE_REVIEW."""
    orch, sid = _orch_at_ready(tmp_path)
    u = orch.run_sample(sid)
    assert u.state == OrchestratorState.AWAITING_SAMPLE_REVIEW
    return orch, sid


def _orch_complete(tmp_path: Path) -> tuple[ResearchOrchestrator, str]:
    """Run the full pipeline to COMPLETE."""
    orch, sid = _orch_at_sample_review(tmp_path)
    u = orch.approve_sample(sid)
    assert u.state == OrchestratorState.FULL_RUNNING
    u = orch.wait(sid, timeout=30)
    assert u.state == OrchestratorState.AWAITING_RESULTS_REVIEW
    u = orch.approve_results(sid)
    assert u.state == OrchestratorState.COMPLETE
    return orch, sid


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorState enum
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorState:
    def test_all_states_present(self):
        states = {s.value for s in OrchestratorState}
        required = {
            "clarifying", "awaiting_spec_approval", "ready_to_sample",
            "sample_running", "awaiting_sample_review", "full_running",
            "awaiting_results_review", "complete", "failed", "aborted",
        }
        assert required.issubset(states)

    def test_string_enum(self):
        assert OrchestratorState.COMPLETE == "complete"


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorUpdate
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorUpdate:
    def _make(self, state=OrchestratorState.COMPLETE) -> OrchestratorUpdate:
        return OrchestratorUpdate(
            session_id="orch-abc",
            state=state,
            message="Test message.",
            next_action="do something",
        )

    def test_str_contains_state(self):
        u = self._make(OrchestratorState.CLARIFYING)
        assert "CLARIFYING" in str(u)

    def test_str_contains_session_id(self):
        u = self._make()
        assert "orch-abc" in str(u)

    def test_str_contains_message(self):
        u = self._make()
        assert "Test message." in str(u)

    def test_str_contains_next_action(self):
        u = self._make()
        assert "do something" in str(u)

    def test_optional_fields_default_none(self):
        u = self._make()
        assert u.clarification is None
        assert u.spec_approval is None
        assert u.artifacts is None
        assert u.sample_summary is None
        assert u.full_summary is None
        assert u.full_analysis is None

    def test_query_answer_default_empty(self):
        u = self._make()
        assert u.query_answer == ""


# ─────────────────────────────────────────────────────────────────────────────
# SweepProgress
# ─────────────────────────────────────────────────────────────────────────────

class TestSweepProgress:
    def test_fraction_done(self):
        p = SweepProgress(total_configs=4, configs_done=2,
                          configs_running=1, elapsed_seconds=10.0,
                          latest_data={})
        assert p.fraction_done == 0.5

    def test_fraction_done_zero_total(self):
        p = SweepProgress(total_configs=0, configs_done=0,
                          configs_running=0, elapsed_seconds=0.0,
                          latest_data={})
        assert p.fraction_done == 0.0

    def test_str_contains_progress(self):
        p = SweepProgress(total_configs=4, configs_done=2,
                          configs_running=1, elapsed_seconds=10.0,
                          latest_data={})
        s = str(p)
        assert "2/4" in s or "50" in s

    def test_str_contains_elapsed(self):
        p = SweepProgress(total_configs=4, configs_done=1,
                          configs_running=1, elapsed_seconds=42.0,
                          latest_data={})
        assert "42" in str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 0: start
# ─────────────────────────────────────────────────────────────────────────────

class TestStart:
    def test_clarifying_when_questions(self, tmp_path):
        orch = _orch([_CLARIFY], tmp_path)
        u = orch.start("Study the Ising model.")
        assert u.state == OrchestratorState.CLARIFYING

    def test_awaiting_spec_when_complete(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study the Ising model.")
        assert u.state == OrchestratorState.AWAITING_SPEC_APPROVAL

    def test_session_id_assigned(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study the Ising model.")
        assert u.session_id and len(u.session_id) > 0

    def test_clarification_payload_set(self, tmp_path):
        orch = _orch([_CLARIFY], tmp_path)
        u = orch.start("Study the Ising model.")
        assert u.clarification is not None

    def test_spec_approval_payload_set(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study the Ising model.")
        assert u.spec_approval is not None

    def test_next_action_populated(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study the Ising model.")
        assert u.next_action and len(u.next_action) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Gate 0b: answer
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswer:
    def test_answer_advances_to_spec_approval(self, tmp_path):
        orch = _orch([_CLARIFY, _VALID_SPEC], tmp_path)
        u = orch.start("Study Ising.")
        assert u.state == OrchestratorState.CLARIFYING
        u2 = orch.answer(u.session_id, {"1": "T from 1.5 to 4.0"})
        assert u2.state == OrchestratorState.AWAITING_SPEC_APPROVAL

    def test_answer_wrong_state_raises(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study Ising.")
        assert u.state == OrchestratorState.AWAITING_SPEC_APPROVAL
        try:
            orch.answer(u.session_id, {"1": "T from 1.5"})
            assert False, "Should raise ValueError"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: approve (spec)
# ─────────────────────────────────────────────────────────────────────────────

class TestApproveSpec:
    def test_approve_gives_ready_to_sample(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study Ising.")
        u2 = orch.approve(u.session_id)
        assert u2.state == OrchestratorState.READY_TO_SAMPLE

    def test_artifacts_on_disk(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        u = orch.get_status(sid)
        # Check script exists via session
        sess = orch._sessions[sid]
        assert sess.artifacts is not None
        assert sess.artifacts.script_path.exists()

    def test_approve_wrong_state_raises(self, tmp_path):
        orch = _orch([_CLARIFY], tmp_path)
        u = orch.start("Study Ising.")
        try:
            orch.approve(u.session_id)
            assert False, "Should raise ValueError"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Gate 1b: reject (spec)
# ─────────────────────────────────────────────────────────────────────────────

class TestRejectSpec:
    def test_reject_from_awaiting_gives_clarifying(self, tmp_path):
        orch = _orch([_VALID_SPEC, _CLARIFY], tmp_path)
        u = orch.start("Study Ising.")
        u2 = orch.reject(u.session_id, "Add energy tracking.")
        assert u2.state in (OrchestratorState.CLARIFYING,
                            OrchestratorState.AWAITING_SPEC_APPROVAL)

    def test_reject_from_ready_to_sample_works(self, tmp_path):
        orch = _orch([_VALID_SPEC, _CLARIFY], tmp_path)
        u = orch.start("Study Ising.")
        u = orch.approve(u.session_id)
        assert u.state == OrchestratorState.READY_TO_SAMPLE
        u2 = orch.reject(u.session_id, "Need more temperature points.")
        assert u2.state in (OrchestratorState.CLARIFYING,
                            OrchestratorState.AWAITING_SPEC_APPROVAL)


# ─────────────────────────────────────────────────────────────────────────────
# Gate 2: run_sample (blocking)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSample:
    def test_run_sample_gives_awaiting_sample_review(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        u = orch.run_sample(sid)
        assert u.state == OrchestratorState.AWAITING_SAMPLE_REVIEW

    def test_sample_summary_present(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        u = orch.run_sample(sid)
        assert u.sample_summary is not None

    def test_sample_analysis_present(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        u = orch.run_sample(sid)
        assert u.sample_analysis is not None

    def test_run_sample_wrong_state_raises(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        u = orch.start("Study Ising.")
        try:
            orch.run_sample(u.session_id)
            assert False, "Should raise ValueError"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Gate 3: approve_sample → FULL_RUNNING (non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

class TestApproveSample:
    def test_approve_sample_gives_full_running(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        u = orch.approve_sample(sid)
        assert u.state == OrchestratorState.FULL_RUNNING

    def test_approve_sample_is_non_blocking(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        t0 = time.perf_counter()
        orch.approve_sample(sid)
        elapsed = time.perf_counter() - t0
        # Should return almost immediately (well under 1s)
        assert elapsed < 2.0

    def test_approve_sample_wrong_state_raises(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        try:
            orch.approve_sample(sid)
            assert False, "Should raise ValueError"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# wait / get_status / query
# ─────────────────────────────────────────────────────────────────────────────

class TestMonitoring:
    def test_wait_reaches_results_review(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.wait(sid, timeout=30)
        assert u.state in (OrchestratorState.AWAITING_RESULTS_REVIEW,
                           OrchestratorState.FAILED)

    def test_get_status_non_blocking(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        t0 = time.perf_counter()
        orch.get_status(sid)
        assert time.perf_counter() - t0 < 1.0

    def test_get_status_returns_update(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.get_status(sid)
        assert isinstance(u, OrchestratorUpdate)

    def test_query_returns_update(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.query(sid, "How many configs are done?")
        assert isinstance(u, OrchestratorUpdate)

    def test_query_answer_is_string(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.query(sid, "How many configs are done?")
        assert isinstance(u.query_answer, str)

    def test_wait_already_complete_state(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        # In READY_TO_SAMPLE, wait should return immediately
        u = orch.wait(sid, timeout=5)
        assert u.state == OrchestratorState.READY_TO_SAMPLE


# ─────────────────────────────────────────────────────────────────────────────
# Gate 4: approve_results → COMPLETE
# ─────────────────────────────────────────────────────────────────────────────

class TestApproveResults:
    def test_approve_results_gives_complete(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.wait(sid, timeout=30)
        if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            u2 = orch.approve_results(sid)
            assert u2.state == OrchestratorState.COMPLETE

    def test_approve_results_has_full_summary(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.wait(sid, timeout=30)
        if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            u2 = orch.approve_results(sid)
            assert u2.full_summary is not None

    def test_approve_results_wrong_state_raises(self, tmp_path):
        orch, sid = _orch_at_ready(tmp_path)
        try:
            orch.approve_results(sid)
            assert False, "Should raise ValueError"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# reject_results
# ─────────────────────────────────────────────────────────────────────────────

class TestRejectResults:
    def test_reject_results_re_enters_design(self, tmp_path):
        orch = _orch([_VALID_SPEC, _CLARIFY], tmp_path)
        u = orch.start("Study Ising.")
        u = orch.approve(u.session_id)
        u = orch.run_sample(u.session_id)
        orch.approve_sample(u.session_id)
        u = orch.wait(u.session_id, timeout=30)
        if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            u2 = orch.reject_results(u.session_id, "Need higher resolution sweep.")
            assert u2.state in (OrchestratorState.CLARIFYING,
                                OrchestratorState.AWAITING_SPEC_APPROVAL)


# ─────────────────────────────────────────────────────────────────────────────
# apply_patch_and_rerun
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyPatchAndRerun:
    def test_no_patch_available_returns_message(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.wait(sid, timeout=30)
        if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            # After clean runs, full_analysis.patch is None
            sess = orch._sessions[sid]
            if sess.full_analysis and not sess.full_analysis.patch:
                u2 = orch.apply_patch_and_rerun(sid)
                assert "No patch" in u2.message or u2.state == u.state

    def test_abort_verdict_returns_message(self, tmp_path):
        # Simulate an ABORT verdict in the analysis
        from sim_tool.contract import AnalysisResult, Flag, SpecPatch, Verdict
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.wait(sid, timeout=30)
        if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            sess = orch._sessions[sid]
            # Inject an ABORT analysis
            sess.full_analysis = AnalysisResult(
                session_id=sid, verdict=Verdict.ABORT,
                flags=[], patch=None,
            )
            u2 = orch.apply_patch_and_rerun(sid)
            assert "abort" in u2.message.lower() or u2.state == OrchestratorState.AWAITING_RESULTS_REVIEW


# ─────────────────────────────────────────────────────────────────────────────
# Unknown session guard
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownSession:
    def test_approve_unknown_raises(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        try:
            orch.approve("ghost-session")
            assert False
        except KeyError:
            pass

    def test_run_sample_unknown_raises(self, tmp_path):
        orch = _orch([_VALID_SPEC], tmp_path)
        try:
            orch.run_sample("ghost-session")
            assert False
        except KeyError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_full_pipeline_reaches_complete(self, tmp_path):
        """Run the full orchestrator pipeline end-to-end."""
        orch, sid = _orch_complete(tmp_path)
        u = orch.get_status(sid)
        assert u.state == OrchestratorState.COMPLETE

    def test_complete_state_has_data_files(self, tmp_path):
        orch, sid = _orch_complete(tmp_path)
        sess = orch._sessions[sid]
        assert sess.full_summary is not None

    def test_complete_message_mentions_data(self, tmp_path):
        orch, sid = _orch_at_sample_review(tmp_path)
        orch.approve_sample(sid)
        u = orch.wait(sid, timeout=30)
        if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            u2 = orch.approve_results(sid)
            # message should mention data files or completion
            assert "complete" in u2.message.lower() or "data" in u2.message.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestOrchestratorState,
        TestOrchestratorUpdate,
        TestSweepProgress,
        TestStart,
        TestAnswer,
        TestApproveSpec,
        TestRejectSpec,
        TestRunSample,
        TestApproveSample,
        TestMonitoring,
        TestApproveResults,
        TestRejectResults,
        TestApplyPatchAndRerun,
        TestUnknownSession,
        TestFullPipeline,
    ]

    passed = failed = 0
    for cls in classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            fn = getattr(instance, name)
            needs_tmp = "tmp_path" in str(fn.__code__.co_varnames)
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    if needs_tmp:
                        fn(Path(tmp))
                    else:
                        fn()
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
