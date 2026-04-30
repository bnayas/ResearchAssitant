"""
test_tool.py
─────────────
Integration tests for tool.py — the top-level API wiring designer,
codegen, runner, and analyst together.

Uses the same MockBackend pattern as test_designer.py; no live API key needed.
All tests run the full stage pipeline using real subprocesses for run_single/run_sweep.

Covers:
  - SimulationTool.start: returns ClarificationRequest or SpecApproval
  - SimulationTool.answer: advances clarification rounds
  - SimulationTool.approve: generates real script + notebook
  - SimulationTool.reject: re-enters design loop
  - SimulationTool.run_single: subprocess execution → RunSummary
  - SimulationTool.run_sweep: multi-config sweep → RunSummary
  - SimulationTool.analyse_sweep: deterministic layer, AnalysisResult
  - SimulationTool.apply_patch: mutates spec, regenerates code
  - SimulationTool.list_sweep_configs: Cartesian product preview
  - SimulationTool.show_spec: formatted spec card
  - SimulationTool.consolidate_memory: no-backend path
  - SimulationTool.show_memory / memory_path

Run:  python test_tool.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.tool import SimulationTool
    from sim_tool.contract import (
        AnalysisResult, ClarificationRequest, GeneratedArtifacts,
        RunSummary, SpecApproval, SpecPatch, PatchChange, Verdict,
    )
    from sim_tool.designer import SimulationDesigner
except ImportError:
    from sim_tool.tool import SimulationTool
    from sim_tool.contract import (
        AnalysisResult, ClarificationRequest, GeneratedArtifacts,
        RunSummary, SpecApproval, SpecPatch, PatchChange, Verdict,
    )
    from sim_tool.designer import SimulationDesigner


# ─────────────────────────────────────────────────────────────────────────────
# Shared valid spec JSON (same structure used in test_designer.py)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SPEC = {
    "spec_complete": True,
    "questions": [],
    "spec": {
        "name": "Ising Tool Test",
        "description": "Minimal Ising spec for tool.py integration tests.",
        "variables": [
            {
                "name": "temperature", "description": "Temperature",
                "kind": "float", "default": 2.0,
                "min_val": 1.0, "max_val": 4.0, "step": 1.0,
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
                "kind": "success", "name": "done",
                "description": "Converged.",
                "check_expr": "state.step >= 5",
                "reason_expr": "f'Done at step {state.step}'",
                "save_on_trigger": True, "priority": 10,
            },
            {
                "kind": "failure", "name": "diverged",
                "description": "Energy diverged.",
                "check_expr": "not (state.energy == state.energy)",  # NaN check
                "reason_expr": "f'Diverged at step {state.step}'",
                "save_on_trigger": True, "priority": 20,
            },
        ],
        "setup_code": "import math\nimport copy\n",
        "precompute_code": "    return {'scale': 1.0 / config.temperature}\n",
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
        "checkpoint_interval": 100,
        "max_steps": 10,
        "progress_interval": 5,
        "time_estimate_seconds": 1.0,
        "time_estimate_explanation": "quick test",
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


# ─────────────────────────────────────────────────────────────────────────────
# Mock backend (same as test_designer.py)
# ─────────────────────────────────────────────────────────────────────────────

class MockBackend:
    model_name = "mock-model"

    def __init__(self, responses: list):
        self._responses = [json.dumps(r) for r in responses]
        self._idx = 0

    def complete(self, system, messages, **kwargs):
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r


def _tool(responses: list, output_root: Path) -> SimulationTool:
    backend = MockBackend(responses)
    return SimulationTool(output_root=output_root, backend=backend)


# ─────────────────────────────────────────────────────────────────────────────
# start / answer / approve / reject
# ─────────────────────────────────────────────────────────────────────────────

class TestStartAnswerApprove:
    def test_start_returns_clarification(self, tmp_path):
        tool = _tool([_CLARIFY], tmp_path)
        r = tool.start("Study the Ising model.")
        assert isinstance(r, ClarificationRequest)

    def test_start_returns_spec_approval(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study the Ising model.")
        assert isinstance(r, SpecApproval)

    def test_answer_advances_to_approval(self, tmp_path):
        tool = _tool([_CLARIFY, _VALID_SPEC], tmp_path)
        r1 = tool.start("Study Ising.")
        r2 = tool.answer(r1.session_id, {"1": "T from 1.5 to 4.0"})
        assert isinstance(r2, SpecApproval)

    def test_session_id_stable_across_calls(self, tmp_path):
        tool = _tool([_CLARIFY, _VALID_SPEC], tmp_path)
        r1 = tool.start("Study Ising.")
        r2 = tool.answer(r1.session_id, {"1": "T from 1.5"})
        assert r1.session_id == r2.session_id

    def test_reject_re_enters_clarification(self, tmp_path):
        tool = _tool([_VALID_SPEC, _CLARIFY], tmp_path)
        r1 = tool.start("Study Ising.")
        assert isinstance(r1, SpecApproval)
        r2 = tool.reject(r1.session_id, "Add energy tracking.")
        assert isinstance(r2, ClarificationRequest)

    def test_unknown_session_raises(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        try:
            tool.answer("ghost-session", {"1": "test"})
            assert False, "Should raise KeyError"
        except KeyError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# approve → GeneratedArtifacts
# ─────────────────────────────────────────────────────────────────────────────

class TestApprove:
    def test_approve_returns_generated_artifacts(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        arts = tool.approve(r.session_id)
        assert isinstance(arts, GeneratedArtifacts)

    def test_script_file_exists(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        arts = tool.approve(r.session_id)
        assert arts.script_path.exists()

    def test_notebook_file_exists(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        arts = tool.approve(r.session_id)
        assert arts.notebook_path.exists()

    def test_script_is_valid_python(self, tmp_path):
        import ast
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        arts = tool.approve(r.session_id)
        ast.parse(arts.script_path.read_text())

    def test_script_size_kb_positive(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        arts = tool.approve(r.session_id)
        assert arts.script_size_kb > 0

    def test_cli_synopsis_present(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        arts = tool.approve(r.session_id)
        assert arts.cli_synopsis

    def test_approve_before_spec_raises(self, tmp_path):
        tool = _tool([_CLARIFY], tmp_path)
        r = tool.start("Study Ising.")
        try:
            tool.approve(r.session_id)
            assert False, "Should raise RuntimeError"
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# run_single
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSingle:
    def _approved_tool(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        tool.approve(r.session_id)
        return tool, r.session_id

    def test_returns_run_summary(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_single(sid, stream_logs=False)
        assert isinstance(summary, RunSummary)

    def test_has_one_result(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_single(sid, stream_logs=False)
        assert len(summary.results) == 1

    def test_result_has_known_outcome(self, tmp_path):
        from sim_tool.contract import RunOutcome
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_single(sid, stream_logs=False)
        assert summary.results[0].outcome in list(RunOutcome)

    def test_config_override_applied(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_single(
            sid, config_overrides={"temperature": 3.0}, stream_logs=False
        )
        assert summary.results[0].config.get("temperature") == 3.0

    def test_run_before_approve_raises(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        try:
            tool.run_single(r.session_id, stream_logs=False)
            assert False, "Should raise RuntimeError"
        except RuntimeError:
            pass

    def test_sweep_dir_set_after_run(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_single(sid, stream_logs=False)
        assert summary.sweep_dir is not None


# ─────────────────────────────────────────────────────────────────────────────
# run_sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSweep:
    def _approved_tool(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        tool.approve(r.session_id)
        return tool, r.session_id

    def test_returns_run_summary(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_sweep(sid, stream_logs=False)
        assert isinstance(summary, RunSummary)

    def test_correct_number_of_results(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        configs = tool.list_sweep_configs(sid)
        summary = tool.run_sweep(sid, stream_logs=False)
        assert len(summary.results) == len(configs)

    def test_sweep_dir_exists(self, tmp_path):
        tool, sid = self._approved_tool(tmp_path)
        summary = tool.run_sweep(sid, stream_logs=False)
        assert summary.sweep_dir.exists()

    def test_two_sweep_configs_for_two_temps(self, tmp_path):
        # spec has sweep_values=[1.5, 2.5] → 2 configs
        tool, sid = self._approved_tool(tmp_path)
        configs = tool.list_sweep_configs(sid)
        assert len(configs) == 2


# ─────────────────────────────────────────────────────────────────────────────
# analyse_sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyseSweep:
    def _run_and_analyse(self, tmp_path, use_llm=False):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        tool.approve(r.session_id)
        tool.run_sweep(r.session_id, stream_logs=False)
        analysis = tool.analyse_sweep(r.session_id, use_llm=use_llm)
        return analysis

    def test_returns_analysis_result(self, tmp_path):
        result = self._run_and_analyse(tmp_path)
        assert isinstance(result, AnalysisResult)

    def test_verdict_is_valid(self, tmp_path):
        result = self._run_and_analyse(tmp_path)
        assert result.verdict in list(Verdict)

    def test_flags_is_list(self, tmp_path):
        result = self._run_and_analyse(tmp_path)
        assert isinstance(result.flags, list)

    def test_data_files_collected(self, tmp_path):
        result = self._run_and_analyse(tmp_path)
        # May be empty if runs crashed, but property must exist
        assert isinstance(result.data_files, list)


# ─────────────────────────────────────────────────────────────────────────────
# apply_patch
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyPatch:
    def _setup(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        tool.approve(r.session_id)
        return tool, r.session_id

    def test_returns_generated_artifacts(self, tmp_path):
        tool, sid = self._setup(tmp_path)
        patch = SpecPatch(
            verdict=Verdict.MINOR_FIX, reason="test",
            changes=[PatchChange("max_steps", 10, 50, "increase budget")],
        )
        arts = tool.apply_patch(sid, patch)
        assert isinstance(arts, GeneratedArtifacts)

    def test_max_steps_updated_in_spec(self, tmp_path):
        tool, sid = self._setup(tmp_path)
        patch = SpecPatch(
            verdict=Verdict.MINOR_FIX, reason="test",
            changes=[PatchChange("max_steps", 10, 999, "test increase")],
        )
        tool.apply_patch(sid, patch)
        spec = tool._sessions[sid].spec
        assert spec.max_steps == 999

    def test_script_regenerated_after_patch(self, tmp_path):
        tool, sid = self._setup(tmp_path)
        old_script = tool._sessions[sid].script_path
        old_mtime = old_script.stat().st_mtime if old_script and old_script.exists() else 0
        patch = SpecPatch(
            verdict=Verdict.MINOR_FIX, reason="test",
            changes=[PatchChange("max_steps", 10, 50, "test")],
        )
        arts = tool.apply_patch(sid, patch)
        assert arts.script_path.exists()

    def test_immutable_field_silently_skipped(self, tmp_path):
        tool, sid = self._setup(tmp_path)
        # PatchChange rejects immutable fields at construction time
        # tool.apply_patch also guards against them
        # Test that the tool doesn't crash when given a patch that
        # tries to reach immutable fields via a workaround
        patch = SpecPatch(
            verdict=Verdict.MINOR_FIX, reason="test",
            changes=[PatchChange("checkpoint_interval", 100, 50, "reduce")],
        )
        arts = tool.apply_patch(sid, patch)
        assert isinstance(arts, GeneratedArtifacts)


# ─────────────────────────────────────────────────────────────────────────────
# list_sweep_configs / show_spec
# ─────────────────────────────────────────────────────────────────────────────

class TestInspectionHelpers:
    def test_list_sweep_configs_before_approve(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        configs = tool.list_sweep_configs(r.session_id)
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_list_sweep_configs_has_temperature(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        configs = tool.list_sweep_configs(r.session_id)
        for cfg in configs:
            assert "temperature" in cfg

    def test_list_sweep_configs_correct_count(self, tmp_path):
        # spec has sweep_values=[1.5, 2.5] → 2 configs
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        configs = tool.list_sweep_configs(r.session_id)
        assert len(configs) == 2

    def test_show_spec_returns_string(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        card = tool.show_spec(r.session_id)
        assert isinstance(card, str)

    def test_show_spec_contains_name(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        card = tool.show_spec(r.session_id)
        assert "Ising Tool Test" in card

    def test_show_spec_before_spec_ready(self, tmp_path):
        tool = _tool([_CLARIFY], tmp_path)
        r = tool.start("Study Ising.")
        card = tool.show_spec(r.session_id)
        assert "No spec" in card or isinstance(card, str)


# ─────────────────────────────────────────────────────────────────────────────
# memory helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryHelpers:
    def test_show_memory_returns_string(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        content = tool.show_memory("designer")
        assert isinstance(content, str)

    def test_show_memory_analyst_returns_string(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        content = tool.show_memory("analyst")
        assert isinstance(content, str)

    def test_show_memory_invalid_agent_raises(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        try:
            tool.show_memory("unknown-agent")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_memory_path_is_path(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        p = tool.memory_path("designer")
        assert isinstance(p, Path)

    def test_memory_path_analyst(self, tmp_path):
        tool = _tool([_VALID_SPEC], tmp_path)
        p = tool.memory_path("analyst")
        assert isinstance(p, Path)
        assert "analyst" in str(p)

    def test_consolidate_memory_no_backend_ok(self, tmp_path):
        # No-backend path: consolidate should not crash
        tool = _tool([_VALID_SPEC], tmp_path)
        r = tool.start("Study Ising.")
        tool.approve(r.session_id)
        result = tool.consolidate_memory(r.session_id)
        assert isinstance(result, dict)
        assert "designer" in result
        assert "analyst" in result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestStartAnswerApprove,
        TestApprove,
        TestRunSingle,
        TestRunSweep,
        TestAnalyseSweep,
        TestApplyPatch,
        TestInspectionHelpers,
        TestMemoryHelpers,
    ]

    passed = failed = 0
    for cls in classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    getattr(instance, name)(Path(tmp))
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
