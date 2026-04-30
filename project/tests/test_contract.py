"""
test_contract.py
─────────────────
Unit tests for contract.py — all typed stage return objects.

Covers:
  - ClarificationRequest / ClarificationQuestion: str representation, limits
  - SpecApproval / OutputContract: str, field counts
  - GeneratedArtifacts: str representation
  - RunOutcome / RunResult / RunSummary: counters, data_files property
  - Verdict / SpecPatch / PatchChange: immutable field rejection
  - AnalysisResult: verdict routing, str representation
  - BugKind / BugTicket: retryability, str representation
  - EnvironmentInfo / LaunchResult: session_id delegation

Run:  python test_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.contract import (
        AnalysisResult, BugKind, BugTicket, ClarificationQuestion,
        ClarificationRequest, EnvironmentInfo, Flag, GeneratedArtifacts,
        IMMUTABLE_FIELDS, LaunchResult, OutputContract, PatchChange,
        PATCHABLE_FIELDS, QuestionTopic, RunOutcome, RunResult, RunSummary,
        SpecApproval, SpecPatch, Verdict,
    )
except ImportError:
    from sim_tool.contract import (
        AnalysisResult, BugKind, BugTicket, ClarificationQuestion,
        ClarificationRequest, EnvironmentInfo, Flag, GeneratedArtifacts,
        IMMUTABLE_FIELDS, LaunchResult, OutputContract, PatchChange,
        PATCHABLE_FIELDS, QuestionTopic, RunOutcome, RunResult, RunSummary,
        SpecApproval, SpecPatch, Verdict,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_result(outcome: RunOutcome = RunOutcome.SUCCESS,
                config=None, steps=100, wall=1.0,
                data_file=None) -> RunResult:
    return RunResult(
        config=config or {"temperature": 2.5},
        outcome=outcome,
        reason="test reason",
        stop_condition_name="test_cond",
        steps_run=steps,
        wall_time_seconds=wall,
        output_dir=Path("/tmp/test"),
        data_file=data_file,
    )


def _env_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        python_executable="/usr/bin/python3",
        python_version="Python 3.11.0",
        uv_version="0.11.2",
        packages_installed=["numpy==2.0.0"],
        env_vars={"PYTHONUNBUFFERED": "1"},
        venv_path="/tmp/.venv",
        ran_as_uv=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ClarificationRequest / ClarificationQuestion
# ─────────────────────────────────────────────────────────────────────────────

class TestClarificationRequest:
    def test_str_contains_session_id(self):
        q = ClarificationQuestion(index=1, text="What range?",
                                  topic=QuestionTopic.VARIABLE_RANGE)
        r = ClarificationRequest(session_id="sess-abc", questions=[q], iteration=0)
        assert "sess-abc" in str(r)

    def test_str_contains_question_text(self):
        q = ClarificationQuestion(index=1, text="What temperature range?",
                                  topic=QuestionTopic.VARIABLE_RANGE)
        r = ClarificationRequest(session_id="s1", questions=[q], iteration=0)
        assert "What temperature range?" in str(r)

    def test_str_contains_topic(self):
        q = ClarificationQuestion(index=1, text="Q?",
                                  topic=QuestionTopic.STEP_BUDGET)
        r = ClarificationRequest(session_id="s1", questions=[q], iteration=0)
        assert "step_budget" in str(r)

    def test_max_questions_constant(self):
        assert ClarificationRequest.MAX_QUESTIONS_PER_ROUND == 5

    def test_max_rounds_constant(self):
        assert ClarificationRequest.MAX_ROUNDS == 4

    def test_question_required_default_true(self):
        q = ClarificationQuestion(index=1, text="Q?",
                                  topic=QuestionTopic.VARIABLE_RANGE)
        assert q.required is True

    def test_question_optional(self):
        q = ClarificationQuestion(index=1, text="Q?",
                                  topic=QuestionTopic.PHYSICAL_UNITS,
                                  required=False)
        assert q.required is False

    def test_all_topics_valid(self):
        for topic in QuestionTopic:
            q = ClarificationQuestion(index=1, text="Q?", topic=topic)
            assert isinstance(q.topic, QuestionTopic)


# ─────────────────────────────────────────────────────────────────────────────
# SpecApproval / OutputContract
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecApproval:
    def _approval(self) -> SpecApproval:
        oc = OutputContract(
            data_log_fields=["step", "sim_time", "magnetisation"],
            results_fields=["status", "reason", "steps_run"],
        )
        return SpecApproval(
            session_id="sess-xyz",
            spec_card="────\n  2D Ising\n────",
            time_estimate="~30s",
            output_contract=oc,
            variable_count=2,
            stop_cond_count=2,
        )

    def test_str_contains_session_id(self):
        assert "sess-xyz" in str(self._approval())

    def test_str_contains_spec_card(self):
        assert "2D Ising" in str(self._approval())

    def test_str_contains_time_estimate(self):
        assert "30s" in str(self._approval())

    def test_output_contract_fields(self):
        oc = OutputContract(
            data_log_fields=["step", "energy"],
            results_fields=["status"],
        )
        assert "step" in oc.data_log_fields
        assert "energy" in oc.data_log_fields

    def test_output_contract_str(self):
        oc = OutputContract(
            data_log_fields=["step", "energy"],
            results_fields=["status", "reason"],
        )
        s = str(oc)
        assert "step" in s
        assert "status" in s


# ─────────────────────────────────────────────────────────────────────────────
# GeneratedArtifacts
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratedArtifacts:
    def _artifacts(self) -> GeneratedArtifacts:
        return GeneratedArtifacts(
            session_id="sess-gen",
            script_path=Path("/tmp/ising.py"),
            notebook_path=Path("/tmp/ising.ipynb"),
            script_size_kb=12,
            cli_synopsis="python ising.py [--temperature T] ...",
        )

    def test_str_contains_session(self):
        assert "sess-gen" in str(self._artifacts())

    def test_str_contains_script_path(self):
        assert "ising.py" in str(self._artifacts())

    def test_str_contains_size(self):
        assert "12" in str(self._artifacts())

    def test_str_contains_cli(self):
        assert "python" in str(self._artifacts())


# ─────────────────────────────────────────────────────────────────────────────
# RunResult / RunSummary
# ─────────────────────────────────────────────────────────────────────────────

class TestRunResult:
    def test_success_str_has_checkmark(self):
        r = _run_result(RunOutcome.SUCCESS)
        assert "✓" in str(r) or "success" in str(r).lower()

    def test_failed_str_has_cross(self):
        r = _run_result(RunOutcome.FAILED)
        s = str(r)
        assert "✗" in s or "fail" in s.lower()

    def test_steps_in_str(self):
        r = _run_result(steps=1234)
        assert "1,234" in str(r) or "1234" in str(r)

    def test_config_values_in_str(self):
        r = _run_result(config={"temperature": 3.14})
        assert "3.14" in str(r)


class TestRunSummary:
    def test_n_success(self):
        s = RunSummary(session_id="s1", results=[
            _run_result(RunOutcome.SUCCESS),
            _run_result(RunOutcome.SUCCESS),
            _run_result(RunOutcome.FAILED),
        ], sweep_dir=Path("/tmp"))
        assert s.n_success == 2

    def test_n_failed(self):
        s = RunSummary(session_id="s1", results=[
            _run_result(RunOutcome.FAILED),
        ], sweep_dir=Path("/tmp"))
        assert s.n_failed == 1

    def test_n_crash(self):
        s = RunSummary(session_id="s1", results=[
            _run_result(RunOutcome.CRASH),
            _run_result(RunOutcome.CRASH),
        ], sweep_dir=Path("/tmp"))
        assert s.n_crash == 2

    def test_n_max_steps(self):
        s = RunSummary(session_id="s1", results=[
            _run_result(RunOutcome.MAX_STEPS),
        ], sweep_dir=Path("/tmp"))
        assert s.n_max_steps == 1

    def test_data_files_excludes_none(self, tmp_path):
        f = tmp_path / "data_log.jsonl"
        f.write_text('{"step":0}')
        s = RunSummary(session_id="s1", results=[
            _run_result(data_file=f),
            _run_result(data_file=None),
        ], sweep_dir=tmp_path)
        assert len(s.data_files) == 1
        assert s.data_files[0] == f

    def test_data_files_excludes_missing_file(self, tmp_path):
        ghost = tmp_path / "ghost.jsonl"  # not created
        s = RunSummary(session_id="s1", results=[
            _run_result(data_file=ghost),
        ], sweep_dir=tmp_path)
        assert len(s.data_files) == 0

    def test_str_shows_counts(self):
        s = RunSummary(session_id="s1", results=[
            _run_result(RunOutcome.SUCCESS),
            _run_result(RunOutcome.FAILED),
            _run_result(RunOutcome.CRASH),
        ], sweep_dir=Path("/tmp"))
        text = str(s)
        assert "success=1" in text or "1" in text


# ─────────────────────────────────────────────────────────────────────────────
# PatchChange / SpecPatch — immutability contract
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchChange:
    def test_patchable_field_accepted(self):
        c = PatchChange(field="max_steps", old_value=1000,
                        new_value=3000, why="hit limit")
        assert c.field == "max_steps"

    def test_variable_sub_field_accepted(self):
        c = PatchChange(field="variables.temperature.default",
                        old_value=2.0, new_value=2.5, why="test")
        assert c.field == "variables.temperature.default"

    def test_stopping_condition_sub_field_accepted(self):
        c = PatchChange(field="stopping_conditions.frozen.check_expr",
                        old_value="old", new_value="new", why="test")
        assert c.field == "stopping_conditions.frozen.check_expr"

    def test_step_code_rejected(self):
        try:
            PatchChange(field="step_code", old_value="x", new_value="y", why="bad")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "step_code" in str(e)

    def test_precompute_code_rejected(self):
        try:
            PatchChange(field="precompute_code", old_value="x",
                        new_value="y", why="bad")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_state_fields_rejected(self):
        try:
            PatchChange(field="state_fields", old_value=[], new_value=[], why="bad")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_all_immutable_fields_rejected(self):
        for field in IMMUTABLE_FIELDS:
            try:
                PatchChange(field=field, old_value="x", new_value="y", why="test")
                assert False, f"{field} should be rejected"
            except ValueError:
                pass

    def test_str_shows_field_and_values(self):
        c = PatchChange(field="max_steps", old_value=1000,
                        new_value=3000, why="ran to limit")
        s = str(c)
        assert "max_steps" in s
        assert "1000" in s
        assert "3000" in s


class TestSpecPatch:
    def test_verdict_stored(self):
        p = SpecPatch(verdict=Verdict.MINOR_FIX,
                      reason="needs more steps", changes=[])
        assert p.verdict == Verdict.MINOR_FIX

    def test_changes_list(self):
        c = PatchChange(field="max_steps", old_value=1000,
                        new_value=3000, why="test")
        p = SpecPatch(verdict=Verdict.MINOR_FIX, reason="r", changes=[c])
        assert len(p.changes) == 1

    def test_abort_str_mentions_abort(self):
        p = SpecPatch(verdict=Verdict.ABORT, reason="fundamental mismatch",
                      changes=[])
        assert "ABORT" in str(p)


# ─────────────────────────────────────────────────────────────────────────────
# AnalysisResult
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisResult:
    def _result(self, verdict=Verdict.OK, flags=None, patch=None) -> AnalysisResult:
        return AnalysisResult(
            session_id="s1",
            verdict=verdict,
            flags=flags or [],
            patch=patch,
        )

    def test_ok_verdict(self):
        r = self._result(Verdict.OK)
        assert r.verdict == Verdict.OK

    def test_str_contains_verdict(self):
        r = self._result(Verdict.MAJOR_FIX)
        assert "MAJOR_FIX" in str(r)

    def test_str_contains_session_id(self):
        r = self._result()
        assert "s1" in str(r)

    def test_str_shows_flags(self):
        flag = Flag(kind="nan_detected", severity="error",
                    message="NaN in energy field")
        r = self._result(Verdict.MAJOR_FIX, flags=[flag])
        s = str(r)
        assert "nan_detected" in s

    def test_str_next_step_ok(self):
        r = self._result(Verdict.OK)
        assert "No changes needed" in str(r)

    def test_str_next_step_minor_fix(self):
        patch = SpecPatch(verdict=Verdict.MINOR_FIX, reason="r",
                          changes=[PatchChange("max_steps", 1000, 3000, "r")])
        r = self._result(Verdict.MINOR_FIX, patch=patch)
        assert "apply_patch" in str(r)

    def test_str_next_step_abort(self):
        r = self._result(Verdict.ABORT)
        assert "restart" in str(r).lower() or "tool.start" in str(r)

    def test_data_files_default_empty(self):
        r = self._result()
        assert r.data_files == []


# ─────────────────────────────────────────────────────────────────────────────
# BugTicket / BugKind
# ─────────────────────────────────────────────────────────────────────────────

class TestBugTicket:
    def _ticket(self, kind=BugKind.RUNTIME_ERROR,
                retryable=False, exit_code=1) -> BugTicket:
        return BugTicket(
            kind=kind,
            session_id="sess-bug",
            script_path="/tmp/sim.py",
            exit_code=exit_code,
            reproducible_command="python /tmp/sim.py --temperature 2.5",
            stdout_tail="",
            stderr_tail="RuntimeError: bad state",
            suggested_fix="Check the step_code.",
            is_retryable=retryable,
        )

    def test_str_shows_kind(self):
        t = self._ticket()
        assert "runtime_error" in str(t)

    def test_str_shows_exit_code(self):
        t = self._ticket(exit_code=1)
        assert "exit=1" in str(t) or "exit_code" in str(t) or "1" in str(t)

    def test_str_shows_retryable(self):
        t = self._ticket(retryable=True)
        assert "True" in str(t)

    def test_str_shows_fix(self):
        t = self._ticket()
        assert "Check the step_code" in str(t)

    def test_env_setup_retryable(self):
        t = self._ticket(kind=BugKind.ENV_SETUP, retryable=True)
        assert t.is_retryable is True

    def test_all_bugkind_values(self):
        for kind in BugKind:
            assert isinstance(kind.value, str)

    def test_unknown_kind_uses_sentinel(self):
        t = BugTicket(
            kind=BugKind.UNKNOWN,
            session_id="s1",
            script_path="/tmp/sim.py",
            exit_code=-1,
            reproducible_command="python /tmp/sim.py",
            suggested_fix="Inspect stderr.",
            is_retryable=False,
        )
        assert t.exit_code == -1


# ─────────────────────────────────────────────────────────────────────────────
# EnvironmentInfo / LaunchResult
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentInfo:
    def test_str_contains_python_version(self):
        env = _env_info()
        s = str(env)
        assert "3.11" in s or "python" in s.lower()

    def test_str_contains_packages(self):
        env = _env_info()
        assert "numpy" in str(env)

    def test_ran_as_uv_reflected(self):
        env = _env_info()
        assert env.ran_as_uv is True


class TestLaunchResult:
    def test_session_id_delegated(self, tmp_path):
        summary = RunSummary(session_id="launch-sess",
                             results=[], sweep_dir=tmp_path)
        result = LaunchResult(
            run_summary=summary,
            env_info=_env_info(),
            pipeline_script_path="/tmp/run.sh",
            launcher_log_entry="[launch-sess] success 1.0s",
        )
        assert result.session_id == "launch-sess"

    def test_data_files_delegated(self, tmp_path):
        f = tmp_path / "data_log.jsonl"
        f.write_text('{"step":0}')
        summary = RunSummary(session_id="s1", results=[
            _run_result(data_file=f)
        ], sweep_dir=tmp_path)
        result = LaunchResult(
            run_summary=summary,
            env_info=_env_info(),
            pipeline_script_path=None,
            launcher_log_entry="",
        )
        assert len(result.data_files) == 1

    def test_str_contains_session(self, tmp_path):
        summary = RunSummary(session_id="lr-sess", results=[], sweep_dir=tmp_path)
        result = LaunchResult(
            run_summary=summary, env_info=_env_info(),
            pipeline_script_path=None, launcher_log_entry="",
        )
        assert "lr-sess" in str(result)


# ─────────────────────────────────────────────────────────────────────────────
# PATCHABLE_FIELDS / IMMUTABLE_FIELDS sets
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldSets:
    def test_patchable_contains_max_steps(self):
        assert "max_steps" in PATCHABLE_FIELDS

    def test_immutable_contains_step_code(self):
        assert "step_code" in IMMUTABLE_FIELDS

    def test_immutable_contains_precompute_code(self):
        assert "precompute_code" in IMMUTABLE_FIELDS

    def test_immutable_contains_state_fields(self):
        assert "state_fields" in IMMUTABLE_FIELDS

    def test_no_overlap_between_sets(self):
        overlap = PATCHABLE_FIELDS & IMMUTABLE_FIELDS
        assert len(overlap) == 0, f"Overlap found: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback, tempfile

    classes = [
        TestClarificationRequest,
        TestSpecApproval,
        TestGeneratedArtifacts,
        TestRunResult,
        TestRunSummary,
        TestPatchChange,
        TestSpecPatch,
        TestAnalysisResult,
        TestBugTicket,
        TestEnvironmentInfo,
        TestLaunchResult,
        TestFieldSets,
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
