"""
test_analyst.py
────────────────
Unit tests for analyst.py — three-layer sweep analysis.

Layer 1: extract_metrics  — parse results.json + data_log.jsonl → RunMetrics
Layer 2: classify_run / classify_sweep — rule-based flags, no LLM
Layer 3: RunAnalyst.analyse_sweep — optional LLM → SpecPatch

Covers:
  - extract_metrics: missing file, malformed JSON, valid run dir
  - classify_run: crash, max_steps, tight budget, NaN, no data
  - classify_sweep: all-crashed, all-failed, high failure rate, zero convergence
  - RunAnalyst (no backend): deterministic verdict from flags
  - RunAnalyst (mock backend): LLM path produces SpecPatch
  - AnalysisResult structure: verdict, flags, patch, data_files

Run:  python test_analyst.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.analyst import (
        RunAnalyst, RunMetrics,
        classify_run, classify_sweep, extract_metrics,
    )
    from sim_tool.contract import AnalysisResult, Flag, Verdict
except ImportError:
    from sim_tool.analyst import (
        RunAnalyst, RunMetrics,
        classify_run, classify_sweep, extract_metrics,
    )
    from sim_tool.contract import AnalysisResult, Flag, Verdict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_run_dir(
    tmp_path: Path,
    tag: str = "run",
    status: str = "success",
    steps: int = 100,
    max_steps: int = 1000,
    data_rows: list = None,
    malformed_json: bool = False,
    no_results: bool = False,
) -> Path:
    """Write a synthetic run directory with results.json and data_log.jsonl."""
    run_dir = tmp_path / tag
    run_dir.mkdir()

    if not no_results:
        result = {
            "status": status,
            "reason": f"test: {status}",
            "stop_condition_name": "test_cond",
            "steps_run": steps,
            "wall_time_seconds": 1.5,
            "config": {"temperature": 2.5, "max_steps": max_steps},
        }
        content = "not valid json!!!" if malformed_json else json.dumps(result)
        (run_dir / "results.json").write_text(content)

    if data_rows is not None:
        dl = run_dir / "data_log.jsonl"
        for row in data_rows:
            dl.open("a").write(json.dumps(row) + "\n")

    return run_dir


def _default_data_rows(n: int = 20) -> list:
    return [{"step": i, "energy": -float(i), "magnetisation": 0.5} for i in range(n)]


def _mock_backend(verdict: str = "ok", changes: list = None) -> MagicMock:
    backend = MagicMock()
    changes = changes or []
    backend.complete.return_value = json.dumps({
        "reasoning": "test reasoning",
        "verdict": verdict,
        "reason": "test reason",
        "changes": changes,
    })
    backend.model_name = "mock-model"
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: extract_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractMetrics:
    def test_missing_results_json_gives_crash(self, tmp_path):
        run_dir = tmp_path / "run"; run_dir.mkdir()
        m = extract_metrics(run_dir)
        assert m.status == "crash"
        assert m.results_found is False

    def test_malformed_results_json_gives_crash(self, tmp_path):
        d = _write_run_dir(tmp_path, malformed_json=True)
        m = extract_metrics(d)
        assert m.status == "crash"

    def test_valid_run_dir_parses_status(self, tmp_path):
        d = _write_run_dir(tmp_path, status="success")
        m = extract_metrics(d)
        assert m.status == "success"

    def test_failed_status_parsed(self, tmp_path):
        d = _write_run_dir(tmp_path, status="failed")
        m = extract_metrics(d)
        assert m.status == "failed"

    def test_max_steps_status_parsed(self, tmp_path):
        d = _write_run_dir(tmp_path, status="max_steps")
        m = extract_metrics(d)
        assert m.status == "max_steps"

    def test_steps_run_parsed(self, tmp_path):
        d = _write_run_dir(tmp_path, steps=247)
        m = extract_metrics(d)
        assert m.steps_run == 247

    def test_wall_time_parsed(self, tmp_path):
        d = _write_run_dir(tmp_path)
        m = extract_metrics(d)
        assert m.wall_time_seconds == 1.5

    def test_config_parsed(self, tmp_path):
        d = _write_run_dir(tmp_path)
        m = extract_metrics(d)
        assert m.config.get("temperature") == 2.5

    def test_data_log_record_count(self, tmp_path):
        rows = _default_data_rows(15)
        d = _write_run_dir(tmp_path, data_rows=rows)
        m = extract_metrics(d)
        assert m.data_log_records == 15

    def test_data_log_summary_computed(self, tmp_path):
        rows = [{"step": i, "energy": float(i)} for i in range(10)]
        d = _write_run_dir(tmp_path, data_rows=rows)
        m = extract_metrics(d)
        assert "energy" in m.data_log_summary
        assert m.data_log_summary["energy"]["min"] == 0.0
        assert m.data_log_summary["energy"]["max"] == 9.0

    def test_nan_fields_detected(self, tmp_path):
        rows = [{"step": i, "energy": float("nan")} for i in range(5)]
        d = _write_run_dir(tmp_path, data_rows=rows)
        m = extract_metrics(d)
        assert "energy" in m.data_log_nan_fields

    def test_inf_fields_detected(self, tmp_path):
        rows = [{"step": i, "energy": float("inf")} for i in range(5)]
        d = _write_run_dir(tmp_path, data_rows=rows)
        m = extract_metrics(d)
        assert "energy" in m.data_log_nan_fields

    def test_no_data_log_means_zero_records(self, tmp_path):
        d = _write_run_dir(tmp_path)  # no data_rows
        m = extract_metrics(d)
        assert m.data_log_records == 0

    def test_outcome_property_maps_correctly(self, tmp_path):
        from sim_tool.contract import RunOutcome
        d = _write_run_dir(tmp_path, status="success")
        m = extract_metrics(d)
        assert m.outcome == RunOutcome.SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: classify_run
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyRun:
    def _run_dir(self, tmp_path, **kw) -> RunMetrics:
        d = _write_run_dir(tmp_path, **kw)
        return extract_metrics(d)

    def test_crash_flag_emitted(self, tmp_path):
        m = self._run_dir(tmp_path, no_results=True)
        flags = classify_run(m)
        kinds = [f.kind for f in flags]
        assert "crash_detected" in kinds

    def test_crash_flag_is_error(self, tmp_path):
        m = self._run_dir(tmp_path, no_results=True)
        flags = classify_run(m)
        crash = next(f for f in flags if f.kind == "crash_detected")
        assert crash.severity == "error"

    def test_max_steps_flag_emitted(self, tmp_path):
        m = self._run_dir(tmp_path, status="max_steps", steps=1000, max_steps=1000)
        flags = classify_run(m)
        kinds = [f.kind for f in flags]
        assert "max_steps_hit" in kinds

    def test_max_steps_flag_is_warning(self, tmp_path):
        m = self._run_dir(tmp_path, status="max_steps", steps=1000, max_steps=1000)
        flags = classify_run(m)
        ms = next(f for f in flags if f.kind == "max_steps_hit")
        assert ms.severity == "warning"

    def test_step_budget_tight_flag(self, tmp_path):
        # Succeeded at 950/1000 steps — tight budget
        m = self._run_dir(tmp_path, status="success", steps=950, max_steps=1000)
        flags = classify_run(m)
        kinds = [f.kind for f in flags]
        assert "step_budget_tight" in kinds

    def test_no_tight_flag_when_early_success(self, tmp_path):
        # Succeeded at 200/1000 — not tight
        m = self._run_dir(tmp_path, status="success", steps=200, max_steps=1000)
        flags = classify_run(m)
        kinds = [f.kind for f in flags]
        assert "step_budget_tight" not in kinds

    def test_nan_flag_emitted(self, tmp_path):
        rows = [{"step": i, "energy": float("nan")} for i in range(5)]
        m = self._run_dir(tmp_path, status="success", data_rows=rows)
        flags = classify_run(m)
        kinds = [f.kind for f in flags]
        assert "nan_detected" in kinds

    def test_nan_flag_is_error(self, tmp_path):
        rows = [{"step": i, "energy": float("nan")} for i in range(5)]
        m = self._run_dir(tmp_path, status="success", data_rows=rows)
        flags = classify_run(m)
        nan = next(f for f in flags if f.kind == "nan_detected")
        assert nan.severity == "error"

    def test_no_data_logged_flag(self, tmp_path):
        m = self._run_dir(tmp_path, status="success")  # no data rows
        flags = classify_run(m)
        kinds = [f.kind for f in flags]
        assert "no_data_logged" in kinds

    def test_clean_run_has_no_flags(self, tmp_path):
        rows = _default_data_rows(10)
        m = self._run_dir(tmp_path, status="success", steps=100,
                          max_steps=1000, data_rows=rows)
        flags = classify_run(m)
        assert len(flags) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: classify_sweep
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySweep:
    def _metrics(self, tmp_path, statuses: list) -> list:
        metrics = []
        for i, status in enumerate(statuses):
            d = _write_run_dir(tmp_path, tag=f"run_{i}", status=status,
                               data_rows=_default_data_rows(5))
            metrics.append(extract_metrics(d))
        return metrics

    def test_empty_metrics_returns_no_flags(self, tmp_path):
        flags = classify_sweep([])
        assert flags == []

    def test_all_crashed_flag(self, tmp_path):
        # Simulate crash by using no_results
        metrics = []
        for i in range(3):
            d = tmp_path / f"r{i}"; d.mkdir()
            metrics.append(extract_metrics(d))
        flags = classify_sweep(metrics)
        kinds = [f.kind for f in flags]
        assert "all_crashed" in kinds

    def test_all_crashed_is_error(self, tmp_path):
        metrics = []
        for i in range(2):
            d = tmp_path / f"r{i}"; d.mkdir()
            metrics.append(extract_metrics(d))
        flags = classify_sweep(metrics)
        flag = next(f for f in flags if f.kind == "all_crashed")
        assert flag.severity == "error"

    def test_all_configs_failed(self, tmp_path):
        metrics = self._metrics(tmp_path, ["failed", "failed", "failed"])
        flags = classify_sweep(metrics)
        kinds = [f.kind for f in flags]
        assert "all_configs_failed" in kinds

    def test_high_failure_rate_flag(self, tmp_path):
        # 3 failed out of 5 > 50%
        metrics = self._metrics(tmp_path,
                                ["failed", "failed", "failed", "success", "success"])
        flags = classify_sweep(metrics)
        kinds = [f.kind for f in flags]
        assert "high_failure_rate" in kinds

    def test_no_high_failure_rate_below_threshold(self, tmp_path):
        # 1 failed out of 5 = 20% — below 50% threshold
        metrics = self._metrics(tmp_path,
                                ["failed", "success", "success", "success", "success"])
        flags = classify_sweep(metrics)
        kinds = [f.kind for f in flags]
        assert "high_failure_rate" not in kinds

    def test_convergence_none_flag(self, tmp_path):
        # All max_steps, no success, no crash
        metrics = self._metrics(tmp_path, ["max_steps", "max_steps"])
        flags = classify_sweep(metrics)
        kinds = [f.kind for f in flags]
        assert "convergence_none" in kinds

    def test_clean_sweep_no_flags(self, tmp_path):
        metrics = self._metrics(tmp_path, ["success", "success", "failed"])
        flags = classify_sweep(metrics)
        # failed < 50%, no crash, no convergence_none
        assert not any(f.severity == "error" for f in flags)


# ─────────────────────────────────────────────────────────────────────────────
# RunAnalyst: deterministic (no backend)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAnalystDeterministic:
    def _make_sweep_dir(self, tmp_path, statuses, data=True):
        sweep_dir = tmp_path / "sweep"
        sweep_dir.mkdir()
        for i, status in enumerate(statuses):
            rows = _default_data_rows(5) if data else None
            _write_run_dir(sweep_dir, tag=f"run_{i}", status=status,
                           steps=100, max_steps=1000, data_rows=rows)
        return sweep_dir

    def test_all_success_gives_ok_verdict(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["success", "success"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.verdict == Verdict.OK

    def test_crash_gives_major_fix_or_abort(self, tmp_path):
        sweep_dir = tmp_path / "sweep"; sweep_dir.mkdir()
        for i in range(3):
            d = sweep_dir / f"run_{i}"; d.mkdir()
            # No results.json = crash
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.verdict in (Verdict.ABORT, Verdict.MAJOR_FIX)

    def test_max_steps_gives_minor_or_major_fix(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["max_steps", "max_steps"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.verdict in (Verdict.MINOR_FIX, Verdict.MAJOR_FIX)

    def test_flags_present_in_result(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["max_steps"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert len(result.flags) > 0

    def test_result_has_session_id(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["success"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="test-session")
        assert result.session_id == "test-session"

    def test_data_files_collected(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["success", "success"], data=True)
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert len(result.data_files) > 0
        for f in result.data_files:
            assert f.exists()

    def test_patch_none_when_ok(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["success"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.patch is None

    def test_patch_present_when_not_ok(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["max_steps"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        if result.verdict != Verdict.OK:
            # patch may or may not have changes, but should exist
            assert result.patch is not None

    def test_deterministic_reasoning_note(self, tmp_path):
        sweep_dir = self._make_sweep_dir(tmp_path, ["success"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert "deterministic" in result.llm_reasoning.lower()

    def test_flag_deduplication(self, tmp_path):
        # Multiple runs with same flag kind — should be deduplicated
        sweep_dir = self._make_sweep_dir(tmp_path,
                                         ["max_steps", "max_steps", "max_steps"])
        analyst = RunAnalyst(backend=None)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        kinds = [f.kind for f in result.flags]
        # max_steps_hit appears once despite 3 runs
        assert kinds.count("max_steps_hit") == 1


# ─────────────────────────────────────────────────────────────────────────────
# RunAnalyst: LLM backend path
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAnalystWithBackend:
    def _sweep_dir(self, tmp_path, statuses):
        sweep_dir = tmp_path / "sweep"; sweep_dir.mkdir()
        for i, s in enumerate(statuses):
            _write_run_dir(sweep_dir, tag=f"r{i}", status=s,
                           data_rows=_default_data_rows(5))
        return sweep_dir

    def test_ok_verdict_from_llm(self, tmp_path):
        sweep_dir = self._sweep_dir(tmp_path, ["success"])
        analyst = RunAnalyst(backend=_mock_backend("ok"))
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.verdict == Verdict.OK

    def test_minor_fix_verdict_from_llm(self, tmp_path):
        sweep_dir = self._sweep_dir(tmp_path, ["max_steps"])
        analyst = RunAnalyst(backend=_mock_backend("minor_fix", changes=[
            {"field": "max_steps", "old_value": 1000, "new_value": 3000,
             "why": "hit limit"}
        ]))
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.verdict == Verdict.MINOR_FIX

    def test_patch_contains_changes(self, tmp_path):
        sweep_dir = self._sweep_dir(tmp_path, ["max_steps"])
        analyst = RunAnalyst(backend=_mock_backend("minor_fix", changes=[
            {"field": "max_steps", "old_value": 1000, "new_value": 3000,
             "why": "hit limit"}
        ]))
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert result.patch is not None
        assert len(result.patch.changes) == 1
        assert result.patch.changes[0].field == "max_steps"

    def test_llm_reasoning_stored(self, tmp_path):
        sweep_dir = self._sweep_dir(tmp_path, ["success"])
        analyst = RunAnalyst(backend=_mock_backend("ok"))
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        assert isinstance(result.llm_reasoning, str)

    def test_llm_failure_falls_back_to_deterministic(self, tmp_path):
        sweep_dir = self._sweep_dir(tmp_path, ["max_steps"])
        backend = MagicMock()
        backend.complete.side_effect = RuntimeError("API down")
        backend.model_name = "mock"
        analyst = RunAnalyst(backend=backend)
        result = analyst.analyse_sweep(sweep_dir, session_id="s1")
        # Should still return a valid AnalysisResult
        assert isinstance(result, AnalysisResult)
        assert result.verdict in list(Verdict)

    def test_immutable_field_patch_rejected(self, tmp_path):
        sweep_dir = self._sweep_dir(tmp_path, ["max_steps"])
        # LLM tries to patch step_code — should raise or be silently dropped
        backend = _mock_backend("minor_fix", changes=[
            {"field": "step_code", "old_value": "old", "new_value": "new",
             "why": "bad idea"}
        ])
        analyst = RunAnalyst(backend=backend)
        # Should not crash — PatchChange.__post_init__ raises ValueError
        try:
            result = analyst.analyse_sweep(sweep_dir, session_id="s1")
            # If it doesn't raise, the change should have been skipped
            if result.patch:
                fields = [c.field for c in result.patch.changes]
                assert "step_code" not in fields
        except ValueError:
            pass  # also acceptable — validator correctly rejects it


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestExtractMetrics,
        TestClassifyRun,
        TestClassifySweep,
        TestRunAnalystDeterministic,
        TestRunAnalystWithBackend,
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
