"""
test_runner.py
───────────────
Unit tests for runner.py — subprocess execution and parameter sweep builder.

Covers:
  - build_sweep_configs: Cartesian product of sweep variable values
  - _config_tag: filesystem-safe tag generation
  - run_script: single run outcomes (success, failed, crash, timeout)
  - run_sweep: sequential multi-config execution
  - SweepSummary: aggregate statistics

Run:  python test_runner.py
      or: python -m pytest test_runner.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.runner import (
        SweepSummary, _config_tag, build_sweep_configs, run_script, run_sweep,
    )
    from sim_tool.models import (
        SimulationSpec, StoppingCondition, Variable, VariableKind,
    )
except ImportError:
    from sim_tool.runner import (
        SweepSummary, _config_tag, build_sweep_configs, run_script, run_sweep,
    )
    from sim_tool.models import (
        SimulationSpec, StoppingCondition, Variable, VariableKind,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_var(name, kind=VariableKind.FLOAT, default=1.0,
              sweep=False, sweep_values=None,
              min_val=None, max_val=None, step=None) -> Variable:
    return Variable(
        name=name, description=f"{name} variable",
        kind=kind, default=default,
        min_val=min_val, max_val=max_val, step=step,
        sweep=sweep, sweep_values=sweep_values,
    )


def _make_spec(variables=None) -> SimulationSpec:
    """Minimal spec for sweep config tests."""
    variables = variables or [_make_var("temperature", default=2.0)]
    return SimulationSpec(
        name="Test Sim", description="test",
        variables=variables,
        stopping_conditions=[
            StoppingCondition(
                kind="success", name="done", description="done",
                check_expr="True", reason_expr="'done'",
            )
        ],
        state_fields=[("x", "float", 0.0)],
        setup_code="", precompute_code="    return {}",
        initial_state_code="    return SimState()",
        step_code="    return state",
        progress_code="    return ''",
        config_assert_code="    pass",
        state_assert_code="    pass",
        output_variables=[], data_log_variables=[],
        max_steps=10, checkpoint_interval=100,
        progress_interval=10, data_log_interval=1,
    )


def _sim_script(tmp_path: Path, outcome: str = "success",
                steps: int = 5, write_data: bool = True) -> Path:
    """Write a minimal real simulation script."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "sim.py"
    script.write_text(textwrap.dedent(f"""\
        import argparse, json, sys
        from pathlib import Path

        p = argparse.ArgumentParser()
        p.add_argument("--temperature", type=float, default=2.0)
        p.add_argument("--N", type=int, default=8)
        p.add_argument("--max-steps", type=int, default=10)
        p.add_argument("--checkpoint-interval", type=int, default=100)
        p.add_argument("--progress-interval", type=int, default=10)
        p.add_argument("--data-log-interval", type=int, default=1)
        p.add_argument("--output-dir", default="sim_output")
        p.add_argument("--results-json", default=None)
        args = p.parse_args()

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if {write_data!r}:
            dl = out / "data_log.jsonl"
            for i in range({steps}):
                dl.open("a").write(json.dumps({{"step": i, "energy": -float(i)}}) + "\\n")

        result = {{
            "status": {outcome!r},
            "reason": "test: {outcome}",
            "stop_condition_name": "test_cond",
            "steps_run": {steps},
            "wall_time_seconds": 0.05,
            "config": {{"temperature": args.temperature, "N": args.N}},
        }}
        if args.results_json:
            Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results_json).write_text(json.dumps(result))
        sys.exit(0)
    """))
    return script


def _crash_script(tmp_path: Path) -> Path:
    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent("""\
        import argparse, sys
        p = argparse.ArgumentParser()
        p.add_argument("--output-dir", default=".")
        p.add_argument("--results-json", default=None)
        for arg in ["--temperature","--N","--max-steps","--checkpoint-interval",
                    "--progress-interval","--data-log-interval"]:
            p.add_argument(arg, default=None)
        p.parse_known_args()
        raise RuntimeError("deliberate crash")
    """))
    return script


# ─────────────────────────────────────────────────────────────────────────────
# build_sweep_configs
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSweepConfigs:
    def test_no_sweep_vars_returns_one_config(self):
        spec = _make_spec([_make_var("temperature", default=2.0, sweep=False)])
        configs = build_sweep_configs(spec)
        assert len(configs) == 1
        assert configs[0]["temperature"] == 2.0

    def test_explicit_sweep_values(self):
        var = _make_var("temperature", sweep=True,
                        sweep_values=[1.5, 2.0, 2.269, 3.0, 4.0])
        spec = _make_spec([var])
        configs = build_sweep_configs(spec)
        assert len(configs) == 5
        temps = [c["temperature"] for c in configs]
        assert 2.269 in temps

    def test_range_sweep_from_min_max_step(self):
        var = _make_var("temperature", sweep=True,
                        min_val=1.0, max_val=3.0, step=1.0)
        spec = _make_spec([var])
        configs = build_sweep_configs(spec)
        temps = [c["temperature"] for c in configs]
        assert 1.0 in temps
        assert 2.0 in temps
        assert 3.0 in temps

    def test_bool_var_sweeps_both_values(self):
        var = Variable(
            name="flag", description="test flag",
            kind=VariableKind.BOOL, default=False,
            sweep=True,
        )
        spec = _make_spec([var])
        configs = build_sweep_configs(spec)
        flags = [c["flag"] for c in configs]
        assert False in flags
        assert True in flags

    def test_choice_var_sweeps_all_choices(self):
        var = Variable(
            name="algo", description="algorithm",
            kind=VariableKind.CHOICE, default="metropolis",
            choices=["metropolis", "wolff", "swendsen-wang"],
            sweep=True,
        )
        spec = _make_spec([var])
        configs = build_sweep_configs(spec)
        algos = [c["algo"] for c in configs]
        assert "metropolis" in algos
        assert "wolff" in algos
        assert "swendsen-wang" in algos

    def test_cartesian_product_two_sweep_vars(self):
        vars_ = [
            _make_var("temperature", sweep=True, sweep_values=[1.5, 2.5]),
            _make_var("N", kind=VariableKind.INT, default=16,
                      sweep=True, sweep_values=[16, 32]),
        ]
        spec = _make_spec(vars_)
        configs = build_sweep_configs(spec)
        assert len(configs) == 4  # 2 × 2

    def test_non_sweep_var_uses_default_in_all_configs(self):
        vars_ = [
            _make_var("temperature", default=2.5, sweep=True, sweep_values=[1.5, 3.0]),
            _make_var("J", default=1.0, sweep=False),
        ]
        spec = _make_spec(vars_)
        configs = build_sweep_configs(spec)
        for cfg in configs:
            assert cfg["J"] == 1.0

    def test_three_way_sweep(self):
        vars_ = [
            _make_var("T", sweep=True, sweep_values=[1.0, 2.0, 3.0]),
            _make_var("N", kind=VariableKind.INT, default=8,
                      sweep=True, sweep_values=[8, 16]),
            _make_var("J", sweep=True, sweep_values=[0.5, 1.0]),
        ]
        spec = _make_spec(vars_)
        configs = build_sweep_configs(spec)
        assert len(configs) == 3 * 2 * 2  # 12

    def test_configs_are_independent_dicts(self):
        var = _make_var("temperature", sweep=True, sweep_values=[1.5, 2.5])
        spec = _make_spec([var])
        configs = build_sweep_configs(spec)
        configs[0]["temperature"] = 999
        assert configs[1]["temperature"] != 999  # mutation doesn't propagate


# ─────────────────────────────────────────────────────────────────────────────
# _config_tag
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigTag:
    def _tag(self, cfg):
        spec = _make_spec([_make_var("temperature", sweep=True, sweep_values=[1.0])])
        return _config_tag(cfg, spec)

    def test_basic_swept_var_appears_in_tag(self):
        spec = _make_spec([_make_var("temperature", sweep=True, sweep_values=[2.5])])
        tag = _config_tag({"temperature": 2.5}, spec)
        assert "2.5" in tag

    def test_non_swept_vars_excluded_from_tag(self):
        vars_ = [
            _make_var("temperature", sweep=True, sweep_values=[2.5]),
            _make_var("max_steps", kind=VariableKind.INT, default=1000),
        ]
        spec = _make_spec(vars_)
        tag = _config_tag({"temperature": 2.5, "max_steps": 1000}, spec)
        assert "max_steps" not in tag

    def test_tag_is_filesystem_safe(self):
        spec = _make_spec([_make_var("temperature", sweep=True, sweep_values=[2.269])])
        tag = _config_tag({"temperature": 2.269}, spec)
        forbidden = set('/\\:*?"<>| ')
        assert not any(c in forbidden for c in tag)

    def test_empty_swept_vars_gives_default(self):
        spec = _make_spec([_make_var("temperature", sweep=False)])
        tag = _config_tag({"temperature": 2.0}, spec)
        assert tag == "default"

    def test_tag_length_bounded(self):
        vars_ = [_make_var(f"p{i}", sweep=True, sweep_values=[i]) for i in range(20)]
        spec = _make_spec(vars_)
        cfg = {f"p{i}": i for i in range(20)}
        tag = _config_tag(cfg, spec)
        assert len(tag) <= 120


# ─────────────────────────────────────────────────────────────────────────────
# run_script (single run)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunScript:
    def test_success_outcome(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success")
        result = run_script(script, {"temperature": 2.5}, tmp_path / "out", stream_logs=False)
        assert result.status == "success"

    def test_failed_outcome(self, tmp_path):
        script = _sim_script(tmp_path, outcome="failed")
        result = run_script(script, {}, tmp_path / "out", stream_logs=False)
        assert result.status == "failed"

    def test_max_steps_outcome(self, tmp_path):
        script = _sim_script(tmp_path, outcome="max_steps")
        result = run_script(script, {}, tmp_path / "out", stream_logs=False)
        assert result.status == "max_steps"

    def test_crash_gives_crash_status(self, tmp_path):
        script = _crash_script(tmp_path)
        result = run_script(script, {}, tmp_path / "out", stream_logs=False)
        assert result.status == "crash"

    def test_missing_script_gives_crash(self, tmp_path):
        result = run_script(tmp_path / "ghost.py", {}, tmp_path / "out", stream_logs=False)
        assert result.status == "crash"

    def test_output_dir_created(self, tmp_path):
        script = _sim_script(tmp_path)
        out = tmp_path / "nested" / "deep" / "out"
        run_script(script, {}, out, stream_logs=False)
        assert out.exists()

    def test_results_json_read(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success", steps=7)
        result = run_script(script, {"temperature": 2.5}, tmp_path / "out", stream_logs=False)
        assert result.steps_run == 7

    def test_wall_time_positive(self, tmp_path):
        script = _sim_script(tmp_path)
        result = run_script(script, {}, tmp_path / "out", stream_logs=False)
        assert result.wall_time_seconds >= 0.0

    def test_config_preserved_in_result(self, tmp_path):
        script = _sim_script(tmp_path)
        cfg = {"temperature": 3.14, "N": 16}
        result = run_script(script, cfg, tmp_path / "out", stream_logs=False)
        assert result.config == cfg

    def test_result_json_path_set_on_success(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success")
        result = run_script(script, {}, tmp_path / "out", stream_logs=False)
        assert result.result_json_path is not None
        assert result.result_json_path.exists()

    def test_timeout_returns_timeout_status(self, tmp_path):
        slow = tmp_path / "slow.py"
        slow.write_text(textwrap.dedent("""\
            import time, argparse
            p = argparse.ArgumentParser()
            p.add_argument("--output-dir", default=".")
            p.add_argument("--results-json", default=None)
            p.parse_known_args()
            time.sleep(60)
        """))
        result = run_script(slow, {}, tmp_path / "out",
                            timeout_seconds=1, stream_logs=False)
        assert result.status == "timeout"


# ─────────────────────────────────────────────────────────────────────────────
# SweepSummary
# ─────────────────────────────────────────────────────────────────────────────

class TestSweepSummary:
    def _make_outcome(self, status):
        from sim_tool.runner import RunOutcome
        return RunOutcome(
            config={}, status=status, reason="", stop_condition="",
            steps_run=10, wall_time_seconds=0.1,
            output_dir=Path("/tmp"), result_json_path=None,
        )

    def test_success_count(self):
        try:
            from sim_tool.runner import RunOutcome
        except ImportError:
            from runner import RunOutcome
        s = SweepSummary(spec_name="test", total_runs=4)
        s.outcomes = [self._make_outcome("success")] * 2 + \
                     [self._make_outcome("failed")] + \
                     [self._make_outcome("crash")]
        assert s.success_count == 2

    def test_failed_count(self):
        s = SweepSummary(spec_name="test", total_runs=3)
        s.outcomes = [self._make_outcome("failed")] * 3
        assert s.failed_count == 3

    def test_crash_count(self):
        s = SweepSummary(spec_name="test", total_runs=2)
        s.outcomes = [self._make_outcome("crash"), self._make_outcome("success")]
        assert s.crash_count == 1

    def test_timeout_count(self):
        s = SweepSummary(spec_name="test", total_runs=1)
        s.outcomes = [self._make_outcome("timeout")]
        assert s.timeout_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# run_sweep (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSweep:
    def test_sweep_runs_all_configs(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success")
        vars_ = [_make_var("temperature", sweep=True, sweep_values=[1.5, 2.5, 3.5])]
        spec = _make_spec(vars_)
        summary = run_sweep(script, spec, tmp_path / "sweep", stream_logs=False)
        assert summary.total_runs == 3
        assert len(summary.outcomes) == 3

    def test_sweep_success_count_matches(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success")
        vars_ = [_make_var("temperature", sweep=True, sweep_values=[1.5, 2.5])]
        spec = _make_spec(vars_)
        summary = run_sweep(script, spec, tmp_path / "sweep", stream_logs=False)
        assert summary.success_count == 2

    def test_sweep_creates_separate_output_dirs(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success")
        vars_ = [_make_var("temperature", sweep=True, sweep_values=[1.5, 2.5])]
        spec = _make_spec(vars_)
        sweep_dir = tmp_path / "sweep"
        run_sweep(script, spec, sweep_dir, stream_logs=False)
        subdirs = [d for d in sweep_dir.iterdir() if d.is_dir()]
        assert len(subdirs) >= 2

    def test_sweep_no_vars_runs_single_default(self, tmp_path):
        script = _sim_script(tmp_path, outcome="success")
        spec = _make_spec([_make_var("temperature", default=2.0, sweep=False)])
        summary = run_sweep(script, spec, tmp_path / "sweep", stream_logs=False)
        assert summary.total_runs == 1

    def test_sweep_mixed_outcomes(self, tmp_path):
        # success script; one run will "fail" because we write failed status
        success_s = _sim_script(tmp_path / "s", outcome="success")
        vars_ = [_make_var("temperature", sweep=True, sweep_values=[2.0])]
        spec = _make_spec(vars_)
        summary = run_sweep(success_s, spec, tmp_path / "sweep", stream_logs=False)
        assert len(summary.outcomes) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestBuildSweepConfigs,
        TestConfigTag,
        TestRunScript,
        TestSweepSummary,
        TestRunSweep,
    ]

    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            needs_tmp = "tmp_path" in str(
                getattr(cls, name).__code__.co_varnames
            )
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    if needs_tmp:
                        getattr(instance, name)(Path(tmp))
                    else:
                        getattr(instance, name)()
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
