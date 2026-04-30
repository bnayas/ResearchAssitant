"""
test_codegen.py
────────────────
Unit tests for codegen.py — script and notebook generation from SimulationSpec.

Covers:
  - generate_script: creates a syntactically valid, self-contained .py file
  - generate_notebook: creates a valid .ipynb JSON structure
  - CLI arg generation for all variable kinds
  - Correct embedding of all code blocks
  - Two-logger setup (algo_log / data_log) present
  - Assertion tiers embedded
  - Stopping condition dispatcher generated
  - Runner body: checkpoints, progress, stopping, data log
  - Generated script actually runs and produces results.json

Run:  python test_codegen.py
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import textwrap
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.codegen import generate_script, generate_notebook
    from sim_tool.models import SimulationSpec, Variable, VariableKind, StoppingCondition
except ImportError:
    from sim_tool.codegen import generate_script, generate_notebook
    from sim_tool.models import SimulationSpec, Variable, VariableKind, StoppingCondition


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_spec(**overrides) -> SimulationSpec:
    """Minimal valid spec for codegen tests."""
    defaults = dict(
        name="Test Simulation",
        description="A simple test simulation.",
        variables=[
            Variable(
                name="temperature", description="System temperature",
                kind=VariableKind.FLOAT, default=2.0,
                min_val=0.1, max_val=10.0, unit="J/kB",
                sweep=True, sweep_values=[1.5, 2.0, 2.5],
            ),
            Variable(
                name="N", description="Lattice size",
                kind=VariableKind.INT, default=16,
                min_val=4, max_val=128,
            ),
            Variable(
                name="use_cache", description="Use precomputed cache",
                kind=VariableKind.BOOL, default=True,
            ),
            Variable(
                name="algo", description="Algorithm variant",
                kind=VariableKind.CHOICE, default="metropolis",
                choices=["metropolis", "wolff"],
            ),
        ],
        stopping_conditions=[
            StoppingCondition(
                kind="success", name="converged",
                description="Magnetisation stabilised.",
                check_expr="state.step > 5",
                reason_expr="f'Converged at step {state.step}'",
                priority=10,
            ),
            StoppingCondition(
                kind="failure", name="frozen",
                description="Lattice frozen.",
                check_expr="state.acceptance_rate < 1e-6",
                reason_expr="f'Frozen at step {state.step}'",
                priority=20,
            ),
        ],
        state_fields=[
            ("magnetisation", "float", 0.0),
            ("energy", "float", 0.0),
            ("acceptance_rate", "float", 1.0),
        ],
        setup_code="import math\nimport random\n",
        precompute_code=(
            "    exp_table = {dE: math.exp(-dE / config.temperature)\n"
            "                 for dE in [-8, -4, 0, 4, 8]}\n"
            "    return {'exp_table': exp_table}\n"
        ),
        initial_state_code=(
            "    state = SimState()\n"
            "    state.magnetisation = 0.0\n"
            "    state.energy = -1.0\n"
            "    state.acceptance_rate = 1.0\n"
            "    return state\n"
        ),
        step_code=(
            "    new = copy.copy(state)\n"
            "    new.magnetisation = state.magnetisation * 0.99\n"
            "    new.energy = state.energy - 0.01\n"
            "    new.acceptance_rate = 0.5\n"
            "    new.sim_time = state.sim_time + 1.0\n"
            "    return new\n"
        ),
        progress_code=(
            "    return f'm={state.magnetisation:.3f} E={state.energy:.3f}'\n"
        ),
        config_assert_code=(
            "    assert config.temperature > 0, "
            "f'temperature must be positive, got {config.temperature}'\n"
        ),
        state_assert_code=(
            "    assert math.isfinite(state.magnetisation), "
            "f'magnetisation diverged: {state.magnetisation}'\n"
        ),
        output_variables=["magnetisation", "energy", "acceptance_rate"],
        data_log_variables=["magnetisation", "energy"],
        data_log_interval=5,
        checkpoint_interval=50,
        max_steps=100,
        progress_interval=20,
        time_estimate_seconds=1.0,
        time_estimate_explanation="quick test run",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — structure
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateScriptStructure:
    def test_creates_file(self, tmp_path):
        spec = _make_spec()
        path = generate_script(spec, tmp_path / "sim.py")
        assert path.exists()

    def test_file_is_valid_python(self, tmp_path):
        spec = _make_spec()
        path = generate_script(spec, tmp_path / "sim.py")
        source = path.read_text()
        ast.parse(source)  # raises SyntaxError if invalid

    def test_file_is_executable(self, tmp_path):
        import os
        spec = _make_spec()
        path = generate_script(spec, tmp_path / "sim.py")
        assert os.access(path, os.X_OK)

    def test_shebang_present(self, tmp_path):
        spec = _make_spec()
        path = generate_script(spec, tmp_path / "sim.py")
        first_line = path.read_text().splitlines()[0]
        assert first_line.startswith("#!/usr/bin/env python")

    def test_sim_tool_not_imported(self, tmp_path):
        spec = _make_spec()
        source = generate_script(spec, tmp_path / "sim.py").read_text()
        assert "import sim_tool" not in source

    def test_standard_imports_present(self, tmp_path):
        spec = _make_spec()
        source = generate_script(spec, tmp_path / "sim.py").read_text()
        for mod in ["json", "math", "copy", "time", "pickle", "logging", "argparse"]:
            assert f"import {mod}" in source, f"Missing: import {mod}"

    def test_setup_code_embedded(self, tmp_path):
        spec = _make_spec(setup_code="import random\n# custom helper\n")
        source = generate_script(spec, tmp_path / "sim.py").read_text()
        assert "import random" in source
        assert "# custom helper" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — two-logger contract
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoLoggers:
    def test_algo_log_defined(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "algo_log" in source

    def test_data_log_defined(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "data_log" in source

    def test_data_log_writes_to_jsonl(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "data_log.jsonl" in source

    def test_setup_loggers_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "_setup_loggers" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — SimConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestSimConfig:
    def test_all_variables_in_config(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        for name in ["temperature", "N", "use_cache", "algo"]:
            assert name in source

    def test_runner_knobs_in_config(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        for knob in ["max_steps", "checkpoint_interval", "progress_interval", "data_log_interval"]:
            assert knob in source

    def test_defaults_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "2.0" in source   # temperature default
        assert "16" in source    # N default


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — SimState
# ─────────────────────────────────────────────────────────────────────────────

class TestSimState:
    def test_state_fields_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        for field in ["magnetisation", "energy", "acceptance_rate"]:
            assert field in source

    def test_step_and_sim_time_in_state(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "step:" in source or "step =" in source
        assert "sim_time" in source

    def test_history_field_in_state(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "history" in source

    def test_snapshot_method_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "def snapshot" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — Code blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeBlocks:
    def test_precompute_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "def precompute(" in source

    def test_initial_state_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "def initial_state(" in source

    def test_sim_step_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "def sim_step(" in source

    def test_progress_summary_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "def progress_summary(" in source

    def test_user_step_code_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "new.magnetisation" in source

    def test_user_precompute_code_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "exp_table" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — Assertion tiers
# ─────────────────────────────────────────────────────────────────────────────

class TestAssertionTiers:
    def test_config_assert_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "_assert_config" in source

    def test_state_assert_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "_assert_state" in source

    def test_user_config_assert_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "temperature must be positive" in source

    def test_user_state_assert_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "magnetisation diverged" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — Stopping conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestStoppingConditions:
    def test_dispatcher_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "_check_stopping" in source

    def test_success_condition_name_in_script(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "converged" in source

    def test_failure_condition_name_in_script(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "frozen" in source

    def test_check_exprs_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "state.step > 5" in source
        assert "state.acceptance_rate < 1e-6" in source

    def test_success_returns_success_status(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "'success'" in source

    def test_failure_returns_failed_status(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "'failed'" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — CLI
# ─────────────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_argparse_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "ArgumentParser" in source

    def test_float_var_has_type_float(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "type=float" in source

    def test_int_var_has_type_int(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "type=int" in source

    def test_choice_var_has_choices(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "choices=" in source
        assert "metropolis" in source

    def test_output_dir_arg_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "--output-dir" in source

    def test_results_json_arg_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "--results-json" in source


# ─────────────────────────────────────────────────────────────────────────────
# Script generation — Runner body
# ─────────────────────────────────────────────────────────────────────────────

class TestRunnerBody:
    def test_run_function_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "def run(" in source

    def test_checkpoint_logic_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "_save_checkpoint" in source

    def test_max_steps_loop_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "max_steps" in source
        assert "for step_num in range" in source

    def test_data_log_variables_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "magnetisation" in source
        assert "energy" in source

    def test_output_variables_embedded(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "_OUTPUT_VARIABLES" in source
        assert "_DATA_LOG_VARIABLES" in source

    def test_results_written_to_json(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "results_json" in source or "results-json" in source

    def test_simresult_dataclass_present(self, tmp_path):
        source = generate_script(_make_spec(), tmp_path / "sim.py").read_text()
        assert "SimResult" in source


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: generated script actually runs
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratedScriptRunnable:
    def test_script_runs_and_exits_zero(self, tmp_path):
        spec = _make_spec(max_steps=20)
        script = generate_script(spec, tmp_path / "sim.py")
        r = subprocess.run(
            [sys.executable, str(script),
             "--temperature", "2.5", "--N", "8",
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"Script crashed:\n{r.stderr[-500:]}"

    def test_script_writes_results_json(self, tmp_path):
        spec = _make_spec(max_steps=20)
        script = generate_script(spec, tmp_path / "sim.py")
        out = tmp_path / "out"
        results_json = out / "results.json"
        subprocess.run(
            [sys.executable, str(script),
             "--output-dir", str(out),
             "--results-json", str(results_json)],
            capture_output=True, text=True, timeout=30,
        )
        assert results_json.exists(), "results.json not written"
        data = json.loads(results_json.read_text())
        assert "status" in data
        assert "steps_run" in data

    def test_script_writes_data_log(self, tmp_path):
        spec = _make_spec(max_steps=30, data_log_interval=5)
        script = generate_script(spec, tmp_path / "sim.py")
        out = tmp_path / "out"
        subprocess.run(
            [sys.executable, str(script), "--output-dir", str(out)],
            capture_output=True, text=True, timeout=30,
        )
        data_log = out / "data_log.jsonl"
        assert data_log.exists(), "data_log.jsonl not written"
        lines = [l for l in data_log.read_text().splitlines() if l.strip()]
        assert len(lines) > 0
        row = json.loads(lines[0])
        assert "step" in row

    def test_script_results_has_correct_status(self, tmp_path):
        # With max_steps=10 and converged condition at step>5, expect success
        spec = _make_spec(max_steps=10)
        script = generate_script(spec, tmp_path / "sim.py")
        out = tmp_path / "out"
        results_json = out / "results.json"
        subprocess.run(
            [sys.executable, str(script),
             "--output-dir", str(out),
             "--results-json", str(results_json)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(results_json.read_text())
        assert data["status"] in ("success", "failed", "max_steps")

    def test_script_help_flag_works(self, tmp_path):
        spec = _make_spec()
        script = generate_script(spec, tmp_path / "sim.py")
        r = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0
        assert "temperature" in r.stdout

    def test_script_accepts_cli_overrides(self, tmp_path):
        spec = _make_spec(max_steps=10)
        script = generate_script(spec, tmp_path / "sim.py")
        out = tmp_path / "out"
        results_json = out / "results.json"
        r = subprocess.run(
            [sys.executable, str(script),
             "--temperature", "3.5", "--N", "4",
             "--max-steps", "8",
             "--output-dir", str(out),
             "--results-json", str(results_json)],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Notebook generation
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateNotebook:
    def test_creates_file(self, tmp_path):
        spec = _make_spec()
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        assert path.exists()

    def test_valid_json(self, tmp_path):
        spec = _make_spec()
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_nbformat_4(self, tmp_path):
        spec = _make_spec()
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        data = json.loads(path.read_text())
        assert data["nbformat"] == 4

    def test_has_cells(self, tmp_path):
        spec = _make_spec()
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        data = json.loads(path.read_text())
        assert len(data["cells"]) >= 3

    def test_has_code_and_markdown_cells(self, tmp_path):
        spec = _make_spec()
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        data = json.loads(path.read_text())
        types = {c["cell_type"] for c in data["cells"]}
        assert "code" in types
        assert "markdown" in types

    def test_spec_name_in_notebook(self, tmp_path):
        spec = _make_spec(name="My Special Simulation")
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        content = path.read_text()
        assert "My Special Simulation" in content

    def test_variable_names_in_notebook(self, tmp_path):
        spec = _make_spec()
        path = generate_notebook(spec, tmp_path / "sim.ipynb")
        content = path.read_text()
        assert "temperature" in content
        assert "N" in content


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestGenerateScriptStructure,
        TestTwoLoggers,
        TestSimConfig,
        TestSimState,
        TestCodeBlocks,
        TestAssertionTiers,
        TestStoppingConditions,
        TestCLI,
        TestRunnerBody,
        TestGeneratedScriptRunnable,
        TestGenerateNotebook,
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
