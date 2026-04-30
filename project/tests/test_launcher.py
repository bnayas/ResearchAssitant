"""
test_launcher.py
─────────────────
Unit tests for Launcher — deterministic environment setup, execution,
failure classification, and pipeline script writing.

Covers:
  - _extract_imports: parses setup_code imports to package names
  - _classify_failure: pattern matching → typed BugTicket
  - _config_to_args: dict → CLI argument string
  - _config_tag: filesystem-safe config tag
  - write_pipeline_script: shell script content and structure
  - Launcher.launch: integration test with a real minimal script
  - Launcher.launch: timeout, crash, and missing-script paths

Run:  python test_launcher.py
      or: python -m pytest test_launcher.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.launcher import (
        Launcher, LaunchRuntimeConfig, _classify_failure, _config_tag, _config_to_args,
        _extract_imports, write_pipeline_script,
    )
    from sim_tool.contract import BugKind, BugTicket, EnvironmentInfo, LaunchResult
    from sim_tool.launcher_log import LauncherLog
except ImportError:
    from sim_tool.launcher import (
        Launcher, LaunchRuntimeConfig, _classify_failure, _config_tag, _config_to_args,
        _extract_imports, write_pipeline_script,
    )
    from sim_tool.contract import BugKind, BugTicket, EnvironmentInfo, LaunchResult
    from sim_tool.launcher_log import LauncherLog


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_info(tmp_path: Path) -> EnvironmentInfo:
    return EnvironmentInfo(
        python_executable=sys.executable,
        python_version=f"Python {sys.version.split()[0]}",
        uv_version="",
        packages_installed=[],
        env_vars={"PYTHONUNBUFFERED": "1"},
        venv_path="",
        ran_as_uv=False,
    )


def _minimal_sim_script(tmp_path: Path, status: str = "success", steps: int = 10) -> Path:
    """Write a self-contained minimal simulation script that produces results.json."""
    script = tmp_path / "minimal_sim.py"
    script.write_text(textwrap.dedent(f"""\
        import argparse, json, sys, time
        from pathlib import Path

        p = argparse.ArgumentParser()
        p.add_argument("--temperature", type=float, default=2.0)
        p.add_argument("--N", type=int, default=8)
        p.add_argument("--output-dir", default="sim_output")
        p.add_argument("--results-json", default=None)
        args = p.parse_args()

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        result = {{
            "status": {status!r},
            "reason": "test run completed",
            "stop_condition_name": "test_condition",
            "steps_run": {steps},
            "wall_time_seconds": 0.1,
            "config": {{"temperature": args.temperature, "N": args.N}},
        }}

        data_log = out / "data_log.jsonl"
        for i in range({steps}):
            data_log.open("a").write(json.dumps({{"step": i, "energy": -1.0 * i}}) + "\\n")

        if args.results_json:
            Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results_json).write_text(json.dumps(result))
        sys.exit(0)
    """))
    return script


def _crashing_script(tmp_path: Path) -> Path:
    script = tmp_path / "crash_sim.py"
    script.write_text(textwrap.dedent("""\
        import argparse, sys
        p = argparse.ArgumentParser()
        p.add_argument("--output-dir", default="sim_output")
        p.add_argument("--results-json", default=None)
        p.parse_known_args()
        raise RuntimeError("intentional crash for testing")
    """))
    return script


def _module_not_found_script(tmp_path: Path) -> Path:
    script = tmp_path / "missing_mod.py"
    script.write_text(textwrap.dedent("""\
        import argparse, sys
        p = argparse.ArgumentParser()
        p.add_argument("--output-dir", default=".")
        p.add_argument("--results-json", default=None)
        p.parse_known_args()
        import this_module_does_not_exist_abc123
    """))
    return script


# ─────────────────────────────────────────────────────────────────────────────
# _extract_imports
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractImports:
    def test_empty_code_returns_empty(self, tmp_path=None):
        assert _extract_imports("") == []

    def test_none_code_returns_empty(self, tmp_path=None):
        assert _extract_imports(None) == []

    def test_numpy_import(self, tmp_path=None):
        code = "import numpy as np\n"
        pkgs = _extract_imports(code)
        assert "numpy" in pkgs

    def test_scipy_import(self, tmp_path=None):
        code = "from scipy import linalg\n"
        pkgs = _extract_imports(code)
        assert "scipy" in pkgs

    def test_pandas_import(self, tmp_path=None):
        code = "import pandas as pd\n"
        pkgs = _extract_imports(code)
        assert "pandas" in pkgs

    def test_unknown_package_not_in_result(self, tmp_path=None):
        code = "import my_custom_lib\n"
        pkgs = _extract_imports(code)
        assert "my_custom_lib" not in pkgs

    def test_stdlib_only_returns_empty(self, tmp_path=None):
        code = "import math\nimport json\nimport os\n"
        pkgs = _extract_imports(code)
        assert pkgs == []

    def test_deduplication(self, tmp_path=None):
        code = "import numpy\nimport numpy as np\n"
        pkgs = _extract_imports(code)
        assert pkgs.count("numpy") == 1

    def test_syntax_error_returns_empty(self, tmp_path=None):
        code = "import {{{broken\n"
        pkgs = _extract_imports(code)
        assert pkgs == []

    def test_sklearn_maps_to_scikit_learn(self, tmp_path=None):
        code = "from sklearn.linear_model import LinearRegression\n"
        pkgs = _extract_imports(code)
        assert "scikit-learn" in pkgs


# ─────────────────────────────────────────────────────────────────────────────
# _config_to_args
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigToArgs:
    def test_single_arg(self, tmp_path=None):
        result = _config_to_args({"temperature": 2.5})
        assert "--temperature 2.5" in result

    def test_multiple_args(self, tmp_path=None):
        result = _config_to_args({"temperature": 2.5, "N": 32})
        assert "--temperature 2.5" in result
        assert "--N 32" in result

    def test_underscore_to_hyphen(self, tmp_path=None):
        result = _config_to_args({"max_steps": 1000})
        assert "--max-steps 1000" in result or "--max_steps 1000" in result

    def test_empty_config(self, tmp_path=None):
        result = _config_to_args({})
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# _config_tag
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigTag:
    def test_basic_tag(self, tmp_path=None):
        cfg = {"temperature": 2.5, "N": 32}
        tag = _config_tag(cfg)
        assert "2.5" in tag or "32" in tag

    def test_tag_is_filesystem_safe(self, tmp_path=None):
        cfg = {"temperature": 2.5, "N": 32}
        tag = _config_tag(cfg)
        forbidden = set('/\\:*?"<>|')
        assert not any(c in forbidden for c in tag)

    def test_tag_max_length(self, tmp_path=None):
        cfg = {f"param_{i}": i for i in range(20)}
        tag = _config_tag(cfg)
        assert len(tag) <= 120

    def test_empty_config_gives_default(self, tmp_path=None):
        tag = _config_tag({})
        assert tag == "default"


# ─────────────────────────────────────────────────────────────────────────────
# _classify_failure
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyFailure:
    def _classify(self, stdout="", stderr="", exit_code=1, tmp_path=None):
        script = tmp_path / "sim.py"
        script.write_text("")
        env = _make_env_info(tmp_path)
        return _classify_failure(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            script_path=script,
            session_id="sess-test",
            config={"temperature": 2.5},
            env_info=env,
            wall_time=1.0,
            pipeline_script_path=None,
            reproducible_command=f"python {script}",
        )

    def test_module_not_found_is_env_setup(self, tmp_path):
        ticket = self._classify(
            stderr="ModuleNotFoundError: No module named 'numpy'",
            tmp_path=tmp_path,
        )
        assert ticket.kind == BugKind.ENV_SETUP
        assert ticket.is_retryable is True

    def test_runtime_error_is_runtime(self, tmp_path):
        ticket = self._classify(
            stderr="Traceback (most recent call last):\n  File sim.py\nRuntimeError: bad state",
            tmp_path=tmp_path,
        )
        assert ticket.kind == BugKind.RUNTIME_ERROR
        assert ticket.is_retryable is False

    def test_syntax_error_classified(self, tmp_path):
        ticket = self._classify(
            stderr="SyntaxError: invalid syntax",
            tmp_path=tmp_path,
        )
        assert ticket.kind == BugKind.SYNTAX_ERROR

    def test_assertion_error_classified(self, tmp_path):
        ticket = self._classify(
            stderr="AssertionError: temperature must be positive, got -1.0",
            tmp_path=tmp_path,
        )
        assert ticket.kind == BugKind.ASSERTION_ERROR
        assert ticket.is_retryable is False

    def test_oom_classified(self, tmp_path):
        ticket = self._classify(
            stderr="MemoryError",
            tmp_path=tmp_path,
        )
        assert ticket.kind == BugKind.OOM
        assert ticket.is_retryable is True

    def test_timeout_classified(self, tmp_path):
        ticket = self._classify(
            stdout="Process timed out",
            tmp_path=tmp_path,
            exit_code=-1,
        )
        assert ticket.kind == BugKind.TIMEOUT
        assert ticket.is_retryable is True

    def test_unknown_exit_produces_ticket(self, tmp_path):
        ticket = self._classify(tmp_path=tmp_path)
        assert isinstance(ticket, BugTicket)
        assert ticket.suggested_fix  # must never be empty

    def test_ticket_always_has_suggested_fix(self, tmp_path):
        ticket = self._classify(
            stderr="some totally unknown error xyz",
            tmp_path=tmp_path,
        )
        assert ticket.suggested_fix.strip() != ""

    def test_reproducible_command_preserved(self, tmp_path):
        script = tmp_path / "sim.py"
        script.write_text("")
        env = _make_env_info(tmp_path)
        ticket = _classify_failure(
            exit_code=1, stdout="", stderr="RuntimeError: x\nTraceback\nError",
            script_path=script, session_id="s1",
            config={}, env_info=env, wall_time=1.0,
            pipeline_script_path=None,
            reproducible_command="python sim.py --temperature 2.5",
        )
        assert "python" in ticket.reproducible_command


# ─────────────────────────────────────────────────────────────────────────────
# write_pipeline_script
# ─────────────────────────────────────────────────────────────────────────────

class TestWritePipelineScript:
    def test_script_is_created(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1", script_path=script,
            config={"temperature": 2.5}, output_dir=tmp_path / "out",
            env_info=env,
        )
        assert pipeline.exists()

    def test_script_is_executable(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1", script_path=script,
            config={}, output_dir=tmp_path / "out",
            env_info=env,
        )
        assert os.access(pipeline, os.X_OK)

    def test_script_contains_shebang(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1", script_path=script,
            config={}, output_dir=tmp_path / "out",
            env_info=env,
        )
        content = pipeline.read_text()
        assert content.startswith("#!/")

    def test_script_contains_config_args(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1", script_path=script,
            config={"temperature": 3.14}, output_dir=tmp_path / "out",
            env_info=env,
        )
        content = pipeline.read_text()
        assert "3.14" in content

    def test_script_contains_session_id(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="unique-session-xyz", script_path=script,
            config={}, output_dir=tmp_path / "out",
            env_info=env,
        )
        content = pipeline.read_text()
        assert "unique-session-xyz" in content

    def test_set_euo_pipefail_present(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1", script_path=script,
            config={}, output_dir=tmp_path / "out",
            env_info=env,
        )
        content = pipeline.read_text()
        assert "set -euo pipefail" in content

    def test_custom_output_path(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        custom_path = tmp_path / "my_pipeline.sh"
        result = write_pipeline_script(
            session_id="s1", script_path=script,
            config={}, output_dir=tmp_path / "out",
            env_info=env, pipeline_path=custom_path,
        )
        assert result == custom_path
        assert custom_path.exists()

    def test_timeout_argument_keeps_script_valid(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1", script_path=script,
            config={}, output_dir=tmp_path / "out",
            env_info=env, timeout_seconds=120,
        )
        content = pipeline.read_text()
        assert "timeout " not in content
        assert "Running simulation" in content

    def test_docker_pipeline_contains_docker_run(self, tmp_path):
        script = tmp_path / "sim.py"; script.write_text("")
        env = _make_env_info(tmp_path)
        pipeline = write_pipeline_script(
            session_id="s1",
            script_path=script,
            config={"temperature": 2.5},
            output_dir=tmp_path / "out",
            env_info=env,
            packages=["numpy"],
            launch_mode="docker",
            runtime=LaunchRuntimeConfig(docker_image="python:3.12-slim"),
        )
        content = pipeline.read_text()
        assert "docker run" in content
        assert "python:3.12-slim" in content


# ─────────────────────────────────────────────────────────────────────────────
# Launcher integration (real subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestLauncherIntegration:
    def test_successful_run_returns_launch_result(self, tmp_path):
        script = _minimal_sim_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="integ-ok",
            script_path=script,
            config={"temperature": 2.5, "N": 8},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert isinstance(result, LaunchResult), f"Expected LaunchResult, got {type(result)}: {result}"

    def test_successful_run_creates_data_log(self, tmp_path):
        script = _minimal_sim_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="integ-data",
            script_path=script,
            config={"temperature": 2.5, "N": 8},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert isinstance(result, LaunchResult)
        data_files = result.run_summary.data_files
        assert len(data_files) > 0
        assert data_files[0].exists()

    def test_crash_returns_bug_ticket(self, tmp_path):
        script = _crashing_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="integ-crash",
            script_path=script,
            config={},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert isinstance(result, BugTicket), f"Expected BugTicket, got {type(result)}"
        assert result.kind == BugKind.RUNTIME_ERROR

    def test_missing_module_gives_env_setup_ticket(self, tmp_path):
        script = _module_not_found_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="integ-env",
            script_path=script,
            config={},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert isinstance(result, BugTicket)
        assert result.kind == BugKind.ENV_SETUP
        assert result.is_retryable is True

    def test_missing_script_gives_env_setup_ticket(self, tmp_path):
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="integ-missing",
            script_path=tmp_path / "nonexistent.py",
            config={},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert isinstance(result, BugTicket)

    def test_timeout_returns_timeout_ticket(self, tmp_path):
        # Script that sleeps longer than our timeout
        script = tmp_path / "slow.py"
        script.write_text(textwrap.dedent("""\
            import time, argparse
            p = argparse.ArgumentParser()
            p.add_argument("--output-dir", default=".")
            p.add_argument("--results-json", default=None)
            p.parse_known_args()
            time.sleep(60)
        """))
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="integ-timeout",
            script_path=script,
            config={},
            output_dir=tmp_path / "out",
            timeout_seconds=1,
            stream_logs=False,
        )
        assert isinstance(result, BugTicket)
        assert result.kind == BugKind.TIMEOUT
        assert result.is_retryable is True

    def test_launcher_log_updated_on_success(self, tmp_path):
        script = _minimal_sim_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        launcher.launch(
            session_id="log-ok",
            script_path=script,
            config={"temperature": 2.0, "N": 4},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert launcher.launcher_log.entry_count >= 1

    def test_launcher_log_updated_on_failure(self, tmp_path):
        script = _crashing_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        launcher.launch(
            session_id="log-fail",
            script_path=script,
            config={},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert launcher.launcher_log.entry_count >= 1

    def test_pipeline_script_written_on_launch(self, tmp_path):
        script = _minimal_sim_script(tmp_path)
        launcher = Launcher(log_dir=tmp_path / "logs")
        result = launcher.launch(
            session_id="pipe-test",
            script_path=script,
            config={"temperature": 2.5, "N": 8},
            output_dir=tmp_path / "out",
            stream_logs=False,
        )
        assert isinstance(result, LaunchResult)
        pipeline = Path(result.pipeline_script_path)
        assert pipeline.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestExtractImports,
        TestConfigToArgs,
        TestConfigTag,
        TestClassifyFailure,
        TestWritePipelineScript,
        TestLauncherIntegration,
    ]

    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                try:
                    getattr(instance, name)(tmp_path)
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
