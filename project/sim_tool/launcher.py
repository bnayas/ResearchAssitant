"""
sim_tool.launcher
─────────────────
Deterministic launch tool. No LLM. No reasoning. Just execution.

Responsibilities
────────────────
1. ENVIRONMENT SETUP
   Detects uv, creates an isolated venv per session, installs packages
   extracted from the spec's setup_code imports.
   Falls back gracefully: uv → venv → sys.executable.

2. PIPELINE SCRIPT WRITING
   Writes a portable shell script that reproduces the full run:
     setup / preprocess / simulate / postprocess
   The script is self-contained — anyone can run it without the Python tool.

3. EXECUTION
   Thin wrapper around subprocess. Streams stdout. Enforces timeout.
   Captures output for failure analysis.

4. FAILURE CLASSIFICATION
   Pattern-matches stdout/stderr against a fixed catalogue of known errors.
   Produces a typed, validated BugTicket — no LLM involved.
   Returns LaunchResult (success) or BugTicket (failure), both validated.

The launcher is intentionally NOT an agent:
  - No LLM calls
  - No multi-turn conversation
  - No memory between calls
  - Deterministic given the same inputs

Usage
─────
  from sim_tool.launcher import Launcher

  launcher = Launcher()
  result = launcher.launch(
      session_id="abc123",
      script_path=Path("sim.py"),
      config={"temperature": 2.5, "N": 32},
      output_dir=Path("./out"),
      preprocess_script=None,
      postprocess_script=None,
      timeout_seconds=300,
  )

  if isinstance(result, LaunchResult):
      print("Success:", result.run_summary)
  else:  # BugTicket
      print("Failed:", result)
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional, Union

from .bug_ticket_validator import BugTicketValidator
from .contract import (
    BugKind, BugTicket, EnvironmentInfo, LaunchResult,
    RunOutcome, RunResult, RunSummary,
)
from .launcher_log import LauncherLog

log = logging.getLogger("sim_tool.launcher")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_STDOUT_TAIL_LINES   = 50
_STDERR_TAIL_LINES   = 50
_DEFAULT_LOG_DIR     = Path.home() / ".sim_tool"

# Env vars always set for subprocess runs
_FIXED_ENV_VARS = {
    "PYTHONUNBUFFERED": "1",      # immediate stdout flush
    "PYTHONDONTWRITEBYTECODE": "1",  # no .pyc clutter
    "MPLBACKEND": "Agg",          # headless matplotlib
}

# Third-party packages that setup_code may import and need installing
_KNOWN_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "sklearn.linear_model": "scikit-learn",
    "networkx": "networkx",
}

# Failure pattern catalogue: (regex, BugKind, is_retryable, fix_template)
# Patterns are checked in order — first match wins.
_FAILURE_PATTERNS: list[tuple[str, BugKind, bool, str]] = [
    # Environment / packaging
    (
        r"ModuleNotFoundError: No module named '(\S+)'",
        BugKind.ENV_SETUP, True,
        "Install the missing package: `uv add {match_1}` (or `pip install {match_1}`). "
        "Then re-run the pipeline script.",
    ),
    (
        r"ImportError: cannot import name '(\S+)' from '(\S+)'",
        BugKind.IMPORT_ERROR, False,
        "The package is installed but the name '{match_1}' could not be imported from "
        "'{match_2}'. Check the package version: `uv pip show {match_2}`. "
        "The generated setup_code may be using an outdated API.",
    ),
    (
        r"No module named pip",
        BugKind.ENV_SETUP, True,
        "pip is not available. Run `uv pip install pip` or recreate the venv.",
    ),
    (
        r"python: command not found|python3: command not found",
        BugKind.ENV_SETUP, True,
        "Python interpreter not found. Install Python 3.9+ or set the PATH correctly.",
    ),
    # Assertion errors (from Tier 1/2 of the contract)
    (
        r"AssertionError: (.+)",
        BugKind.ASSERTION_ERROR, False,
        "An assertion in the generated code fired: {match_1}. "
        "This means a config or state invariant was violated. "
        "Check the config values and the config_assert_code / state_assert_code "
        "in the spec. This is a coder-level bug.",
    ),
    # Syntax (should have been caught by validator — escalate)
    (
        r"SyntaxError: (.+)",
        BugKind.SYNTAX_ERROR, False,
        "SyntaxError in the generated script: {match_1}. "
        "This should have been caught by SpecValidator. "
        "Re-run the validator on the spec and check for B-tier errors.",
    ),
    # Runtime errors
    (
        r"RecursionError",
        BugKind.RUNTIME_ERROR, False,
        "Recursion limit exceeded. The simulation step_code calls itself recursively "
        "or has very deep call chains. Refactor to use iteration.",
    ),
    (
        r"MemoryError|Cannot allocate memory|Killed",
        BugKind.OOM, True,
        "Out of memory. Reduce N or max_steps, or run on a machine with more RAM.",
    ),
    (
        r"OSError: \[Errno 28\]|No space left on device",
        BugKind.DISK_FULL, True,
        "Disk full. Free space in the output directory or redirect to a larger volume.",
    ),
    (
        r"TimeoutExpired|Process timed out",
        BugKind.TIMEOUT, True,
        "Simulation exceeded the timeout. Increase timeout_seconds, reduce max_steps, "
        "or check for an infinite loop in step_code.",
    ),
    # Signals
    (
        r"Segmentation fault|signal 11|SIGSEGV",
        BugKind.SIGNAL, False,
        "Segmentation fault (SIGSEGV). This is a C-level crash, likely in a native "
        "extension (numpy, scipy). Check for out-of-bounds array access.",
    ),
    (
        r"Aborted|signal 6|SIGABRT",
        BugKind.SIGNAL, False,
        "Process aborted (SIGABRT). Check for numpy/scipy assertion failures "
        "or double-free in native extensions.",
    ),
    # Generic runtime (catch-all before UNKNOWN)
    (
        r"(Error|Exception|Traceback).*\n.*\n.*",
        BugKind.RUNTIME_ERROR, False,
        "An unclassified exception occurred. See stderr_tail for the full traceback. "
        "The most common causes are: wrong data types in step_code, "
        "list index out of range, or key not found in precomputed dict.",
    ),
]

_ticket_validator = BugTicketValidator()


@dataclass(frozen=True)
class LaunchRuntimeConfig:
    primary_mode: str = "docker"
    fallback_mode: str = "process"
    docker_binary: str = "docker"
    docker_image: str = "research-platform-launcher:latest"
    docker_workdir: str = "/workspace"
    docker_auto_pull: bool = False

    @classmethod
    def from_env(cls) -> "LaunchRuntimeConfig":
        return cls(
            primary_mode=os.environ.get("SIM_TOOL_LAUNCH_MODE", "docker").strip().lower(),
            fallback_mode=os.environ.get("SIM_TOOL_LAUNCH_FALLBACK", "process").strip().lower(),
            docker_binary=os.environ.get("SIM_TOOL_DOCKER_BINARY", "docker").strip() or "docker",
            docker_image=os.environ.get("SIM_TOOL_DOCKER_IMAGE", "research-platform-launcher:latest").strip() or "research-platform-launcher:latest",
            docker_workdir=os.environ.get("SIM_TOOL_DOCKER_WORKDIR", "/workspace").strip() or "/workspace",
            docker_auto_pull=_parse_bool(os.environ.get("SIM_TOOL_DOCKER_AUTO_PULL", "")),
        )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────────────
# Environment probe
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _EnvProbe:
    uv_path:    Optional[str]
    uv_version: str
    python:     str   # the interpreter to use
    has_venv:   bool
    venv_dir:   Optional[Path]


def _probe_environment(session_dir: Path) -> _EnvProbe:
    """Detect available tools. Deterministic — reads filesystem and runs --version."""
    uv_path = shutil.which("uv")
    uv_version = ""
    if uv_path:
        try:
            out = subprocess.check_output(
                [uv_path, "--version"], text=True, timeout=5
            ).strip()
            uv_version = out.split()[-1] if out else ""
        except (subprocess.SubprocessError, OSError):
            uv_path = None

    venv_dir = session_dir / ".venv"
    has_venv = (venv_dir / "bin" / "python").exists() or (venv_dir / "Scripts" / "python.exe").exists()

    # Prefer: uv's managed python > venv python > sys.executable
    if uv_path:
        python = uv_path  # will be called as `uv run python`
    elif has_venv:
        python_bin = venv_dir / "bin" / "python"
        python = str(python_bin) if python_bin.exists() else sys.executable
    else:
        python = sys.executable

    return _EnvProbe(
        uv_path=uv_path,
        uv_version=uv_version,
        python=python,
        has_venv=has_venv,
        venv_dir=venv_dir if has_venv else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Import extraction from setup_code
# ─────────────────────────────────────────────────────────────────────────────

def _extract_imports(setup_code: str) -> list[str]:
    """
    Parse setup_code and return the top-level package names being imported.
    Returns installable package names (e.g. "numpy", "scikit-learn").
    """
    if not setup_code or not setup_code.strip():
        return []
    packages: list[str] = []
    try:
        tree = ast.parse(textwrap.dedent(setup_code), mode="exec")
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in _KNOWN_PACKAGES:
                        packages.append(_KNOWN_PACKAGES[top])
            else:
                top = (node.module or "").split(".")[0]
                if top in _KNOWN_PACKAGES:
                    packages.append(_KNOWN_PACKAGES[top])
    return list(dict.fromkeys(packages))  # deduplicated, order preserved


# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_environment(
    env: _EnvProbe,
    packages: list[str],
    session_dir: Path,
) -> tuple[EnvironmentInfo, list[str]]:
    """
    Install missing packages and return (EnvironmentInfo, base_cmd_prefix).
    base_cmd_prefix: list of tokens to prepend to every subprocess command.
    """
    installed: list[str] = []
    base_cmd: list[str] = []

    if env.uv_path:
        # Use uv — create a per-session venv if it doesn't exist
        venv_dir = session_dir / ".venv"
        if not (venv_dir / "bin" / "python").exists():
            log.info(f"Creating uv venv at {venv_dir}")
            subprocess.run(
                [env.uv_path, "venv", str(venv_dir)],
                check=True, capture_output=True, text=True,
            )
        if packages:
            log.info(f"Installing packages via uv: {packages}")
            result = subprocess.run(
                [env.uv_path, "pip", "install", "--quiet",
                 f"--python={venv_dir / 'bin' / 'python'}", *packages],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                installed = packages
            else:
                log.warning(f"uv pip install failed: {result.stderr[:200]}")

        python_in_venv = str(venv_dir / "bin" / "python")
        base_cmd = [python_in_venv]

        # Get python version
        try:
            ver = subprocess.check_output(
                [python_in_venv, "--version"], text=True, timeout=5
            ).strip()
        except Exception:
            ver = "unknown"

        env_info = EnvironmentInfo(
            python_executable=python_in_venv,
            python_version=ver,
            uv_version=env.uv_version,
            packages_installed=installed,
            env_vars=dict(_FIXED_ENV_VARS),
            venv_path=str(venv_dir),
            ran_as_uv=True,
            execution_backend="process",
        )

    else:
        # Fall back to sys.executable
        if packages:
            log.info(f"Installing packages via pip: {packages}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", *packages],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                installed = packages
            else:
                log.warning(f"pip install failed: {result.stderr[:200]}")

        base_cmd = [sys.executable]
        try:
            ver = subprocess.check_output(
                [sys.executable, "--version"], text=True, timeout=5
            ).strip()
        except Exception:
            ver = "unknown"

        env_info = EnvironmentInfo(
            python_executable=sys.executable,
            python_version=ver,
            uv_version="",
            packages_installed=installed,
            env_vars=dict(_FIXED_ENV_VARS),
            venv_path="",
            ran_as_uv=False,
            execution_backend="process",
        )

    return env_info, base_cmd


def _docker_ready(runtime: LaunchRuntimeConfig) -> bool:
    docker_bin = shutil.which(runtime.docker_binary)
    if not docker_bin:
        return False
    try:
        info = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    if info.returncode != 0:
        return False
    if runtime.docker_auto_pull:
        return True
    try:
        inspect = subprocess.run(
            [docker_bin, "image", "inspect", runtime.docker_image],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return inspect.returncode == 0


def _select_launch_mode(runtime: LaunchRuntimeConfig) -> str:
    primary = runtime.primary_mode or "process"
    if primary == "docker" and _docker_ready(runtime):
        return "docker"
    if primary == "process":
        return "process"
    fallback = runtime.fallback_mode or "process"
    if fallback == "docker" and _docker_ready(runtime):
        return "docker"
    return "process"


def _build_docker_env_info(
    runtime: LaunchRuntimeConfig,
    packages: list[str],
    extra_env: Optional[dict] = None,
) -> EnvironmentInfo:
    env_vars = {**_FIXED_ENV_VARS, **(extra_env or {})}
    return EnvironmentInfo(
        python_executable="python",
        python_version="container-managed",
        uv_version="",
        packages_installed=list(packages),
        env_vars=env_vars,
        venv_path="",
        ran_as_uv=False,
        execution_backend="docker",
        container_image=runtime.docker_image,
    )


def _path_anchor(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists() and resolved.is_dir():
        return resolved
    return resolved.parent


def _shared_mount_root(*paths: Optional[Path]) -> Path:
    anchors = [str(_path_anchor(Path(p))) for p in paths if p is not None]
    if not anchors:
        return Path.cwd().resolve()
    return Path(os.path.commonpath(anchors))


def _container_path(host_path: Path, mount_root: Path, container_root: str) -> str:
    relative = Path(host_path).resolve().relative_to(mount_root)
    return str(PurePosixPath(container_root) / PurePosixPath(relative.as_posix()))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline script writer
# ─────────────────────────────────────────────────────────────────────────────

def write_pipeline_script(
    session_id: str,
    script_path: Path,
    config: dict,
    output_dir: Path,
    env_info: EnvironmentInfo,
    preprocess_script: Optional[Path] = None,
    postprocess_script: Optional[Path] = None,
    timeout_seconds: Optional[float] = None,
    pipeline_path: Optional[Path] = None,
    packages: Optional[list[str]] = None,
    launch_mode: str = "process",
    runtime: Optional[LaunchRuntimeConfig] = None,
) -> Path:
    """
    Write a self-contained shell script that reproduces the full pipeline.

    The script:
      - Sets required env vars
      - Optionally activates the venv
      - Runs preprocess → simulation → postprocess with proper exit handling
      - Logs start/end time and exit codes
      - Is idempotent: safe to re-run from scratch
    """
    runtime = runtime or LaunchRuntimeConfig.from_env()
    packages = list(packages or [])
    if pipeline_path is None:
        pipeline_path = output_dir / "run_pipeline.sh"

    python_bin = shlex.quote(env_info.python_executable)
    sim_args = _config_to_args(config)
    output_dir_q = shlex.quote(str(output_dir))
    results_json_q = shlex.quote(str(output_dir / "results.json"))
    sim_cmd = (
        f"{python_bin} {shlex.quote(str(script_path))} {sim_args} "
        f"--output-dir {output_dir_q} "
        f"--results-json {results_json_q}"
    ).strip()

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# ─────────────────────────────────────────────────────────────────",
        f"# sim_tool pipeline — session {session_id}",
        f"# Script  : {script_path}",
        f"# Python  : {python_bin}",
        f"# Backend : {launch_mode}",
        f"# Outputs : {output_dir}",
        f"# Generated: auto — do not edit by hand",
        "# ─────────────────────────────────────────────────────────────────",
        "",
        "set -euo pipefail",
        "",
    ]

    if launch_mode == "docker":
        mount_root = _shared_mount_root(
            script_path,
            output_dir,
            preprocess_script,
            postprocess_script,
            pipeline_path,
        )
        container_script = _container_path(script_path, mount_root, runtime.docker_workdir)
        container_output = _container_path(output_dir, mount_root, runtime.docker_workdir)
        container_results = _container_path(output_dir / "results.json", mount_root, runtime.docker_workdir)
        inner_lines = [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(container_output)}",
        ]
        if packages:
            pkg_args = " ".join(shlex.quote(pkg) for pkg in packages)
            inner_lines.append(
                f"python -m pip install --quiet --disable-pip-version-check {pkg_args}"
            )
        if preprocess_script:
            preprocess_container = _container_path(
                preprocess_script, mount_root, runtime.docker_workdir
            )
            inner_lines += [
                f'echo "[launcher] Running preprocess: {preprocess_script.name}"',
                f"python {shlex.quote(preprocess_container)} --output-dir {shlex.quote(container_output)}",
            ]
        inner_sim = (
            f"python {shlex.quote(container_script)} {sim_args} "
            f"--output-dir {shlex.quote(container_output)} "
            f"--results-json {shlex.quote(container_results)}"
        ).strip()
        inner_lines += [
            f'echo "[launcher] Running simulation: {script_path.name}"',
            "set +e",
            inner_sim,
            "SIM_EXIT=$?",
            "set -e",
            'echo "[launcher] Simulation exit code: $SIM_EXIT"',
            'if [ "$SIM_EXIT" -ne 0 ]; then',
            '    echo "[launcher] ERROR: simulation failed with exit $SIM_EXIT" >&2',
            '    exit "$SIM_EXIT"',
            "fi",
        ]
        if postprocess_script:
            postprocess_container = _container_path(
                postprocess_script, mount_root, runtime.docker_workdir
            )
            inner_lines += [
                f'echo "[launcher] Running postprocess: {postprocess_script.name}"',
                f"python {shlex.quote(postprocess_container)} "
                f"--input-dir {shlex.quote(container_output)} "
                f"--output-dir {shlex.quote(container_output)}",
            ]
        inner_script = "\n".join(inner_lines)
        docker_cmd_parts = [
            shlex.quote(runtime.docker_binary),
            "run",
            "--rm",
            "-v",
            shlex.quote(f"{mount_root}:{runtime.docker_workdir}"),
            "-w",
            shlex.quote(runtime.docker_workdir),
        ]
        for key, value in env_info.env_vars.items():
            docker_cmd_parts += ["-e", shlex.quote(f"{key}={value}")]
        docker_cmd_parts += [
            shlex.quote(runtime.docker_image),
            "bash",
            "-lc",
            shlex.quote(inner_script),
        ]
        lines += [
            "# ── Docker ───────────────────────────────────────────────────────",
            f'OUTPUT_DIR="{output_dir}"',
            'mkdir -p "$OUTPUT_DIR"',
            'START_TIME=$(date +%s)',
            'echo "[launcher] Pipeline started: $(date)"',
            f'echo "[launcher] Docker image: {runtime.docker_image}"',
            'echo "[launcher] Output directory: $OUTPUT_DIR"',
            "",
            "set +e",
            " ".join(docker_cmd_parts),
            "SIM_EXIT=$?",
            "set -e",
            'echo "[launcher] Container exit code: $SIM_EXIT"',
            'if [ "$SIM_EXIT" -ne 0 ]; then',
            '    echo "[launcher] ERROR: container failed with exit $SIM_EXIT" >&2',
            '    exit "$SIM_EXIT"',
            "fi",
            "",
        ]
    else:
        lines += [
            "# ── Environment ──────────────────────────────────────────────────",
        ]
        for key, value in env_info.env_vars.items():
            lines.append(f'export {key}="{value}"')

        if env_info.venv_path:
            lines += [
                "",
                f'VENV="{env_info.venv_path}"',
                'if [ -f "$VENV/bin/activate" ]; then',
                '    source "$VENV/bin/activate"',
                '    echo "[launcher] venv activated: $VENV"',
                'else',
                '    echo "[launcher] WARNING: venv not found at $VENV" >&2',
                'fi',
            ]

        lines += [
            "",
            f'OUTPUT_DIR="{output_dir}"',
            'mkdir -p "$OUTPUT_DIR"',
            "",
            'START_TIME=$(date +%s)',
            'echo "[launcher] Pipeline started: $(date)"',
            'echo "[launcher] Output directory: $OUTPUT_DIR"',
            "",
        ]

        if preprocess_script:
            lines += [
                "# ── Preprocess ───────────────────────────────────────────────────",
                f'echo "[launcher] Running preprocess: {preprocess_script}"',
                f"{python_bin} {shlex.quote(str(preprocess_script))} --output-dir \"$OUTPUT_DIR\"",
                'echo "[launcher] Preprocess complete"',
                "",
            ]

        lines += [
            "# ── Simulation ───────────────────────────────────────────────────",
            f'echo "[launcher] Running simulation: {script_path.name}"',
            "set +e",
            sim_cmd,
            "SIM_EXIT=$?",
            "set -e",
            'echo "[launcher] Simulation exit code: $SIM_EXIT"',
            'if [ "$SIM_EXIT" -ne 0 ]; then',
            '    echo "[launcher] ERROR: simulation failed with exit $SIM_EXIT" >&2',
            '    exit "$SIM_EXIT"',
            "fi",
            "",
        ]

        if postprocess_script:
            lines += [
                "# ── Postprocess ──────────────────────────────────────────────────",
                f'echo "[launcher] Running postprocess: {postprocess_script}"',
                f"{python_bin} {shlex.quote(str(postprocess_script))} "
                f"--input-dir \"$OUTPUT_DIR\" --output-dir \"$OUTPUT_DIR\"",
                'echo "[launcher] Postprocess complete"',
                "",
            ]

    lines += [
        "# ── Summary ──────────────────────────────────────────────────────",
        'END_TIME=$(date +%s)',
        'ELAPSED=$((END_TIME - START_TIME))',
        'echo "[launcher] Pipeline finished in ${ELAPSED}s"',
        'echo "[launcher] Results: $OUTPUT_DIR/results.json"',
        'echo "[launcher] Data   : $OUTPUT_DIR/data_log.jsonl"',
    ]

    script_text = "\n".join(lines) + "\n"
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.write_text(script_text, encoding="utf-8")
    pipeline_path.chmod(pipeline_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
    log.info(f"Pipeline script written: {pipeline_path}")
    return pipeline_path


def _config_to_args(config: dict) -> str:
    """Convert a config dict to CLI argument string."""
    parts: list[str] = []
    for k, v in config.items():
        flag = f"--{k.replace('_', '-')}"
        parts.append(f"{flag} {shlex.quote(str(v))}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Failure classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_failure(
    exit_code: int,
    stdout: str,
    stderr: str,
    script_path: Path,
    session_id: str,
    config: dict,
    env_info: EnvironmentInfo,
    wall_time: float,
    pipeline_script_path: Optional[str],
    reproducible_command: str,
) -> BugTicket:
    """
    Pattern-match stdout/stderr to produce a typed, validated BugTicket.
    Returns a valid ticket or raises if ticket validation itself fails (bug in launcher).
    """
    combined = stderr + "\n" + stdout
    stdout_tail = _tail(stdout, _STDOUT_TAIL_LINES)
    stderr_tail = _tail(stderr, _STDERR_TAIL_LINES)

    kind        = BugKind.UNKNOWN
    pattern_str = ""
    error_line  = ""
    fix         = "Inspect the stderr_tail for the root cause and fix the generated code."
    is_retryable = False

    for regex, bug_kind, retryable, fix_template in _FAILURE_PATTERNS:
        m = re.search(regex, combined, re.MULTILINE | re.DOTALL)
        if m:
            kind = bug_kind
            is_retryable = retryable
            pattern_str = regex
            # Fill in {match_1}, {match_2} placeholders in fix template
            fix = fix_template
            for i, group in enumerate(m.groups(), 1):
                fix = fix.replace(f"{{match_{i}}}", group or "")
            # Extract the specific error line
            for line in combined.splitlines():
                if re.search(regex.split(r"\n")[0], line):
                    error_line = line.strip()
                    break
            break

    ticket = BugTicket(
        kind=kind,
        session_id=session_id,
        script_path=str(script_path),
        pipeline_script_path=pipeline_script_path,
        exit_code=exit_code,
        wall_time_seconds=wall_time,
        python_executable=env_info.python_executable,
        uv_version=env_info.uv_version,
        env_vars_set=env_info.env_vars,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        failure_pattern=pattern_str,
        error_line=error_line,
        reproducible_command=reproducible_command,
        suggested_fix=fix,
        is_retryable=is_retryable,
    )

    validation = _ticket_validator.validate(ticket)
    if not validation.passed:
        # This is a bug in the launcher itself — log loudly and patch the ticket
        log.error(
            f"[{session_id}] BugTicket failed its own validation:\n"
            f"{validation.summary()}"
        )
        # Patch rather than crash — the ticket still contains useful information
        if not ticket.suggested_fix.strip():
            ticket = BugTicket(
                **{**ticket.__dict__,
                   "suggested_fix": "Inspect the stderr_tail for the root cause."}
            )

    return ticket


def _tail(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


# ─────────────────────────────────────────────────────────────────────────────
# Launcher
# ─────────────────────────────────────────────────────────────────────────────

class Launcher:
    """
    Deterministic launch tool. No LLM. Given the same inputs, produces the
    same outputs (modulo filesystem state and network).

    Parameters
    ----------
    log_dir  : Directory for launcher_log.md. Default: ~/.sim_tool
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        runtime: Optional[LaunchRuntimeConfig] = None,
    ) -> None:
        self._log = LauncherLog(log_dir=log_dir)
        self._runtime = runtime or LaunchRuntimeConfig.from_env()
        log.info(
            f"Launcher ready | "
            f"preferred_mode={self._runtime.primary_mode} | "
            f"log={self._log.path}"
        )

    @property
    def launcher_log(self) -> LauncherLog:
        return self._log

    # ── Public API ────────────────────────────────────────────────────────────

    def launch(
        self,
        session_id: str,
        script_path: Path,
        config: dict,
        output_dir: Path,
        setup_code: str = "",
        preprocess_script: Optional[Path] = None,
        postprocess_script: Optional[Path] = None,
        timeout_seconds: Optional[float] = None,
        stream_logs: bool = True,
        extra_env: Optional[dict] = None,
    ) -> Union[LaunchResult, BugTicket]:
        """
        Launch a simulation script with full environment setup.

        Returns LaunchResult on success, BugTicket on any failure.
        Both are validated before return.

        Parameters
        ----------
        session_id        : Orchestrator session identifier.
        script_path       : Path to the generated .py simulation script.
        config            : Dict of variable name → value (CLI args).
        output_dir        : Directory for results.json and data_log.jsonl.
        setup_code        : The spec's setup_code block (used to extract
                            required packages for auto-install).
        preprocess_script : Optional script to run before the simulation.
        postprocess_script: Optional script to run after the simulation.
        timeout_seconds   : Kill the subprocess after this many seconds.
        stream_logs       : Echo subprocess stdout to our logger.
        extra_env         : Additional environment variables to set.
        """
        script_path = Path(script_path).resolve()
        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        session_dir = output_dir  # venv lives next to output

        log.info(f"[{session_id}] Launcher.launch() → {script_path.name}")

        packages = _extract_imports(setup_code)
        launch_mode = _select_launch_mode(self._runtime)
        if launch_mode != self._runtime.primary_mode:
            log.info(
                f"[{session_id}] Preferred mode {self._runtime.primary_mode!r} "
                f"unavailable; falling back to {launch_mode!r}"
            )
        log.info(
            f"[{session_id}] Mode={launch_mode} | packages needed: {packages or '(none)'}"
        )

        if launch_mode == "docker":
            env_info = _build_docker_env_info(self._runtime, packages, extra_env=extra_env)
        else:
            env = _probe_environment(session_dir)
            env_info, _ = _setup_environment(env, packages, session_dir)
            env_info.env_vars.update(extra_env or {})

        # ── Step 2: Write pipeline script ─────────────────────────────────────
        pipeline_path = write_pipeline_script(
            session_id=session_id,
            script_path=script_path,
            config=config,
            output_dir=output_dir,
            env_info=env_info,
            preprocess_script=preprocess_script,
            postprocess_script=postprocess_script,
            timeout_seconds=timeout_seconds,
            packages=packages,
            launch_mode=launch_mode,
            runtime=self._runtime,
        )

        results_json = output_dir / "results.json"
        cmd = ["bash", str(pipeline_path)]
        reproducible_cmd = " ".join(shlex.quote(str(t)) for t in cmd)

        log.info(f"[{session_id}] Command: {reproducible_cmd}")
        log.debug(f"[{session_id}] Pipeline script: {pipeline_path}")

        # ── Step 4: Execute ───────────────────────────────────────────────────
        wall_start = time.perf_counter()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            # Stream stdout, capture both
            import threading

            def _reader(stream, store):
                for line in iter(stream.readline, ""):
                    store.append(line.rstrip())
                    if stream_logs:
                        log.info(f"[sim] {line.rstrip()}")
                stream.close()

            t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_lines))
            t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_lines))
            t_out.start(); t_err.start()

            if timeout_seconds:
                try:
                    proc.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    t_out.join(timeout=2); t_err.join(timeout=2)
                    wall_time = time.perf_counter() - wall_start
                    ticket = _classify_failure(
                        exit_code=-1,
                        stdout="\n".join(stdout_lines) + "\nProcess timed out",
                        stderr="\n".join(stderr_lines),
                        script_path=script_path,
                        session_id=session_id,
                        config=config,
                        env_info=env_info,
                        wall_time=wall_time,
                        pipeline_script_path=str(pipeline_path),
                        reproducible_command=reproducible_cmd,
                    )
                    self._log.record_failure(session_id, str(script_path), ticket)
                    return ticket
            else:
                proc.wait()

            t_out.join(); t_err.join()
            exit_code = proc.returncode

        except FileNotFoundError as exc:
            wall_time = time.perf_counter() - wall_start
            log.error(f"[{session_id}] Executable not found: {exc}")
            ticket = _classify_failure(
                exit_code=127,
                stdout="",
                stderr=f"python: command not found\n{exc}",
                script_path=script_path,
                session_id=session_id,
                config=config,
                env_info=env_info,
                wall_time=wall_time,
                pipeline_script_path=str(pipeline_path),
                reproducible_command=reproducible_cmd,
            )
            self._log.record_failure(session_id, str(script_path), ticket)
            return ticket

        wall_time = time.perf_counter() - wall_start
        stdout_text = "\n".join(stdout_lines)
        stderr_text = "\n".join(stderr_lines)

        # ── Step 5: Classify outcome ──────────────────────────────────────────
        if exit_code != 0:
            log.warning(
                f"[{session_id}] Process exited {exit_code} after {wall_time:.1f}s"
            )
            ticket = _classify_failure(
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                script_path=script_path,
                session_id=session_id,
                config=config,
                env_info=env_info,
                wall_time=wall_time,
                pipeline_script_path=str(pipeline_path),
                reproducible_command=reproducible_cmd,
            )
            self._log.record_failure(session_id, str(script_path), ticket)
            return ticket

        # ── Step 6: Collect results ───────────────────────────────────────────
        run_result = self._collect_result(
            results_json=results_json,
            output_dir=output_dir,
            config=config,
            wall_time=wall_time,
        )
        summary = RunSummary(
            session_id=session_id,
            results=[run_result],
            sweep_dir=output_dir,
        )
        if env_info.execution_backend == "docker":
            env_summary = f"docker:{env_info.container_image}"
        else:
            env_summary = (
                f"uv {env_info.uv_version}" if env_info.ran_as_uv
                else env_info.python_executable
            )
        self._log.record_success(
            session_id=session_id,
            script_path=str(script_path),
            env_summary=env_summary,
            wall_time=wall_time,
            configs_run=1,
        )
        log.info(
            f"[{session_id}] Launch succeeded: "
            f"status={run_result.outcome.value} in {wall_time:.1f}s"
        )
        entry = f"[{session_id}] success {wall_time:.1f}s"
        return LaunchResult(
            run_summary=summary,
            env_info=env_info,
            pipeline_script_path=str(pipeline_path),
            launcher_log_entry=entry,
        )

    def launch_sweep(
        self,
        session_id: str,
        script_path: Path,
        configs: list[dict],
        base_output_dir: Path,
        setup_code: str = "",
        preprocess_script: Optional[Path] = None,
        postprocess_script: Optional[Path] = None,
        timeout_per_run: Optional[float] = None,
        max_workers: int = 1,
    ) -> tuple[list[LaunchResult], list[BugTicket]]:
        """
        Launch a parameter sweep. Returns (successes, failures).

        Runs sequentially (max_workers=1) or in parallel.
        Each config gets its own output subdirectory and pipeline script.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        base_output_dir = Path(base_output_dir)

        def _run_one(cfg: dict) -> Union[LaunchResult, BugTicket]:
            tag = _config_tag(cfg)
            out = base_output_dir / tag
            return self.launch(
                session_id=session_id,
                script_path=script_path,
                config=cfg,
                output_dir=out,
                setup_code=setup_code,
                preprocess_script=preprocess_script,
                postprocess_script=postprocess_script,
                timeout_seconds=timeout_per_run,
                stream_logs=(max_workers == 1),
            )

        successes: list[LaunchResult] = []
        failures:  list[BugTicket]   = []

        log.info(
            f"[{session_id}] Sweep: {len(configs)} configs, "
            f"{max_workers} worker(s)"
        )

        if max_workers <= 1:
            for i, cfg in enumerate(configs, 1):
                log.info(f"[{session_id}] Config {i}/{len(configs)}: {cfg}")
                result = _run_one(cfg)
                (successes if isinstance(result, LaunchResult) else failures).append(result)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        (successes if isinstance(result, LaunchResult) else failures).append(result)
                    except Exception as exc:
                        log.error(f"[{session_id}] Sweep worker raised: {exc}")

        log.info(
            f"[{session_id}] Sweep complete: "
            f"{len(successes)} ok, {len(failures)} failed"
        )
        return successes, failures

    # ── Internal ──────────────────────────────────────────────────────────────

    def _collect_result(
        self,
        results_json: Path,
        output_dir: Path,
        config: dict,
        wall_time: float,
    ) -> RunResult:
        """Read results.json and return a RunResult."""
        data_file = output_dir / "data_log.jsonl"
        if results_json.exists():
            try:
                data = json.loads(results_json.read_text())
                return RunResult(
                    config=config,
                    outcome=RunOutcome(data.get("status", "max_steps")),
                    reason=data.get("reason", ""),
                    stop_condition_name=data.get("stop_condition_name", ""),
                    steps_run=int(data.get("steps_run", 0)),
                    wall_time_seconds=data.get("wall_time_seconds", wall_time),
                    output_dir=results_json.parent,
                    data_file=data_file if data_file.exists() else None,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        return RunResult(
            config=config,
            outcome=RunOutcome.MAX_STEPS,
            reason="results.json not found or unreadable",
            stop_condition_name="",
            steps_run=0,
            wall_time_seconds=wall_time,
            output_dir=output_dir,
            data_file=data_file if data_file.exists() else None,
        )


def _config_tag(config: dict) -> str:
    """Filesystem-safe tag for a config dict."""
    swept_items = [(k, v) for k, v in config.items()
                   if k not in ("max_steps", "checkpoint_interval",
                                "progress_interval", "data_log_interval")]
    if not swept_items:
        return "default"
    tag = "__".join(f"{k}={v}" for k, v in swept_items)
    return "".join(c if c.isalnum() or c in "-_=." else "_" for c in tag)[:120]
