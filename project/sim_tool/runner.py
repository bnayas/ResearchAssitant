"""
sim_tool.runner
───────────────
Executes generated simulation scripts as isolated subprocesses.

Features:
  • Single-run execution with live stdout streaming
  • Parameter sweep across variable ranges (sequential or parallel)
  • Timeout enforcement
  • Structured result collection from JSON output files
  • Sweep summary with pass/fail/timeout breakdown
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Optional

from .models import SimulationSpec, Variable, VariableKind

log = logging.getLogger("sim_tool.runner")


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunOutcome:
    config:           dict[str, Any]
    status:           str             # "success" | "failed" | "max_steps" | "timeout" | "crash"
    reason:           str
    stop_condition:   str
    steps_run:        int
    wall_time_seconds: float
    output_dir:       Path
    result_json_path: Optional[Path]
    history_length:   int = 0
    error_message:    str = ""        # populated on crash/timeout


@dataclass
class SweepSummary:
    spec_name:    str
    total_runs:   int
    outcomes:     list[RunOutcome] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "success")

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    @property
    def timeout_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "timeout")

    @property
    def crash_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "crash")

    def print_summary(self) -> None:
        log.info("=" * 70)
        log.info(f"SWEEP SUMMARY : {self.spec_name}")
        log.info(f"  Total runs  : {self.total_runs}")
        log.info(f"  Success     : {self.success_count}")
        log.info(f"  Failed      : {self.failed_count}  (stopped early — bad config)")
        log.info(f"  Max steps   : {sum(1 for o in self.outcomes if o.status == 'max_steps')}")
        log.info(f"  Timeout     : {self.timeout_count}")
        log.info(f"  Crash       : {self.crash_count}")
        log.info("─" * 70)
        log.info("Individual outcomes:")
        for o in self.outcomes:
            cfg_str = " | ".join(f"{k}={v}" for k, v in o.config.items())
            log.info(f"  [{o.status.upper():<9}] {cfg_str}")
            log.info(f"             reason={o.reason!r}  steps={o.steps_run:,}  wall={o.wall_time_seconds:.1f}s")
        log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_script(
    script_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    timeout_seconds: Optional[float] = None,
    stream_logs: bool = True,
) -> RunOutcome:
    """
    Execute a generated simulation script with `config` as CLI arguments.

    Args:
        script_path:     Path to the generated .py file.
        config:          Dict of variable name → value (CLI args).
        output_dir:      Directory for checkpoints and result JSON.
        timeout_seconds: Kill the process after this many seconds (None = unlimited).
        stream_logs:     If True, echo the subprocess stdout to our logger in real-time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "results.json"

    cmd = [sys.executable, str(script_path)]
    for k, v in config.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    cmd += ["--output-dir", str(output_dir), "--results-json", str(results_json)]

    log.info("─" * 70)
    log.info(f"LAUNCHING : {script_path.name}")
    log.info(f"Config    : {config}")
    log.info(f"Output    : {output_dir}")
    if timeout_seconds:
        log.info(f"Timeout   : {timeout_seconds}s")
    log.debug(f"Full cmd  : {' '.join(cmd)}")

    wall_start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        stdout_lines: list[str] = []
        def _reader() -> None:
            try:
                for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
                    line = line.rstrip()
                    stdout_lines.append(line)
                    if stream_logs:
                        log.info(f"[sim] {line}")
            finally:
                if proc.stdout:
                    proc.stdout.close()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            reader.join(timeout=2)
            log.warning(f"TIMEOUT: Process killed after {timeout_seconds}s")
            return RunOutcome(
                config=config,
                status="timeout",
                reason=f"Killed after {timeout_seconds}s timeout",
                stop_condition="timeout",
                steps_run=0,
                wall_time_seconds=time.perf_counter() - wall_start,
                output_dir=output_dir,
                result_json_path=None,
                error_message="Process timed out",
            )

        reader.join()
        elapsed = time.perf_counter() - wall_start

    except FileNotFoundError:
        return RunOutcome(
            config=config,
            status="crash",
            reason="Script not found",
            stop_condition="crash",
            steps_run=0,
            wall_time_seconds=0,
            output_dir=output_dir,
            result_json_path=None,
            error_message=f"Script not found: {script_path}",
        )
    except Exception as exc:
        return RunOutcome(
            config=config,
            status="crash",
            reason=f"Unexpected error: {exc}",
            stop_condition="crash",
            steps_run=0,
            wall_time_seconds=time.perf_counter() - wall_start,
            output_dir=output_dir,
            result_json_path=None,
            error_message=str(exc),
        )

    # ── Read JSON results ─────────────────────────────────────────────────────
    if results_json.exists():
        try:
            data = json.loads(results_json.read_text())
            return RunOutcome(
                config=config,
                status=data.get("status", "unknown"),
                reason=data.get("reason", ""),
                stop_condition=data.get("stop_condition_name", ""),
                steps_run=data.get("steps_run", 0),
                wall_time_seconds=data.get("wall_time_seconds", elapsed),
                output_dir=output_dir,
                result_json_path=results_json,
                history_length=len(data.get("history", [])),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning(f"Could not parse results JSON: {exc}")

    # Fallback: derive status from exit code
    status = "success" if proc.returncode == 0 else "crash"
    return RunOutcome(
        config=config,
        status=status,
        reason=f"Exit code {proc.returncode}",
        stop_condition="",
        steps_run=0,
        wall_time_seconds=elapsed,
        output_dir=output_dir,
        result_json_path=None,
        error_message="" if proc.returncode == 0 else f"Non-zero exit {proc.returncode}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweep
# ─────────────────────────────────────────────────────────────────────────────

def build_sweep_configs(spec: SimulationSpec) -> list[dict[str, Any]]:
    """
    Build the Cartesian product of all sweep variable values.
    Non-swept variables use their defaults.
    """
    base = {v.name: v.default for v in spec.variables}
    sweep_vars = [v for v in spec.variables if v.sweep]

    if not sweep_vars:
        return [base]

    sweep_axes: list[list[tuple[str, Any]]] = []
    for v in sweep_vars:
        values = _get_sweep_values(v)
        sweep_axes.append([(v.name, val) for val in values])

    configs = []
    for combo in product(*sweep_axes):
        cfg = dict(base)
        for name, val in combo:
            cfg[name] = val
        configs.append(cfg)

    log.info(
        f"Sweep: {len(sweep_vars)} variables × "
        f"{' × '.join(str(len(ax)) for ax in sweep_axes)} = "
        f"{len(configs)} configurations"
    )
    return configs


def _get_sweep_values(v: Variable) -> list[Any]:
    if v.sweep_values:
        return v.sweep_values
    if v.kind == VariableKind.BOOL:
        return [False, True]
    if v.kind == VariableKind.CHOICE:
        return v.choices or [v.default]
    if v.min_val is not None and v.max_val is not None:
        step = v.step or (v.max_val - v.min_val) / 9
        vals = []
        cur = v.min_val
        while cur <= v.max_val + 1e-9:
            vals.append(int(round(cur)) if v.kind == VariableKind.INT else cur)
            cur += step
        return vals
    return [v.default]


def run_sweep(
    script_path: Path,
    spec: SimulationSpec,
    base_output_dir: Path,
    timeout_seconds: Optional[float] = None,
    max_workers: int = 1,
    stream_logs: bool = False,
) -> SweepSummary:
    """
    Run a full parameter sweep for all configurations implied by the spec.

    Args:
        max_workers: 1 = sequential; >1 = parallel (use with care on shared machines).
        stream_logs: Echo each subprocess's logs. Noisy for parallel runs.
    """
    configs = build_sweep_configs(spec)
    summary = SweepSummary(spec_name=spec.name, total_runs=len(configs))
    base_output_dir = Path(base_output_dir)

    log.info("=" * 70)
    log.info(f"PARAMETER SWEEP : {spec.name}")
    log.info(f"Configurations  : {len(configs)}")
    log.info(f"Parallelism     : {max_workers} worker(s)")
    log.info("=" * 70)

    if max_workers <= 1:
        for i, cfg in enumerate(configs, 1):
            cfg_tag = _config_tag(cfg, spec)
            out_dir = base_output_dir / cfg_tag
            log.info(f"Run {i}/{len(configs)} — {cfg_tag}")
            outcome = run_script(
                script_path, cfg, out_dir, timeout_seconds, stream_logs
            )
            summary.outcomes.append(outcome)
    else:
        futures_map: dict = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for cfg in configs:
                cfg_tag = _config_tag(cfg, spec)
                out_dir = base_output_dir / cfg_tag
                fut = pool.submit(
                    run_script, script_path, cfg, out_dir, timeout_seconds, False
                )
                futures_map[fut] = cfg_tag
            for fut in as_completed(futures_map):
                tag = futures_map[fut]
                try:
                    outcome = fut.result()
                except Exception as exc:
                    log.error(f"Run {tag} raised: {exc}")
                    outcome = RunOutcome(
                        config={},
                        status="crash",
                        reason=str(exc),
                        stop_condition="crash",
                        steps_run=0,
                        wall_time_seconds=0,
                        output_dir=base_output_dir / tag,
                        result_json_path=None,
                        error_message=str(exc),
                    )
                log.info(f"Completed [{outcome.status.upper()}] {tag}")
                summary.outcomes.append(outcome)

    summary.print_summary()
    return summary


def _config_tag(config: dict, spec: SimulationSpec) -> str:
    """Generate a short, filesystem-safe tag for a config dict."""
    sweep_names = {v.name for v in spec.variables if v.sweep}
    parts = [
        f"{k}={v}" for k, v in config.items()
        if k in sweep_names
    ]
    tag = "__".join(parts) or "default"
    # Sanitize for filesystem
    return "".join(c if c.isalnum() or c in "-_=." else "_" for c in tag)[:120]
