"""
sim_tool.codegen
────────────────
Generates a self-contained Python simulation script and Jupyter notebook
from a SimulationSpec.

Generated script features
─────────────────────────
  • Zero sim_tool imports — ships standalone.
  • Two loggers:
      algo_log — stdout, human-readable, for debugging the algorithm.
      data_log — append-only .jsonl file, for research data collection.
  • Three assertion tiers baked in: config validation, state invariants,
    plus stopping-condition guards.
  • One precompute() call before the loop — returns a dict used by step_code,
    stopping conditions, and progress_code.
  • Checkpoints every N steps as pickle files.
  • Full CLI with --help auto-generated from variables.
  • JSON results file written on exit.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .models import SimulationSpec, Variable, VariableKind, StoppingCondition


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def generate_script(spec: SimulationSpec, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_script(spec))
    output_path.chmod(0o755)
    return output_path


def generate_notebook(spec: SimulationSpec, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_build_notebook(spec), indent=2))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Script renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_script(spec: SimulationSpec) -> str:
    parts = [
        _render_header(spec),
        _FIXED_IMPORTS,
    ]
    if spec.setup_code.strip():
        parts += [
            "# ── User setup ───────────────────────────────────────────────────────────",
            textwrap.dedent(spec.setup_code).strip(),
        ]
    parts += [
        _LOGGING_BLOCK,
        _render_config(spec),
        _render_state(spec),
        _RESULT_DATACLASS,
        _render_assertions(spec),
        _render_precompute(spec),
        _render_sim_functions(spec),
        _render_stopping_dispatcher(spec),
        _render_runner(spec),
        _render_cli(spec),
    ]
    return "\n\n\n".join(parts) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Section renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_header(spec: SimulationSpec) -> str:
    out = ", ".join(spec.output_variables) or "(none)"
    data = ", ".join(spec.data_log_variables) or "(none)"
    return (
        f'#!/usr/bin/env python3\n'
        f'"""\n'
        f'{"═"*70}\n'
        f'  SIMULATION : {spec.name}\n'
        f'{"─"*70}\n'
        f'  {spec.description}\n'
        f'{"─"*70}\n'
        f'  Max steps        : {spec.max_steps:,}\n'
        f'  Est. time        : {spec.time_estimate_explanation or "unknown"}\n'
        f'  Checkpoint every : {spec.checkpoint_interval:,} steps\n'
        f'  Progress every   : {spec.progress_interval:,} steps\n'
        f'{"═"*70}\n\n'
        f'VARIABLES\n{"─"*9}\n{spec.format_variable_docs()}\n\n'
        f'STOPPING CONDITIONS\n{"─"*19}\n{spec.format_stop_docs()}\n\n'
        f'IN-MEMORY HISTORY  : {out}\n'
        f'RESEARCH DATA LOG  : {data}  (every {spec.data_log_interval} steps → data_log.jsonl)\n\n'
        f'USAGE\n{"─"*5}\n'
        f'  python {spec.name.lower().replace(" ","_")}.py --help\n'
        f'  python {spec.name.lower().replace(" ","_")}.py [--var VALUE ...]\n'
        f'         [--output-dir sim_output] [--results-json results.json]\n'
        f'"""\n'
    )


_FIXED_IMPORTS = """\
# ── Standard library ─────────────────────────────────────────────────────────
import sys
import json
import math
import copy
import time
import pickle
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional"""


_LOGGING_BLOCK = '''\
# ── Logging — TWO SEPARATE LOGGERS ────────────────────────────────────────────
#
#  algo_log : Software / algorithm tracing (stdout).
#             Use this to trace the algorithm's progress and debug issues.
#             Levels: DEBUG for per-step detail, INFO for milestones,
#                     WARNING for anomalies, ERROR for failures.
#             Rule: messages must be self-contained — include variable values.
#             Example:  algo_log.debug(f"step {s}: energy={E:.4f}, delta={d:.2e}")
#
#  data_log : Research data (append-only .jsonl file beside output dir).
#             One JSON object per line, written every data_log_interval steps.
#             Keys must be stable and self-descriptive.
#             Example:  {"step":5000,"energy_J":3.14,"temp_K":298.0,"accepted_frac":0.42}
#             Load in Python: pd.read_json("data_log.jsonl", lines=True)
#
def _setup_loggers(output_dir: Path) -> tuple[logging.Logger, logging.Logger]:
    # ── algo_log (stdout) ────────────────────────────────────────────────────
    algo = logging.getLogger("algo")
    if not algo.handlers:
        algo.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        algo.addHandler(ch)

    # ── data_log (jsonl file) ────────────────────────────────────────────────
    data = logging.getLogger("data")
    data_path = output_dir / "data_log.jsonl"
    if not data.handlers:
        data.setLevel(logging.INFO)
        fh = logging.FileHandler(data_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))  # raw JSON only
        data.addHandler(fh)

    return algo, data

# Module-level placeholders — replaced in run()
algo_log: logging.Logger = logging.getLogger("algo")
data_log: logging.Logger = logging.getLogger("data")'''


def _render_config(spec: SimulationSpec) -> str:
    type_map = {
        VariableKind.FLOAT: "float", VariableKind.INT: "int",
        VariableKind.BOOL: "bool", VariableKind.CHOICE: "str",
        VariableKind.STRING: "str",
    }
    lines = [
        "# ── Configuration ────────────────────────────────────────────────────────────",
        "@dataclass",
        "class SimConfig:",
    ]
    for v in spec.variables:
        comment = " | ".join(filter(None, [
            v.description,
            f"[{v.unit}]" if v.unit else "",
            v.range_str(),
        ]))
        lines += [
            f"    # {comment}",
            f"    {v.name}: {type_map[v.kind]} = {v.default!r}",
        ]
    lines += [
        "    # ── Runner knobs ─────────────────────────────────────────────────────────",
        f"    max_steps:           int = {spec.max_steps}",
        f"    checkpoint_interval: int = {spec.checkpoint_interval}",
        f"    progress_interval:   int = {spec.progress_interval}",
        f"    data_log_interval:   int = {spec.data_log_interval}",
    ]
    return "\n".join(lines)


def _render_state(spec: SimulationSpec) -> str:
    lines = [
        "# ── State ────────────────────────────────────────────────────────────────────",
        "@dataclass",
        "class SimState:",
        "    step:     int   = 0",
        "    sim_time: float = 0.0",
    ]
    for name, type_hint, default in spec.state_fields:
        if isinstance(default, list):
            lines.append(f"    {name}: {type_hint} = field(default_factory=list)")
        elif isinstance(default, dict):
            lines.append(f"    {name}: {type_hint} = field(default_factory=dict)")
        else:
            lines.append(f"    {name}: {type_hint} = {default!r}")
    lines += [
        "    history: list = field(default_factory=list)",
        "",
        "    def snapshot(self, keys: list[str]) -> dict:",
        "        d = {\"step\": self.step, \"sim_time\": self.sim_time}",
        "        for k in keys:",
        "            d[k] = getattr(self, k, None)",
        "        return d",
    ]
    return "\n".join(lines)


_RESULT_DATACLASS = """\
# ── Result ────────────────────────────────────────────────────────────────────
@dataclass
class SimResult:
    \"\"\"
    status: "success" | "failed" | "max_steps"
      success   — a success stopping condition was met
      failed    — a failure stopping condition was met (bad config; stopped early)
      max_steps — ran to the step limit without a decisive outcome
    \"\"\"
    status:              str
    reason:              str
    stop_condition_name: str
    steps_run:           int
    wall_time_seconds:   float
    final_state:         dict
    history:             list
    partial_save_paths:  list[str]
    data_log_path:       Optional[str]
    config:              dict
    metadata:            dict = field(default_factory=dict)"""


def _render_assertions(spec: SimulationSpec) -> str:
    config_body = textwrap.indent(
        textwrap.dedent(spec.config_assert_code).strip() or "pass", "    "
    )
    state_body = textwrap.indent(
        textwrap.dedent(spec.state_assert_code).strip() or "pass", "    "
    )
    return "\n".join([
        "# ── Assertions ───────────────────────────────────────────────────────────────",
        "#",
        "# TIER 1: config validation — called once before the loop.",
        "# Every assert must include the offending value in its message.",
        "def _assert_config(config: SimConfig) -> None:",
        config_body,
        "",
        "",
        "# TIER 2: state invariants — called every step. Keep O(1).",
        "# Catches NaN/Inf blowups and physical constraint violations early.",
        "def _assert_state(config: SimConfig, state: SimState) -> None:",
        state_body,
    ])


def _render_precompute(spec: SimulationSpec) -> str:
    body = textwrap.indent(
        textwrap.dedent(spec.precompute_code).strip() or "return {}", "    "
    )
    return "\n".join([
        "# ── Precompute ───────────────────────────────────────────────────────────────",
        "# Runs ONCE before the loop. Anything that depends only on config belongs here.",
        "# Return a plain dict; it is passed as `precomputed` to step/stop/progress.",
        "def precompute(config: SimConfig) -> dict:",
        body,
    ])


def _render_sim_functions(spec: SimulationSpec) -> str:
    def ibody(code: str) -> str:
        return textwrap.indent(textwrap.dedent(code).strip(), "    ")

    return "\n\n".join([
        (
            "# ── Simulation logic (generated) ─────────────────────────────────────────────\n"
            "def initial_state(config: SimConfig, precomputed: dict) -> SimState:\n"
            + ibody(spec.initial_state_code)
        ),
        (
            "def sim_step(config: SimConfig, state: SimState, precomputed: dict) -> SimState:\n"
            + ibody(spec.step_code)
        ),
        (
            "def progress_summary(state: SimState, precomputed: dict) -> str:\n"
            + ibody(spec.progress_code)
        ),
    ])


def _render_stopping_dispatcher(spec: SimulationSpec) -> str:
    lines = [
        "# ── Stopping conditions (generated) ──────────────────────────────────────────",
        "# TIER 3 assertions: detect physically hopeless states.",
        "# Conditions are evaluated in priority order (highest first).",
        "# `precomputed` is available alongside `config` and `state`.",
        "def _check_stopping(",
        "    config: SimConfig, state: SimState, precomputed: dict",
        ") -> tuple[Optional[str], str, str]:",
        '    """Returns (status, reason, condition_name) or (None, "", "") to continue."""',
    ]

    for cond in sorted(spec.stopping_conditions, key=lambda s: -s.priority):
        status = "success" if cond.kind == "success" else "failed"
        lines += [
            f"    # [{status.upper()}] {cond.name} — {cond.description}",
            f"    try:",
            f"        if {cond.check_expr}:",
            f"            return {status!r}, {cond.reason_expr}, {cond.name!r}",
            f"    except Exception as _e:",
            f'        algo_log.debug("Stopping check {cond.name} raised: %s", _e)',
        ]

    lines.append('    return None, "", ""')
    return "\n".join(lines)


def _render_runner(spec: SimulationSpec) -> str:
    out_vars = repr(spec.output_variables)
    data_vars = repr(spec.data_log_variables)
    stop_doc = "\n".join(
        f"      [{s.kind.upper()}] {s.name}: {s.description}"
        for s in spec.stopping_conditions
    )
    return (
        f"# ── Runner ──────────────────────────────────────────────────────────────────\n"
        f"_OUTPUT_VARIABLES   = {out_vars}\n"
        f"_DATA_LOG_VARIABLES = {data_vars}\n"
        f"\n"
        f"def run(config: SimConfig, output_dir: Path = Path('sim_output')) -> SimResult:\n"
        f'    """\n'
        f"    Execute the simulation with automatic stopping, checkpointing, and logging.\n"
        f"\n"
        f"    Stopping conditions:\n"
        f"{stop_doc}\n"
        f'    """\n'
        + _RUNNER_BODY
    )


_RUNNER_BODY = '''\
    global algo_log, data_log

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_log, data_log = _setup_loggers(output_dir)
    data_log_path = str(output_dir / "data_log.jsonl")

    partial_saves: list[str] = []
    config_dict = asdict(config)
    max_steps = config.max_steps

    # ── Tier 1: Validate config before anything else ──────────────────────────
    algo_log.debug("Validating config before simulation start...")
    try:
        _assert_config(config)
        algo_log.debug("Config validation passed.")
    except AssertionError as exc:
        algo_log.error(f"CONFIG VALIDATION FAILED: {exc}")
        raise

    # ── Banner ────────────────────────────────────────────────────────────────
    algo_log.info("=" * 70)
    algo_log.info("SIMULATION START")
    algo_log.info(f"Output dir       : {output_dir.resolve()}")
    algo_log.info(f"Max steps        : {max_steps:,}")
    algo_log.info(f"Checkpoint every : {config.checkpoint_interval:,} steps")
    algo_log.info(f"Progress every   : {config.progress_interval:,} steps")
    algo_log.info(f"Data log every   : {config.data_log_interval:,} steps → {data_log_path}")
    algo_log.info("─" * 70)
    algo_log.info("Configuration:")
    for k, v in config_dict.items():
        algo_log.info(f"  {k:<34} = {v}")
    algo_log.info("=" * 70)

    # ── Precompute ────────────────────────────────────────────────────────────
    algo_log.info("Running precompute()...")
    wall_start = time.perf_counter()
    precomputed = precompute(config)
    algo_log.info(
        f"Precompute done in {time.perf_counter()-wall_start:.3f}s. "
        f"Keys: {list(precomputed.keys())}"
    )

    # ── Initial state ─────────────────────────────────────────────────────────
    state = initial_state(config, precomputed)
    algo_log.debug(f"Initial state created. Fields: {list(asdict(state).keys())}")

    # Write initial data-log entry
    if _DATA_LOG_VARIABLES:
        data_log.info(json.dumps(state.snapshot(_DATA_LOG_VARIABLES)))

    wall_start = time.perf_counter()

    for step_num in range(max_steps):
        state.step = step_num

        # ── Tier 3: Stopping conditions ───────────────────────────────────────
        stop_status, stop_reason, stop_name = _check_stopping(config, state, precomputed)
        if stop_status is not None:
            elapsed = time.perf_counter() - wall_start
            _log_stop(stop_status, stop_name, stop_reason, step_num, elapsed)
            _save_checkpoint(state, output_dir, step_num, partial_saves, label=stop_status)
            if _DATA_LOG_VARIABLES:
                data_log.info(json.dumps({
                    **state.snapshot(_DATA_LOG_VARIABLES),
                    "_event": f"stop:{stop_name}", "_reason": stop_reason,
                }))
            return SimResult(
                status=stop_status, reason=stop_reason,
                stop_condition_name=stop_name, steps_run=step_num,
                wall_time_seconds=elapsed, final_state=asdict(state),
                history=state.history, partial_save_paths=partial_saves,
                data_log_path=data_log_path, config=config_dict,
            )

        # ── Advance ───────────────────────────────────────────────────────────
        state = sim_step(config, state, precomputed)

        # ── Tier 2: State invariants ───────────────────────────────────────────
        try:
            _assert_state(config, state)
        except AssertionError as exc:
            elapsed = time.perf_counter() - wall_start
            algo_log.error(f"STATE INVARIANT VIOLATED at step {step_num}: {exc}")
            _save_checkpoint(state, output_dir, step_num, partial_saves, label="invariant_failure")
            return SimResult(
                status="failed",
                reason=f"State invariant violated at step {step_num}: {exc}",
                stop_condition_name="state_assert",
                steps_run=step_num, wall_time_seconds=elapsed,
                final_state=asdict(state), history=state.history,
                partial_save_paths=partial_saves, data_log_path=data_log_path,
                config=config_dict,
            )

        # ── In-memory history ─────────────────────────────────────────────────
        if _OUTPUT_VARIABLES:
            state.history.append(state.snapshot(_OUTPUT_VARIABLES))

        # ── Research data log ─────────────────────────────────────────────────
        if _DATA_LOG_VARIABLES and step_num % config.data_log_interval == 0:
            data_log.info(json.dumps(state.snapshot(_DATA_LOG_VARIABLES)))

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step_num > 0 and step_num % config.checkpoint_interval == 0:
            _save_checkpoint(state, output_dir, step_num, partial_saves)

        # ── Progress ──────────────────────────────────────────────────────────
        if step_num > 0 and step_num % config.progress_interval == 0:
            elapsed = time.perf_counter() - wall_start
            rate = step_num / elapsed if elapsed > 0 else 0
            pct = 100.0 * step_num / max_steps
            eta_secs = (max_steps - step_num) / rate if rate > 0 else float("inf")
            eta_str = (
                f"{eta_secs:.0f}s" if eta_secs < 3600 else f"{eta_secs/3600:.1f}h"
            ) if math.isfinite(eta_secs) else "∞"
            algo_log.info(
                f"Progress : {step_num:>9,}/{max_steps:,}  ({pct:5.1f}%)  |  "
                f"{elapsed:7.1f}s elapsed  |  {rate:8.0f} steps/s  |  ETA {eta_str}  |  "
                + progress_summary(state, precomputed)
            )

    # ── Max steps exhausted ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start
    algo_log.warning("=" * 70)
    algo_log.warning(f"SIMULATION STOP  : max_steps limit reached ({max_steps:,})")
    algo_log.warning(f"Wall time        : {elapsed:.2f}s")
    algo_log.warning("=" * 70)
    _save_checkpoint(state, output_dir, max_steps, partial_saves, label="final")

    return SimResult(
        status="max_steps",
        reason=f"Reached maximum step limit of {max_steps:,} without a decisive outcome.",
        stop_condition_name="max_steps", steps_run=max_steps,
        wall_time_seconds=elapsed, final_state=asdict(state),
        history=state.history, partial_save_paths=partial_saves,
        data_log_path=data_log_path, config=config_dict,
    )


def _log_stop(status: str, name: str, reason: str, steps: int, elapsed: float) -> None:
    lvl = logging.INFO if status == "success" else logging.WARNING
    algo_log.log(lvl, "=" * 70)
    algo_log.log(lvl, f"SIMULATION STOP  : {name}")
    algo_log.log(lvl, f"Status           : {status.upper()}")
    algo_log.log(lvl, f"Reason           : {reason}")
    algo_log.log(lvl, f"Steps completed  : {steps:,}")
    algo_log.log(lvl, f"Wall time        : {elapsed:.2f}s")
    algo_log.log(lvl, "=" * 70)


def _save_checkpoint(
    state: SimState, output_dir: Path, step: int,
    partial_saves: list[str], label: str = "",
) -> None:
    tag = f"_{label}" if label else ""
    path = output_dir / f"checkpoint{tag}_{step:010d}.pkl"
    try:
        with open(path, "wb") as fh:
            pickle.dump(asdict(state), fh, protocol=pickle.HIGHEST_PROTOCOL)
        partial_saves.append(str(path))
        algo_log.debug(f"Checkpoint saved : {path.name}")
    except Exception as exc:
        algo_log.warning(f"Checkpoint failed: {exc}")
'''


def _render_cli(spec: SimulationSpec) -> str:
    type_map = {
        VariableKind.FLOAT: "float", VariableKind.INT: "int",
        VariableKind.BOOL: "bool", VariableKind.CHOICE: "str",
        VariableKind.STRING: "str",
    }
    arg_lines = []
    for v in spec.variables:
        unit_note = f" [{v.unit}]" if v.unit else ""
        help_str = f"{v.description}{unit_note} | {v.range_str()}"
        if v.kind == VariableKind.BOOL:
            arg_lines.append(
                f'    p.add_argument("--{v.name.replace(chr(95), chr(45))}", dest={v.name!r}, type=lambda x: x.lower()=="true", '
                f'default={v.default!r}, help={help_str!r})'
            )
        elif v.kind == VariableKind.CHOICE:
            arg_lines.append(
                f'    p.add_argument("--{v.name.replace(chr(95), chr(45))}", dest={v.name!r}, choices={v.choices!r}, '
                f'default={v.default!r}, help={help_str!r})'
            )
        else:
            arg_lines.append(
                f'    p.add_argument("--{v.name.replace(chr(95), chr(45))}", dest={v.name!r}, type={type_map[v.kind]}, '
                f'default={v.default!r}, help={help_str!r})'
            )
    arg_lines += [
        f'    p.add_argument("--max-steps", type=int, default={spec.max_steps})',
        f'    p.add_argument("--checkpoint-interval", type=int, default={spec.checkpoint_interval})',
        f'    p.add_argument("--progress-interval", type=int, default={spec.progress_interval})',
        f'    p.add_argument("--data-log-interval", type=int, default={spec.data_log_interval})',
    ]
    arg_block = "\n".join(arg_lines)

    return (
        f'# ── CLI ─────────────────────────────────────────────────────────────────────\n'
        f'def _build_cli() -> argparse.ArgumentParser:\n'
        f'    p = argparse.ArgumentParser(\n'
        f'        description={spec.description!r},\n'
        f'        formatter_class=argparse.ArgumentDefaultsHelpFormatter,\n'
        f'    )\n'
        f'{arg_block}\n'
        f'    p.add_argument("--output-dir", default="sim_output")\n'
        f'    p.add_argument("--results-json", default=None)\n'
        f'    return p\n'
        f'\n\nif __name__ == "__main__":\n'
        f'    _args = _build_cli().parse_args()\n'
        f'    _cfg_kwargs = {{\n'
        f'        k: v for k, v in vars(_args).items()\n'
        f'        if k not in ("output_dir", "results_json")\n'
        f'    }}\n'
        f'    # Normalise hyphenated keys from argparse\n'
        f'    _cfg_kwargs = {{k.replace("-","_"): v for k, v in _cfg_kwargs.items()}}\n'
        f'    _config = SimConfig(**_cfg_kwargs)\n'
        f'    _result = run(_config, output_dir=Path(_args.output_dir))\n'
        f'\n'
        f'    algo_log.info("─" * 70)\n'
        f'    algo_log.info("FINAL SUMMARY")\n'
        f'    algo_log.info(f"  Status           : {{_result.status.upper()}}")\n'
        f'    algo_log.info(f"  Reason           : {{_result.reason}}")\n'
        f'    algo_log.info(f"  Steps run        : {{_result.steps_run:,}}")\n'
        f'    algo_log.info(f"  Wall time        : {{_result.wall_time_seconds:.2f}}s")\n'
        f'    algo_log.info(f"  Data log         : {{_result.data_log_path}}")\n'
        f'    algo_log.info(f"  Checkpoints      : {{len(_result.partial_save_paths)}}")\n'
        f'    algo_log.info("─" * 70)\n'
        f'\n'
        f'    if _args.results_json:\n'
        f'        _out = Path(_args.results_json)\n'
        f'        _out.parent.mkdir(parents=True, exist_ok=True)\n'
        f'        with open(_out, "w") as _fh:\n'
        f'            json.dump(asdict(_result), _fh, indent=2, default=str)\n'
        f'        algo_log.info(f"Full results → {{_out}}")\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Notebook builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_notebook(spec: SimulationSpec) -> dict:
    def md_cell(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source}

    def code_cell(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": source}

    full_script = _render_script(spec)
    cli_marker = "# ── CLI "
    if cli_marker in full_script:
        full_script = full_script[:full_script.index(cli_marker)].rstrip()

    config_lines = ["config_params = {"]
    for v in spec.variables:
        config_lines.append(f"    {v.name!r}: {v.default!r},  # {v.description}")
    config_lines += [
        f"    'max_steps': {spec.max_steps},",
        f"    'checkpoint_interval': {spec.checkpoint_interval},",
        f"    'progress_interval': {spec.progress_interval},",
        f"    'data_log_interval': {spec.data_log_interval},",
        "}",
        "config = SimConfig(**config_params)",
        "print('Config:', config)",
    ]

    plot_vars = spec.output_variables[:4]
    plot_lines = [
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n\n",
        "# ── In-memory history plot ───────────────────────────────────────────────────\n",
        "history = result.history\n",
        "if history:\n",
        "    steps = [h['step'] for h in history]\n",
    ]
    if plot_vars:
        n = len(plot_vars)
        plot_lines += [
            f"    fig, axes = plt.subplots({n}, 1, figsize=(10, {3*n}), sharex=True)\n",
            f"    if {n} == 1: axes = [axes]\n",
        ]
        for i, var in enumerate(plot_vars):
            plot_lines += [
                f"    axes[{i}].plot(steps, [h.get({var!r}) for h in history], label={var!r})\n",
                f"    axes[{i}].set_ylabel({var!r}); axes[{i}].legend()\n",
            ]
        plot_lines += [
            "    axes[-1].set_xlabel('step')\n",
            f"    fig.suptitle({spec.name!r})\n",
            "    plt.tight_layout(); plt.show()\n",
        ]

    plot_lines += [
        "\n# ── Research data log ────────────────────────────────────────────────────────\n",
        "from pathlib import Path\n",
        "data_log_path = Path(result.data_log_path) if result.data_log_path else None\n",
        "if data_log_path and data_log_path.exists():\n",
        "    df = pd.read_json(data_log_path, lines=True)\n",
        "    print(df.describe())\n",
        "    display(df.head())\n",
        "else:\n",
        "    print('No data log found.')\n",
    ]

    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            md_cell(
                f"# {spec.name}\n\n_{spec.description}_\n\n"
                f"**Est. time**: {spec.time_estimate_explanation or 'unknown'}  \n"
                f"**Max steps**: {spec.max_steps:,}\n\n---\n"
                f"## Variables\n{spec.format_variable_docs()}\n\n"
                f"## Stopping Conditions\n{spec.format_stop_docs()}\n\n"
                f"## Logging\n"
                f"- **algo_log** → stdout (algorithm trace)\n"
                f"- **data_log** → `data_log.jsonl` (research data)\n"
            ),
            code_cell(
                "# Uncomment to install dependencies\n"
                "# import subprocess, sys\n"
                "# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'pandas', 'matplotlib'])"
            ),
            code_cell(full_script),
            code_cell("\n".join(config_lines)),
            code_cell(
                "from pathlib import Path\n\n"
                "result = run(config, output_dir=Path('sim_output_notebook'))\n\n"
                "print(f'\\nStatus  : {result.status.upper()}')\n"
                "print(f'Reason  : {result.reason}')\n"
                "print(f'Steps   : {result.steps_run:,}')\n"
                "print(f'Time    : {result.wall_time_seconds:.2f}s')\n"
                "print(f'Data log: {result.data_log_path}')"
            ),
            code_cell("".join(plot_lines)),
        ],
    }
