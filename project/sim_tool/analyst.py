"""
sim_tool.analyst
────────────────
Analyses run outputs and returns a SpecPatch (or OK verdict).

Contract
────────
  Input:  RunSummary (from tool.run_sweep)
  Output: AnalysisResult
            .verdict  ∈ {OK, MINOR_FIX, MAJOR_FIX, ABORT}
            .patch    SpecPatch | None

The analyst CANNOT and WILL NOT:
  ✗ Rewrite step_code, precompute_code, initial_state_code
  ✗ Add or remove state_fields
  ✗ Change the simulation algorithm
  ✗ Start a new design conversation

A SpecPatch contains PatchChange entries that target ONLY:
  max_steps, checkpoint_interval, progress_interval, data_log_interval,
  variables.<name>.default / .min_val / .max_val / .sweep_values,
  stopping_conditions.<name>.check_expr / .reason_expr / .priority

Architecture (three layers, always in order)
────────────────────────────────────────────
Layer 1  extract_metrics()    — parse results.json + data_log.jsonl → RunMetrics
Layer 2  classify_*()         — rule-based flags, no LLM
Layer 3  _llm_to_patch()      — LLM reads compact summary → SpecPatch (if flags exist)

Memory
──────
Reads analyst_memory.md at init. Appended via tool.consolidate_memory().
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .contract import (
    AnalysisResult, Flag, PatchChange, RunOutcome,
    RunResult, RunSummary, SpecPatch, Verdict,
    IMMUTABLE_FIELDS, PATCHABLE_FIELDS,
)
from .memory import AgentMemory

log = logging.getLogger("sim_tool.analyst")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    run_dir:             Path
    config:              dict
    status:              str
    reason:              str
    stop_condition_name: str
    steps_run:           int
    wall_time_seconds:   float
    data_log_records:    int     = 0
    data_log_nan_fields: list[str] = field(default_factory=list)
    data_log_summary:    dict    = field(default_factory=dict)
    results_found:       bool    = True

    @property
    def outcome(self) -> RunOutcome:
        try:
            return RunOutcome(self.status)
        except ValueError:
            return RunOutcome.CRASH


def extract_metrics(run_dir: Path) -> RunMetrics:
    """Layer 1: Parse a run directory into RunMetrics. Never raises."""
    run_dir = Path(run_dir)
    results_json = run_dir / "results.json"
    data_log_path = run_dir / "data_log.jsonl"

    if not results_json.exists():
        log.warning(f"No results.json in {run_dir}")
        return RunMetrics(
            run_dir=run_dir, config={}, status="crash",
            reason="results.json not found — process crashed before writing",
            stop_condition_name="missing", steps_run=0, wall_time_seconds=0,
            results_found=False,
        )

    try:
        data = json.loads(results_json.read_text())
    except json.JSONDecodeError as exc:
        return RunMetrics(
            run_dir=run_dir, config={}, status="crash",
            reason=f"Malformed results.json: {exc}",
            stop_condition_name="malformed", steps_run=0, wall_time_seconds=0,
        )

    m = RunMetrics(
        run_dir=run_dir, config=data.get("config", {}),
        status=data.get("status", "unknown"), reason=data.get("reason", ""),
        stop_condition_name=data.get("stop_condition_name", ""),
        steps_run=int(data.get("steps_run", 0)),
        wall_time_seconds=float(data.get("wall_time_seconds", 0)),
    )

    if data_log_path.exists():
        records = []
        for line in data_log_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        m.data_log_records = len(records)
        if records:
            skip = {"step", "sim_time", "_event", "_reason"}
            numeric_fields = {
                k for rec in records for k, v in rec.items()
                if k not in skip and isinstance(v, (int, float))
            }
            nan_fields: list[str] = []
            summary: dict = {}
            for fname in sorted(numeric_fields):
                vals = [rec[fname] for rec in records
                        if fname in rec and isinstance(rec[fname], (int, float))]
                if not vals:
                    continue
                if any(not math.isfinite(v) for v in vals):
                    nan_fields.append(fname)
                finite = [v for v in vals if math.isfinite(v)]
                if finite:
                    summary[fname] = dict(
                        min=min(finite), max=max(finite),
                        mean=sum(finite)/len(finite), last=finite[-1], n=len(finite),
                    )
            m.data_log_nan_fields = nan_fields
            m.data_log_summary = summary
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Rule-based flags
# ─────────────────────────────────────────────────────────────────────────────

def classify_run(m: RunMetrics) -> list[Flag]:
    flags: list[Flag] = []
    max_s = m.config.get("max_steps", 0)

    if m.outcome == RunOutcome.CRASH:
        flags.append(Flag(kind="crash_detected", severity="error",
            message=(f"Run in {m.run_dir.name} crashed (status={m.status}). "
                     f"{m.reason}")))

    if m.outcome == RunOutcome.MAX_STEPS:
        flags.append(Flag(kind="max_steps_hit", severity="warning",
            message=(f"Reached max_steps={max_s:,} without a decisive outcome. "
                     f"Stopping conditions were never satisfied."),
            config_hint=f"max_steps={max_s*3}" if max_s else None,
            affects=["max_steps"]))

    if m.outcome == RunOutcome.SUCCESS and max_s and m.steps_run > 0.9 * max_s:
        flags.append(Flag(kind="step_budget_tight", severity="info",
            message=(f"Succeeded at step {m.steps_run:,} ({100*m.steps_run/max_s:.0f}% "
                     f"of max_steps={max_s:,}). Consider increasing max_steps."),
            config_hint=f"max_steps={max_s*2}", affects=["max_steps"]))

    if m.data_log_nan_fields:
        flags.append(Flag(kind="nan_detected", severity="error",
            message=(f"NaN/Inf in data log fields: {m.data_log_nan_fields}. "
                     f"Numerical instability — check dt, coupling strength, "
                     f"or missing state_assert."),
            affects=m.data_log_nan_fields))

    if m.data_log_records == 0:
        flags.append(Flag(kind="no_data_logged", severity="warning",
            message=(f"No data log records in {m.run_dir.name}. "
                     f"Check that data_log_variables is non-empty.")))
    return flags


def classify_sweep(metrics: list[RunMetrics]) -> list[Flag]:
    n = len(metrics)
    if n == 0:
        return []
    flags: list[Flag] = []
    n_success = sum(1 for m in metrics if m.outcome == RunOutcome.SUCCESS)
    n_failed  = sum(1 for m in metrics if m.outcome == RunOutcome.FAILED)
    n_crash   = sum(1 for m in metrics if m.outcome == RunOutcome.CRASH)

    if n_crash == n:
        flags.append(Flag(kind="all_crashed", severity="error",
            message=f"Every run crashed ({n}/{n}). The generated script has a bug."))
    elif n_failed == n:
        flags.append(Flag(kind="all_configs_failed", severity="error",
            message=(f"Every configuration hit a failure condition ({n}/{n}). "
                     f"Parameter space may be entirely outside valid range.")))
    elif n_failed > n * 0.5:
        flags.append(Flag(kind="high_failure_rate", severity="warning",
            message=(f"{n_failed}/{n} configurations failed early. "
                     f"Failure conditions may be too aggressive, or sweep range "
                     f"includes many degenerate configurations.")))
    if n_success == 0 and n_crash < n:
        flags.append(Flag(kind="convergence_none", severity="warning",
            message=(f"Zero configurations converged successfully "
                     f"({n - n_failed - n_crash} hit max_steps, {n_failed} failed early)."),
            affects=["max_steps"]))
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: LLM → SpecPatch
# ─────────────────────────────────────────────────────────────────────────────

_ANALYST_SYSTEM_TEMPLATE = """
{memory_prefix}
You are a simulation diagnostician. You receive a structured summary of run outcomes
and diagnostic flags. You must decide whether the spec needs changes and, if so,
produce a SpecPatch.

━━━ YOUR OUTPUT CONTRACT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You return ONLY a JSON object with this structure:
{{
  "reasoning": "<step-by-step thinking>",
  "verdict": "ok" | "minor_fix" | "major_fix" | "abort",
  "reason": "<plain English explanation of overall verdict>",
  "changes": [
    {{
      "field": "<dot-notation field path>",
      "old_value": <current value>,
      "new_value": <proposed value>,
      "why": "<plain English — shown to agent before applying>"
    }}
  ]
}}

━━━ VERDICT RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ok         — all runs look fine, no changes needed
  minor_fix  — safe to apply automatically (only numerical changes, high confidence)
  major_fix  — changes are warranted but need human review first
  abort      — the sweep parameter space is fundamentally wrong; restart design

━━━ PATCHABLE FIELDS (exhaustive list) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You may ONLY propose changes to these fields:
  max_steps
  checkpoint_interval
  progress_interval
  data_log_interval
  variables.<name>.default
  variables.<name>.min_val
  variables.<name>.max_val
  variables.<name>.sweep_values
  stopping_conditions.<name>.check_expr
  stopping_conditions.<name>.reason_expr
  stopping_conditions.<name>.priority

━━━ IMMUTABLE FIELDS — NEVER TOUCH THESE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  step_code, precompute_code, initial_state_code, progress_code,
  setup_code, state_fields, config_assert_code, state_assert_code

If the problem requires changing these, set verdict="abort" and explain why
in "reason". Do NOT attempt to patch them.

━━━ CONSERVATISM RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Only change what you are confident about
  - Prefer small multipliers (×3, ×5) over guessing exact values
  - If the cause is unclear, prefer major_fix over minor_fix
  - If nothing is wrong, say ok — do not invent changes
""".strip()


def _build_analyst_prompt(flags: list[Flag], metrics: list[RunMetrics]) -> str:
    lines = ["RUN SUMMARY\n"]
    n = len(metrics)
    n_ok   = sum(1 for m in metrics if m.outcome == RunOutcome.SUCCESS)
    n_fail = sum(1 for m in metrics if m.outcome == RunOutcome.FAILED)
    n_max  = sum(1 for m in metrics if m.outcome == RunOutcome.MAX_STEPS)
    n_err  = sum(1 for m in metrics if m.outcome == RunOutcome.CRASH)
    lines += [
        f"Total runs: {n}  |  success: {n_ok}  |  failed: {n_fail}  "
        f"|  max_steps: {n_max}  |  crash: {n_err}", "",
    ]
    for m in metrics:
        cfg = {k: v for k, v in m.config.items()
               if k not in ("checkpoint_interval","progress_interval","data_log_interval")}
        lines.append(f"[{m.status.upper():>10}] {cfg}  steps={m.steps_run:,}  wall={m.wall_time_seconds:.1f}s")
        lines.append(f"  reason: {m.reason}")
        for fname, stats in list(m.data_log_summary.items())[:3]:
            lines.append(f"  {fname}: min={stats['min']:.4g} max={stats['max']:.4g} "
                         f"mean={stats['mean']:.4g} last={stats['last']:.4g}")
    if flags:
        lines.append("\nDIAGNOSTIC FLAGS")
        for f in flags:
            lines.append(f"  [{f.severity.upper()}] {f.kind}: {f.message}")
            if f.config_hint:
                lines.append(f"    hint: {f.config_hint}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RunAnalyst — public API
# ─────────────────────────────────────────────────────────────────────────────

class RunAnalyst:
    """
    Analyses run outputs and produces an AnalysisResult.
    The LLM backend is optional — without it, only Layers 1+2 run.
    """

    def __init__(self, backend=None) -> None:
        self._backend = backend
        self._memory = AgentMemory("analyst")
        self._system = _ANALYST_SYSTEM_TEMPLATE.format(
            memory_prefix=self._memory.as_prompt_prefix()
        )
        log.info(
            f"Analyst initialised | "
            f"backend={'none' if backend is None else backend.model_name} | "
            f"memory: {self._memory.entry_count} entries"
        )

    @property
    def memory(self) -> AgentMemory:
        return self._memory

    def analyse_sweep(
        self,
        sweep_dir: Path,
        run_summary: Optional[RunSummary] = None,
        session_id: str = "",
    ) -> AnalysisResult:
        """Analyse all run subdirectories inside a sweep directory."""
        sweep_dir = Path(sweep_dir)
        run_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
        if not run_dirs:
            run_dirs = [sweep_dir]

        log.info(f"Analysing {len(run_dirs)} run(s) in {sweep_dir}")
        all_metrics = [extract_metrics(d) for d in run_dirs]

        # Layer 2: collect and deduplicate flags
        flags: list[Flag] = []
        for m in all_metrics:
            flags.extend(classify_run(m))
        flags.extend(classify_sweep(all_metrics))
        seen: dict[str, Flag] = {}
        for f in flags:
            if f.kind not in seen or f.severity == "error":
                seen[f.kind] = f
        flags = list(seen.values())

        # Determine verdict from flags alone if no LLM
        has_error   = any(f.severity == "error" for f in flags)
        has_warning = any(f.severity == "warning" for f in flags)
        has_abort   = any(f.kind == "all_crashed" for f in flags)

        if self._backend is None or not flags:
            verdict = (
                Verdict.ABORT      if has_abort else
                Verdict.MAJOR_FIX  if has_error else
                Verdict.MINOR_FIX  if has_warning else
                Verdict.OK
            )
            patch = None
            if verdict != Verdict.OK:
                # Build a deterministic patch from config hints in flags
                changes = _flags_to_changes(flags, all_metrics)
                patch = SpecPatch(
                    verdict=verdict,
                    reason=" | ".join(f.message[:80] for f in flags[:3]),
                    changes=changes,
                )
            data_files = [
                m.run_dir / "data_log.jsonl"
                for m in all_metrics
                if (m.run_dir / "data_log.jsonl").exists()
            ]
            return AnalysisResult(
                session_id=session_id,
                verdict=verdict,
                flags=flags,
                patch=patch,
                llm_reasoning="(deterministic only — no LLM backend provided)",
                data_files=data_files,
            )

        # Layer 3: LLM call
        prompt = _build_analyst_prompt(flags, all_metrics)
        log.info(f"Calling LLM analyst: {len(flags)} flags, {len(all_metrics)} runs")
        try:
            raw = self._backend.complete(
                system=self._system,
                messages=[{"role": "user", "content": prompt}],
            )
            m_json = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m_json.group()) if m_json else {}
        except Exception as exc:
            log.error(f"LLM analyst call failed: {exc}")
            data = {}

        reasoning = data.get("reasoning", "")
        verdict_str = data.get("verdict", "ok")
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.MINOR_FIX if flags else Verdict.OK

        patch = None
        if verdict != Verdict.OK:
            raw_changes = data.get("changes", [])
            changes: list[PatchChange] = []
            for c in raw_changes:
                try:
                    changes.append(PatchChange(
                        field=c["field"],
                        old_value=c.get("old_value"),
                        new_value=c["new_value"],
                        why=c.get("why", ""),
                    ))
                except (KeyError, ValueError) as exc:
                    log.warning(f"Skipping invalid patch change: {exc}")
            patch = SpecPatch(
                verdict=verdict,
                reason=data.get("reason", "Changes recommended."),
                changes=changes,
            )

        data_files = [
            m.run_dir / "data_log.jsonl"
            for m in all_metrics
            if (m.run_dir / "data_log.jsonl").exists()
        ]
        log.info(f"Analysis verdict: {verdict.value} | {len(patch.changes) if patch else 0} change(s)")
        return AnalysisResult(
            session_id=session_id,
            verdict=verdict,
            flags=flags,
            patch=patch,
            llm_reasoning=reasoning,
            data_files=data_files,
        )


def _flags_to_changes(flags: list[Flag], metrics: list[RunMetrics]) -> list[PatchChange]:
    """Deterministic patch generation from flags (used when no LLM backend)."""
    changes: list[PatchChange] = []
    for flag in flags:
        if flag.kind == "max_steps_hit":
            current = metrics[0].config.get("max_steps", 10000) if metrics else 10000
            changes.append(PatchChange(
                field="max_steps",
                old_value=current,
                new_value=current * 3,
                why=f"Simulation hit max_steps={current:,} without decisive outcome. ×3 increase.",
            ))
            break  # one change per flag kind
    return changes
