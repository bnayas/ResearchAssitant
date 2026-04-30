"""
sim_tool.tool
─────────────
Top-level API. The only module the coding agent imports.

Every method returns a typed contract object. No string-matching required.

Stage flow
──────────
  tool.start(description)           → ClarificationRequest | SpecApproval
  tool.answer(session_id, answers)  → ClarificationRequest | SpecApproval
  tool.approve(session_id)          → GeneratedArtifacts
  tool.reject(session_id, feedback) → ClarificationRequest | SpecApproval
  tool.generate(session_id)         → GeneratedArtifacts
  tool.run_single(session_id, ...)  → RunSummary
  tool.run_sweep(session_id, ...)   → RunSummary
  tool.analyse_sweep(session_id)    → AnalysisResult
  tool.apply_patch(session_id, p)   → GeneratedArtifacts
  tool.consolidate_memory(sess_id)  → list[str]  (rules added to AGENTS.md)
  tool.show_memory(agent)           → str  (contents of AGENTS.md)
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .analyst import RunAnalyst
from .codegen import generate_notebook, generate_script
from .contract import (
    AnalysisResult, ClarificationRequest, GeneratedArtifacts,
    OutputContract, RunOutcome, RunResult, RunSummary,
    SpecApproval, SpecPatch, Verdict,
)
from .designer import SimulationDesigner
from .launcher import Launcher, _config_tag
from .llm import LLMBackend, make_backend
from .memory import AgentMemory
from .models import SimulationSpec
from .runner import build_sweep_configs

log = logging.getLogger("sim_tool")


# ─────────────────────────────────────────────────────────────────────────────
# Internal session record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Session:
    session_id:    str
    spec:          Optional[SimulationSpec] = None
    script_path:   Optional[Path]          = None
    notebook_path: Optional[Path]          = None
    approved:      bool                    = False
    last_sweep_dir: Optional[Path]         = None


# ─────────────────────────────────────────────────────────────────────────────
# SimulationTool
# ─────────────────────────────────────────────────────────────────────────────

class SimulationTool:
    """
    The single entry point for a coding agent.

    Args:
        output_root: Directory for all generated code and run outputs.
        backend:     Optional compatibility shortcut that applies the same
                     backend to both designer and analyst.
        designer_backend / analyst_backend:
                     Independently configurable LLM backends. When omitted,
                     each service resolves its own LM Studio-first default.
        default_timeout: Per-run subprocess timeout in seconds.
    """

    def __init__(
        self,
        output_root: Union[str, Path] = "./simulations",
        backend: Optional[LLMBackend] = None,
        designer_backend: Optional[LLMBackend] = None,
        analyst_backend: Optional[LLMBackend] = None,
        launcher: Optional[Launcher] = None,
        default_timeout: Optional[float] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.default_timeout = default_timeout
        if backend is not None:
            designer_backend = designer_backend or backend
            analyst_backend = analyst_backend or backend
        self._designer_backend = designer_backend or make_backend(service="designer")
        self._analyst_backend = analyst_backend or make_backend(service="analyst")
        self._designer = SimulationDesigner(backend=self._designer_backend)
        self._analyst  = RunAnalyst(backend=self._analyst_backend)
        self._launcher = launcher or Launcher()
        self._sessions: dict[str, _Session] = {}
        _setup_logging()
        log.info(
            f"SimulationTool ready | "
            f"designer_backend={self._designer_backend.model_name} | "
            f"analyst_backend={self._analyst_backend.model_name} | "
            f"output_root={self.output_root}"
        )

    # ── Phase 1: Design ───────────────────────────────────────────────────────

    def start(self, description: str) -> Union[ClarificationRequest, SpecApproval]:
        """
        Begin a new simulation design session.
        Returns ClarificationRequest (if the LLM has questions) or
        SpecApproval (if the description was complete enough).
        """
        result = self._designer.start(description)
        sid = result.session_id
        self._sessions[sid] = _Session(session_id=sid)
        if isinstance(result, SpecApproval) and result.spec_card:
            self._sessions[sid].spec = self._designer.get_spec(sid)
        return result

    def answer(
        self,
        session_id: str,
        answers: dict[str, str],
    ) -> Union[ClarificationRequest, SpecApproval]:
        """
        Provide answers to the designer's clarifying questions.
        answers: {"1": "temperature from 1.5 to 4.0", "2": "max_steps=5000"}
        """
        result = self._designer.answer(session_id, answers)
        if isinstance(result, SpecApproval):
            self._sessions[session_id].spec = self._designer.get_spec(session_id)
        return result

    def approve(self, session_id: str) -> GeneratedArtifacts:
        """
        Approve the spec and generate the Python script + Jupyter notebook.
        Combines the old approve() + generate() into one call for clarity.
        """
        sess = self._get(session_id)
        self._designer.approve(session_id)   # validates spec is ready
        sess.approved = True
        sess.spec = self._designer.get_spec(session_id)
        return self._generate(session_id)

    def reject(
        self,
        session_id: str,
        feedback: str,
    ) -> Union[ClarificationRequest, SpecApproval]:
        """
        Reject the proposed spec and continue refining.
        feedback: plain English describing what needs to change.
        """
        self._sessions[session_id].approved = False
        return self._designer.reject(session_id, feedback)

    # ── Phase 2: Run ──────────────────────────────────────────────────────────

    def run_single(
        self,
        session_id: str,
        config_overrides: Optional[dict] = None,
        timeout: Optional[float] = None,
        stream_logs: bool = True,
    ) -> RunSummary:
        """
        Run the simulation once with default config (+ optional overrides).
        Returns RunSummary with one RunResult.

        data_log.jsonl is at RunSummary.data_files[0] — load with pandas.
        """
        sess = self._get(session_id)
        if sess.script_path is None:
            raise RuntimeError(f"[{session_id}] No script — call approve() first.")

        spec = sess.spec
        config = {v.name: v.default for v in spec.variables}
        if config_overrides:
            config.update(config_overrides)

        out_dir = self.output_root / session_id / "run_single"
        launched = self._launcher.launch(
            session_id=session_id,
            script_path=sess.script_path,
            config=config,
            output_dir=out_dir,
            setup_code=spec.setup_code,
            timeout_seconds=timeout or self.default_timeout,
            stream_logs=stream_logs,
        )
        sess.last_sweep_dir = out_dir.parent  # parent so analyse_sweep finds it

        if hasattr(launched, "run_summary"):
            summary = launched.run_summary
        else:
            summary = RunSummary(
                session_id=session_id,
                results=[_ticket_to_run_result(launched, config=config, output_dir=out_dir)],
                sweep_dir=out_dir,
            )
        return summary

    def run_sweep(
        self,
        session_id: str,
        max_workers: int = 1,
        timeout: Optional[float] = None,
        stream_logs: bool = False,
    ) -> RunSummary:
        """
        Run the parameter sweep across all variables marked sweep=True.
        Returns RunSummary with one RunResult per configuration.

        Call tool.analyse_sweep(session_id) next.
        """
        sess = self._get(session_id)
        if sess.script_path is None:
            raise RuntimeError(f"[{session_id}] No script — call approve() first.")

        out_dir = self.output_root / session_id / "sweep"
        configs = build_sweep_configs(sess.spec)
        results_by_index: list[Optional[RunResult]] = [None] * len(configs)

        def _launch_one(index: int, cfg: dict) -> tuple[int, RunResult]:
            run_dir = out_dir / _config_tag(cfg)
            launched = self._launcher.launch(
                session_id=session_id,
                script_path=sess.script_path,
                config=cfg,
                output_dir=run_dir,
                setup_code=sess.spec.setup_code,
                timeout_seconds=timeout or self.default_timeout,
                stream_logs=stream_logs and max_workers == 1,
            )
            if hasattr(launched, "run_summary"):
                run_result = launched.run_summary.results[0]
            else:
                run_result = _ticket_to_run_result(launched, config=cfg, output_dir=run_dir)
            return index, run_result

        if max_workers <= 1:
            for index, cfg in enumerate(configs):
                _, run_result = _launch_one(index, cfg)
                results_by_index[index] = run_result
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_launch_one, index, cfg)
                    for index, cfg in enumerate(configs)
                ]
                for future in as_completed(futures):
                    index, run_result = future.result()
                    results_by_index[index] = run_result

        sess.last_sweep_dir = out_dir

        results = [result for result in results_by_index if result is not None]
        return RunSummary(
            session_id=session_id,
            results=results,
            sweep_dir=out_dir,
        )

    # ── Phase 3: Analyse ──────────────────────────────────────────────────────

    def analyse_sweep(
        self,
        session_id: str,
        use_llm: bool = True,
    ) -> AnalysisResult:
        """
        Analyse the most recent sweep output.
        Returns AnalysisResult with .verdict and optional .patch.

        Layers:
          1. Deterministic extraction of metrics from results.json + data_log.jsonl
          2. Rule-based flag classification (no LLM)
          3. LLM produces SpecPatch if flags exist (only if use_llm=True)

        Next steps:
          Verdict.OK          → read .data_files with pandas
          Verdict.MINOR_FIX   → tool.apply_patch(session_id, result.patch)
          Verdict.MAJOR_FIX   → review .patch.changes, then apply_patch
          Verdict.ABORT       → call tool.start() with revised description
        """
        sess = self._get(session_id)
        sweep_dir = sess.last_sweep_dir or (self.output_root / session_id / "sweep")

        analyst = RunAnalyst(
            backend=self._analyst_backend if use_llm else None
        )
        report = analyst.analyse_sweep(
            sweep_dir=sweep_dir,
            session_id=session_id,
        )
        log.info(
            f"[{session_id}] Analysis: verdict={report.verdict.value} | "
            f"{len(report.flags)} flags | "
            f"{len(report.patch.changes) if report.patch else 0} patch changes"
        )
        return report

    def apply_patch(
        self,
        session_id: str,
        patch: SpecPatch,
    ) -> GeneratedArtifacts:
        """
        Apply a SpecPatch to the current spec and regenerate code.

        The patch is validated against the PATCHABLE_FIELDS contract.
        Immutable fields (step_code, precompute_code, etc.) are silently rejected.
        Returns GeneratedArtifacts pointing to the updated script.
        """
        sess = self._get(session_id)
        if sess.spec is None:
            raise RuntimeError(f"[{session_id}] No spec available.")

        spec = sess.spec
        applied: list[str] = []

        for change in patch.changes:
            parts = change.field.split(".")
            top = parts[0]

            if top in ("step_code", "precompute_code", "initial_state_code",
                       "progress_code", "setup_code", "state_fields",
                       "config_assert_code", "state_assert_code"):
                log.warning(
                    f"[{session_id}] Refusing to patch immutable field {change.field!r}. "
                    f"Analyst should have returned verdict=abort for algorithmic changes."
                )
                continue

            if top == "max_steps":
                spec.max_steps = int(change.new_value)
                applied.append(f"max_steps → {spec.max_steps:,}")
            elif top == "checkpoint_interval":
                spec.checkpoint_interval = int(change.new_value)
                applied.append(f"checkpoint_interval → {spec.checkpoint_interval}")
            elif top == "progress_interval":
                spec.progress_interval = int(change.new_value)
                applied.append(f"progress_interval → {spec.progress_interval}")
            elif top == "data_log_interval":
                spec.data_log_interval = int(change.new_value)
                applied.append(f"data_log_interval → {spec.data_log_interval}")
            elif top == "variables" and len(parts) == 3:
                var_name, sub_field = parts[1], parts[2]
                for v in spec.variables:
                    if v.name == var_name:
                        setattr(v, sub_field, change.new_value)
                        applied.append(f"variables.{var_name}.{sub_field} → {change.new_value!r}")
                        break
            elif top == "stopping_conditions" and len(parts) == 3:
                cond_name, sub_field = parts[1], parts[2]
                for c in spec.stopping_conditions:
                    if c.name == cond_name:
                        setattr(c, sub_field, change.new_value)
                        applied.append(f"stopping_conditions.{cond_name}.{sub_field} → {change.new_value!r}")
                        break
            else:
                log.warning(f"[{session_id}] Unknown patch field {change.field!r} — skipped")

        if applied:
            log.info(f"[{session_id}] Patch applied: {'; '.join(applied)}")
        else:
            log.warning(f"[{session_id}] No changes were applied from patch")

        sess.approved = True
        return self._generate(session_id)

    # ── Memory ────────────────────────────────────────────────────────────────

    def consolidate_memory(
        self,
        session_id: str,
        run_summary: Optional[RunSummary] = None,
        analysis: Optional[AnalysisResult] = None,
    ) -> dict[str, list[str]]:
        """
        Ask each agent to extract lessons from this session and write them
        to their respective AGENTS.md files.

        Returns {"designer": [rules], "analyst": [rules]}.
        Call this after a session completes (successful or not).
        """
        sess = self._get(session_id)
        sim_name = sess.spec.name if sess.spec else "unknown"
        sim_type = _infer_sim_type(sess.spec)

        # Build session facts for the analyst's memory
        analyst_facts: dict = {"outcomes": [], "flags": [], "patch_changes": []}
        if run_summary:
            analyst_facts["outcomes"] = [
                {"status": r.outcome.value, "reason": r.reason}
                for r in run_summary.results
            ]
        if analysis:
            analyst_facts["flags"] = [
                {"kind": f.kind, "severity": f.severity, "message": f.message}
                for f in analysis.flags
            ]
            if analysis.patch:
                analyst_facts["patch_changes"] = [
                    {"field": c.field, "why": c.why}
                    for c in analysis.patch.changes
                ]

        designer_rules = self._designer.memory.consolidate_session(
            session_id=session_id, sim_name=sim_name, sim_type=sim_type,
            session_facts=analyst_facts, backend=self._designer_backend,
        )
        analyst_rules = self._analyst.memory.consolidate_session(
            session_id=session_id, sim_name=sim_name, sim_type=sim_type,
            session_facts=analyst_facts, backend=self._analyst_backend,
        )
        return {"designer": designer_rules, "analyst": analyst_rules}

    def show_memory(self, agent: str = "designer") -> str:
        """
        Return the contents of an agent's AGENTS.md.
        agent: "designer" | "analyst"
        """
        if agent == "designer":
            return self._designer.memory.show()
        elif agent == "analyst":
            return self._analyst.memory.show()
        else:
            raise ValueError(f"Unknown agent {agent!r}. Use 'designer' or 'analyst'.")

    def memory_path(self, agent: str = "designer") -> Path:
        """Return the filesystem path to an agent's AGENTS.md file."""
        if agent == "designer":
            return self._designer.memory.path
        return self._analyst.memory.path

    # ── Inspection helpers ────────────────────────────────────────────────────

    def list_sweep_configs(self, session_id: str) -> list[dict]:
        """Preview all configurations that will be run in a sweep."""
        sess = self._get(session_id)
        if sess.spec is None:
            return []
        return build_sweep_configs(sess.spec)

    def show_spec(self, session_id: str) -> str:
        """Return a formatted spec summary."""
        sess = self._get(session_id)
        if sess.spec is None:
            return f"[{session_id}] No spec yet."
        from .designer import _format_spec_card
        return _format_spec_card(sess.spec)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, session_id: str) -> _Session:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id!r}")
        return self._sessions[session_id]

    def _generate(self, session_id: str) -> GeneratedArtifacts:
        sess = self._get(session_id)
        spec = sess.spec
        run_dir = self.output_root / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        safe = spec.name.lower().replace(" ", "_").replace("—","").replace("-","_")
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in safe).strip("_")

        script_path   = run_dir / f"{safe}.py"
        notebook_path = run_dir / f"{safe}.ipynb"

        log.info(f"[{session_id}] Generating script  → {script_path}")
        generate_script(spec, script_path)
        log.info(f"[{session_id}] Generating notebook → {notebook_path}")
        generate_notebook(spec, notebook_path)

        sess.script_path   = script_path
        sess.notebook_path = notebook_path

        kb = script_path.stat().st_size // 1024 + 1
        cli_synopsis = f"python {script_path.name} [--VAR VALUE ...] [--output-dir DIR]"
        return GeneratedArtifacts(
            session_id=session_id,
            script_path=script_path,
            notebook_path=notebook_path,
            script_size_kb=kb,
            cli_synopsis=cli_synopsis,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_run_result(outcome) -> RunResult:
    """Convert runner.RunOutcome (old internal type) to contract.RunResult."""
    try:
        ro = RunOutcome(outcome.status)
    except ValueError:
        ro = RunOutcome.CRASH
    data_file = None
    if outcome.result_json_path:
        candidate = Path(outcome.output_dir) / "data_log.jsonl"
        if candidate.exists():
            data_file = candidate
    return RunResult(
        config=outcome.config,
        outcome=ro,
        reason=outcome.reason,
        stop_condition_name=getattr(outcome, "stop_condition", ""),
        steps_run=outcome.steps_run,
        wall_time_seconds=outcome.wall_time_seconds,
        output_dir=Path(outcome.output_dir),
        data_file=data_file,
    )


def _ticket_to_run_result(ticket, config: dict, output_dir: Path) -> RunResult:
    outcome = (
        RunOutcome.TIMEOUT
        if getattr(getattr(ticket, "kind", None), "value", "") == RunOutcome.TIMEOUT.value
        else RunOutcome.CRASH
    )
    data_file = output_dir / "data_log.jsonl"
    return RunResult(
        config=config,
        outcome=outcome,
        reason=getattr(ticket, "error_line", "") or getattr(ticket, "suggested_fix", "") or str(ticket),
        stop_condition_name=getattr(getattr(ticket, "kind", None), "value", "launch_failure"),
        steps_run=0,
        wall_time_seconds=float(getattr(ticket, "wall_time_seconds", 0.0)),
        output_dir=Path(output_dir),
        data_file=data_file if data_file.exists() else None,
    )


def _infer_sim_type(spec: Optional[SimulationSpec]) -> str:
    if spec is None:
        return "unknown"
    desc = (spec.name + " " + spec.description).lower()
    if any(w in desc for w in ("monte carlo", "metropolis", "ising", "markov")):
        return "Monte Carlo"
    if any(w in desc for w in ("euler", "runge-kutta", "ode", "pde")):
        return "ODE/PDE"
    if any(w in desc for w in ("random walk", "diffusion", "brownian")):
        return "Random Walk"
    if any(w in desc for w in ("projectile", "trajectory", "motion")):
        return "Classical Mechanics"
    return "General"


def _setup_logging() -> None:
    root = logging.getLogger("sim_tool")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(ch)
