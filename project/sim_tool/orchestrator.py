"""
sim_tool.orchestrator
─────────────────────
Stateful research orchestrator that drives the full lifecycle from research
goal to final results, with explicit user review gates at each stage.

State machine
─────────────
  CLARIFYING              Designer is asking questions; user answers them.
  AWAITING_SPEC_APPROVAL  Spec ready; user reviews before any code runs.
  READY_TO_SAMPLE         Code generated; user can open and inspect the script.
  SAMPLE_RUNNING          Quick trial (1 config, ~5% of steps) running.
  AWAITING_SAMPLE_REVIEW  Sample finished; user decides whether to proceed.
  FULL_RUNNING            Full sweep running in background thread.
  AWAITING_RESULTS_REVIEW Full sweep done; user reviews analysis.
  COMPLETE                Session finished; results available.
  FAILED / ABORTED        Terminal error or user abort.

User review gates
─────────────────
  Gate 1 — AWAITING_SPEC_APPROVAL
    User sees: spec card, output contract, time estimate, script path.
    User can: approve() → READY_TO_SAMPLE
              reject(feedback) → CLARIFYING

  Gate 2 — READY_TO_SAMPLE
    User sees: script path (can open, read, edit, validate independently).
    User can: run_sample() → SAMPLE_RUNNING
              reject(feedback) → back to designer

  Gate 3 — AWAITING_SAMPLE_REVIEW
    User sees: sample RunSummary + quick AnalysisResult.
    User can: approve_sample() → FULL_RUNNING
              reject(feedback) → AWAITING_SPEC_APPROVAL

  Gate 4 — AWAITING_RESULTS_REVIEW
    User sees: full RunSummary + AnalysisResult + data file paths.
    User can: approve_results() → COMPLETE
              reject_results(feedback) → AWAITING_SPEC_APPROVAL

Progress queries (work in any state, especially SAMPLE_RUNNING / FULL_RUNNING)
─────────────────────────────────────────────────────────────────────────────
  orchestrator.query(session_id, "What is the acceptance rate at T=2.269?")
  → Reads partial data_log.jsonl files from the sweep directory.
  → Computes per-config statistics from whatever has been written so far.
  → LLM answers the specific question with those statistics as context.
  → Returns OrchestratorUpdate with .query_answer set.

Usage
─────
  from sim_tool.orchestrator import ResearchOrchestrator, OrchestratorUpdate

  orch = ResearchOrchestrator(tool)

  u = orch.start("Study the Ising phase transition as a function of temperature")
  # u.state == OrchestratorState.CLARIFYING
  print(u)

  # Answer questions (same as tool.answer but routed through orchestrator)
  u = orch.answer(u.session_id, {"1": "T from 1.5 to 4.0", ...})
  # u.state == OrchestratorState.AWAITING_SPEC_APPROVAL
  print(u)

  # Gate 1: Review spec
  u = orch.approve(u.session_id)                # approve spec + generate code
  # u.state == OrchestratorState.READY_TO_SAMPLE
  print(f"Script at: {u.artifacts.script_path}")  # open and inspect it

  # Gate 2: Review script, then start sample
  u = orch.run_sample(u.session_id)             # blocks until sample done
  # u.state == OrchestratorState.AWAITING_SAMPLE_REVIEW
  print(u)

  # Gate 3: Review sample, then launch full sweep (background)
  u = orch.approve_sample(u.session_id)         # non-blocking, sweep in thread
  # u.state == OrchestratorState.FULL_RUNNING

  # Query while sweep runs
  u = orch.query(u.session_id, "Is the low-temperature run converging?")
  print(u.query_answer)

  # Check status
  u = orch.get_status(u.session_id)             # non-blocking
  print(u)

  # Wait for completion and collect results
  u = orch.wait(u.session_id)                   # blocks until done
  # u.state == OrchestratorState.AWAITING_RESULTS_REVIEW
  print(u)

  u = orch.approve_results(u.session_id)
  # u.state == OrchestratorState.COMPLETE
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from .contract import (
    AnalysisResult, ClarificationRequest, Flag,
    GeneratedArtifacts, RunSummary, SpecApproval, Verdict,
)
from .tool import SimulationTool

log = logging.getLogger("sim_tool.orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorState(str, Enum):
    # Design
    CLARIFYING             = "clarifying"
    AWAITING_SPEC_APPROVAL = "awaiting_spec_approval"
    # Review script
    READY_TO_SAMPLE        = "ready_to_sample"
    # Sample
    SAMPLE_RUNNING         = "sample_running"
    AWAITING_SAMPLE_REVIEW = "awaiting_sample_review"
    # Full sweep
    FULL_RUNNING           = "full_running"
    AWAITING_RESULTS_REVIEW= "awaiting_results_review"
    # Terminal
    COMPLETE               = "complete"
    FAILED                 = "failed"
    ABORTED                = "aborted"

# What the user should do in each state
_NEXT_ACTION = {
    OrchestratorState.CLARIFYING:
        'orch.answer(session_id, {"1": "...", "2": "..."}) — answer questions',
    OrchestratorState.AWAITING_SPEC_APPROVAL:
        'orch.approve(session_id) or orch.reject(session_id, feedback) — review spec first',
    OrchestratorState.READY_TO_SAMPLE:
        'orch.run_sample(session_id) — optionally inspect the script first',
    OrchestratorState.SAMPLE_RUNNING:
        'orch.query(session_id, "...") or orch.get_status(session_id) — sample running',
    OrchestratorState.AWAITING_SAMPLE_REVIEW:
        'orch.approve_sample(session_id) or orch.reject(session_id, feedback)',
    OrchestratorState.FULL_RUNNING:
        'orch.query(session_id, "...") or orch.get_status(session_id) or orch.wait(session_id)',
    OrchestratorState.AWAITING_RESULTS_REVIEW:
        'orch.approve_results(session_id) or orch.reject_results(session_id, feedback)',
    OrchestratorState.COMPLETE:
        'Session complete. Use .data_files for research data.',
    OrchestratorState.FAILED:
        'Session failed. Check .message for details.',
    OrchestratorState.ABORTED:
        'Session aborted by user.',
}


# ─────────────────────────────────────────────────────────────────────────────
# Update object (returned from every orchestrator call)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorUpdate:
    """
    Return type for every orchestrator method.

    Read .state to know what happened and what to do next.
    Read .message for a human-readable summary.
    Read .next_action for the exact call to make.

    Payloads (at most one is non-None per update):
      .clarification    — when state == CLARIFYING
      .spec_approval    — when state == AWAITING_SPEC_APPROVAL
      .artifacts        — when state == READY_TO_SAMPLE (script path to inspect)
      .sample_summary   — when state == AWAITING_SAMPLE_REVIEW
      .sample_analysis  — when state == AWAITING_SAMPLE_REVIEW
      .full_summary     — when state == AWAITING_RESULTS_REVIEW or COMPLETE
      .full_analysis    — when state == AWAITING_RESULTS_REVIEW or COMPLETE
      .query_answer     — when called via .query()
      .progress         — when called via .get_status()
    """
    session_id:      str
    state:           OrchestratorState
    message:         str
    next_action:     str

    clarification:   Optional[ClarificationRequest]  = None
    spec_approval:   Optional[SpecApproval]          = None
    artifacts:       Optional[GeneratedArtifacts]    = None
    sample_summary:  Optional[RunSummary]            = None
    sample_analysis: Optional[AnalysisResult]        = None
    full_summary:    Optional[RunSummary]             = None
    full_analysis:   Optional[AnalysisResult]        = None
    query_answer:    str                             = ""
    progress:        Optional["SweepProgress"]       = None

    def __str__(self) -> str:
        lines = [
            f"{'─'*60}",
            f"[{self.state.value.upper()}]  session={self.session_id}",
            f"{self.message}",
            f"{'─'*60}",
        ]
        if self.clarification:
            lines.append(str(self.clarification))
        if self.spec_approval:
            lines.append(str(self.spec_approval))
        if self.artifacts:
            lines.append(str(self.artifacts))
        if self.sample_summary:
            lines.append("SAMPLE RESULTS:")
            lines.append(str(self.sample_summary))
        if self.sample_analysis:
            lines.append("SAMPLE ANALYSIS:")
            lines.append(str(self.sample_analysis))
        if self.full_summary:
            lines.append("FULL SWEEP RESULTS:")
            lines.append(str(self.full_summary))
        if self.full_analysis:
            lines.append("FULL ANALYSIS:")
            lines.append(str(self.full_analysis))
        if self.query_answer:
            lines.append(f"ANSWER: {self.query_answer}")
        if self.progress:
            lines.append(str(self.progress))
        lines.append(f"Next: {self.next_action}")
        return "\n".join(lines)


@dataclass
class SweepProgress:
    """Live progress snapshot from a running sweep."""
    total_configs:   int
    configs_done:    int
    configs_running: int
    elapsed_seconds: float
    latest_data:     dict[str, dict]   # config_tag → last data_log row

    @property
    def fraction_done(self) -> float:
        return self.configs_done / max(self.total_configs, 1)

    def __str__(self) -> str:
        pct = 100 * self.fraction_done
        eta = (
            (self.elapsed_seconds / self.fraction_done * (1 - self.fraction_done))
            if self.fraction_done > 0 else float("inf")
        )
        eta_str = f"{eta:.0f}s" if math.isfinite(eta) and eta < 3600 else (
                  f"{eta/3600:.1f}h" if math.isfinite(eta) else "∞")
        lines = [
            f"Progress: {self.configs_done}/{self.total_configs} configs done "
            f"({pct:.0f}%)  elapsed={self.elapsed_seconds:.0f}s  ETA≈{eta_str}",
        ]
        for tag, row in list(self.latest_data.items())[:5]:
            row_str = "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
                if k not in ("step", "sim_time", "_event", "_reason")
            )
            lines.append(f"  {tag[:30]:<30} {row_str}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal session record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _OrchestratorSession:
    session_id:      str
    research_goal:   str
    tool_session_id: str
    state:           OrchestratorState = OrchestratorState.CLARIFYING

    # Accumulated results (set as stages complete)
    spec_approval:   Optional[SpecApproval]       = None
    artifacts:       Optional[GeneratedArtifacts] = None
    sample_summary:  Optional[RunSummary]         = None
    sample_analysis: Optional[AnalysisResult]     = None
    full_summary:    Optional[RunSummary]         = None
    full_analysis:   Optional[AnalysisResult]     = None

    # Background execution
    _future:         Optional[Future]             = field(default=None, repr=False)
    _executor:       Optional[ThreadPoolExecutor] = field(default=None, repr=False)
    _sweep_start:    float                        = field(default=0.0, repr=False)
    _sweep_dir:      Optional[Path]               = field(default=None, repr=False)
    _lock:           threading.Lock               = field(default_factory=threading.Lock, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts for progress queries
# ─────────────────────────────────────────────────────────────────────────────

_QUERY_SYSTEM = """
You are the progress assistant for a running scientific simulation.
You receive:
  - The research goal
  - The simulation specification (variables, stopping conditions)
  - A snapshot of partial data from data_log.jsonl files (whatever has been written so far)
  - The user's question

Answer the question concisely and precisely based only on the data provided.
If the data does not yet contain enough information to answer, say so clearly.
Do not speculate beyond what the data shows.
Keep your answer under 150 words.
Reference specific numbers from the data wherever possible.
""".strip()

_SAMPLE_REVIEW_SYSTEM = """
You are reviewing the results of a quick sample run for a scientific simulation.
The sample ran with default parameters at reduced step count (5–10% of full budget).
Its purpose is to validate the generated code is correct and the physics is reasonable.

You receive:
  - The research goal
  - The sample RunSummary (outcomes, steps, wall time)
  - The AnalysisResult (flags, verdict)
  - Partial data statistics from data_log.jsonl

Write a brief (≤200 words) review covering:
  1. Did the code run correctly? (crash, assertion errors, NaN?)
  2. Does the physics look reasonable? (magnetisation in [-1,1]? energy finite?)
  3. Should the user proceed to the full sweep, or is something wrong?

End with exactly one line:
  VERDICT: PROCEED  or  VERDICT: FIX_FIRST: <what to fix>
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# ResearchOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ResearchOrchestrator:
    """
    Drives the full research workflow from goal to results.

    Args:
        tool:         SimulationTool instance (already configured with backend).
        sample_steps: Max steps for the trial run (default: 200).
                      Set to 0 to skip the sample phase entirely.
    """

    def __init__(
        self,
        tool: SimulationTool,
        sample_steps: int = 200,
    ) -> None:
        self._tool = tool
        self._sample_steps = sample_steps
        self._sessions: dict[str, _OrchestratorSession] = {}
        log.info(
            f"ResearchOrchestrator ready | "
            f"backend={tool._designer._backend.model_name} | "
            f"sample_steps={sample_steps}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Design
    # ─────────────────────────────────────────────────────────────────────────

    def start(self, research_goal: str) -> OrchestratorUpdate:
        """
        Begin a new research session.

        research_goal: Plain English description of what you want to study.
        The orchestrator passes this to the designer LLM which will ask
        clarifying questions if needed.
        """
        log.info(f"New research session | goal: {research_goal[:60]!r}…")

        # The designer takes the research goal as a simulation description.
        # They are capable of asking what they need.
        result = self._tool.start(research_goal)
        tool_session_id = result.session_id

        orch_id = uuid.uuid4().hex[:8]
        sess = _OrchestratorSession(
            session_id=orch_id,
            research_goal=research_goal,
            tool_session_id=tool_session_id,
        )
        self._sessions[orch_id] = sess

        return self._route_designer_result(sess, result)

    def answer(
        self,
        session_id: str,
        answers: dict[str, str],
    ) -> OrchestratorUpdate:
        """
        Answer the designer's clarifying questions.
        answers: {"1": "temperature from 1.5 to 4.0", ...}
        """
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.CLARIFYING)
        result = self._tool.answer(sess.tool_session_id, answers)
        return self._route_designer_result(sess, result)

    def approve(self, session_id: str) -> OrchestratorUpdate:
        """
        Approve the proposed spec. Generates script + notebook.
        Returns READY_TO_SAMPLE with the script path for inspection.
        """
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.AWAITING_SPEC_APPROVAL)

        artifacts = self._tool.approve(sess.tool_session_id)
        sess.artifacts = artifacts
        sess.state = OrchestratorState.READY_TO_SAMPLE

        log.info(f"[{session_id}] Spec approved. Script: {artifacts.script_path}")
        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=(
                f"Spec approved. Script generated ({artifacts.script_size_kb} KB).\n"
                f"You may now open and inspect the script before running the sample.\n"
                f"  Script   : {artifacts.script_path}\n"
                f"  Notebook : {artifacts.notebook_path}\n"
                f"  Inspect  : less {artifacts.script_path}\n"
                + (f"  Try it   : python {artifacts.script_path.name} --help\n"
                   if artifacts.script_path else "")
            ),
            next_action=_NEXT_ACTION[sess.state],
            artifacts=artifacts,
        )

    def reject(self, session_id: str, feedback: str) -> OrchestratorUpdate:
        """
        Reject the current spec and restart design with feedback.
        Works from: AWAITING_SPEC_APPROVAL, READY_TO_SAMPLE,
                    AWAITING_SAMPLE_REVIEW.
        """
        sess = self._get(session_id)
        allowed = {
            OrchestratorState.AWAITING_SPEC_APPROVAL,
            OrchestratorState.READY_TO_SAMPLE,
            OrchestratorState.AWAITING_SAMPLE_REVIEW,
        }
        if sess.state not in allowed:
            raise ValueError(
                f"Cannot reject in state {sess.state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        result = self._tool.reject(sess.tool_session_id, feedback)
        return self._route_designer_result(sess, result)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Sample run
    # ─────────────────────────────────────────────────────────────────────────

    def run_sample(self, session_id: str) -> OrchestratorUpdate:
        """
        Run a quick trial: single default config, ~5% of max_steps.
        Blocks until complete. Validates code correctness before the full sweep.

        Returns AWAITING_SAMPLE_REVIEW with sample results.
        """
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.READY_TO_SAMPLE)
        sess.state = OrchestratorState.SAMPLE_RUNNING

        log.info(
            f"[{session_id}] Starting sample run "
            f"(max_steps={self._sample_steps}, single config)"
        )

        # Run with reduced steps; no sweep — default config only
        overrides: dict = {}
        if self._sample_steps > 0:
            overrides["max_steps"] = self._sample_steps

        try:
            summary = self._tool.run_single(
                sess.tool_session_id,
                config_overrides=overrides,
                stream_logs=False,
            )
        except Exception as exc:
            sess.state = OrchestratorState.FAILED
            return self._fail(session_id, f"Sample run failed with exception: {exc}")

        sess.sample_summary = summary
        analysis = self._tool.analyse_sweep(
            sess.tool_session_id, use_llm=False   # fast deterministic-only for sample
        )
        sess.sample_analysis = analysis
        sess.state = OrchestratorState.AWAITING_SAMPLE_REVIEW

        # Ask LLM to write a brief sample review
        review = self._write_sample_review(sess)

        log.info(
            f"[{session_id}] Sample done: "
            f"{summary.n_success} success, {summary.n_crash} crash, "
            f"{summary.n_max_steps} max_steps"
        )
        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=f"Sample run complete.\n\n{review}",
            next_action=_NEXT_ACTION[sess.state],
            sample_summary=summary,
            sample_analysis=analysis,
        )

    def approve_sample(self, session_id: str) -> OrchestratorUpdate:
        """
        Approve the sample results and launch the full parameter sweep.
        The sweep runs in a background thread — this call returns immediately.
        Use query() or get_status() to monitor progress, wait() to block.
        """
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.AWAITING_SAMPLE_REVIEW)

        configs = self._tool.list_sweep_configs(sess.tool_session_id)
        log.info(
            f"[{session_id}] Launching full sweep: "
            f"{len(configs)} configuration(s) in background thread"
        )

        sess.state = OrchestratorState.FULL_RUNNING
        sess._sweep_start = time.perf_counter()
        sess._sweep_dir = (
            self._tool.output_root / sess.tool_session_id / "sweep"
        )

        # Launch sweep in background thread
        sess._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep")
        sess._future = sess._executor.submit(self._run_full_sweep, sess)

        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=(
                f"Full sweep launched in background ({len(configs)} configuration(s)).\n"
                f"This call returned immediately. The sweep is running.\n"
                f"  Sweep output : {sess._sweep_dir}\n"
                f"  Configs      : "
                + ", ".join(
                    str(list({k: v for k, v in c.items()
                              if k not in ("max_steps","checkpoint_interval",
                                           "progress_interval","data_log_interval")}.values()))
                    for c in configs[:5]
                )
                + (f"  … and {len(configs)-5} more" if len(configs) > 5 else "")
            ),
            next_action=_NEXT_ACTION[sess.state],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Monitor
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, session_id: str, question: str) -> OrchestratorUpdate:
        """
        Ask a question about the current state of the session.
        Works in any state, especially useful during FULL_RUNNING.

        During FULL_RUNNING: reads partial data_log.jsonl files from the sweep
        directory (the subprocess is writing to them continuously) and asks the
        LLM to answer the question with that live data.
        """
        sess = self._get(session_id)

        # Check for completion first
        if sess.state == OrchestratorState.FULL_RUNNING:
            self._check_sweep_done(sess)

        answer = self._answer_question(sess, question)
        log.debug(f"[{session_id}] Query: {question!r} → {len(answer)} chars")

        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=f"Q: {question}",
            next_action=_NEXT_ACTION[sess.state],
            query_answer=answer,
        )

    def get_status(self, session_id: str) -> OrchestratorUpdate:
        """
        Get a live progress snapshot. Non-blocking.
        If the background sweep just finished, transitions to AWAITING_RESULTS_REVIEW.
        """
        sess = self._get(session_id)

        if sess.state == OrchestratorState.FULL_RUNNING:
            self._check_sweep_done(sess)

        progress = self._read_progress(sess)

        if sess.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
            # Just transitioned — return the full results
            return OrchestratorUpdate(
                session_id=session_id,
                state=sess.state,
                message="Full sweep complete. Results ready for review.",
                next_action=_NEXT_ACTION[sess.state],
                full_summary=sess.full_summary,
                full_analysis=sess.full_analysis,
                progress=progress,
            )

        msg = f"State: {sess.state.value}"
        if progress:
            msg += f"\n{progress}"

        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=msg,
            next_action=_NEXT_ACTION[sess.state],
            progress=progress,
        )

    def wait(self, session_id: str, timeout: Optional[float] = None) -> OrchestratorUpdate:
        """
        Block until the background sweep finishes, then return results.
        Polls every 2 seconds to remain interruptible via KeyboardInterrupt.
        """
        sess = self._get(session_id)
        if sess.state not in (OrchestratorState.FULL_RUNNING,
                               OrchestratorState.SAMPLE_RUNNING):
            return self.get_status(session_id)

        deadline = time.perf_counter() + timeout if timeout else None
        while True:
            self._check_sweep_done(sess)
            if sess.state != OrchestratorState.FULL_RUNNING:
                break
            if deadline and time.perf_counter() > deadline:
                log.warning(f"[{session_id}] wait() timed out after {timeout}s")
                break
            time.sleep(2.0)

        return self.get_status(session_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Results review
    # ─────────────────────────────────────────────────────────────────────────

    def approve_results(self, session_id: str) -> OrchestratorUpdate:
        """Accept the results and close the session."""
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.AWAITING_RESULTS_REVIEW)

        # Write lessons to AGENTS.md
        self._tool.consolidate_memory(
            sess.tool_session_id,
            run_summary=sess.full_summary,
            analysis=sess.full_analysis,
        )

        sess.state = OrchestratorState.COMPLETE
        data_files = sess.full_summary.data_files if sess.full_summary else []
        log.info(f"[{session_id}] Session complete. {len(data_files)} data files.")

        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=(
                f"Research session complete.\n"
                f"  Data files for analysis ({len(data_files)}):\n"
                + "\n".join(f"    {p}  ← pd.read_json(path, lines=True)"
                            for p in data_files)
                + f"\n  Memory updated: {self._tool.memory_path('designer')}"
            ),
            next_action=_NEXT_ACTION[sess.state],
            full_summary=sess.full_summary,
            full_analysis=sess.full_analysis,
        )

    def reject_results(
        self, session_id: str, feedback: str
    ) -> OrchestratorUpdate:
        """
        Reject the full results and revise the simulation spec.
        Returns to AWAITING_SPEC_APPROVAL with the designer's revised spec.
        """
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.AWAITING_RESULTS_REVIEW)
        log.info(f"[{session_id}] Results rejected — revising spec: {feedback!r}")
        result = self._tool.reject(sess.tool_session_id, feedback)
        return self._route_designer_result(sess, result)

    # ─────────────────────────────────────────────────────────────────────────
    # Patch application (shortcut from analysis)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_patch_and_rerun(self, session_id: str) -> OrchestratorUpdate:
        """
        Apply the recommended SpecPatch from the full analysis and re-run.
        Only works if the analysis verdict is MINOR_FIX or MAJOR_FIX.
        Returns READY_TO_SAMPLE so user can inspect the patched script.
        """
        sess = self._get(session_id)
        self._require_state(sess, OrchestratorState.AWAITING_RESULTS_REVIEW)

        analysis = sess.full_analysis
        if not analysis or not analysis.patch:
            return OrchestratorUpdate(
                session_id=session_id, state=sess.state,
                message="No patch available. Use approve_results() to finish.",
                next_action=_NEXT_ACTION[sess.state],
            )

        if analysis.verdict == Verdict.ABORT:
            return OrchestratorUpdate(
                session_id=session_id, state=sess.state,
                message=(
                    "Analyst returned ABORT verdict — the parameter space is "
                    "fundamentally incompatible. Use reject_results() with feedback "
                    "and redesign the simulation."
                ),
                next_action=_NEXT_ACTION[sess.state],
            )

        artifacts = self._tool.apply_patch(sess.tool_session_id, analysis.patch)
        sess.artifacts = artifacts
        sess.full_summary = None
        sess.full_analysis = None
        sess.state = OrchestratorState.READY_TO_SAMPLE

        return OrchestratorUpdate(
            session_id=session_id,
            state=sess.state,
            message=(
                f"Patch applied and code regenerated.\n"
                f"  Changes: "
                + "; ".join(str(c) for c in analysis.patch.changes[:3])
                + f"\n  New script: {artifacts.script_path}"
            ),
            next_action=_NEXT_ACTION[sess.state],
            artifacts=artifacts,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get(self, session_id: str) -> _OrchestratorSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown orchestrator session: {session_id!r}")
        return self._sessions[session_id]

    def _require_state(
        self, sess: _OrchestratorSession, expected: OrchestratorState
    ) -> None:
        if sess.state != expected:
            raise ValueError(
                f"Expected state {expected.value!r}, "
                f"but session is in {sess.state.value!r}."
            )

    def _fail(self, session_id: str, message: str) -> OrchestratorUpdate:
        sess = self._sessions[session_id]
        sess.state = OrchestratorState.FAILED
        log.error(f"[{session_id}] FAILED: {message}")
        return OrchestratorUpdate(
            session_id=session_id, state=sess.state,
            message=message, next_action=_NEXT_ACTION[sess.state],
        )

    def _route_designer_result(
        self,
        sess: _OrchestratorSession,
        result: Union[ClarificationRequest, SpecApproval],
    ) -> OrchestratorUpdate:
        """Route a designer result to the appropriate orchestrator state."""
        sid = sess.session_id
        if isinstance(result, ClarificationRequest):
            sess.state = OrchestratorState.CLARIFYING
            topic_counts = {}
            for q in result.questions:
                topic_counts[q.topic.value] = topic_counts.get(q.topic.value, 0) + 1
            return OrchestratorUpdate(
                session_id=sid,
                state=sess.state,
                message=(
                    f"Designer needs {len(result.questions)} clarification(s).\n"
                    f"Topics: {topic_counts}"
                ),
                next_action=_NEXT_ACTION[sess.state],
                clarification=result,
            )
        else:
            # SpecApproval
            sess.spec_approval = result
            sess.state = OrchestratorState.AWAITING_SPEC_APPROVAL
            return OrchestratorUpdate(
                session_id=sid,
                state=sess.state,
                message=(
                    f"Spec ready for review.\n"
                    f"  Variables: {result.variable_count}  "
                    f"Stop conditions: {result.stop_cond_count}\n"
                    f"  Time estimate: {result.time_estimate}"
                ),
                next_action=_NEXT_ACTION[sess.state],
                spec_approval=result,
            )

    def _run_full_sweep(self, sess: _OrchestratorSession) -> RunSummary:
        """Called in background thread. Runs sweep, then analyses, updates state."""
        log.info(f"[{sess.session_id}] Background sweep started")
        try:
            summary = self._tool.run_sweep(
                sess.tool_session_id,
                max_workers=1,
                stream_logs=False,
            )
            analysis = self._tool.analyse_sweep(
                sess.tool_session_id, use_llm=True
            )
            with sess._lock:
                sess.full_summary = summary
                sess.full_analysis = analysis
                sess.state = OrchestratorState.AWAITING_RESULTS_REVIEW
            log.info(
                f"[{sess.session_id}] Background sweep done: "
                f"verdict={analysis.verdict.value}"
            )
            return summary
        except Exception as exc:
            with sess._lock:
                sess.state = OrchestratorState.FAILED
            log.error(f"[{sess.session_id}] Background sweep failed: {exc}")
            raise

    def _check_sweep_done(self, sess: _OrchestratorSession) -> None:
        """Check if background future is done and update state accordingly."""
        if sess._future is None or not sess._future.done():
            return
        if sess.state in (OrchestratorState.AWAITING_RESULTS_REVIEW,
                          OrchestratorState.COMPLETE, OrchestratorState.FAILED):
            return   # already transitioned
        try:
            sess._future.result()  # re-raise exceptions from thread
        except Exception as exc:
            with sess._lock:
                sess.state = OrchestratorState.FAILED
            log.error(f"[{sess.session_id}] Sweep thread raised: {exc}")

    def _read_progress(self, sess: _OrchestratorSession) -> Optional[SweepProgress]:
        """
        Read partial data_log.jsonl files from the sweep directory.
        Returns a SweepProgress snapshot showing what has been written so far.
        Works during both SAMPLE_RUNNING and FULL_RUNNING.
        """
        sweep_dir = sess._sweep_dir
        if not sweep_dir or not sweep_dir.exists():
            # For single runs, check the run_single dir
            single_dir = self._tool.output_root / sess.tool_session_id
            if not single_dir.exists():
                return None
            sweep_dir = single_dir

        configs = self._tool.list_sweep_configs(sess.tool_session_id)
        total = max(len(configs), 1)

        # Find all data_log.jsonl files
        data_files = list(sweep_dir.rglob("data_log.jsonl"))
        configs_done = sum(
            1 for d in sweep_dir.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        ) if sweep_dir.is_dir() else 0

        latest_data: dict[str, dict] = {}
        for data_file in data_files[:10]:  # cap to avoid slow glob
            lines = []
            try:
                lines = data_file.read_text().strip().splitlines()
            except OSError:
                continue
            if lines:
                try:
                    last_row = json.loads(lines[-1])
                    tag = data_file.parent.name[:30]
                    latest_data[tag] = last_row
                except json.JSONDecodeError:
                    pass

        elapsed = time.perf_counter() - sess._sweep_start if sess._sweep_start else 0
        return SweepProgress(
            total_configs=total,
            configs_done=configs_done,
            configs_running=len(data_files) - configs_done,
            elapsed_seconds=elapsed,
            latest_data=latest_data,
        )

    def _answer_question(self, sess: _OrchestratorSession, question: str) -> str:
        """
        Answer a user's progress question using LLM + partial data.
        Falls back to deterministic answer if no LLM backend.
        """
        # Collect context
        context_parts: list[str] = [
            f"Research goal: {sess.research_goal}",
            f"Current state: {sess.state.value}",
        ]

        # Add spec summary
        if sess.tool_session_id in self._tool._sessions:
            spec_str = self._tool.show_spec(sess.tool_session_id)
            context_parts.append(f"Simulation spec:\n{spec_str}")

        # Add partial data
        progress = self._read_progress(sess)
        if progress:
            context_parts.append(f"Live progress:\n{progress}")
            if progress.latest_data:
                context_parts.append("Partial data (latest row per config):")
                for tag, row in progress.latest_data.items():
                    context_parts.append(f"  {tag}: {json.dumps(row)}")

        # Add sample results if available
        if sess.sample_summary:
            context_parts.append("Sample run results:")
            for r in sess.sample_summary.results:
                context_parts.append(f"  [{r.outcome.value}] {r.reason}")

        # Add full results if available
        if sess.full_analysis:
            context_parts.append(f"Analysis verdict: {sess.full_analysis.verdict.value}")
            if sess.full_analysis.flags:
                context_parts.append("Flags: " + ", ".join(f.kind for f in sess.full_analysis.flags))

        context = "\n\n".join(context_parts)
        prompt = f"{context}\n\nQuestion: {question}"

        backend = self._tool._designer._backend
        try:
            answer = backend.complete(
                system=_QUERY_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return answer.strip()
        except Exception as exc:
            log.warning(f"LLM query failed: {exc}")
            # Deterministic fallback
            return self._deterministic_answer(sess, question, progress)

    def _deterministic_answer(
        self,
        sess: _OrchestratorSession,
        question: str,
        progress: Optional[SweepProgress],
    ) -> str:
        """Fallback answer without LLM — reports raw numbers from data."""
        parts = [f"State: {sess.state.value}"]
        if progress:
            parts.append(str(progress))
        if not progress or not progress.latest_data:
            parts.append("No data available yet.")
        return "\n".join(parts)

    def _write_sample_review(self, sess: _OrchestratorSession) -> str:
        """Ask LLM to review sample results, or produce deterministic review."""
        summary = sess.sample_summary
        analysis = sess.sample_analysis
        if not summary or not analysis:
            return "No sample data to review."

        # Build compact context
        context_parts = [
            f"Research goal: {sess.research_goal}",
            f"Sample results ({len(summary.results)} run(s)):",
        ]
        for r in summary.results:
            context_parts.append(
                f"  [{r.outcome.value}] steps={r.steps_run:,} "
                f"wall={r.wall_time_seconds:.1f}s  {r.reason}"
            )
        if analysis.flags:
            context_parts.append(
                "Flags: " + "; ".join(f"{f.kind}({f.severity})" for f in analysis.flags)
            )
        # Add data stats
        for r in summary.results:
            data_file = r.data_file
            if data_file and data_file.exists():
                lines = data_file.read_text().strip().splitlines()
                if lines:
                    try:
                        last = json.loads(lines[-1])
                        context_parts.append(
                            f"Last data row: {json.dumps(last)}"
                        )
                    except json.JSONDecodeError:
                        pass

        context = "\n".join(context_parts)
        backend = self._tool._designer._backend

        try:
            return backend.complete(
                system=_SAMPLE_REVIEW_SYSTEM,
                messages=[{"role": "user", "content": context}],
                max_tokens=400,
            ).strip()
        except Exception as exc:
            log.warning(f"LLM sample review failed: {exc}")
            # Deterministic fallback
            n_ok = summary.n_success + summary.n_max_steps
            n_bad = summary.n_crash + summary.n_failed
            verdict = "PROCEED" if n_bad == 0 else f"FIX_FIRST: {n_bad} runs crashed or failed"
            return (
                f"Sample run: {n_ok}/{len(summary.results)} completed without crash.\n"
                f"Flags: {[f.kind for f in (analysis.flags if analysis else [])]}\n"
                f"VERDICT: {verdict}"
            )
