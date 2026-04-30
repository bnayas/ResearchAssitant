#!/usr/bin/env python3
"""
example_orchestrator.py
═══════════════════════
Demonstrates the full ResearchOrchestrator workflow for the 2D Ising model.

The orchestrator drives the complete research lifecycle:

  orch.start(goal)          → CLARIFYING
  orch.answer(sid, answers) → AWAITING_SPEC_APPROVAL
  orch.approve(sid)         → READY_TO_SAMPLE  ← inspect script here
  orch.run_sample(sid)      → AWAITING_SAMPLE_REVIEW
  orch.approve_sample(sid)  → FULL_RUNNING  ← non-blocking
  orch.query(sid, question) → answer from partial data_log.jsonl
  orch.get_status(sid)      → live progress snapshot
  orch.wait(sid)            → AWAITING_RESULTS_REVIEW
  orch.approve_results(sid) → COMPLETE

Usage:
    python example_orchestrator.py                     # scripted, LM Studio default
    python example_orchestrator.py --backend mock     # fully offline mock
    python example_orchestrator.py --backend anthropic
    python example_orchestrator.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from sim_tool import SimulationTool
from sim_tool.llm import make_backend
from sim_tool.orchestrator import OrchestratorState, ResearchOrchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Research goal — the ONLY domain input
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_GOAL = """
Study the ferromagnetic phase transition in the 2D Ising model.

I want to understand how the magnetisation and energy behave across a range
of temperatures, specifically near the theoretical critical temperature
Tc ≈ 2.269 J/kB.

The simulation should use the Metropolis Monte Carlo algorithm on a 32×32
square lattice with periodic boundary conditions. The Boltzmann acceptance
weights should be precomputed (only 5 distinct ΔE values exist).

I want results for temperatures from 1.5 to 4.0 in steps of 0.5 J/kB.
Track per-site magnetisation, energy, and acceptance rate.
The simulation should stop early when magnetisation has stabilised
(success) or when the lattice freezes (failure at low T).
"""

SCRIPTED_ANSWERS = {
    "1": "N=32 fixed. T swept 1.5 to 4.0 step 0.5. J=1.0. equil=500. prec=0.01. seed=42.",
    "2": "max_steps=3000. checkpoint=500. progress=300. data_log every 5 sweeps.",
    "3": "State: spins(list), energy(float), magnetisation(float), acceptance_rate(float), in_equilibration(bool), m_window(list). Pure Python stdlib.",
    "4": "Use sensible defaults.",
}

# Questions the user asks while the sweep is running
PROGRESS_QUERIES = [
    "How many temperature configurations have finished so far?",
    "What is the current magnetisation at the lowest and highest temperatures?",
    "Is the T=1.5 run converging or has it frozen?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(interactive: bool, backend_name: str, model: str) -> None:

    # ── Backend ───────────────────────────────────────────────────────────────
    use_mock = backend_name == "mock"
    if use_mock:
        print("NOTE: Using MockBackend for offline demo.\n")
        from sim_tool.llm_mock import MockBackend
        backend = MockBackend("ising")
    else:
        backend = make_backend(backend_name, **({} if not model else {"model": model}))

    tool = SimulationTool(output_root=Path("./orch_output"), backend=backend)
    orch = ResearchOrchestrator(tool, sample_steps=200)

    print(f"{'═'*65}")
    print(f"  ISING MODEL — RESEARCH ORCHESTRATOR DEMO")
    print(f"  Backend: {backend.model_name}")
    print(f"{'═'*65}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 0: Submit research goal
    # ─────────────────────────────────────────────────────────────────────────
    section("Submitting research goal")
    print(RESEARCH_GOAL.strip())
    print()

    u = orch.start(RESEARCH_GOAL)
    print(u)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 0b: Clarifying questions (if any)
    # ─────────────────────────────────────────────────────────────────────────
    while u.state == OrchestratorState.CLARIFYING:
        assert u.clarification is not None
        section(f"Clarifying questions (round {u.clarification.iteration+1})")
        print(f"Topics: {[q.topic.value for q in u.clarification.questions]}\n")

        answers: dict[str, str] = {}
        for q in u.clarification.questions:
            if interactive:
                print(f"  Q{q.index} [{q.topic.value}]: {q.text}")
                answers[str(q.index)] = input("  Your answer: ").strip()
            else:
                ans = SCRIPTED_ANSWERS.get(str(q.index), "Use physics defaults")
                print(f"  Q{q.index} [{q.topic.value}]: {q.text}")
                print(f"  A: [scripted] {ans}")
                answers[str(q.index)] = ans
        print()
        u = orch.answer(u.session_id, answers)
        print(u)
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 1: Review spec — user sees variables, stopping conditions, contract
    # ─────────────────────────────────────────────────────────────────────────
    assert u.state == OrchestratorState.AWAITING_SPEC_APPROVAL
    section("GATE 1 — Review spec before any code runs")
    assert u.spec_approval is not None
    print(u.spec_approval)
    print()

    if interactive:
        choice = input("Approve spec? [y/n/feedback]: ").strip().lower()
        if choice not in ("y", "yes", ""):
            feedback = input("Feedback: ").strip()
            u = orch.reject(u.session_id, feedback)
            print(u)
            sys.exit(0)
    else:
        print("[AUTO-APPROVE spec]")

    u = orch.approve(u.session_id)
    print(u)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 2: Inspect script — code is on disk, user can open it
    # ─────────────────────────────────────────────────────────────────────────
    assert u.state == OrchestratorState.READY_TO_SAMPLE
    assert u.artifacts is not None
    section("GATE 2 — Inspect the generated script before the trial run")

    script = Path(u.artifacts.script_path)
    print(f"  Script   : {script}  ({u.artifacts.script_size_kb} KB)")
    print(f"  Notebook : {u.artifacts.notebook_path}")
    print(f"  CLI      : {u.artifacts.cli_synopsis}")
    print()
    print(f"  First 20 lines of generated simulation code:")
    print("  " + "─" * 55)
    lines = script.read_text().splitlines()
    for line in lines[:20]:
        print(f"  {line}")
    print("  …")
    print()

    if interactive:
        print(f"  Full script at: {script}")
        print(f"  Open it, read it, verify it looks correct.")
        input("  Press Enter when ready to run the sample… ")
    else:
        print("  [Script looks good — proceeding to sample run]")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # SAMPLE RUN — quick trial (single config, reduced steps)
    # ─────────────────────────────────────────────────────────────────────────
    section("Running sample (single config, 200 steps — validates generated code)")
    u = orch.run_sample(u.session_id)
    print(u)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 3: Review sample — was the code correct? physics reasonable?
    # ─────────────────────────────────────────────────────────────────────────
    assert u.state == OrchestratorState.AWAITING_SAMPLE_REVIEW
    section("GATE 3 — Review sample results before full sweep")

    if u.sample_summary:
        print("Sample outcomes:")
        for r in u.sample_summary.results:
            icon = "✓" if r.outcome.value == "success" else (
                   "✗" if r.outcome.value in ("crash","failed") else "⚠")
            print(f"  {icon} [{r.outcome.value}]  steps={r.steps_run:,}  {r.reason}")
        if u.sample_summary.data_files:
            df = u.sample_summary.data_files[0]
            lines = df.read_text().strip().splitlines()
            print(f"\n  Sample data log: {len(lines)} records → {df}")
            if lines:
                import json
                last = json.loads(lines[-1])
                print(f"  Last record: {last}")
    print()

    if u.sample_analysis:
        verdict = u.sample_analysis.verdict
        print(f"  Sample analysis verdict: {verdict.value.upper()}")
        for flag in u.sample_analysis.flags:
            icon = {"error": "✗", "warning": "⚠", "info": "i"}[flag.severity]
            print(f"  [{icon}] {flag.kind}: {flag.message[:80]}")
    print()

    if interactive:
        choice = input("Proceed to full sweep? [y/n]: ").strip().lower()
        if choice not in ("y", "yes", ""):
            feedback = input("What needs fixing? ").strip()
            u = orch.reject(u.session_id, feedback)
            print(u)
            sys.exit(0)
    else:
        print("[AUTO-APPROVE sample — proceeding to full sweep]")

    # ─────────────────────────────────────────────────────────────────────────
    # FULL SWEEP — background thread, returns immediately
    # ─────────────────────────────────────────────────────────────────────────
    section("Launching full sweep in background thread")
    u = orch.approve_sample(u.session_id)
    print(u)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # PROGRESS QUERIES — while sweep is running
    # ─────────────────────────────────────────────────────────────────────────
    section("Querying progress while sweep runs")
    time.sleep(1.0)  # give the first config a moment to start writing

    for i, question in enumerate(PROGRESS_QUERIES, 1):
        if i > 1:
            time.sleep(2.0)  # let more data accumulate between queries
        print(f"  User query {i}: {question!r}")
        u_q = orch.query(u.session_id, question)
        print(f"  Answer: {u_q.query_answer}")
        print()

    # Also show a status snapshot
    section("Live status snapshot")
    u_s = orch.get_status(u.session_id)
    if u_s.progress:
        print(u_s.progress)
    else:
        print(f"  State: {u_s.state.value}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Wait for completion
    # ─────────────────────────────────────────────────────────────────────────
    section("Waiting for full sweep to complete…")
    u = orch.wait(u.session_id)
    print(u)
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # GATE 4: Review full results and analysis
    # ─────────────────────────────────────────────────────────────────────────
    assert u.state in (OrchestratorState.AWAITING_RESULTS_REVIEW,
                       OrchestratorState.FULL_RUNNING)   # may still be running in mock
    section("GATE 4 — Review full results and analysis")

    if u.full_summary:
        print("Full sweep outcomes:")
        for r in u.full_summary.results:
            icon = "✓" if r.outcome.value == "success" else (
                   "✗" if r.outcome.value in ("crash","failed") else "⚠")
            T = r.config.get("temperature", "?")
            print(f"  {icon} T={T}  [{r.outcome.value:>10}]  "
                  f"steps={r.steps_run:,}  {r.reason[:50]}")
        print()
        if u.full_summary.data_files:
            print(f"Data files for research analysis:")
            for p in u.full_summary.data_files:
                print(f"  {p}")
            print(f"\n  Load: import pandas as pd")
            print(f"  df = pd.read_json('{u.full_summary.data_files[0]}', lines=True)")
            print(f"  df.plot(x='sim_time', y='magnetisation')")

    if u.full_analysis:
        print(f"\nAnalysis verdict: {u.full_analysis.verdict.value.upper()}")
        for flag in u.full_analysis.flags:
            icon = {"error": "✗", "warning": "⚠", "info": "i"}[flag.severity]
            print(f"  [{icon}] {flag.kind}: {flag.message[:80]}")
        if u.full_analysis.patch:
            print(f"\nRecommended patch ({len(u.full_analysis.patch.changes)} change(s)):")
            for c in u.full_analysis.patch.changes:
                print(f"  {c.field}: {c.old_value!r} → {c.new_value!r}  ({c.why[:60]})")
    print()

    if interactive:
        choice = input("Accept results? [y=accept / p=apply patch and rerun / n=reject]: ").strip().lower()
        if choice == "p" and u.full_analysis and u.full_analysis.patch:
            u = orch.apply_patch_and_rerun(u.session_id)
            print(u)
            sys.exit(0)  # would continue the loop in a real agent
        elif choice not in ("y", "yes", ""):
            feedback = input("What needs changing? ").strip()
            u = orch.reject_results(u.session_id, feedback)
            print(u)
            sys.exit(0)
    else:
        print("[AUTO-ACCEPT results]")

    if u.state == OrchestratorState.AWAITING_RESULTS_REVIEW:
        u = orch.approve_results(u.session_id)
        print(u)

    print()
    print(f"{'═'*65}")
    print(f"  DONE | session={u.session_id} | state={u.state.value}")
    print(f"  Memory: {tool.memory_path('designer')}")
    print(f"{'═'*65}\n")


def section(title: str) -> None:
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Ising model research orchestrator demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--interactive", action="store_true",
                   help="Stop at each gate for user input")
    p.add_argument("--backend", default="lmstudio",
                   choices=["mock", "anthropic", "lmstudio", "ollama", "openai", "openai-compat", "custom"])
    p.add_argument("--model", default="")
    args = p.parse_args()
    main(interactive=args.interactive, backend_name=args.backend, model=args.model)
