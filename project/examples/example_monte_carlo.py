#!/usr/bin/env python3
"""
example_monte_carlo.py — 2D Ising Model, full agent workflow.

Demonstrates the contract-typed API:

  tool.start()       → ClarificationRequest | SpecApproval
  tool.answer()      → ClarificationRequest | SpecApproval
  tool.approve()     → GeneratedArtifacts
  tool.run_sweep()   → RunSummary
  tool.analyse_sweep() → AnalysisResult  (verdict + SpecPatch)
  tool.apply_patch() → GeneratedArtifacts  (if MINOR_FIX)
  tool.consolidate_memory() → rules written to ~/.sim_tool/AGENTS.md
"""
import argparse, sys
from pathlib import Path
from sim_tool import SimulationTool, ClarificationRequest, SpecApproval, Verdict
from sim_tool.llm import make_backend

DESCRIPTION = """
Simulate the 2D Ising model on a square NxN lattice using the Metropolis
Monte Carlo algorithm.

Spins are ±1 with nearest-neighbour coupling J and periodic boundary conditions.
One step = one full lattice sweep (N² spin-flip attempts).

Goal: study the phase transition by sweeping temperature T.
Critical temperature Tc ≈ 2.269 J/kB.

OPTIMISATION: exp(-ΔE/T) has only 5 possible values — precompute them once.
Also precompute neighbour index arrays to avoid modulo inside the inner loop.

Track per-site magnetisation m, energy E/N², and acceptance rate.
Data log: every 5 sweeps. In-memory history: all sweeps.

Stopping:
  SUCCESS: |m| stable within tolerance over 100 sweeps after equilibration.
  FAILURE: acceptance_rate ≈ 0 for many sweeps (lattice frozen, low T).
  FAILURE: energy non-finite (numerical failure).

Temperature sweep: T from 1.5 to 4.0 step 0.5 (J/kB). N=32 fixed. seed=42.
"""

SCRIPTED_ANSWERS = {
    "1": "N=32 fixed. T swept 1.5 to 4.0 step 0.5. J=1.0. equil=500. prec=0.01. seed=42.",
    "2": "max_steps=3000. checkpoint every 500. progress every 300. data_log every 5.",
    "3": "State: spins(list), energy(float), magnetisation(float), acceptance_rate(float), in_equilibration(bool), m_window(list). Pure Python stdlib only.",
    "4": "Use reasonable defaults.",
}

def main(interactive, backend_name, model):
    use_mock = backend_name == "mock"
    if use_mock:
        print("NOTE: Using MockBackend\n")
        from sim_tool.llm_mock import MockBackend
        backend = MockBackend("ising")
    else:
        backend = make_backend(backend_name, **({} if not model else {"model": model}))

    tool = SimulationTool(output_root=Path("./ising_output"), backend=backend)

    print(f"\n{'═'*60}\n  2D ISING MODEL | backend: {backend.model_name}\n{'═'*60}\n")

    # ── STEP 1: Start ─────────────────────────────────────────────────────────
    print("STEP 1 — Describing simulation to LLM designer\n")
    r = tool.start(DESCRIPTION)
    print(r); print()

    # ── STEP 2: Clarify ───────────────────────────────────────────────────────
    round_n = 0
    while isinstance(r, ClarificationRequest):
        round_n += 1
        print(f"STEP 2.{round_n} — {len(r.questions)} question(s) | topics: "
              f"{[q.topic.value for q in r.questions]}\n")
        answers = {}
        for q in r.questions:
            if interactive:
                print(f"  Q{q.index} [{q.topic.value}]: {q.text}")
                answers[str(q.index)] = input("  A: ").strip()
            else:
                ans = SCRIPTED_ANSWERS.get(str(q.index), "Use physics defaults")
                print(f"  Q{q.index} [{q.topic.value}]: {q.text}")
                print(f"  A: [scripted] {ans}")
                answers[str(q.index)] = ans
        print()
        r = tool.answer(r.session_id, answers)
        print(r); print()

    assert isinstance(r, SpecApproval), f"Expected SpecApproval, got {type(r)}"

    # ── STEP 3: Review spec ───────────────────────────────────────────────────
    session_id = r.session_id
    print(f"STEP 3 — Spec ready | {r.variable_count} vars | {r.stop_cond_count} stop conds")
    print(f"  Time estimate    : {r.time_estimate}")
    print(f"  Output contract  :\n{r.output_contract}")
    print()
    if interactive and input("Approve? [y/n]: ").lower() not in ("y","yes",""):
        feedback = input("Feedback: ")
        tool.reject(session_id, feedback); sys.exit(0)
    else:
        print("  [auto-approve]")

    # ── STEP 4: Approve → GeneratedArtifacts ──────────────────────────────────
    artifacts = tool.approve(session_id)
    print(f"\nSTEP 4 — Code generated\n{artifacts}\n")

    # ── STEP 5: Run sweep → RunSummary ────────────────────────────────────────
    configs = tool.list_sweep_configs(session_id)
    print(f"STEP 5 — Sweep: {len(configs)} configuration(s)")
    summary = tool.run_sweep(session_id, max_workers=1, stream_logs=False)
    print(summary)
    print(f"\nData files ({len(summary.data_files)}):")
    for f in summary.data_files:
        print(f"  {f}  ← pd.read_json(path, lines=True)")
    print()

    # ── STEP 6: Analyse → AnalysisResult ─────────────────────────────────────
    print("STEP 6 — Analysing sweep results")
    analysis = tool.analyse_sweep(session_id, use_llm=(not use_mock))
    print(analysis)
    print()

    # ── STEP 7: Apply patch if needed ─────────────────────────────────────────
    if analysis.verdict in (Verdict.MINOR_FIX, Verdict.MAJOR_FIX) and analysis.patch:
        print(f"STEP 7 — Applying {analysis.verdict.value} patch")
        new_artifacts = tool.apply_patch(session_id, analysis.patch)
        print(new_artifacts)
        print("\nRe-running with patched spec...")
        summary2 = tool.run_sweep(session_id, max_workers=1, stream_logs=False)
        print(summary2)
    elif analysis.verdict == Verdict.OK:
        print("✓ All runs successful — no patch needed.")
    elif analysis.verdict == Verdict.ABORT:
        print("⚠ ABORT verdict — restart with a revised description.")

    # ── STEP 8: Consolidate memory ────────────────────────────────────────────
    print("\nSTEP 8 — Consolidating session lessons into AGENTS.md")
    rules = tool.consolidate_memory(session_id, summary, analysis)
    print(f"  Designer rules added: {len(rules['designer'])}")
    print(f"  Analyst rules added : {len(rules['analyst'])}")
    print(f"  Memory files: {tool.memory_path('designer')}")
    print()
    print(f"{'═'*60}\n  DONE | session={session_id}\n{'═'*60}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--backend", default="lmstudio",
                   choices=["mock", "anthropic", "lmstudio", "ollama", "openai", "openai-compat", "custom"])
    p.add_argument("--model", default="")
    args = p.parse_args()
    main(args.interactive, args.backend, args.model)
