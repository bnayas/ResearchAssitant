# AGENTS.md — Research Platform Agent Registry

This file describes every agent in the platform: its role, contract boundaries,
what it is allowed to do, and what it must never do. Read this before modifying
any agent. Keep it up to date when adding new agents or changing contracts.

---

## Architecture overview

```
Director (orchestrator)
├── sim_tool/
│   ├── SymbolicAgent        — mathematical formalization
│   ├── SimulationDesigner   — LLM spec designer
│   ├── SpecValidator        — deterministic spec gate (28 checks)
│   ├── Launcher             — deterministic executor
│   ├── RunAnalyst           — three-layer sweep analysis
│   └── ResearchOrchestrator — stateful 9-state pipeline driver
│
├── literature_review/
│   ├── LiteratureReviewer   — multi-round search + synthesis
│   └── LiteratureAuditor    — deterministic scope re-validator
│
├── reviewer/
│   ├── JournalReviewer      — article review with math verification
│   └── MathAgentRegistry    — routes math claims to CAS agents
│
└── writer/
    ├── AcademicWriter       — multi-phase paper authoring
    ├── GroundingValidator   — artifact grounding gate
    └── PromiseRegistry      — cross-section forward-reference tracker
```

---

## sim_tool agents

### SymbolicAgent
**File:** `sim_tool/symbolic_agent.py`
**Role:** Formalizes a simulation description into a `SymbolicSpec` (symbols,
equations, assumptions, unresolved gaps) using an LLM + SymPy CAS validation.

**Allowed:**
- Ask clarifying questions about physics (not implementation)
- Call SymPy for symbolic verification
- Return `SymbolicResult` with validated spec or error

**Never:**
- Generate simulation code
- Access external databases
- Make assumptions about numerical parameters not stated in the description

**Contract output:** `SymbolicResult` → gated by `SymbolicValidator` (32 checks, 4 tiers)

---

### SimulationDesigner
**File:** `sim_tool/designer.py`
**Role:** Multi-turn LLM agent that converts a natural-language simulation
description into a validated `SimulationSpec`.

**Allowed questions (exactly 5 topics):**
- `variable_range` — defaults, min/max, sweep points
- `stopping_threshold` — convergence tolerance, failure cutoffs
- `step_budget` — max_steps, data logging frequency
- `physical_units` — units, constants
- `sweep_target` — which variables to sweep

**Forbidden question topics:**
- Which library to use (numpy vs stdlib)
- Code style or data structure preference
- Anything already stated in the description

**Limits:** MAX 5 questions/round, MAX 4 rounds. Forces `spec_complete=true` at MAX_ROUNDS.

**Contract output:** `ClarificationRequest | SpecApproval`
**Auto-repair:** One LLM retry on validator failure before surfacing errors.

---

### SpecValidator
**File:** `sim_tool/contract_validator.py`
**Role:** Deterministic gate between designer LLM output and the director. No LLM.

**Tiers:**
- **A** (4 checks) — presence: required fields non-empty
- **B** (9 checks) — syntax: all code blocks compile as valid Python
- **C** (11 checks) — semantic: return statements, valid identifiers, no bare assignment in check_expr
- **D** (8 checks) — quality warnings (non-blocking): no print(), non-trivial asserts

**Rule:** A/B/C errors block. D warnings pass. `passed = len(A+B+C errors) == 0`.

---

### Launcher
**File:** `sim_tool/launcher.py`
**Role:** Deterministic executor. No LLM. No reasoning. Given same inputs → same outputs.

**Responsibilities:**
1. Runtime selection (Docker-first local execution with process fallback)
2. Environment setup inside the selected runtime
3. Pipeline script writing (portable reproducible `.sh`)
4. Execution with timeout enforcement
5. Failure classification via 14-pattern catalogue → typed `BugTicket`

**Contract output:** `LaunchResult` (success) | `BugTicket` (failure)
Both are validated by `BugTicketValidator` (8 checks) before return.

**Never:** LLM calls, multi-turn conversation, session state between calls.

---

### RunAnalyst
**File:** `sim_tool/analyst.py`
**Role:** Analyses run outputs via three layers and returns a `SpecPatch`.

**Layers (always in order):**
1. **extract_metrics()** — parse `results.json` + `data_log.jsonl` → `RunMetrics`
2. **classify_run/sweep()** — rule-based flags (crash, max_steps, NaN, high failure rate)
3. **_llm_to_patch()** — LLM reads compact summary → `SpecPatch` (only if flags exist)

**Patchable fields (exhaustive):**
`max_steps`, `checkpoint_interval`, `progress_interval`, `data_log_interval`,
`variables.<name>.default/min_val/max_val/sweep_values`,
`stopping_conditions.<name>.check_expr/reason_expr/priority`

**Immutable fields (never patch):**
`step_code`, `precompute_code`, `initial_state_code`, `progress_code`,
`state_fields`, `setup_code`, `config_assert_code`, `state_assert_code`

**Verdict rules:**
- `ok` — no flags → no patch
- `minor_fix` — numerical tweak, high confidence, auto-applicable
- `major_fix` — needs human review before applying
- `abort` — algorithmic incompatibility; requires `tool.start()` restart

---

### ResearchOrchestrator
**File:** `sim_tool/orchestrator.py`
**Role:** 9-state machine driving the full research lifecycle from goal to results.

**States:**
```
CLARIFYING → AWAITING_SPEC_APPROVAL → READY_TO_SAMPLE
  → SAMPLE_RUNNING → AWAITING_SAMPLE_REVIEW
  → FULL_RUNNING → AWAITING_RESULTS_REVIEW
  → COMPLETE | FAILED | ABORTED
```

**Gates (always required):**
1. `AWAITING_SPEC_APPROVAL` — PI reviews spec before any code runs
2. `READY_TO_SAMPLE` — PI can inspect script before sample
3. `AWAITING_SAMPLE_REVIEW` — PI reviews sample run before full sweep
4. `AWAITING_RESULTS_REVIEW` — PI reviews final analysis

**Threading:** Full sweep runs in a background thread. `approve_sample()` is non-blocking.
**Wrong-state calls:** Raise `ValueError` — never silently ignore.

---

## literature_review agents

### LiteratureReviewer
**File:** `literature_review/reviewer.py`
**Role:** Multi-round academic search with per-paper scope validation and synthesis.

**Search progression:**
- Round 1: primary keyword sets (LLM-extracted)
- Rounds 2..N: variation keyword sets (broader/synonym-based)
- Early stop: when `accepted >= min_papers_threshold`
- Hard stop: `max_rounds` total regardless

**Scope validation:** Deterministic year gates first (no LLM), then LLM topic check.
**Deduplication:** URL-based across all rounds.
**Streaming:** All events via `StreamEvent` to GUI in real time.

**Contract output:** `LiteratureReviewArtifact` → passed to `LiteratureAuditor`

---

### LiteratureAuditor
**File:** `literature_review/auditor.py`
**Role:** Deterministic post-hoc audit. No LLM. 8 checks.

**Checks (D1–D8):**
- D1/D2: Year min/max violations in accepted papers
- D3: Accepted count exceeds `scope.max_papers`
- D4: Papers with empty abstracts
- D5: Papers with no URL
- D6: Empty synthesis
- D7: `is_sufficient` vs `status` consistency
- D8: Empty search log

**Rule:** All checks blocking. `passed = all(D1..D8 pass)`.

---

## reviewer agents

### JournalReviewer
**File:** `reviewer/journal_reviewer.py`
**Role:** Four-phase article review agent. Judges ONLY from article text.

**Phases:**
1. **PARSE** — split article into `ArticleView` sections
2. **ANALYSE** — score review dimensions, extract math claims
3. **VERIFY** — dispatch `MathCheckRequest`s concurrently to `MathAgentRegistry`
4. **DECIDE** — synthesize `ReviewFindings` → emit `ReviewDecision` + optional diff

**Decision kinds:** `accepted` | `rejected` | `correction_needed`
**`correction_needed`** always carries an `AnnotatedArticle` with unified diff.

**Hard constraints:**
- No access to upstream simulation/analysis artifacts
- No external databases or internet
- Math verification via agents only, not LLM guessing

---

### MathAgentRegistry
**File:** `reviewer/math_tools.py`
**Role:** Routes `MathCheckRequest` to the correct CAS agent by `claim_kind`.

**Routing table:**
| claim_kind | agent |
|---|---|
| `algebraic_identity`, `bound_or_complexity`, `optimization_claim` | AlgebraVerifier (SymPy) |
| `numerical_result`, `differential_equation` | NumericalVerifier (scipy) |
| `statistical_claim` | StatisticsVerifier (scipy.stats) |
| `proof_step` | ProofChecker (Z3/Lean) |

**Unregistered kind** → `status="unsupported"` (never raises).
**Timeout** → `status="timeout"`.
**Exception in agent** → `status="uncertain"` (never propagates).

**Agent contract (all four):**
- `request_id` preserved verbatim
- `timeout_seconds` respected via `asyncio.wait_for`
- `confidence ∈ [0, 1]`
- Stateless between requests
- No internet access

---

## writer agents

### AcademicWriter
**File:** `writer/academic_writer.py`
**Role:** Four-phase academic paper authoring agent.

**Phases:**
1. **PLAN** — produce `TOCPlan` (sections, figures, initial promises)
2. **DRAFT** — section-by-section in dependency order, with auto-repair loop
3. **AUDIT** — run `GroundingValidator` across all sections
4. **REVIEW** — accept `ReviewerDemand`, revise targeted sections

**Promise system:** Cross-section forward references survive context compression
as `<<promise:prom-xxxx>>` tags. `PromiseRegistry` resolves them at assembly.

**Auto-repair:** Up to `max_repair_rounds` (default 3) on validator failure per section.

**Grounding rule:** Every factual assertion must have a `GroundingClaim` referencing
a real artifact in the `ArtifactStore`. Unsupported claims block.

---

### GroundingValidator
**File:** `writer/validator.py`
**Role:** Artifact grounding gate. No LLM. Four tiers.

**Tiers:**
- **G01–G04** (Tier 1, hard errors): artifact exists, agent consistent, excerpt non-empty, confidence ≥ 0.5
- **G05–G08** (Tier 2, errors+warnings): claim count, density (≥0.5/100 words), ratio (≥80% high-confidence), deduplication
- **P01–P03** (Tier 3, promise integrity): broken promises, pending in completed sections, origin section exists
- **L01–L03** (Tier 4, paper-level warnings): all sections drafted, figures grounded, abstract present

---

### PromiseRegistry
**File:** `writer/promise_registry.py`
**Role:** Tracks all cross-section forward/backward references.

**Lifecycle:** `pending → resolved | broken | waived`
**Tag format:** `<<promise:prom-xxxx>>` — embedded in section text, survives context compression.
**Materialise:** Replaces tags with resolved text, `[PENDING:...]`, or `[BROKEN PROMISE]` at assembly.
**Auto-audit:** `audit_section_completion(sid, text)` breaks any pending promises the section didn't fulfill.

---

## Cross-cutting rules

### Logging
- `algo_log` → stdout (algorithm trace, DEBUG/INFO). Never for research data.
- `data_log` → `.jsonl` file (research data, one JSON object per line). Never for trace.
- Generated scripts must use ONLY these two loggers. No `print()`.

### Memory
- Each agent has `~/.sim_tool/<agent>_memory.md` (AGENTS.md per agent).
- Read at session start, written after session via `consolidate_memory()`.
- Lessons are specific, actionable, falsifiable. Not "be careful with edge cases."

### Contracts
- Every handoff has a typed return type. No string-matching by the director.
- Deterministic validators gate every stage before returning to the director.
- Advisory-only checks are insufficient — blocking errors must block.

### Immutability
- `step_code`, `precompute_code`, `initial_state_code`, `state_fields`, `setup_code`,
  `config_assert_code`, `state_assert_code` are never patched by the analyst.
  Algorithmic changes require `ABORT` verdict and a new design session.

### No hardcoding
- Examples and demos must use the real agent workflow (`tool.start()` → `tool.answer()` → etc.).
- Manually constructing spec objects in demos is forbidden — it bypasses the contract.

---

## Adding a new agent

1. Define its typed output contract in the relevant `contract.py`.
2. Write a deterministic validator for its outputs (mirror `SpecValidator` pattern).
3. Add it to `AGENTS.md` (this file) with: role, allowed/forbidden, contract output.
4. Write a test file covering the validator with ≥20 cases.
5. Update `__init__.py` exports.
6. Do **not** add it to the Director before the validator passes all tests.
