"""
sim_tool.contract
─────────────────
Typed contracts for every stage of the tool's lifecycle.

The coding agent ONLY sees these types — it never needs to inspect strings
or read source code to know what to do next. Each stage has exactly one
return type. Every field that matters to the agent is explicit.

Stage flow
──────────
  tool.start(description)
      → ClarificationRequest   (status="need_clarification")
        OR SpecApproval        (status="need_approval")  if no questions needed

  tool.answer(session_id, answers)
      → ClarificationRequest   (more questions remain)
        OR SpecApproval        (spec is ready)

  tool.approve(session_id)   /  tool.reject(session_id, feedback)
      → GeneratedArtifacts    (after approve + generate)

  tool.run_single(...)  /  tool.run_sweep(...)
      → RunSummary

  tool.analyse_sweep(...)
      → AnalysisResult
          .verdict == Verdict.OK           → nothing to do
          .verdict == Verdict.MINOR_FIX   → .patch is auto-applicable
          .verdict == Verdict.MAJOR_FIX   → .patch needs human review
          .verdict == Verdict.ABORT       → parameter space is wrong, restart

  tool.apply_patch(session_id, patch)
      → GeneratedArtifacts    (patched spec re-generated, ready to re-run)

Question contract
─────────────────
The designer is only allowed to ask questions about these five topics:
  VARIABLE_RANGE    — default values, min/max, sweep points
  STOPPING_THRESHOLD— convergence tolerance, failure cutoffs
  STEP_BUDGET       — max_steps, data_log_interval, checkpoint_interval
  PHYSICAL_UNITS    — units, constants (g, kB, …)
  SWEEP_TARGET      — which variable(s) to sweep, in what range

Questions about implementation (numpy vs stdlib, data structures, code style)
are NOT allowed and will be refused by the validator.

Analyst contract
────────────────
The analyst ONLY returns a SpecPatch. It cannot:
  ✗ Rewrite step_code or precompute_code
  ✗ Add or remove state_fields
  ✗ Change the simulation algorithm
  ✗ Initiate a new design conversation

A SpecPatch contains a list of PatchChange entries, each targeting one
of these patchable fields:
  max_steps, checkpoint_interval, progress_interval, data_log_interval,
  variables[name].default, variables[name].min_val, variables[name].max_val,
  variables[name].sweep_values,
  stopping_conditions[name].check_expr,
  stopping_conditions[name].reason_expr,
  stopping_conditions[name].priority

Output contract (what a run produces on disk)
─────────────────────────────────────────────
  <output_dir>/
    results.json          — RunResult serialised (machine-readable summary)
    data_log.jsonl        — one JSON object per line (research data)
                            load with: pd.read_json("data_log.jsonl", lines=True)
    checkpoint_*.pkl      — full SimState at N-step intervals (internal use)

The coding agent should read data_log.jsonl for research data.
The coding agent should read results.json for run outcome.
Checkpoints are for resumption only — not part of the public contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Question topics — the ONLY things the designer may ask about
# ─────────────────────────────────────────────────────────────────────────────

class QuestionTopic(str, Enum):
    VARIABLE_RANGE     = "variable_range"
    STOPPING_THRESHOLD = "stopping_threshold"
    STEP_BUDGET        = "step_budget"
    PHYSICAL_UNITS     = "physical_units"
    SWEEP_TARGET       = "sweep_target"

ALLOWED_TOPICS = set(QuestionTopic)

TOPIC_DESCRIPTIONS = {
    QuestionTopic.VARIABLE_RANGE: (
        "Ranges, defaults, or valid values for simulation variables. "
        "e.g. 'What range should temperature cover?'"
    ),
    QuestionTopic.STOPPING_THRESHOLD: (
        "Numerical thresholds for success/failure conditions. "
        "e.g. 'What convergence tolerance is acceptable?'"
    ),
    QuestionTopic.STEP_BUDGET: (
        "How long to run: max_steps, data logging frequency, checkpoints. "
        "e.g. 'How many sweeps before giving up?'"
    ),
    QuestionTopic.PHYSICAL_UNITS: (
        "Physical constants or unit systems. "
        "e.g. 'Is temperature in Kelvin or reduced units?'"
    ),
    QuestionTopic.SWEEP_TARGET: (
        "Which variables to sweep and over what values. "
        "e.g. 'Should I sweep both temperature and coupling strength?'"
    ),
}


@dataclass
class ClarificationQuestion:
    index:    int           # 1-based, for easy dict keying {"1": "answer"}
    text:     str           # the question text
    topic:    QuestionTopic # which category this falls into
    required: bool = True   # if False, agent may skip with empty string


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 return: ClarificationRequest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClarificationRequest:
    """
    Returned by tool.start() or tool.answer() when more information is needed.
    The agent MUST call tool.answer(session_id, answers) to continue.

    answers format:
        {"1": "temperature from 1.5 to 4.0", "2": "max_steps=5000"}
        Keys are question indices as strings. Optional questions may be skipped.
    """
    session_id: str
    questions:  list[ClarificationQuestion]
    iteration:  int   # how many clarification rounds have happened so far

    MAX_QUESTIONS_PER_ROUND = 5
    MAX_ROUNDS = 4

    def __str__(self) -> str:
        lines = [
            f"[NEED_CLARIFICATION] session={self.session_id} "
            f"(round {self.iteration}/{self.MAX_ROUNDS})",
        ]
        for q in self.questions:
            req = "" if q.required else " (optional)"
            lines.append(f"  {q.index}. [{q.topic.value}]{req} {q.text}")
        lines.append(
            f'\nAnswer: tool.answer("{self.session_id}", '
            '{"1": "...", "2": "..."})'
        )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 return: SpecApproval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OutputContract:
    """
    Exactly what will appear on disk after a run. Declared before running.
    The agent can rely on these paths and formats unconditionally.
    """
    data_log_fields:   list[str]   # field names guaranteed in every data_log.jsonl row
    results_fields:    list[str]   # keys guaranteed in results.json
    checkpoint_format: str = "pickle — dict(SimState); load with pickle.load(f)"

    def __str__(self) -> str:
        return (
            f"  data_log.jsonl fields : {self.data_log_fields}\n"
            f"  results.json fields   : {self.results_fields}\n"
            f"  checkpoints           : {self.checkpoint_format}"
        )


@dataclass
class SpecApproval:
    """
    Returned when the designer has a complete spec ready for review.
    The agent must call tool.approve() or tool.reject(feedback) to continue.

    The agent should inspect:
        .spec_card        — one-screen human readable summary
        .time_estimate    — expected wall-clock cost
        .output_contract  — exactly what files will be produced

    To modify: tool.reject(session_id, feedback) re-enters clarification.
    To proceed: tool.approve(session_id) → GeneratedArtifacts.
    """
    session_id:      str
    spec_card:       str            # formatted, fits on one terminal screen
    time_estimate:   str            # e.g. "~30s for 7 configurations"
    output_contract: OutputContract
    variable_count:  int
    stop_cond_count: int

    def __str__(self) -> str:
        return (
            f"[NEED_APPROVAL] session={self.session_id}\n"
            f"{self.spec_card}\n\n"
            f"Time estimate    : {self.time_estimate}\n"
            f"Output contract  :\n{self.output_contract}\n\n"
            f"  tool.approve(\"{self.session_id}\")  — proceed\n"
            f"  tool.reject(\"{self.session_id}\", feedback)  — revise"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 return: GeneratedArtifacts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedArtifacts:
    """
    Returned after tool.approve() + tool.generate(), or after tool.apply_patch().
    The agent may inspect .script_path and .notebook_path.
    The script is fully self-contained and executable independently.
    """
    session_id:    str
    script_path:   Path
    notebook_path: Path
    script_size_kb: int
    cli_synopsis:  str    # one-line usage from --help

    def __str__(self) -> str:
        return (
            f"[GENERATED] session={self.session_id}\n"
            f"  Script   : {self.script_path} ({self.script_size_kb} KB)\n"
            f"  Notebook : {self.notebook_path}\n"
            f"  Usage    : {self.cli_synopsis}\n\n"
            f"  tool.run_single(\"{self.session_id}\")  — run with defaults\n"
            f"  tool.run_sweep(\"{self.session_id}\")   — sweep all marked variables"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 return: RunResult / RunSummary
# ─────────────────────────────────────────────────────────────────────────────

class RunOutcome(str, Enum):
    SUCCESS   = "success"    # a success stopping condition fired
    FAILED    = "failed"     # a failure stopping condition fired (bad config)
    MAX_STEPS = "max_steps"  # ran to limit without decisive outcome
    CRASH     = "crash"      # script exited non-zero or was missing
    TIMEOUT   = "timeout"    # killed by runner timeout


@dataclass
class RunResult:
    """
    Outcome of a single simulation run.

    The agent reads research data from .data_file (jsonl), not from this object.
    This object carries only the control-flow outcome.

    data_file is guaranteed to exist if outcome != CRASH and != TIMEOUT.
    """
    config:             dict
    outcome:            RunOutcome
    reason:             str
    stop_condition_name: str
    steps_run:          int
    wall_time_seconds:  float
    output_dir:         Path
    data_file:          Optional[Path]   # <output_dir>/data_log.jsonl

    def __str__(self) -> str:
        icon = "✓" if self.outcome == RunOutcome.SUCCESS else (
               "✗" if self.outcome == RunOutcome.FAILED else
               "⚠" if self.outcome == RunOutcome.MAX_STEPS else "!")
        cfg_str = " ".join(f"{k}={v}" for k, v in self.config.items()
                           if k not in ("max_steps","checkpoint_interval",
                                        "progress_interval","data_log_interval"))
        return (
            f"  {icon} [{self.outcome.value:>10}] {cfg_str}\n"
            f"    {self.reason}  ({self.steps_run:,} steps, {self.wall_time_seconds:.1f}s)"
        )


@dataclass
class RunSummary:
    """
    Returned by tool.run_single() or tool.run_sweep().
    Contains one RunResult per configuration executed.

    The agent should next call tool.analyse_sweep(session_id).
    """
    session_id:   str
    results:      list[RunResult]
    sweep_dir:    Path

    @property
    def n_success(self)   -> int: return sum(1 for r in self.results if r.outcome == RunOutcome.SUCCESS)
    @property
    def n_failed(self)    -> int: return sum(1 for r in self.results if r.outcome == RunOutcome.FAILED)
    @property
    def n_max_steps(self) -> int: return sum(1 for r in self.results if r.outcome == RunOutcome.MAX_STEPS)
    @property
    def n_crash(self)     -> int: return sum(1 for r in self.results if r.outcome == RunOutcome.CRASH)

    @property
    def data_files(self) -> list[Path]:
        """All data_log.jsonl paths from successful/completed runs."""
        return [r.data_file for r in self.results if r.data_file and r.data_file.exists()]

    def __str__(self) -> str:
        lines = [
            f"[RUN_SUMMARY] session={self.session_id}  "
            f"{len(self.results)} run(s)",
            f"  ✓ success={self.n_success}  "
            f"✗ failed={self.n_failed}  "
            f"⚠ max_steps={self.n_max_steps}  "
            f"! crash={self.n_crash}",
        ]
        for r in self.results:
            lines.append(str(r))
        if self.data_files:
            lines.append(f"\nData files ({len(self.data_files)}):")
            for p in self.data_files:
                lines.append(f"  {p}")
        lines.append(f'\ntool.analyse_sweep("{self.session_id}") — diagnose and get patch')
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 return: AnalysisResult + SpecPatch
# ─────────────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    OK          = "ok"           # all runs completed, no changes needed
    MINOR_FIX   = "minor_fix"   # numerical tweak, auto-applicable
    MAJOR_FIX   = "major_fix"   # needs human review before applying
    ABORT       = "abort"        # parameter space is fundamentally wrong; restart design


# Fields the analyst IS allowed to patch (exhaustive list)
PATCHABLE_FIELDS = frozenset({
    "max_steps",
    "checkpoint_interval",
    "progress_interval",
    "data_log_interval",
    # variable sub-fields, keyed as "variables.NAME.FIELD"
    # e.g. "variables.temperature.default", "variables.temperature.min_val"
    # stopping condition sub-fields: "stopping_conditions.NAME.check_expr"
    # "stopping_conditions.NAME.priority"
})

# Fields the analyst is NEVER allowed to touch
IMMUTABLE_FIELDS = frozenset({
    "step_code",
    "precompute_code",
    "initial_state_code",
    "progress_code",
    "state_fields",
    "setup_code",
    "config_assert_code",
    "state_assert_code",
})


@dataclass
class PatchChange:
    """
    One atomic change to the spec.
    field follows dot-notation: "max_steps", "variables.N.default",
    "stopping_conditions.lattice_frozen.check_expr"
    """
    field:     str
    old_value: object
    new_value: object
    why:       str    # plain English, shown to agent before applying

    def __post_init__(self) -> None:
        # Validate field is patchable
        top = self.field.split(".")[0]
        if top in IMMUTABLE_FIELDS:
            raise ValueError(
                f"Analyst tried to patch immutable field {self.field!r}. "
                f"Immutable fields are: {sorted(IMMUTABLE_FIELDS)}. "
                f"Only patchable fields are allowed: {sorted(PATCHABLE_FIELDS)}."
            )

    def __str__(self) -> str:
        return f"  {self.field}: {self.old_value!r} → {self.new_value!r}\n    why: {self.why}"



@dataclass
class Flag:
    """A typed diagnostic signal from Layer 2 (rule-based classification)."""
    kind:        str
    severity:    str          # "error" | "warning" | "info"
    message:     str
    config_hint: Optional[str] = None
    affects:     list[str]    = field(default_factory=list)


@dataclass
class SpecPatch:
    """
    The ONLY output of the analyst when changes are needed.

    A patch is a typed diff — not a redesign. It targets specific numerical
    fields in the spec. The agent applies it with tool.apply_patch(session_id, patch).

    If verdict == MINOR_FIX: safe to apply automatically (auto_applicable=True on all changes).
    If verdict == MAJOR_FIX: show changes to human before applying.
    If verdict == ABORT:     patch is None; restart with tool.start() with new description.
    """
    verdict:  Verdict
    reason:   str             # overall explanation of why changes are needed
    changes:  list[PatchChange] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"[SPEC_PATCH] verdict={self.verdict.value}",
            f"  {self.reason}",
        ]
        if self.changes:
            lines.append(f"  Changes ({len(self.changes)}):")
            for c in self.changes:
                lines.append(str(c))
        if self.verdict == Verdict.ABORT:
            lines.append(
                "\n  ⚠ ABORT: The parameter space is fundamentally incompatible "
                "with the stopping conditions. Restart with a revised description."
            )
        return "\n".join(lines)


@dataclass
class AnalysisResult:
    """
    Returned by tool.analyse_sweep(session_id).

    The agent inspects .verdict to decide what to do:
      Verdict.OK         → done, read .data_files for research output
      Verdict.MINOR_FIX  → call tool.apply_patch(session_id, .patch) and re-run
      Verdict.MAJOR_FIX  → review .patch.changes, then apply_patch if acceptable
      Verdict.ABORT      → call tool.start() with a revised description

    .flags contains the deterministic diagnostic signals (Layer 2).
    .patch contains the typed diff (Layer 3, may be None if verdict==OK).
    .llm_reasoning is the LLM's step-by-step thinking (for audit/trust).
    """
    session_id:    str
    verdict:       Verdict
    flags:         list       # list[Flag] from analyst.py Layer 2
    patch:         Optional[SpecPatch]
    llm_reasoning: str = ""   # LLM chain-of-thought, empty if no LLM was called
    data_files:    list[Path] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"[ANALYSIS] session={self.session_id}  verdict={self.verdict.value.upper()}",
        ]
        if self.flags:
            lines.append(f"  Flags ({len(self.flags)}):")
            for f in self.flags:
                icon = {"error":"✗","warning":"⚠","info":"i"}[f.severity]
                lines.append(f"    [{icon}] {f.kind}: {f.message}")
        if self.patch:
            lines.append("")
            lines.append(str(self.patch))
        if self.data_files:
            lines.append(f"\n  Data files for research use:")
            for p in self.data_files:
                lines.append(f"    {p}  (load: pd.read_json(p, lines=True))")
        # Next action hint
        lines.append("\n  Next step:")
        if self.verdict == Verdict.OK:
            lines.append("    ✓ No changes needed. Use .data_files for analysis.")
        elif self.verdict == Verdict.MINOR_FIX:
            lines.append(f'    tool.apply_patch("{self.session_id}", result.patch) → re-run')
        elif self.verdict == Verdict.MAJOR_FIX:
            lines.append("    Review .patch.changes, then apply_patch if acceptable.")
        else:  # ABORT
            lines.append("    tool.start(revised_description) — restart design.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Launcher contract types
# ─────────────────────────────────────────────────────────────────────────────

class BugKind(str, Enum):
    """
    Exhaustive taxonomy of launch failures. Determines default retryability
    and the suggested fix template.
    """
    ENV_SETUP       = "env_setup"        # missing package, bad Python version
    IMPORT_ERROR    = "import_error"     # module present but can't be imported
    SYNTAX_ERROR    = "syntax_error"     # should have been caught by validator
    ASSERTION_ERROR = "assertion_error"  # Tier 1/2 assert fired at runtime
    RUNTIME_ERROR   = "runtime_error"   # general exception inside sim loop
    TIMEOUT         = "timeout"          # killed by launcher timeout
    OOM             = "oom"              # out of memory / killed by OS
    DISK_FULL       = "disk_full"        # ENOSPC on checkpoint/log write
    SIGNAL          = "signal"           # SIGKILL, SIGSEGV, etc.
    UNKNOWN         = "unknown"          # exit non-zero, no pattern matched


# Fields that every BugTicket must have (validated deterministically)
_BUG_TICKET_REQUIRED = frozenset({
    "kind", "session_id", "script_path", "exit_code",
    "reproducible_command", "stdout_tail", "stderr_tail",
})


@dataclass
class BugTicket:
    """
    Returned by the Launcher when a run fails.
    Passed to the Coder (or Director) instead of a RunSummary.

    Contract guarantees (checked by BugTicketValidator before return):
      - All required fields are non-empty / non-None
      - kind is a known BugKind value
      - reproducible_command is a non-empty string that starts with a valid
        executable name (uv, python, python3, bash, sh)
      - stdout_tail and stderr_tail are strings (may be empty if process
        produced no output, but the field must exist)
      - is_retryable is explicitly set (not defaulted silently)

    is_retryable:
      True  — ENV_SETUP, TIMEOUT, OOM, DISK_FULL (fix env and retry)
      False — SYNTAX_ERROR, ASSERTION_ERROR, RUNTIME_ERROR, SIGNAL, UNKNOWN
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    kind:                  BugKind
    session_id:            str
    script_path:           str          # absolute path to the failing script
    pipeline_script_path:  Optional[str] = None   # .sh that reproduces the run

    # ── Execution context ─────────────────────────────────────────────────────
    exit_code:             int          = -1
    wall_time_seconds:     float        = 0.0
    python_executable:     str          = ""   # which python was used
    uv_version:            str          = ""   # "" if uv was not used
    env_vars_set:          dict         = field(default_factory=dict)

    # ── Failure evidence ──────────────────────────────────────────────────────
    stdout_tail:           str          = ""   # last ≤50 lines of stdout
    stderr_tail:           str          = ""   # last ≤50 lines of stderr
    failure_pattern:       str          = ""   # which regex matched
    error_line:            str          = ""   # the specific line that triggered

    # ── Actionable output ─────────────────────────────────────────────────────
    reproducible_command:  str          = ""   # exact shell command to reproduce
    suggested_fix:         str          = ""   # deterministic, no LLM
    is_retryable:          bool         = False

    def __str__(self) -> str:
        lines = [
            f"[BUG_TICKET]  kind={self.kind.value}  exit={self.exit_code}  "
            f"retryable={self.is_retryable}",
            f"  script   : {self.script_path}",
            f"  command  : {self.reproducible_command}",
        ]
        if self.error_line:
            lines.append(f"  error    : {self.error_line}")
        if self.suggested_fix:
            lines.append(f"  fix      : {self.suggested_fix}")
        if self.stderr_tail.strip():
            lines.append("  stderr (tail):")
            for line in self.stderr_tail.strip().splitlines()[-8:]:
                lines.append(f"    {line}")
        if self.pipeline_script_path:
            lines.append(f"  pipeline : {self.pipeline_script_path}")
        return "\n".join(lines)


@dataclass
class EnvironmentInfo:
    """Records the exact environment that was used to run a simulation."""
    python_executable:  str
    python_version:     str
    uv_version:         str          # "" if not used
    packages_installed: list[str]    # ["numpy==2.1.0", ...]
    env_vars:           dict         # env vars that were explicitly set
    venv_path:          str          # "" if not using a venv
    ran_as_uv:          bool         # True if launched via `uv run`
    execution_backend:  str = "process"
    container_image:    str = ""

    def __str__(self) -> str:
        runner = f"uv {self.uv_version}" if self.ran_as_uv else self.python_executable
        lines = [
            f"  backend  : {self.execution_backend}",
            f"  runner   : {runner}",
            f"  python   : {self.python_version}",
        ]
        if self.container_image:
            lines.append(f"  image    : {self.container_image}")
        if self.packages_installed:
            lines.append(f"  packages : {', '.join(self.packages_installed[:6])}"
                         + (" …" if len(self.packages_installed) > 6 else ""))
        if self.env_vars:
            for k, v in self.env_vars.items():
                lines.append(f"  env      : {k}={v}")
        return "\n".join(lines)


@dataclass
class LaunchResult:
    """
    Returned by the Launcher on successful completion.
    Wraps RunSummary with additional environment and pipeline metadata.

    The director treats this as equivalent to RunSummary for analysis purposes.
    """
    run_summary:          "RunSummary"
    env_info:             EnvironmentInfo
    pipeline_script_path: Optional[str]   # path to the generated .sh
    launcher_log_entry:   str             # the markdown entry that was written

    @property
    def session_id(self) -> str:
        return self.run_summary.session_id

    @property
    def data_files(self) -> list:
        return self.run_summary.data_files

    def __str__(self) -> str:
        lines = [
            f"[LAUNCH_RESULT]  session={self.session_id}",
            str(self.env_info),
            str(self.run_summary),
        ]
        if self.pipeline_script_path:
            lines.append(f"  pipeline : {self.pipeline_script_path}")
        return "\n".join(lines)
