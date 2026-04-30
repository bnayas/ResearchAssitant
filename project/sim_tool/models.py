"""
sim_tool.models
───────────────
All data structures for simulation specification, designer state, and results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Variable
# ─────────────────────────────────────────────────────────────────────────────

class VariableKind(str, Enum):
    FLOAT  = "float"
    INT    = "int"
    BOOL   = "bool"
    CHOICE = "choice"
    STRING = "string"


@dataclass
class Variable:
    name:         str
    description:  str
    kind:         VariableKind
    default:      Any
    min_val:      Optional[float] = None
    max_val:      Optional[float] = None
    step:         Optional[float] = None
    choices:      Optional[list]  = None
    unit:         Optional[str]   = None
    sweep:        bool            = False
    sweep_values: Optional[list]  = None

    def range_str(self) -> str:
        if self.choices:
            return f"choices={self.choices}"
        parts = []
        if self.min_val is not None:
            parts.append(f"min={self.min_val}")
        if self.max_val is not None:
            parts.append(f"max={self.max_val}")
        if self.step is not None:
            parts.append(f"step={self.step}")
        return ", ".join(parts) if parts else "unbounded"

    def unit_str(self) -> str:
        return f" [{self.unit}]" if self.unit else ""


@dataclass
class StoppingCondition:
    """
    check_expr  : Python bool expression; `config`, `state`, `precomputed` in scope.
    reason_expr : Python str expression; same scope.
    """
    kind:            str    # "success" | "failure"
    name:            str
    description:     str
    check_expr:      str
    reason_expr:     str
    save_on_trigger: bool = True
    priority:        int  = 0


@dataclass
class SimulationSpec:
    """
    Complete, self-contained description of a simulation.

    Logging model (two separate loggers in generated code)
    ───────────────────────────────────────────────────────
    algo_log  — software-level tracing (stdout, DEBUG/INFO):
                 step decisions, precompute results, convergence checks,
                 anything useful for debugging the algorithm itself.

    data_log  — research-level events (append-only .jsonl file):
                 measurements, per-step statistics, stopping events.
                 Designed to be read by pandas/numpy for analysis.
                 Written at data_log_interval frequency.

    Assertion model (three tiers)
    ──────────────────────────────
    Tier 1  config_assert_code   — run once; validates all config inputs
                                   (ranges, mutual constraints, physical sense)
    Tier 2  state_assert_code    — run every step; lightweight state invariants
                                   (no NaN, positive masses, prob sums to 1, …)
    Tier 3  stopping conditions  — checked before every step; detect unrecoverable
                                   states (stalled, underground, diverged)

    Precompute model
    ─────────────────
    precompute_code runs once before the loop and returns a plain dict.
    The dict is passed as `precomputed` to step_code, stopping conditions,
    and progress_code. Use it for:
      - Transition probability matrices (Markov chains)
      - Neighbour adjacency lists (lattice sims)
      - Lookup tables for expensive functions
      - Precomputed constants that depend on config
    """
    name:                      str
    description:               str
    variables:                 list[Variable]
    stopping_conditions:       list[StoppingCondition]
    state_fields:              list[tuple[str, str, Any]]   # (name, type_hint, default)

    setup_code:                str   # top-level imports + module-level helpers
    precompute_code:           str   # body of precompute(config) -> dict
    initial_state_code:        str   # body of initial_state(config, precomputed) -> SimState
    step_code:                 str   # body of sim_step(config, state, precomputed) -> SimState
    progress_code:             str   # body of progress_summary(state, precomputed) -> str

    config_assert_code:        str   # statements; assert with descriptive messages
    state_assert_code:         str   # statements; assert with descriptive messages

    output_variables:          list[str]   # fields recorded in in-memory history list
    data_log_variables:        list[str]   # fields written to .jsonl research log
    data_log_interval:         int  = 1

    checkpoint_interval:       int   = 100
    max_steps:                 int   = 10_000
    progress_interval:         int   = 500
    time_estimate_seconds:     float = 0.0
    time_estimate_explanation: str   = ""

    def format_variable_docs(self) -> str:
        lines = []
        for v in self.variables:
            sweep_note = " [SWEPT]" if v.sweep else ""
            lines.append(
                f"  {v.name:<28} {v.kind.value:<8} "
                f"default={v.default!r:<12} {v.range_str()}{v.unit_str()}{sweep_note}"
            )
            lines.append(f"    └─ {v.description}")
        return "\n".join(lines)

    def format_stop_docs(self) -> str:
        lines = []
        for s in sorted(self.stopping_conditions, key=lambda s: -s.priority):
            lines.append(f"  [{s.kind.upper()}] {s.name}")
            lines.append(f"    └─ {s.description}")
        return "\n".join(lines)


@dataclass
class DesignerState:
    original_description: str
    session_id:           str            = field(default_factory=lambda: uuid.uuid4().hex[:8])
    conversation:         list[dict]     = field(default_factory=list)
    cumulative_answers:   dict[str, str] = field(default_factory=dict)
    spec:                 Optional[SimulationSpec] = None
    iteration:            int            = 0


@dataclass
class ToolResponse:
    status:            str
    session_id:        str                   = ""
    message:           str                   = ""
    questions:         list[str]             = field(default_factory=list)
    spec:              Optional[SimulationSpec] = None
    time_estimate_str: str                   = ""
    script_path:       Optional[str]         = None
    notebook_path:     Optional[str]         = None
    results:           Optional[dict]        = None

    def __str__(self) -> str:
        lines = [f"[{self.status.upper()}]"]
        if self.session_id:
            lines[0] += f"  session={self.session_id}"
        if self.message:
            lines.append(self.message)
        if self.questions:
            lines.append("Questions:")
            for i, q in enumerate(self.questions, 1):
                lines.append(f"  {i}. {q}")
        if self.time_estimate_str:
            lines.append(f"Time estimate: {self.time_estimate_str}")
        if self.script_path:
            lines.append(f"Script:   {self.script_path}")
        if self.notebook_path:
            lines.append(f"Notebook: {self.notebook_path}")
        return "\n".join(lines)
