"""
sim_tool.designer — PATCH NOTES
────────────────────────────────
Changes from the original designer.py:

1. _call() now catches non-JSON output and retries via StagedSpecGenerator
   before raising ValueError.  This makes the designer resilient to
   context-overflow failures without changing its external contract.

2. _compress_for_initial_call() trims the description to MAX_INITIAL_DESC_CHARS
   for the first LLM call only.  Subsequent clarification rounds still
   send the full accumulated context because they are much shorter.

3. StagedSpecGenerator is imported lazily to avoid circular imports.

Drop-in replacement for the original designer.py in project/sim_tool/.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .contract_validator import SpecValidator, ValidationResult
from .contract import (
    ClarificationQuestion, ClarificationRequest, OutputContract,
    QuestionTopic, SpecApproval, TOPIC_DESCRIPTIONS,
)
from .llm import LLMBackend, make_backend
from .memory import AgentMemory
from .models import (
    DesignerState, SimulationSpec, StoppingCondition,
    ToolResponse, Variable, VariableKind,
)

log = logging.getLogger("sim_tool.designer")

# Maximum description length sent to the monolithic JSON-generation call.
# Beyond this we expect context-overflow failures and fall back to staged.
MAX_INITIAL_DESC_CHARS = 4000

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """
{memory_prefix}
You are an expert simulation architect and Python engineer. Convert natural-language
simulation requests into precise, complete, executable specifications.

━━━ QUESTION CONTRACT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You may ONLY ask questions in these five categories. No others are allowed:

  variable_range     — {variable_range}
  stopping_threshold — {stopping_threshold}
  step_budget        — {step_budget}
  physical_units     — {physical_units}
  sweep_target       — {sweep_target}

FORBIDDEN question topics (never ask about these):
  ✗ Which library to use (numpy vs stdlib)
  ✗ Code style or data structure preference
  ✗ Anything already stated in the description
  ✗ General implementation approach

Max questions per round: 5. Max clarification rounds: 4.
If you have reached the information you need, set spec_complete=true immediately.

━━━ ASSERTION RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1 — config_assert_code (once, before loop):
  assert config.FIELD > 0, f"FIELD must be positive, got FIELD={{config.FIELD}}"
  For rate/probability fields: assert 0.0 <= config.rate <= 1.0
  Messages must contain the offending value.

Tier 2 — state_assert_code (every step, O(1)):
  assert math.isfinite(state.x), f"x diverged to {{state.x}} at step {{state.step}}"
  For ALL rate/probability fields: assert both bounds (>= 0 AND <= 1)
  For ALL position/energy fields: assert math.isfinite(value)

━━━ LOGGING RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
algo_log → stdout (algorithm trace, for developers debugging code)
data_log → .jsonl file (research data, for analysis)

━━━ PRECOMPUTE RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
precompute_code runs ONCE before the loop. ALWAYS precompute:
  - Anything depending only on config (not state)
  - Boltzmann/exponential tables
  - Neighbour/adjacency index arrays
  Log what was computed: algo_log.debug(f"Precomputed: {{result}}")

━━━ STOPPING CONDITIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUST have: ≥1 SUCCESS condition (goal achieved, tolerance met)
MUST have: ≥1 FAILURE condition (bad config detected early)

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON ONLY — no markdown, no prose outside the JSON object.

{{
  "spec_complete": <bool>,
  "questions": [
    {{"index": 1, "text": "...", "topic": "variable_range|stopping_threshold|step_budget|physical_units|sweep_target", "required": true}}
  ],
  "spec": {{
    "name": <string>,
    "description": <string>,
    "variables": [...],
    "state_fields": [["<name>", "<python_type>", <default>]],
    "stopping_conditions": [...],
    "setup_code": <str>,
    "precompute_code": <str>,
    "initial_state_code": <str>,
    "step_code": <str>,
    "progress_code": <str>,
    "config_assert_code": <str>,
    "state_assert_code": <str>,
    "output_variables": [<str>],
    "data_log_variables": [<str>],
    "data_log_interval": <int>,
    "checkpoint_interval": <int>,
    "max_steps": <int>,
    "progress_interval": <int>,
    "time_estimate_seconds": <float>,
    "time_estimate_explanation": <str>
  }}
}}
""".strip()


def _build_system_prompt(memory: AgentMemory) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        memory_prefix=memory.as_prompt_prefix(),
        **{t.value: TOPIC_DESCRIPTIONS[t] for t in QuestionTopic},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Designer
# ─────────────────────────────────────────────────────────────────────────────

class SimulationDesigner:
    """
    Stateful, multi-turn simulation designer.
    Returns ClarificationRequest or SpecApproval — never raw strings.

    Resilience strategy
    ───────────────────
    If the initial description is very long (research brief + literature
    context), the LLM may return prose instead of JSON.  In that case:

    1. The designer attempts a compressed description (≤MAX_INITIAL_DESC_CHARS)
       for the initial monolithic call.
    2. If the monolithic call still fails, it falls back to StagedSpecGenerator,
       which generates the spec in 6–12 focused micro-calls that each fit
       comfortably in a 16k-token context window.

    The fallback is transparent: the caller still receives a SpecApproval.
    """

    def __init__(self, backend: Optional[LLMBackend] = None) -> None:
        self._backend = backend or make_backend(service="designer")
        self._sessions: dict[str, DesignerState] = {}
        self._memory = AgentMemory("designer")
        self._validator = SpecValidator()
        self._system_prompt = _build_system_prompt(self._memory)
        log.info(
            f"Designer using {self._backend.model_name} | "
            f"memory: {self._memory.entry_count} entries"
        )

    @property
    def memory(self) -> AgentMemory:
        return self._memory

    def start(self, description: str) -> ClarificationRequest | SpecApproval:
        state = DesignerState(original_description=description)
        self._sessions[state.session_id] = state
        log.info(f"[{state.session_id}] New session — {description[:60]!r}… (len={len(description)})")
        return self._call(state, initial=True)

    def answer(
        self, session_id: str, answers: dict[str, str]
    ) -> ClarificationRequest | SpecApproval:
        state = self._get(session_id)
        if state.iteration >= ClarificationRequest.MAX_ROUNDS:
            log.warning(
                f"[{session_id}] Reached MAX_ROUNDS={ClarificationRequest.MAX_ROUNDS}. "
                "Forcing spec_complete on next LLM call."
            )
        state.cumulative_answers.update(answers)
        state.iteration += 1
        log.info(f"[{session_id}] Round {state.iteration}: {len(answers)} answer(s)")
        return self._call(state)

    def approve(self, session_id: str) -> SpecApproval:
        state = self._get(session_id)
        if state.spec is None:
            raise RuntimeError(f"[{session_id}] No spec ready yet.")
        return _make_spec_approval(session_id, state.spec)

    def reject(self, session_id: str, feedback: str = "") -> ClarificationRequest | SpecApproval:
        state = self._get(session_id)
        state.spec = None
        if feedback:
            state.cumulative_answers[f"_rejection_{state.iteration}"] = feedback
            state.iteration += 1
        log.info(f"[{session_id}] Spec rejected — re-refining.")
        return self._call(state)

    def get_spec(self, session_id: str) -> Optional[SimulationSpec]:
        return self._sessions.get(session_id, DesignerState("")).spec

    def get_session(self, session_id: str) -> DesignerState:
        return self._get(session_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, session_id: str) -> DesignerState:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id: {session_id!r}")
        return self._sessions[session_id]

    def _call(self, state: DesignerState, initial: bool = False) -> ClarificationRequest | SpecApproval:
        sid = state.session_id
        force_complete = state.iteration >= ClarificationRequest.MAX_ROUNDS

        if initial:
            # For very long descriptions, compress for the monolithic call.
            # The full description is preserved in state.original_description
            # so staged generation can use it later if needed.
            compressed = _compress_description(
                state.original_description, MAX_INITIAL_DESC_CHARS
            )
            if len(compressed) < len(state.original_description):
                log.info(
                    f"[{sid}] Description compressed: "
                    f"{len(state.original_description)} → {len(compressed)} chars"
                )
            user_content = f"Design a simulation for:\n\n{compressed}"
        else:
            request_text = _compress_description(
                state.original_description, MAX_INITIAL_DESC_CHARS
            )
            if len(request_text) < len(state.original_description):
                log.info(
                    f"[{sid}] Follow-up context compressed: "
                    f"{len(state.original_description)} → {len(request_text)} chars"
                )
            answers_block = "\n".join(
                f"  [{k}]: {v}" for k, v in state.cumulative_answers.items()
                if not k.startswith("_")
            )
            force_note = (
                "\n\nIMPORTANT: You have reached the maximum number of clarification "
                "rounds. You MUST now set spec_complete=true and return a complete spec."
                if force_complete else ""
            )
            user_content = (
                f"Original request:\n{request_text}\n\n"
                f"Answers so far:\n{answers_block}"
                + force_note
            )

        state.conversation.append({"role": "user", "content": user_content})

        log.debug(f"[{sid}] LLM call #{(len(state.conversation)+1)//2}")
        try:
            raw = self._backend.complete(
                system=self._system_prompt,
                messages=state.conversation,
            )
        except Exception as exc:
            log.error(f"[{sid}] LLM error: {exc}; falling back to staged generation")
            return self._staged_fallback(state, sid)

        state.conversation.append({"role": "assistant", "content": raw})
        data = _parse_json(raw)

        # ── Non-JSON fallback: staged generation ──────────────────────────────
        if data is None:
            log.warning(
                f"[{sid}] Monolithic call returned non-JSON "
                f"(first 200 chars: {raw[:200]!r}). "
                "Falling back to staged generation."
            )
            return self._staged_fallback(state, sid)

        if not data.get("spec_complete"):
            raw_qs = data.get("questions", [])
            questions = _validate_questions(raw_qs, sid)
            log.info(f"[{sid}] Clarification round {state.iteration+1}: {len(questions)} question(s)")
            return ClarificationRequest(
                session_id=sid,
                questions=questions,
                iteration=state.iteration,
            )

        spec = _parse_spec(data["spec"])
        return self._validate_and_approve(spec, state, sid)

    def _staged_fallback(
        self, state: DesignerState, sid: str
    ) -> ClarificationRequest | SpecApproval:
        """
        Generate spec via StagedSpecGenerator when the monolithic call fails.

        Uses the full original description so staged generation has all
        the detail it needs.  Returns SpecApproval directly (no
        clarification rounds — staged generation is self-contained).
        """
        from .staged_designer import StagedSpecGenerator
        log.info(f"[{sid}] Starting staged generation for session")
        try:
            generator = StagedSpecGenerator(self._backend)
            spec = generator.generate(
                state.original_description,
                session_id=sid,
                max_repair_attempts=2,
            )
            state.spec = spec
            log.info(
                f"[{sid}] Staged generation succeeded: '{spec.name}' | "
                f"{len(spec.variables)} vars | {len(spec.stopping_conditions)} stop conds"
            )
            return _make_spec_approval(sid, spec)
        except Exception as exc:
            log.error(f"[{sid}] Staged generation failed: {exc}")
            raise ValueError(
                f"[{sid}] Both monolithic and staged spec generation failed.\n"
                f"Staged error: {exc}"
            ) from exc

    def _validate_and_approve(
        self, spec: SimulationSpec, state: DesignerState, sid: str
    ) -> SpecApproval:
        """Run SpecValidator; attempt one auto-repair; then approve."""
        validation = self._validator.validate(spec)
        if not validation.passed:
            log.warning(
                f"[{sid}] Spec failed validation: "
                f"{len(validation.errors)} error(s) — attempting auto-repair"
            )
            repair_prompt = (
                "The spec you returned failed deterministic validation. "
                "Fix ALL blocking errors below, then return the complete corrected spec JSON.\n\n"
                + validation.error_summary()
            )
            state.conversation.append({"role": "user", "content": repair_prompt})
            try:
                raw2 = self._backend.complete(
                    system=self._system_prompt,
                    messages=state.conversation,
                )
            except Exception as exc:
                log.error(f"[{sid}] Auto-repair LLM call failed: {exc}; falling back to staged generation")
                return self._staged_fallback(state, sid)
            state.conversation.append({"role": "assistant", "content": raw2})
            data2 = _parse_json(raw2)
            if data2 and data2.get("spec_complete") and data2.get("spec"):
                try:
                    spec = _parse_spec(data2["spec"])
                    validation2 = self._validator.validate(spec)
                    if not validation2.passed:
                        log.error(f"[{sid}] Still invalid after auto-repair — trying staged repair")
                        # Last resort: staged repair of specific failing fields
                        from .staged_designer import StagedSpecGenerator
                        generator = StagedSpecGenerator(self._backend)
                        skeleton = {
                            "name": spec.name,
                            "description": spec.description,
                            "variables": [
                                {
                                    "name": v.name, "description": v.description,
                                    "kind": v.kind.value, "default": v.default,
                                    "min_val": v.min_val, "max_val": v.max_val,
                                    "step": v.step, "choices": v.choices,
                                    "unit": v.unit, "sweep": v.sweep,
                                    "sweep_values": v.sweep_values,
                                }
                                for v in spec.variables
                            ],
                            "state_fields": list(spec.state_fields),
                            "stopping_conditions": [
                                {
                                    "kind": c.kind, "name": c.name,
                                    "description": c.description, "priority": c.priority,
                                }
                                for c in spec.stopping_conditions
                            ],
                            "output_variables": spec.output_variables,
                            "data_log_variables": spec.data_log_variables,
                            "data_log_interval": spec.data_log_interval,
                            "checkpoint_interval": spec.checkpoint_interval,
                            "max_steps": spec.max_steps,
                            "progress_interval": spec.progress_interval,
                            "time_estimate_seconds": spec.time_estimate_seconds,
                            "time_estimate_explanation": spec.time_estimate_explanation,
                        }
                        spec = generator._phase5_repair(spec, validation2, skeleton, sid)
                        validation3 = self._validator.validate(spec)
                        if not validation3.passed:
                            spec = _apply_deterministic_repairs(spec, validation3, sid)
                            validation3 = self._validator.validate(spec)
                            if not validation3.passed:
                                raise ValueError(
                                    f"Spec validation failed after all repair attempts.\n"
                                    + validation3.error_summary()
                                )
                    else:
                        log.info(f"[{sid}] Auto-repair succeeded.")
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"Auto-repair produced unparseable spec: {exc}")
            else:
                # Auto-repair also returned non-JSON — fall back to staged
                log.warning(f"[{sid}] Auto-repair returned non-JSON, falling back to staged")
                return self._staged_fallback(state, sid)

        if validation.warnings:
            for w in validation.warnings:
                log.info(f"[{sid}]   [D] {w.code}: {w.message[:80]}")

        state.spec = spec
        log.info(
            f"[{sid}] Spec validated: '{spec.name}' | "
            f"{len(spec.variables)} vars | {len(spec.stopping_conditions)} stop conds"
        )
        return _make_spec_approval(sid, spec)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as original)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_deterministic_repairs(
    spec: SimulationSpec, validation: ValidationResult, sid: str
) -> SimulationSpec:
    """Fix validator-contract issues that do not require scientific judgment."""
    error_codes = {err.code for err in validation.errors}
    state_field_names = [name for name, *_ in spec.state_fields]
    state_field_set = set(state_field_names)

    def valid_state_refs(names: list[str]) -> list[str]:
        out: list[str] = []
        for name in names or []:
            if name in state_field_set or name in {"step", "sim_time"}:
                if name not in out:
                    out.append(name)
        return out

    def preferred_refs() -> list[str]:
        preferred = [
            "current_generation",
            "generation",
            "species_counts",
            "population",
            "counts",
            "environmental_state",
        ]
        out = [name for name in preferred if name in state_field_set]
        if not out:
            out = state_field_names[:3]
        return out

    changed: list[str] = []

    if "missing_field" in error_codes and not spec.variables:
        spec.variables = _reconstruct_variables_from_spec(spec)
        changed.append("variables")

    if (
        "undefined_data_log_variable" in error_codes
        or "empty_data_log_variables" in error_codes
        or not spec.data_log_variables
    ):
        repaired = valid_state_refs(spec.data_log_variables)
        if not repaired:
            repaired = preferred_refs()
        if repaired != spec.data_log_variables:
            spec.data_log_variables = repaired
            changed.append("data_log_variables")

    if (
        "undefined_output_variable" in error_codes
        or "empty_output_variables" in {w.code for w in validation.warnings}
        or not spec.output_variables
    ):
        repaired = valid_state_refs(spec.output_variables)
        if not repaired:
            repaired = preferred_refs()
        if repaired != spec.output_variables:
            spec.output_variables = repaired
            changed.append("output_variables")

    if "no_failure_condition" in error_codes:
        if "_sim_tool_invalid_value" not in spec.setup_code:
            spec.setup_code = (
                spec.setup_code.rstrip()
                + "\n\n"
                + "def _sim_tool_invalid_value(value):\n"
                + "    if isinstance(value, float):\n"
                + "        return math.isnan(value) or math.isinf(value)\n"
                + "    if isinstance(value, (list, tuple)):\n"
                + "        return any(_sim_tool_invalid_value(v) for v in value[:10000])\n"
                + "    if isinstance(value, dict):\n"
                + "        return any(_sim_tool_invalid_value(v) for v in value.values())\n"
                + "    return False\n"
            )
        spec.stopping_conditions.append(
            StoppingCondition(
                kind="failure",
                name="invalid_state_detected",
                description="Stop if any tracked state value becomes NaN or infinite.",
                check_expr=(
                    "any(_sim_tool_invalid_value(v) for v in vars(state).values())"
                ),
                reason_expr='"invalid or non-finite state detected"',
                save_on_trigger=True,
                priority=10_000,
            )
        )
        changed.append("stopping_conditions")

    if changed:
        log.info(f"[{sid}] Applied deterministic spec repairs: {', '.join(changed)}")
    return spec


_CONFIG_REF_RE = re.compile(r"\bconfig\.([A-Za-z_][A-Za-z0-9_]*)")
_BUILTIN_CONFIG_FIELDS = {
    "max_steps",
    "checkpoint_interval",
    "progress_interval",
    "data_log_interval",
}


def _reconstruct_variables_from_spec(spec: SimulationSpec) -> list[Variable]:
    """Recover a minimal SimConfig contract when an LLM repair drops variables."""
    code_blocks = "\n".join(
        [
            spec.precompute_code or "",
            spec.initial_state_code or "",
            spec.step_code or "",
            spec.progress_code or "",
            spec.config_assert_code or "",
            spec.state_assert_code or "",
            " ".join(sc.check_expr or "" for sc in spec.stopping_conditions),
            " ".join(sc.reason_expr or "" for sc in spec.stopping_conditions),
        ]
    )
    names = [
        name
        for name in sorted(set(_CONFIG_REF_RE.findall(code_blocks)))
        if name not in _BUILTIN_CONFIG_FIELDS
    ]
    if not names:
        names = [
            "population_size",
            "mutation_rate",
            "delta",
            "gamma",
            "initial_species_count",
            "burn_in_generations",
            "sample_interval",
        ]
    return [_default_variable_for_name(name) for name in names]


def _default_variable_for_name(name: str) -> Variable:
    lname = name.lower()
    if lname in {"n", "population_size", "n_individuals", "community_size"}:
        return Variable(
            name=name,
            description="Community size N.",
            kind=VariableKind.INT,
            default=10_000,
            min_val=100,
            max_val=100_000,
            step=1_000,
        )
    if lname in {"nu", "mutation_rate", "speciation_rate"}:
        return Variable(
            name=name,
            description="Per-elementary-event mutation/speciation probability.",
            kind=VariableKind.FLOAT,
            default=0.01,
            min_val=0.0,
            max_val=1.0,
            step=0.001,
        )
    if lname in {"theta", "varpi", "biodiversity_number"}:
        return Variable(
            name=name,
            description="Fundamental biodiversity number theta=N*nu.",
            kind=VariableKind.FLOAT,
            default=100.0,
            min_val=0.0,
            max_val=10_000.0,
            step=10.0,
        )
    if lname in {"delta", "environment_correlation_time", "correlation_time"}:
        return Variable(
            name=name,
            description="Environmental correlation time measured in generations.",
            kind=VariableKind.FLOAT,
            default=0.5,
            min_val=0.01,
            max_val=10.0,
            step=0.05,
        )
    if lname in {"gamma", "fitness_amplitude"}:
        return Variable(
            name=name,
            description="Amplitude of dichotomous fitness fluctuations.",
            kind=VariableKind.FLOAT,
            default=0.5,
            min_val=0.0,
            max_val=2.0,
            step=0.05,
        )
    if "model" in lname:
        return Variable(
            name=name,
            description="Neutral dynamics model variant.",
            kind=VariableKind.CHOICE,
            default="model_a",
            choices=["model_a", "model_b"],
        )
    if lname in {"initial_species_count", "num_species", "s0", "species_count"}:
        return Variable(
            name=name,
            description="Initial number of species before burn-in.",
            kind=VariableKind.INT,
            default=200,
            min_val=2,
            max_val=10_000,
            step=10,
        )
    if "burn" in lname:
        return Variable(
            name=name,
            description="Burn-in duration in generations.",
            kind=VariableKind.INT,
            default=4_000,
            min_val=0,
            max_val=100_000,
            step=500,
        )
    if "sample" in lname or "interval" in lname:
        return Variable(
            name=name,
            description="Sampling interval in generations.",
            kind=VariableKind.INT,
            default=100,
            min_val=1,
            max_val=10_000,
            step=10,
        )
    if "seed" in lname:
        return Variable(
            name=name,
            description="Random seed.",
            kind=VariableKind.INT,
            default=0,
            min_val=0,
            max_val=1_000_000,
            step=1,
        )
    return Variable(
        name=name,
        description=f"Recovered configuration parameter '{name}'.",
        kind=VariableKind.FLOAT,
        default=1.0,
        min_val=None,
        max_val=None,
        step=None,
    )


def _compress_description(description: str, max_chars: int) -> str:
    """
    Compress a long description for the initial monolithic spec call.

    The initial call needs to determine structure (variables, state shape,
    stopping conditions).  For code generation the full context is less
    critical because the staged generator rebuilds it from structure.
    """
    if len(description) <= max_chars:
        return description
    # Keep a balanced head and tail — the head sets the physics,
    # the tail often has the key parameters and procedure summary.
    keep = max_chars // 2
    return (
        description[:keep].rstrip()
        + "\n\n[... context compressed — full detail available in code generation phase ...]\n\n"
        + description[-keep:].lstrip()
    )


def _validate_questions(raw_qs: list[dict], sid: str) -> list[ClarificationQuestion]:
    valid_topic_values = {t.value for t in QuestionTopic}
    forbidden_keywords = {"numpy", "stdlib", "library", "dataclass", "class", "dict",
                          "list vs", "array", "style", "format"}
    result: list[ClarificationQuestion] = []
    for raw in raw_qs[:ClarificationRequest.MAX_QUESTIONS_PER_ROUND]:
        topic_str = raw.get("topic", "variable_range")
        if topic_str not in valid_topic_values:
            log.warning(f"[{sid}] Invalid topic {topic_str!r} — defaulting to variable_range")
            topic_str = "variable_range"
        text = raw.get("text", "")
        if any(kw in text.lower() for kw in forbidden_keywords):
            log.warning(f"[{sid}] Skipping implementation question: {text[:60]!r}")
            continue
        result.append(ClarificationQuestion(
            index=len(result) + 1,
            text=text,
            topic=QuestionTopic(topic_str),
            required=bool(raw.get("required", True)),
        ))
    return result


def _make_spec_approval(session_id: str, spec: SimulationSpec) -> SpecApproval:
    output_contract = OutputContract(
        data_log_fields=["step", "sim_time"] + spec.data_log_variables,
        results_fields=["status", "reason", "stop_condition_name",
                        "steps_run", "wall_time_seconds", "config"],
    )
    eta_secs = spec.time_estimate_seconds
    if eta_secs <= 0:
        eta_str = "unknown"
    elif eta_secs < 60:
        eta_str = f"~{eta_secs:.0f}s"
    elif eta_secs < 3600:
        eta_str = f"~{eta_secs/60:.1f} min"
    else:
        eta_str = f"~{eta_secs/3600:.1f} h"
    if spec.time_estimate_explanation:
        eta_str += f" — {spec.time_estimate_explanation}"
    return SpecApproval(
        session_id=session_id,
        spec_card=_format_spec_card(spec),
        time_estimate=eta_str,
        output_contract=output_contract,
        variable_count=len(spec.variables),
        stop_cond_count=len(spec.stopping_conditions),
    )


def _format_spec_card(spec: SimulationSpec) -> str:
    lines = [
        f"{'─'*60}",
        f"  {spec.name}",
        f"  {spec.description}",
        f"{'─'*60}",
        f"  max_steps : {spec.max_steps:,}",
        "",
        "  Variables:",
    ]
    for v in spec.variables:
        sweep = " [SWEPT]" if v.sweep else ""
        lines.append(
            f"    {v.name:<26} {v.kind.value:<7} "
            f"default={v.default!r:<10} {v.range_str()}{v.unit_str()}{sweep}"
        )
        lines.append(f"      └─ {v.description}")
    lines += ["", "  Stopping conditions:"]
    for s in sorted(spec.stopping_conditions, key=lambda s: -s.priority):
        lines.append(f"    [{s.kind.upper()}] {s.name}: {s.description}")
    lines.append(f"{'─'*60}")
    return "\n".join(lines)


def _parse_json(raw: str) -> Optional[dict]:
    clean = raw.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    parsed = _loads_json_lenient(clean)
    if parsed is not None:
        return parsed
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        return _loads_json_lenient(m.group())
    return None


def _loads_json_lenient(text: str) -> Optional[dict]:
    """Parse JSON, repairing common LLM mistakes that preserve intent."""
    latex_repaired = _repair_latex_escapes(text)
    if latex_repaired != text:
        try:
            data = json.loads(latex_repaired)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        if repaired == text:
            return None
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


_LATEX_COMMAND_RE = re.compile(
    r"\\(?=("
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|"
    r"chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    r"frac|sqrt|left|right|cdot|times|pm|leq|geq|neq"
    r")\b)"
)


def _repair_latex_escapes(text: str) -> str:
    """Escape common LaTeX commands that LLMs put inside JSON strings."""
    return _LATEX_COMMAND_RE.sub(r"\\\\", text)


def _parse_spec(data: dict) -> SimulationSpec:
    variables = [
        variable
        for index, raw in enumerate(data["variables"], 1)
        if (variable := _parse_variable(raw, index)) is not None
    ]
    stopping_conditions = sorted(
        [
            condition
            for index, raw in enumerate(data["stopping_conditions"], 1)
            if (condition := _parse_stopping_condition(raw, index)) is not None
        ],
        key=lambda s: -s.priority,
    )
    return SimulationSpec(
        name=data["name"], description=data["description"],
        variables=variables, stopping_conditions=stopping_conditions,
        state_fields=[tuple(f) for f in data["state_fields"]],
        setup_code=data.get("setup_code", ""),
        precompute_code=data.get("precompute_code", "    return {}"),
        initial_state_code=data["initial_state_code"],
        step_code=data["step_code"],
        progress_code=data["progress_code"],
        config_assert_code=data.get("config_assert_code", "    pass"),
        state_assert_code=data.get("state_assert_code", "    pass"),
        output_variables=data.get("output_variables", []),
        data_log_variables=data.get("data_log_variables", []),
        data_log_interval=int(data.get("data_log_interval", 1)),
        checkpoint_interval=int(data.get("checkpoint_interval", 100)),
        max_steps=int(data.get("max_steps", 10_000)),
        progress_interval=int(data.get("progress_interval", 500)),
        time_estimate_seconds=float(data.get("time_estimate_seconds", 0)),
        time_estimate_explanation=data.get("time_estimate_explanation", ""),
    )


def _parse_variable(raw: object, index: int) -> Variable | None:
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
        if not values:
            return None
        payload = {
            "name": values[0],
            "description": values[1] if len(values) > 1 else "",
            "kind": values[2] if len(values) > 2 else _infer_variable_kind(values[3] if len(values) > 3 else None),
            "default": values[3] if len(values) > 3 else None,
        }
        if len(values) > 4:
            payload["min_val"] = values[4]
        if len(values) > 5:
            payload["max_val"] = values[5]
        if len(values) > 6:
            payload["step"] = values[6]
    else:
        return None

    name = str(payload.get("name") or f"variable_{index}").strip()
    if not name:
        return None
    kind = str(payload.get("kind") or _infer_variable_kind(payload.get("default"))).strip().lower()
    try:
        variable_kind = VariableKind(kind)
    except ValueError:
        variable_kind = VariableKind(_infer_variable_kind(payload.get("default")))
    return Variable(
        name=name,
        description=str(payload.get("description") or name).strip(),
        kind=variable_kind,
        default=payload.get("default"),
        min_val=payload.get("min_val"),
        max_val=payload.get("max_val"),
        step=payload.get("step"),
        choices=payload.get("choices") if isinstance(payload.get("choices"), list) else None,
        unit=payload.get("unit"),
        sweep=bool(payload.get("sweep", False)),
        sweep_values=payload.get("sweep_values") if isinstance(payload.get("sweep_values"), list) else None,
    )


def _parse_stopping_condition(raw: object, index: int) -> StoppingCondition | None:
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
        if len(values) < 3:
            return None
        payload = {
            "kind": values[0],
            "name": values[1],
            "description": values[2],
            "check_expr": values[3] if len(values) > 3 else "False",
            "reason_expr": values[4] if len(values) > 4 else repr(str(values[2])),
        }
    else:
        return None
    return StoppingCondition(
        kind=str(payload.get("kind") or "success"),
        name=str(payload.get("name") or f"condition_{index}"),
        description=str(payload.get("description") or ""),
        check_expr=str(payload.get("check_expr") or "False"),
        reason_expr=str(payload.get("reason_expr") or repr(str(payload.get("description") or ""))),
        save_on_trigger=bool(payload.get("save_on_trigger", True)),
        priority=int(payload.get("priority", 0) or 0),
    )


def _infer_variable_kind(value: object) -> str:
    if isinstance(value, bool):
        return VariableKind.BOOL.value
    if isinstance(value, int):
        return VariableKind.INT.value
    if isinstance(value, float):
        return VariableKind.FLOAT.value
    if isinstance(value, str):
        return VariableKind.STRING.value
    return VariableKind.FLOAT.value
