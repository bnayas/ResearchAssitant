"""
sim_tool.designer
─────────────────
Multi-turn LLM designer: natural language → SimulationSpec.

Contract
────────
  Returns:  ClarificationRequest  when questions remain
            SpecApproval          when spec is complete and ready for review

  Questions are constrained to five topics (see contract.QuestionTopic).
  The LLM is forbidden from asking about implementation details.

  Hard limits per ClarificationContract:
    MAX_QUESTIONS_PER_ROUND = 5
    MAX_ROUNDS = 4

  Memory: reads designer_memory.md at session start, so lessons from
  previous sessions are applied automatically.
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

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
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
  For mutual constraints: assert config.max_steps > config.equil + 100
  Messages must contain the offending value.

Tier 2 — state_assert_code (every step, O(1)):
  assert math.isfinite(state.x), f"x diverged to {{state.x}} at step {{state.step}}"
  For ALL rate/probability fields: assert both bounds (>= 0 AND <= 1)
  For ALL position/energy fields: assert math.isfinite(value)

━━━ LOGGING RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
algo_log → stdout (algorithm trace, for developers debugging code)
  algo_log.debug(f"step {{state.step}}: energy={{state.E:.4f}}")  ← include values
  algo_log.info(f"Equilibration complete at step {{state.step}}")

data_log → .jsonl file (research data, for analysis)
  data_log.info(json.dumps({{"step": state.step, "energy_J": state.energy}}))
  Keys must be stable and descriptive. Include units in key name.

━━━ PRECOMPUTE RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
precompute_code runs ONCE before the loop. ALWAYS precompute:
  - Anything depending only on config (not state)
  - Boltzmann/exponential tables (only N unique values exist)
  - Neighbour/adjacency index arrays
  - Trig components from angles
  - Precomputed constants
  Log what was computed: algo_log.debug(f"Precomputed table: {{result}}")

━━━ STOPPING CONDITIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUST have: ≥1 SUCCESS condition (goal achieved, tolerance met)
MUST have: ≥1 FAILURE condition (bad config detected early)
Failure conditions save compute — they exit bad configs in tens of steps.

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
    "variables": [
      {{"name": <str>, "description": <str>, "kind": "float|int|bool|choice|string",
        "default": <value>, "min_val": <number|null>, "max_val": <number|null>,
        "step": <number|null>, "choices": <list|null>, "unit": <str|null>,
        "sweep": <bool>, "sweep_values": <list|null>}}
    ],
    "state_fields": [["<name>", "<python_type>", <default>]],
    "stopping_conditions": [
      {{"kind": "success|failure", "name": <str>, "description": <str>,
        "check_expr": <str>, "reason_expr": <str>,
        "save_on_trigger": <bool>, "priority": <int>}}
    ],
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
        log.info(f"[{state.session_id}] New session — {description[:60]!r}…")
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
        """Re-return the current spec approval (no new LLM call)."""
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
            user_content = f"Design a simulation for:\n\n{state.original_description}"
        else:
            answers_block = "\n".join(
                f"  [{k}]: {v}" for k, v in state.cumulative_answers.items()
                if not k.startswith("_")
            )
            force_note = (
                "\n\nIMPORTANT: You have reached the maximum number of clarification "
                "rounds. You MUST now set spec_complete=true and return a complete spec, "
                "using reasonable defaults for anything still unclear."
                if force_complete else ""
            )
            user_content = (
                f"Original request:\n{state.original_description}\n\n"
                f"Answers so far:\n{answers_block}"
                + force_note
            )

        state.conversation.append({"role": "user", "content": user_content})

        log.debug(
            f"[{sid}] LLM call #{(len(state.conversation)+1)//2} "
            f"({'FORCED COMPLETE' if force_complete else 'normal'})"
        )
        try:
            raw = self._backend.complete(
                system=self._system_prompt,
                messages=state.conversation,
                max_tokens=4096,
            )
        except Exception as exc:
            log.error(f"[{sid}] LLM error: {exc}")
            raise

        state.conversation.append({"role": "assistant", "content": raw})
        data = _parse_json(raw)
        if data is None:
            raise ValueError(f"[{sid}] LLM returned non-JSON output")

        if not data.get("spec_complete"):
            raw_qs = data.get("questions", [])
            # Validate and cap questions
            questions = _validate_questions(raw_qs, sid)
            log.info(f"[{sid}] Clarification round {state.iteration+1}: {len(questions)} question(s)")
            return ClarificationRequest(
                session_id=sid,
                questions=questions,
                iteration=state.iteration,
            )

        spec = _parse_spec(data["spec"])

        # ── Deterministic validation gate ─────────────────────────────────────
        # Runs before returning to the director. If it fails, we feed the
        # structured errors back to the LLM for one auto-repair attempt.
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
                    max_tokens=4096,
                )
            except Exception as exc:
                log.error(f"[{sid}] Auto-repair LLM call failed: {exc}")
                raise
            state.conversation.append({"role": "assistant", "content": raw2})
            data2 = _parse_json(raw2)
            if data2 and data2.get("spec_complete") and data2.get("spec"):
                try:
                    spec = _parse_spec(data2["spec"])
                    validation2 = self._validator.validate(spec)
                    if not validation2.passed:
                        log.error(
                            f"[{sid}] Spec still invalid after auto-repair: "
                            f"{len(validation2.errors)} error(s)"
                        )
                        # Surface to director with all errors so user can see them
                        raise ValueError(
                            f"Spec validation failed after auto-repair attempt.\n"
                            + validation2.error_summary()
                        )
                    else:
                        log.info(f"[{sid}] Auto-repair succeeded.")
                        validation = validation2
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"Auto-repair produced unparseable spec: {exc}")
            else:
                raise ValueError(
                    f"[{sid}] Auto-repair did not return a complete spec."
                )

        if validation.warnings:
            log.info(
                f"[{sid}] Spec passed with {len(validation.warnings)} warning(s):"
            )
            for w in validation.warnings:
                log.info(f"  [D] {w.code}: {w.message[:80]}")

        state.spec = spec
        log.info(
            f"[{sid}] Spec validated and complete — '{spec.name}' | "
            f"{len(spec.variables)} vars | {len(spec.stopping_conditions)} stop conds"
        )
        return _make_spec_approval(sid, spec)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_questions(raw_qs: list[dict], sid: str) -> list[ClarificationQuestion]:
    """
    Validate questions against the contract.
    - Cap at MAX_QUESTIONS_PER_ROUND
    - Validate topic is an allowed QuestionTopic
    - Filter out implementation questions (best-effort)
    """
    valid_topic_values = {t.value for t in QuestionTopic}
    forbidden_keywords = {"numpy", "stdlib", "library", "dataclass", "class", "dict",
                          "list vs", "array", "style", "format"}

    result: list[ClarificationQuestion] = []
    for raw in raw_qs[:ClarificationRequest.MAX_QUESTIONS_PER_ROUND]:
        topic_str = raw.get("topic", "variable_range")
        if topic_str not in valid_topic_values:
            log.warning(f"[{sid}] Question has invalid topic {topic_str!r} — defaulting to variable_range")
            topic_str = "variable_range"

        text = raw.get("text", "")
        # Check for forbidden implementation topics
        text_lower = text.lower()
        if any(kw in text_lower for kw in forbidden_keywords):
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
    """Build a SpecApproval from a validated spec."""
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
    clean_resp = raw.strip()
    if clean_resp.startswith("```json"):
        clean_resp = clean_resp[7:]
    elif clean_resp.startswith("```"):
        clean_resp = clean_resp[3:]
    if clean_resp.endswith("```"):
        clean_resp = clean_resp[:-3]
    clean_resp = clean_resp.strip()

    try:
        return json.loads(clean_resp)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean_resp, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError as e:
                log.warning("JSON decode failed on extracted block: %s. Raw was: %r", e, raw[:500])
        else:
            log.warning("No JSON object could be extracted. Raw was: %r", raw[:500])
    return None


def _parse_spec(data: dict) -> SimulationSpec:
    variables = [
        Variable(
            name=v["name"], description=v["description"],
            kind=VariableKind(v["kind"]), default=v["default"],
            min_val=v.get("min_val"), max_val=v.get("max_val"),
            step=v.get("step"), choices=v.get("choices"),
            unit=v.get("unit"), sweep=v.get("sweep", False),
            sweep_values=v.get("sweep_values"),
        )
        for v in data["variables"]
    ]
    stopping_conditions = sorted(
        [StoppingCondition(
            kind=s["kind"], name=s["name"], description=s["description"],
            check_expr=s["check_expr"], reason_expr=s["reason_expr"],
            save_on_trigger=s.get("save_on_trigger", True),
            priority=s.get("priority", 0),
        ) for s in data["stopping_conditions"]],
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
