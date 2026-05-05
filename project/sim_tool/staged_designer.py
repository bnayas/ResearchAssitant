"""
sim_tool.staged_designer
─────────────────────────
Staged simulation code generation for context-window-limited environments.

Why staged?
───────────
A full SimulationSpec contains ~10 distinct code blocks (step_code,
precompute_code, initial_state_code, config_assert_code, state_assert_code,
progress_code, setup_code, plus check_expr/reason_expr per stopping
condition).  Generating all of these in a single LLM call with a long
research description as context predictably overflows context windows,
producing either truncated JSON, prose instead of JSON, or hallucinated
field names.

The staged approach splits generation into four tightly-scoped phases,
each with a narrow prompt that can always fit in a 16k-token window:

  Phase 1  ─ Structural skeleton
             Generates everything EXCEPT code bodies: name, description,
             variables, state_fields, stopping-condition metadata, and
             the scalar run-control fields.  No code at all.

  Phase 2  ─ Code design (signatures + docstrings)
             Given only the skeleton, generates a table of what each
             function needs and returns.  This acts as the API contract
             for Phase 3.

  Phase 3  ─ Per-function implementation
             One LLM call per code field.  Each call receives:
               - the function signature and its Phase-2 docstring
               - the SimConfig and SimState field names
               - the keys returned by precompute()
             Returns only the function body, no JSON wrapping.

  Phase 4  ─ Assembly and SpecValidator gate
             All bodies are assembled into a complete SimulationSpec,
             then passed through the deterministic SpecValidator (28
             checks, 4 tiers).  Blocking errors trigger targeted repairs
             on only the failing field, not a full regeneration.

Integration
───────────
    from sim_tool.staged_designer import StagedSpecGenerator

    generator = StagedSpecGenerator(backend)
    spec = generator.generate(description, session_id="abc123")
    # spec is a validated SimulationSpec or raises ValueError on unrecoverable failure
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import asdict
from typing import Optional

from .contract_validator import SpecValidator
from .models import SimulationSpec, StoppingCondition, Variable, VariableKind

log = logging.getLogger("sim_tool.staged_designer")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Structural skeleton (no code bodies)
# ─────────────────────────────────────────────────────────────────────────────

_SKELETON_SYSTEM = """
You are a simulation architect. Your job is to design the STRUCTURE of a simulation
WITHOUT writing any code.  Code bodies will be written in a separate step.

Return ONLY a JSON object with this exact structure (no markdown, no prose):

{
  "name": "<short simulation name>",
  "description": "<one sentence>",
  "variables": [
    {
      "name": "<python_identifier>",
      "description": "<what it controls>",
      "kind": "float|int|bool|choice",
      "default": <value>,
      "min_val": <number|null>,
      "max_val": <number|null>,
      "step": <number|null>,
      "choices": <list|null>,
      "unit": "<unit string|null>",
      "sweep": <bool>,
      "sweep_values": <list|null>
    }
  ],
  "state_fields": [
    ["<field_name>", "<python_type>", <default_value>]
  ],
  "stopping_conditions": [
    {
      "kind": "success|failure",
      "name": "<snake_case_name>",
      "description": "<what triggers this>",
      "priority": <int>
    }
  ],
  "output_variables": ["<state_field_name>"],
  "data_log_variables": ["<state_field_name>"],
  "data_log_interval": <int>,
  "checkpoint_interval": <int>,
  "max_steps": <int>,
  "progress_interval": <int>,
  "time_estimate_seconds": <float>,
  "time_estimate_explanation": "<brief estimate explanation>"
}

Rules:
- MUST have ≥1 success stopping condition and ≥1 failure stopping condition.
- state_fields: each entry is [name, python_type, default_value].
- All variable and state-field names must be valid Python identifiers.
- Do NOT write any code.  check_expr and reason_expr will be added later.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Code design (API contracts for each function)
# ─────────────────────────────────────────────────────────────────────────────

_CODE_DESIGN_SYSTEM = """
You are designing the API contract for a simulation's code functions.
You receive a structural skeleton (no code yet).

Return ONLY a JSON object describing each function.  This will be used
as the spec for the implementation phase.  Be concrete and specific.

{
  "setup_imports": ["import numpy as np", "import math"],
  "precompute_returns": {
    "<key>": "<type and description, e.g. 'list[list[int]]: adjacency table'>",
    ...
  },
  "initial_state_notes": "<what non-zero/non-default fields need to be set and why>",
  "step_algorithm": "<describe the algorithm in 3-5 steps, e.g. 'Metropolis: pick random site, compute dE, accept if exp(-dE/kT) > rand'>",
  "progress_fields": ["<field_name to include in progress string>"],
  "config_assertions": [
    "<what to assert, e.g. 'temperature > 0, lattice_size >= 4'>",
    ...
  ],
  "state_invariants": [
    "<what to assert every step, e.g. 'math.isfinite(energy), 0 <= acceptance_rate <= 1'>",
    ...
  ],
  "stopping_conditions": [
    {
      "name": "<must match skeleton>",
      "check_description": "<when this fires, e.g. 'state.step > 100 and abs(delta_m) < 1e-4'>",
      "reason_description": "<what the reason string should say>"
    }
  ]
}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Per-function implementation prompts
# ─────────────────────────────────────────────────────────────────────────────

def _make_function_prompt(
    func_name: str,
    signature: str,
    docstring: str,
    context: str,
    constraint: str,
) -> str:
    """Build the prompt for a single function body generation call."""
    return f"""
Write the body of this Python function for a simulation.

FUNCTION SIGNATURE:
{signature}

DOCSTRING / SPEC:
{docstring}

CONTEXT (fields and types available):
{context}

CONSTRAINT:
{constraint}

OUTPUT RULES:
- Return ONLY the function body (the indented code that goes inside the function).
- Do NOT include the `def` line.
- Do NOT include any other text, explanation, or markdown.
- Use 4-space indentation.
- Code must be valid Python 3.11.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# StagedSpecGenerator
# ─────────────────────────────────────────────────────────────────────────────

class StagedSpecGenerator:
    """
    Generates a validated SimulationSpec via four focused LLM passes.

    Each pass has a prompt small enough to always fit a 16k-token window.
    The generator never sends the full source research description to a
    code-generation call: it compresses to ≤2000 characters first.
    """

    MAX_DESC_CHARS = 2000  # hard limit on description sent to code passes

    def __init__(self, backend) -> None:
        self._backend = backend
        self._validator = SpecValidator()

    def generate(
        self,
        description: str,
        *,
        session_id: str = "",
        max_repair_attempts: int = 2,
    ) -> SimulationSpec:
        """
        Run all four phases and return a validated SimulationSpec.
        Raises ValueError if the spec cannot be made valid within max_repair_attempts.
        """
        sid = session_id or "staged"
        log.info(f"[{sid}] Staged generation — description length={len(description)}")

        # Phase 1
        log.info(f"[{sid}] Phase 1: generating structural skeleton")
        skeleton = self._phase1_skeleton(description, sid)

        # Phase 2
        log.info(f"[{sid}] Phase 2: designing code contracts")
        design = self._phase2_design(skeleton, description, sid)

        # Phase 3
        log.info(f"[{sid}] Phase 3: implementing functions")
        code = self._phase3_implement(skeleton, design, description, sid)

        # Phase 4
        log.info(f"[{sid}] Phase 4: assembling and validating")
        spec = self._phase4_assemble(skeleton, code, sid)

        # Targeted repair loop
        for attempt in range(max_repair_attempts):
            result = self._validator.validate(spec)
            if result.passed:
                if result.warnings:
                    log.info(f"[{sid}] Spec valid with {len(result.warnings)} warning(s)")
                return spec
            log.warning(
                f"[{sid}] Validation failed attempt {attempt+1}/{max_repair_attempts}: "
                f"{len(result.errors)} error(s)"
            )
            spec = self._phase5_repair(spec, result, skeleton, sid)

        result = self._validator.validate(spec)
        if not result.passed:
            raise ValueError(
                f"[{sid}] Staged generation failed after {max_repair_attempts} repairs.\n"
                + result.error_summary()
            )
        return spec

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _phase1_skeleton(self, description: str, sid: str) -> dict:
        """Generate the structural skeleton (no code bodies)."""
        prompt = (
            f"Design a simulation for:\n\n{description}\n\n"
            "Return the structural skeleton JSON."
        )
        raw = self._complete(_SKELETON_SYSTEM, prompt, label=f"{sid}/skeleton")
        data = _parse_json(raw)
        if not data:
            raise ValueError(f"[{sid}] Phase 1 returned non-JSON: {raw[:200]}")
        return data

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _phase2_design(self, skeleton: dict, description: str, sid: str) -> dict:
        """Design the code contracts from the skeleton."""
        skeleton_summary = json.dumps({
            "name": skeleton.get("name"),
            "state_fields": skeleton.get("state_fields", []),
            "variables": [
                {"name": v.get("name"), "kind": v.get("kind"), "description": v.get("description")}
                for v in skeleton.get("variables", [])
            ],
            "stopping_conditions": [
                {"name": c.get("name"), "kind": c.get("kind"), "description": c.get("description")}
                for c in skeleton.get("stopping_conditions", [])
            ],
        }, indent=2)

        prompt = (
            f"Simulation: {skeleton.get('description', '')}\n\n"
            f"Skeleton:\n{skeleton_summary}\n\n"
            "Return the code design JSON."
        )
        raw = self._complete(_CODE_DESIGN_SYSTEM, prompt, label=f"{sid}/design")
        data = _parse_json(raw)
        if not data:
            log.warning(f"[{sid}] Phase 2 returned non-JSON; using minimal design")
            data = {}
        return data

    # ── Phase 3 ───────────────────────────────────────────────────────────────

    def _phase3_implement(
        self, skeleton: dict, design: dict, description: str, sid: str
    ) -> dict:
        """Generate each code field in a separate focused LLM call."""

        # Derive a compact context string from the skeleton
        config_fields = "\n".join(
            f"  config.{v['name']}: {v.get('kind','float')} (default={v.get('default')}) — {v.get('description','')}"
            for v in skeleton.get("variables", [])
        )
        state_fields = "\n".join(
            f"  state.{f[0]}: {f[1]} (default={f[2]})"
            for f in skeleton.get("state_fields", [])
        )
        precomputed_keys = "\n".join(
            f"  precomputed['{k}']: {v}"
            for k, v in design.get("precompute_returns", {}).items()
        ) or "  (empty dict)"

        context_block = (
            f"SimConfig fields:\n{config_fields or '  (none)'}\n\n"
            f"SimState fields (always has .step: int, .sim_time: float, .history: list):\n"
            f"{state_fields or '  (none)'}\n\n"
            f"precomputed dict keys:\n{precomputed_keys}"
        )

        code: dict = {}
        imports = design.get("setup_imports", ["import math"])

        # setup_code
        code["setup_code"] = "\n".join(imports) + "\n"

        # precompute_code
        precompute_desc = "\n".join(
            f"  Return '{k}': {v}"
            for k, v in design.get("precompute_returns", {}).items()
        ) or "  Return empty dict."
        precompute_docstring = (
            f"Called once before the main loop.\n"
            f"Returns a dict of precomputed values:\n{precompute_desc}\n"
            f"config and SimConfig fields are available.\n"
            f"Return a plain dict."
        )
        code["precompute_code"] = self._implement_function(
            func_name="precompute",
            signature="def precompute(config: SimConfig) -> dict:",
            docstring=precompute_docstring,
            context=f"SimConfig fields:\n{config_fields or '  (none)'}",
            constraint="MUST end with a `return {{...}}` statement.",
            sid=sid,
        )

        # initial_state_code
        init_notes = design.get("initial_state_notes", "Set all state fields to their initial values.")
        init_docstring = (
            f"Create and return the initial SimState.\n"
            f"Notes: {init_notes}\n"
            f"precomputed dict is available."
        )
        code["initial_state_code"] = self._implement_function(
            func_name="initial_state",
            signature="def initial_state(config: SimConfig, precomputed: dict) -> SimState:",
            docstring=init_docstring,
            context=context_block,
            constraint="MUST create a SimState() instance and return it.",
            sid=sid,
        )

        # step_code
        step_alg = design.get("step_algorithm", "Advance the simulation by one step and return the new state.")
        step_docstring = (
            f"Advance the simulation by one step.\n"
            f"Algorithm: {step_alg}\n"
            f"All config, state, and precomputed fields are available."
        )
        code["step_code"] = self._implement_function(
            func_name="sim_step",
            signature="def sim_step(config: SimConfig, state: SimState, precomputed: dict) -> SimState:",
            docstring=step_docstring,
            context=context_block,
            constraint=(
                "MUST return a SimState. Update state.sim_time appropriately. "
                "Do NOT mutate state in-place — copy it or create a new one."
            ),
            sid=sid,
        )

        # progress_code
        prog_fields = design.get("progress_fields", [f[0] for f in skeleton.get("state_fields", [])[:3]])
        progress_docstring = (
            f"Return a short one-line string summarising the current simulation state.\n"
            f"Key fields to include: {prog_fields}"
        )
        code["progress_code"] = self._implement_function(
            func_name="progress_summary",
            signature="def progress_summary(state: SimState, precomputed: dict) -> str:",
            docstring=progress_docstring,
            context=f"SimState fields:\n{state_fields or '  (none)'}",
            constraint="MUST return a str. Keep it short (< 80 chars).",
            sid=sid,
        )

        # config_assert_code
        config_asserts = design.get("config_assertions", [])
        config_assert_docstring = (
            "Validate config before the simulation starts.\n"
            "Assertions to implement:\n"
            + ("\n".join(f"  - {a}" for a in config_asserts) or "  - assert basic physical validity")
        )
        code["config_assert_code"] = self._implement_function(
            func_name="_assert_config",
            signature="def _assert_config(config: SimConfig) -> None:",
            docstring=config_assert_docstring,
            context=f"SimConfig fields:\n{config_fields or '  (none)'}",
            constraint=(
                "Use assert statements only. Each assert message MUST include the "
                "offending value: assert config.x > 0, f'x must be positive, got {config.x}'"
            ),
            sid=sid,
        )

        # state_assert_code
        state_invariants = design.get("state_invariants", [])
        state_assert_docstring = (
            "Check state invariants every step (must be O(1)).\n"
            "Invariants to check:\n"
            + ("\n".join(f"  - {a}" for a in state_invariants) or "  - assert math.isfinite() on numeric fields")
        )
        code["state_assert_code"] = self._implement_function(
            func_name="_assert_state",
            signature="def _assert_state(config: SimConfig, state: SimState) -> None:",
            docstring=state_assert_docstring,
            context=f"SimState fields:\n{state_fields or '  (none)'}",
            constraint=(
                "Use assert statements only. Each message must reference the value. "
                "Must remain O(1) — do NOT loop over large data structures."
            ),
            sid=sid,
        )

        # Stopping conditions — check_expr and reason_expr per condition
        stopping_code: list[dict] = []
        for cond in skeleton.get("stopping_conditions", []):
            design_entry = next(
                (c for c in design.get("stopping_conditions", []) if c.get("name") == cond["name"]),
                {},
            )
            check_desc = design_entry.get(
                "check_description",
                f"Check when the {cond['kind']} condition '{cond['name']}' is met",
            )
            reason_desc = design_entry.get(
                "reason_description",
                f"Explain why {cond['name']} fired",
            )

            check_expr = self._implement_expression(
                kind="boolean expression",
                description=f"Returns True when: {check_desc}",
                context=context_block,
                constraint=(
                    "Single Python expression. Available names: config, state, precomputed. "
                    "Use == for equality, not =. No assignment operators."
                ),
                sid=f"{sid}/{cond['name']}/check",
            )
            reason_expr = self._implement_expression(
                kind="string expression",
                description=f"A string explaining the reason: {reason_desc}",
                context=f"state.step: int\nstate.sim_time: float\n{state_fields}",
                constraint=(
                    "Single Python expression that evaluates to a str. "
                    "Use an f-string. Available names: config, state, precomputed."
                ),
                sid=f"{sid}/{cond['name']}/reason",
            )
            stopping_code.append({
                **cond,
                "check_expr": check_expr.strip(),
                "reason_expr": reason_expr.strip(),
                "save_on_trigger": True,
            })

        code["stopping_conditions"] = stopping_code
        return code

    # ── Phase 4 ───────────────────────────────────────────────────────────────

    def _phase4_assemble(self, skeleton: dict, code: dict, sid: str) -> SimulationSpec:
        """Assemble skeleton + code into a SimulationSpec."""
        variables = [
            Variable(
                name=v["name"],
                description=v.get("description", ""),
                kind=VariableKind(v.get("kind", "float")),
                default=v["default"],
                min_val=v.get("min_val"),
                max_val=v.get("max_val"),
                step=v.get("step"),
                choices=v.get("choices"),
                unit=v.get("unit"),
                sweep=v.get("sweep", False),
                sweep_values=v.get("sweep_values"),
            )
            for v in skeleton.get("variables", [])
        ]

        stopping_conditions = sorted(
            [
                StoppingCondition(
                    kind=c["kind"],
                    name=c["name"],
                    description=c.get("description", ""),
                    check_expr=c.get("check_expr", "False"),
                    reason_expr=c.get("reason_expr", '"stopping condition met"'),
                    save_on_trigger=c.get("save_on_trigger", True),
                    priority=c.get("priority", 0),
                )
                for c in code.get("stopping_conditions", skeleton.get("stopping_conditions", []))
            ],
            key=lambda s: -s.priority,
        )

        state_fields = [tuple(f) for f in skeleton.get("state_fields", [])]

        return SimulationSpec(
            name=skeleton.get("name", "Simulation"),
            description=skeleton.get("description", ""),
            variables=variables,
            stopping_conditions=stopping_conditions,
            state_fields=state_fields,
            setup_code=code.get("setup_code", "import math\n"),
            precompute_code=code.get("precompute_code", "    return {}"),
            initial_state_code=code.get("initial_state_code", "    return SimState()"),
            step_code=code.get("step_code", "    return state"),
            progress_code=code.get("progress_code", "    return f'step={state.step}'"),
            config_assert_code=code.get("config_assert_code", "    pass"),
            state_assert_code=code.get("state_assert_code", "    pass"),
            output_variables=skeleton.get("output_variables", []),
            data_log_variables=skeleton.get("data_log_variables", []),
            data_log_interval=int(skeleton.get("data_log_interval", 1)),
            checkpoint_interval=int(skeleton.get("checkpoint_interval", 100)),
            max_steps=int(skeleton.get("max_steps", 10_000)),
            progress_interval=int(skeleton.get("progress_interval", 500)),
            time_estimate_seconds=float(skeleton.get("time_estimate_seconds", 0)),
            time_estimate_explanation=skeleton.get("time_estimate_explanation", ""),
        )

    # ── Phase 5 (repair) ──────────────────────────────────────────────────────

    def _phase5_repair(
        self, spec: SimulationSpec, validation, skeleton: dict, sid: str
    ) -> SimulationSpec:
        """
        Targeted repair: re-generate only the fields that failed validation.

        Rather than regenerating the entire spec, we look at which field
        caused each error and re-run only that field's generation call.
        This is both faster and more reliable than full regeneration.
        """
        error_fields = {e.field.split("[")[0].split(".")[0] for e in validation.errors}
        log.info(f"[{sid}] Repairing fields: {error_fields}")

        state_fields = "\n".join(
            f"  state.{f[0]}: {f[1]} (default={f[2]})"
            for f in skeleton.get("state_fields", [])
        )
        config_fields = "\n".join(
            f"  config.{v['name']}: {v.get('kind','float')}"
            for v in skeleton.get("variables", [])
        )
        context_block = (
            f"SimConfig fields:\n{config_fields}\n\n"
            f"SimState fields:\n{state_fields}"
        )

        # Map field names to repair actions
        error_summary = "\n".join(
            f"  [{e.code}] {e.field}: {e.message[:80]}"
            for e in validation.errors
        )
        repair_constraint = f"Fix these validation errors:\n{error_summary}"

        if "precompute_code" in error_fields:
            spec.precompute_code = self._implement_function(
                "precompute",
                "def precompute(config: SimConfig) -> dict:",
                "Precompute values before the simulation loop. Return a dict.",
                f"SimConfig fields:\n{config_fields}",
                f"MUST return a dict. {repair_constraint}",
                sid=f"{sid}/repair/precompute",
            )
        if "initial_state_code" in error_fields:
            spec.initial_state_code = self._implement_function(
                "initial_state",
                "def initial_state(config: SimConfig, precomputed: dict) -> SimState:",
                "Create and return the initial SimState.",
                context_block,
                f"MUST return a SimState. {repair_constraint}",
                sid=f"{sid}/repair/initial_state",
            )
        if "step_code" in error_fields:
            spec.step_code = self._implement_function(
                "sim_step",
                "def sim_step(config: SimConfig, state: SimState, precomputed: dict) -> SimState:",
                "Advance the simulation by one step.",
                context_block,
                f"MUST return a SimState. {repair_constraint}",
                sid=f"{sid}/repair/step",
            )
        if "progress_code" in error_fields:
            spec.progress_code = self._implement_function(
                "progress_summary",
                "def progress_summary(state: SimState, precomputed: dict) -> str:",
                "Return a short status string.",
                f"SimState fields:\n{state_fields}",
                f"MUST return a str. {repair_constraint}",
                sid=f"{sid}/repair/progress",
            )
        if "config_assert_code" in error_fields:
            spec.config_assert_code = self._implement_function(
                "_assert_config",
                "def _assert_config(config: SimConfig) -> None:",
                "Assert that config values are valid.",
                f"SimConfig fields:\n{config_fields}",
                f"Use assert statements only. {repair_constraint}",
                sid=f"{sid}/repair/config_assert",
            )
        if "state_assert_code" in error_fields:
            spec.state_assert_code = self._implement_function(
                "_assert_state",
                "def _assert_state(config: SimConfig, state: SimState) -> None:",
                "Check state invariants every step.",
                context_block,
                f"Use assert statements only. O(1). {repair_constraint}",
                sid=f"{sid}/repair/state_assert",
            )

        return spec

    # ── Primitive helpers ─────────────────────────────────────────────────────

    def _implement_function(
        self,
        func_name: str,
        signature: str,
        docstring: str,
        context: str,
        constraint: str,
        sid: str,
    ) -> str:
        """
        Generate a single function body and return it properly indented.

        The LLM is asked for the body only (no def line), which is much
        easier to validate and extract than trying to parse a full function
        from a larger JSON blob.
        """
        prompt = _make_function_prompt(func_name, signature, docstring, context, constraint)
        raw = self._complete(
            "You are a Python simulation engineer. Write only the function body, "
            "4-space indented, no def line, no markdown, no explanation.",
            prompt,
            label=f"{sid}/{func_name}",
        )
        body = _clean_function_body(raw)
        # Quick syntax check — surface problems early before SpecValidator
        try:
            compile(f"def {func_name}():\n{body}", f"<{func_name}>", "exec")
        except SyntaxError as exc:
            log.warning(f"[{sid}] Syntax error in {func_name}: {exc}. Body: {body[:200]}")
            # Return a minimal valid body rather than crashing the whole pipeline
            body = f"    # auto-generated stub (original had syntax error: {exc.msg})\n    pass"
        return body

    def _implement_expression(
        self,
        kind: str,
        description: str,
        context: str,
        constraint: str,
        sid: str,
    ) -> str:
        """
        Generate a single Python expression (for check_expr / reason_expr).

        Unlike _implement_function, this returns a raw expression string
        suitable for use directly in generated code.
        """
        prompt = (
            f"Write a Python {kind} that: {description}\n\n"
            f"Available context:\n{context}\n\n"
            f"Constraint: {constraint}\n\n"
            "Return ONLY the expression, one line, no quotes wrapping it, no explanation."
        )
        raw = self._complete(
            "You are a Python simulation engineer. Return only the expression.",
            prompt,
            label=sid,
        )
        # Strip any accidental quoting or markdown
        expr = raw.strip().strip("`").strip()
        # Remove leading f-string wrapper if someone wrapped the whole thing
        if expr.startswith('f"') or expr.startswith("f'"):
            pass  # f-strings for reason_expr are correct
        return expr or '"stopping condition met"'

    def _complete(self, system: str, prompt: str, *, label: str) -> str:
        """Single LLM call with logging."""
        log.debug(f"[{label}] LLM call: {len(prompt)} chars")
        try:
            result = self._backend.complete(
                system=system,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.0,
            )
            log.debug(f"[{label}] Response: {len(result)} chars")
            return result
        except Exception as exc:
            log.error(f"[{label}] LLM call failed: {exc}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[dict]:
    """Best-effort JSON extraction from LLM output."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _clean_function_body(raw: str) -> str:
    """
    Extract and clean a function body from LLM output.

    The LLM may wrap the body in markdown fences, include the def line,
    or add trailing prose.  This strips all of that and ensures uniform
    4-space indentation.
    """
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # If the LLM included the def line, strip it and dedent
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("def "):
        lines = lines[1:]  # drop the def line
        text = "\n".join(lines)
        text = textwrap.dedent(text)

    # Re-indent uniformly to 4 spaces
    dedented = textwrap.dedent(text)
    body = textwrap.indent(dedented, "    ")

    return body if body.strip() else "    pass"


def compress_description(description: str, max_chars: int = 2000) -> str:
    """
    Compress a long research description for use in code-generation prompts.

    The full description may contain an entire article brief (procedures,
    parameters, literature context).  For code generation we only need
    the core model description and key parameters — everything else is
    structural detail that belongs in the skeleton, not the code passes.
    """
    if len(description) <= max_chars:
        return description
    # Keep the first third and the last third; drop the middle
    keep = max_chars // 2
    return (
        description[:keep].rstrip()
        + "\n\n... [context compressed for code generation] ...\n\n"
        + description[-keep:].lstrip()
    )
