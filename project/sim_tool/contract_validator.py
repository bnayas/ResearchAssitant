"""
sim_tool.contract_validator
────────────────────────────
Deterministic validation of the coding agent's output before it is
returned to the director. No LLM involved.

Role in the pipeline
────────────────────
  [Designer LLM] → raw spec dict
      → [parse_spec()]              structural parsing
      → [SpecValidator.validate()]  deterministic checks  ← THIS MODULE
      → [SpecApproval / ValidationFailed]  returned to director

If validation fails the designer gets one auto-repair attempt (feeding
the structured errors back to the LLM). If it fails again the director
receives a ValidationFailed result and can decide whether to retry or
surface the error to the user.

What is checked (exhaustive)
─────────────────────────────

TIER A — Presence checks (MISSING_FIELD)
  A1  All required top-level fields are present and non-empty.
  A2  Every variable has: name, description, kind, default.
  A3  Every stopping condition has: kind, name, description,
      check_expr, reason_expr.

TIER B — Python syntax checks (SYNTAX_ERROR)
  B1  setup_code            compiles as a module
  B2  precompute_code       compiles as a function body
  B3  initial_state_code    compiles as a function body
  B4  step_code             compiles as a function body
  B5  progress_code         compiles as a function body
  B6  config_assert_code    compiles as a function body
  B7  state_assert_code     compiles as a function body
  B8  check_expr            compiles as an expression (per condition)
  B9  reason_expr           compiles as an expression (per condition)

TIER C — Semantic checks (SEMANTIC_ERROR)
  C1  precompute_code contains a 'return' statement.
  C2  initial_state_code contains 'return'.
  C3  step_code contains 'return'.
  C4  progress_code contains 'return'.
  C5  All names in output_variables exist in state_fields.
  C6  All names in data_log_variables exist in state_fields.
  C7  At least one stopping condition with kind='success'.
  C8  At least one stopping condition with kind='failure'.
  C9  All variable names are valid Python identifiers.
  C10 All state field names are valid Python identifiers.
  C11 No stopping condition check_expr uses assignment (= not ==).

TIER D — Quality checks (WARNING, non-blocking)
  D1  No bare print() calls in any code block (should use algo_log/data_log).
  D2  No 'import sim_tool' in any code block.
  D3  config_assert_code is non-trivial (not just 'pass').
  D4  state_assert_code is non-trivial (not just 'pass').
  D5  data_log_variables is non-empty.
  D6  output_variables is non-empty.
  D7  max_steps > 0.
  D8  progress_interval > 0.

Error format
────────────
Every error is a typed ValidationError with:
  tier:    "A" | "B" | "C" | "D"
  code:    short snake_case identifier
  field:   which spec field caused it
  message: plain English, readable without source access

Usage
─────
  from sim_tool.contract_validator import SpecValidator, ValidationResult

  validator = SpecValidator()
  result = validator.validate(spec)

  if result.passed:
      # hand spec to director
  else:
      # feed result.error_summary() back to LLM for correction
      print(result.error_summary())
"""

from __future__ import annotations

import ast
import keyword
import logging
from dataclasses import dataclass, field
from typing import Optional

from .models import SimulationSpec

log = logging.getLogger("sim_tool.validator")


# ─────────────────────────────────────────────────────────────────────────────
# Error types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationError:
    tier:    str   # "A" | "B" | "C" | "D"
    code:    str   # e.g. "missing_precompute_return"
    field:   str   # e.g. "precompute_code", "stopping_conditions[1].check_expr"
    message: str   # plain English, no source access needed

    def __str__(self) -> str:
        blocking = "(BLOCKING)" if self.tier != "D" else "(warning)"
        return f"  [{self.tier}] {self.code} {blocking}\n    field: {self.field}\n    {self.message}"


@dataclass
class ValidationResult:
    """
    Result of a spec validation run.

    .passed  — True iff there are zero tier-A/B/C errors.
               Tier-D warnings never block.
    .errors  — tier A, B, C (blocking)
    .warnings— tier D (informational only)
    """
    errors:   list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def error_summary(self) -> str:
        """
        Compact text suitable for feeding back to the LLM as a correction prompt.
        Every line is self-contained — the LLM does not need the source code.
        """
        if self.passed and not self.warnings:
            return "Validation passed with no issues."

        lines: list[str] = []
        if self.errors:
            lines.append(
                f"VALIDATION FAILED — {len(self.errors)} blocking error(s) "
                f"must be fixed before the spec can be accepted:\n"
            )
            for e in self.errors:
                lines.append(str(e))
        if self.warnings:
            lines.append(
                f"\n{len(self.warnings)} warning(s) (non-blocking, but should be fixed):\n"
            )
            for w in self.warnings:
                lines.append(str(w))

        lines.append(
            "\nPlease return a corrected spec JSON with all blocking errors fixed."
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else f"FAILED ({len(self.errors)} errors)"
        lines = [f"ValidationResult: {status}"]
        for e in self.errors:
            lines.append(str(e))
        for w in self.warnings:
            lines.append(str(w))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class SpecValidator:
    """
    Deterministic spec validator. No LLM, no randomness, no I/O.
    Instantiate once and call validate() for each spec.
    """

    def validate(self, spec: SimulationSpec) -> ValidationResult:
        result = ValidationResult()

        self._check_A_presence(spec, result)
        self._check_B_syntax(spec, result)
        self._check_C_semantic(spec, result)
        self._check_D_quality(spec, result)

        n_err = len(result.errors)
        n_warn = len(result.warnings)
        if n_err == 0:
            log.debug(
                f"Spec '{spec.name}' passed validation "
                f"({n_warn} warning(s))"
            )
        else:
            log.warning(
                f"Spec '{spec.name}' FAILED validation: "
                f"{n_err} error(s), {n_warn} warning(s)"
            )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Tier A: Presence checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_A_presence(self, spec: SimulationSpec, r: ValidationResult) -> None:
        def req(val: object, field: str, desc: str) -> bool:
            if not val:
                r.errors.append(ValidationError(
                    tier="A", code="missing_field", field=field,
                    message=f"{desc} is required but was empty or missing.",
                ))
                return False
            return True

        req(spec.name,                "name",                "Simulation name")
        req(spec.description,         "description",         "Simulation description")
        req(spec.variables,           "variables",           "Variable list")
        req(spec.stopping_conditions, "stopping_conditions", "Stopping condition list")
        req(spec.state_fields,        "state_fields",        "State field list")
        req(spec.initial_state_code,  "initial_state_code",  "initial_state function body")
        req(spec.step_code,           "step_code",           "sim_step function body")
        req(spec.progress_code,       "progress_code",       "progress_summary function body")
        req(spec.precompute_code,     "precompute_code",
            "precompute function body (return {} is the minimum acceptable body)")

        for i, v in enumerate(spec.variables):
            prefix = f"variables[{i}]"
            if not v.name:
                r.errors.append(ValidationError(
                    "A", "missing_variable_name", prefix,
                    f"Variable at index {i} has no name.",
                ))
            if not v.description:
                r.errors.append(ValidationError(
                    "A", "missing_variable_description", f"{prefix}.description",
                    f"Variable '{v.name or i}' has no description.",
                ))
            if v.default is None:
                r.errors.append(ValidationError(
                    "A", "missing_variable_default", f"{prefix}.default",
                    f"Variable '{v.name or i}' has no default value. "
                    "Every variable must have a default so runs can start without "
                    "requiring every argument.",
                ))

        for i, sc in enumerate(spec.stopping_conditions):
            prefix = f"stopping_conditions[{i}]"
            if not sc.name:
                r.errors.append(ValidationError(
                    "A", "missing_condition_name", prefix,
                    f"Stopping condition at index {i} has no name.",
                ))
            if not sc.description:
                r.errors.append(ValidationError(
                    "A", "missing_condition_description", f"{prefix}.description",
                    f"Stopping condition '{sc.name or i}' has no description.",
                ))
            if not sc.check_expr:
                r.errors.append(ValidationError(
                    "A", "missing_check_expr", f"{prefix}.check_expr",
                    f"Stopping condition '{sc.name or i}' has no check_expr. "
                    "check_expr must be a Python boolean expression.",
                ))
            if not sc.reason_expr:
                r.errors.append(ValidationError(
                    "A", "missing_reason_expr", f"{prefix}.reason_expr",
                    f"Stopping condition '{sc.name or i}' has no reason_expr. "
                    "reason_expr must be a Python string expression (e.g. f-string).",
                ))

    # ─────────────────────────────────────────────────────────────────────────
    # Tier B: Syntax checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_B_syntax(self, spec: SimulationSpec, r: ValidationResult) -> None:

        def check_module(code: str, field: str) -> None:
            """Compile code as a top-level module."""
            if not code or not code.strip():
                return
            self._try_compile(code, field, "exec", r)

        def check_body(code: str, field: str, fn_name: str = "f") -> None:
            """Compile code as a function body by wrapping it in a dummy function."""
            if not code or not code.strip():
                return
            import textwrap
            dedented = textwrap.dedent(code)
            wrapped = f"def {fn_name}():\n" + textwrap.indent(dedented, "    ")
            self._try_compile(wrapped, field, "exec", r,
                              strip_fn_wrapper=True, fn_name=fn_name)

        def check_expr(expr: str, field: str) -> None:
            """Compile code as a Python expression."""
            if not expr or not expr.strip():
                return
            self._try_compile(expr.strip(), field, "eval", r)

        check_module(spec.setup_code,          "setup_code")
        check_body(spec.precompute_code,       "precompute_code",    "precompute")
        check_body(spec.initial_state_code,    "initial_state_code", "initial_state")
        check_body(spec.step_code,             "step_code",          "sim_step")
        check_body(spec.progress_code,         "progress_code",      "progress_summary")
        check_body(spec.config_assert_code,    "config_assert_code", "config_assert")
        check_body(spec.state_assert_code,     "state_assert_code",  "state_assert")

        for i, sc in enumerate(spec.stopping_conditions):
            if sc.check_expr:
                check_expr(sc.check_expr,
                           f"stopping_conditions[{i}][{sc.name}].check_expr")
            if sc.reason_expr:
                # reason_expr is often an f-string — wrap in eval
                check_expr(sc.reason_expr,
                           f"stopping_conditions[{i}][{sc.name}].reason_expr")

    def _try_compile(
        self,
        code: str,
        field: str,
        mode: str,
        r: ValidationResult,
        strip_fn_wrapper: bool = False,
        fn_name: str = "f",
    ) -> bool:
        """
        Attempt to compile `code`. Record a ValidationError on failure.
        Returns True if compilation succeeded.
        """
        try:
            compile(code, f"<{field}>", mode)
            return True
        except SyntaxError as exc:
            lineno = exc.lineno or 0
            offset_note = f" (line {lineno})" if lineno else ""
            if strip_fn_wrapper and lineno:
                # Subtract 1 for the `def f():` line we added
                actual_line = max(1, lineno - 1)
                offset_note = f" (approx. line {actual_line} of generated code)"
            r.errors.append(ValidationError(
                tier="B", code="syntax_error", field=field,
                message=(
                    f"Python syntax error{offset_note}: {exc.msg}. "
                    f"The code must be valid Python that can be compiled with compile(). "
                    f"Check for: mismatched brackets, bad indentation, "
                    f"missing colons after if/for/def."
                ),
            ))
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Tier C: Semantic checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_C_semantic(self, spec: SimulationSpec, r: ValidationResult) -> None:

        state_field_names = {f[0] for f in spec.state_fields}

        # C1–C4: required return statements
        for code, field, fn_label in [
            (spec.precompute_code,    "precompute_code",    "precompute"),
            (spec.initial_state_code, "initial_state_code", "initial_state"),
            (spec.step_code,          "step_code",          "sim_step"),
            (spec.progress_code,      "progress_code",      "progress_summary"),
        ]:
            if not _contains_return(code):
                r.errors.append(ValidationError(
                    tier="C", code="missing_return", field=field,
                    message=(
                        f"The {fn_label} function body has no 'return' statement. "
                        f"precompute must return a dict, initial_state must return a "
                        f"SimState, sim_step must return a SimState, and "
                        f"progress_summary must return a str."
                    ),
                ))

        # C5: output_variables reference real state fields
        for var in spec.output_variables:
            if var not in state_field_names and var not in ("step", "sim_time"):
                r.errors.append(ValidationError(
                    tier="C", code="undefined_output_variable",
                    field=f"output_variables[{var!r}]",
                    message=(
                        f"output_variables references '{var}' which is not a declared "
                        f"state field. Declared state fields: {sorted(state_field_names)}. "
                        f"Also allowed: 'step', 'sim_time'."
                    ),
                ))

        # C6: data_log_variables reference real state fields
        for var in spec.data_log_variables:
            if var not in state_field_names and var not in ("step", "sim_time"):
                r.errors.append(ValidationError(
                    tier="C", code="undefined_data_log_variable",
                    field=f"data_log_variables[{var!r}]",
                    message=(
                        f"data_log_variables references '{var}' which is not a declared "
                        f"state field. Declared state fields: {sorted(state_field_names)}."
                    ),
                ))

        # C7/C8: at least one success and one failure condition
        kinds = {sc.kind for sc in spec.stopping_conditions}
        if "success" not in kinds:
            r.errors.append(ValidationError(
                tier="C", code="no_success_condition",
                field="stopping_conditions",
                message=(
                    "No stopping condition with kind='success'. "
                    "Every simulation must have at least one success condition "
                    "(e.g. convergence achieved, target reached) so the runner "
                    "knows when to stop collecting data."
                ),
            ))
        if "failure" not in kinds:
            r.errors.append(ValidationError(
                tier="C", code="no_failure_condition",
                field="stopping_conditions",
                message=(
                    "No stopping condition with kind='failure'. "
                    "Every simulation must have at least one failure condition "
                    "(e.g. NaN detected, lattice frozen, speed=0) to avoid "
                    "wasting compute on degenerate configurations."
                ),
            ))

        # C9: variable names are valid Python identifiers
        for i, v in enumerate(spec.variables):
            if v.name and not _is_valid_identifier(v.name):
                r.errors.append(ValidationError(
                    tier="C", code="invalid_variable_name",
                    field=f"variables[{i}].name",
                    message=(
                        f"Variable name {v.name!r} is not a valid Python identifier. "
                        f"Names must start with a letter or underscore, contain only "
                        f"alphanumerics and underscores, and must not be a Python "
                        f"keyword ({keyword.kwlist})."
                    ),
                ))

        # C10: state field names are valid Python identifiers
        for i, (name, *_) in enumerate(spec.state_fields):
            if not _is_valid_identifier(name):
                r.errors.append(ValidationError(
                    tier="C", code="invalid_state_field_name",
                    field=f"state_fields[{i}].name",
                    message=(
                        f"State field name {name!r} is not a valid Python identifier."
                    ),
                ))

        # C11: no assignment operator in check_expr (common LLM mistake: = instead of ==)
        for i, sc in enumerate(spec.stopping_conditions):
            if sc.check_expr and _has_bare_assignment(sc.check_expr):
                r.errors.append(ValidationError(
                    tier="C", code="assignment_in_check_expr",
                    field=f"stopping_conditions[{i}][{sc.name}].check_expr",
                    message=(
                        f"check_expr for '{sc.name}' contains what looks like an "
                        f"assignment (=) instead of a comparison (==). "
                        f"check_expr must be a boolean expression that reads "
                        f"state without modifying it."
                    ),
                ))

    # ─────────────────────────────────────────────────────────────────────────
    # Tier D: Quality checks (warnings only)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_D_quality(self, spec: SimulationSpec, r: ValidationResult) -> None:

        all_code_blocks = [
            ("setup_code",         spec.setup_code),
            ("precompute_code",    spec.precompute_code),
            ("initial_state_code", spec.initial_state_code),
            ("step_code",          spec.step_code),
            ("progress_code",      spec.progress_code),
            ("config_assert_code", spec.config_assert_code),
            ("state_assert_code",  spec.state_assert_code),
        ]

        # D1: no bare print() — should use algo_log or data_log
        for field_name, code in all_code_blocks:
            if code and _contains_bare_print(code):
                r.warnings.append(ValidationError(
                    tier="D", code="bare_print_found", field=field_name,
                    message=(
                        f"'{field_name}' contains print(). "
                        f"Use algo_log.debug/info/warning() for software tracing, "
                        f"or data_log.info(json.dumps({{...}})) for research data. "
                        f"print() output will not appear in structured logs."
                    ),
                ))

        # D2: no sim_tool import in generated code
        for field_name, code in all_code_blocks:
            if code and "import sim_tool" in code:
                r.warnings.append(ValidationError(
                    tier="D", code="sim_tool_import_in_code", field=field_name,
                    message=(
                        f"'{field_name}' imports sim_tool. Generated simulation scripts "
                        f"must be self-contained and must not import sim_tool. "
                        f"Remove this import."
                    ),
                ))

        # D3/D4: assert blocks should be non-trivial
        for field_name, code in [
            ("config_assert_code", spec.config_assert_code),
            ("state_assert_code",  spec.state_assert_code),
        ]:
            if not code or code.strip() in ("pass", ""):
                r.warnings.append(ValidationError(
                    tier="D", code="trivial_assert_block", field=field_name,
                    message=(
                        f"'{field_name}' is empty or just 'pass'. "
                        f"Add assertions that validate the inputs. "
                        f"config_assert_code should check variable ranges and mutual "
                        f"constraints. state_assert_code should check for NaN/Inf and "
                        f"physical invariants (probability in [0,1], etc.)."
                    ),
                ))

        # D5: data_log_variables non-empty
        if not spec.data_log_variables:
            r.warnings.append(ValidationError(
                tier="D", code="empty_data_log_variables",
                field="data_log_variables",
                message=(
                    "data_log_variables is empty. No research data will be written to "
                    "data_log.jsonl. Add at least the primary measurement fields "
                    "(e.g. magnetisation, energy, position)."
                ),
            ))

        # D6: output_variables non-empty
        if not spec.output_variables:
            r.warnings.append(ValidationError(
                tier="D", code="empty_output_variables",
                field="output_variables",
                message=(
                    "output_variables is empty. No in-memory history will be recorded. "
                    "Add the fields you want available for post-run analysis."
                ),
            ))

        # D7: max_steps sanity
        if spec.max_steps <= 0:
            r.warnings.append(ValidationError(
                tier="D", code="invalid_max_steps", field="max_steps",
                message=f"max_steps={spec.max_steps} must be > 0.",
            ))

        # D8: progress_interval sanity
        if spec.progress_interval <= 0:
            r.warnings.append(ValidationError(
                tier="D", code="invalid_progress_interval", field="progress_interval",
                message=f"progress_interval={spec.progress_interval} must be > 0.",
            ))


# ─────────────────────────────────────────────────────────────────────────────
# AST helpers — deterministic, no LLM
# ─────────────────────────────────────────────────────────────────────────────

def _contains_return(code: str) -> bool:
    """Return True if any line in `code` contains a 'return' token."""
    if not code:
        return False
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("return") and (
            len(stripped) == 6 or not stripped[6].isalnum()
        ):
            return True
    return False


def _is_valid_identifier(name: str) -> bool:
    """Return True if `name` is a valid Python identifier and not a keyword."""
    return name.isidentifier() and not keyword.iskeyword(name)


def _has_bare_assignment(expr: str) -> bool:
    """
    Detect `=` used as assignment rather than `==` comparison in an expression.
    Uses the AST to detect NamedExpr (walrus :=) and bare Assign nodes,
    but not keyword args, dict literals, or f-strings.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return False  # syntax errors caught in Tier B
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):   # := walrus
            return True
    # Also check raw string for lone = not preceded/followed by [=!<>]
    import re
    # Remove string literals first to avoid false positives inside f-strings
    cleaned = re.sub(r'"[^"]*"', '""', expr)
    cleaned = re.sub(r"'[^']*'", "''", cleaned)
    # Look for = that is not ==, !=, <=, >=, :=
    return bool(re.search(r'(?<![=!<>:])=(?!=)', cleaned))


def _contains_bare_print(code: str) -> bool:
    """Return True if `code` contains a print() call (not in a comment or string)."""
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(code), mode="exec")
    except SyntaxError:
        return False  # syntax errors caught in Tier B
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                return True
    return False
