"""
tests/test_contract_validator.py
─────────────────────────────────
Unit tests for SpecValidator — the deterministic gate between
the coding agent and the director.

Each test:
  - Builds a minimal valid spec (or a deliberately broken one)
  - Runs the validator
  - Asserts exactly the expected errors / no errors

Run with:  python -m pytest tests/ -v
       or: python tests/test_contract_validator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tool.contract_validator import SpecValidator, ValidationError
from sim_tool.models import SimulationSpec, Variable, VariableKind, StoppingCondition


# ─────────────────────────────────────────────────────────────────────────────
# Minimal valid spec fixture
# ─────────────────────────────────────────────────────────────────────────────

def _make_valid_spec(**overrides) -> SimulationSpec:
    """Build the minimal spec that passes all validation tiers."""
    defaults = dict(
        name="Test Simulation",
        description="A simple test simulation for validator unit tests.",
        variables=[
            Variable(
                name="dt", description="Time step", kind=VariableKind.FLOAT,
                default=0.01, min_val=1e-5, max_val=1.0, unit="s",
            ),
            Variable(
                name="n_steps", description="Number of steps", kind=VariableKind.INT,
                default=100, min_val=1, max_val=100000,
            ),
        ],
        stopping_conditions=[
            StoppingCondition(
                kind="success", name="target_reached",
                description="Position reached target.",
                check_expr="state.x >= config.target",
                reason_expr="f'Reached x={state.x:.4f}'",
                priority=10,
            ),
            StoppingCondition(
                kind="failure", name="diverged",
                description="Position went to infinity.",
                check_expr="not math.isfinite(state.x)",
                reason_expr="f'x diverged to {state.x}'",
                priority=20,
            ),
        ],
        state_fields=[
            ("x",     "float", 0.0),
            ("v",     "float", 0.0),
            ("energy","float", 0.0),
        ],
        setup_code="import math\n",
        precompute_code="    return {'target': config.n_steps * config.dt}\n",
        initial_state_code=(
            "    state = SimState()\n"
            "    state.x = 0.0\n"
            "    state.v = 1.0\n"
            "    state.energy = 0.5\n"
            "    return state\n"
        ),
        step_code=(
            "    new = copy.copy(state)\n"
            "    new.x = state.x + state.v * config.dt\n"
            "    new.sim_time = state.sim_time + config.dt\n"
            "    return new\n"
        ),
        progress_code=(
            "    return f'x={state.x:.3f}  v={state.v:.3f}'\n"
        ),
        config_assert_code=(
            "    assert config.dt > 0, f'dt must be positive, got {config.dt}'\n"
            "    assert config.n_steps > 0, f'n_steps must be positive, got {config.n_steps}'\n"
        ),
        state_assert_code=(
            "    import math\n"
            "    assert math.isfinite(state.x), f'x diverged: {state.x}'\n"
            "    assert math.isfinite(state.v), f'v diverged: {state.v}'\n"
        ),
        output_variables=["x", "v", "energy"],
        data_log_variables=["x", "v", "energy"],
        data_log_interval=10,
        checkpoint_interval=100,
        max_steps=1000,
        progress_interval=100,
        time_estimate_seconds=1.0,
        time_estimate_explanation="quick test",
    )
    defaults.update(overrides)
    return SimulationSpec(**defaults)


validator = SpecValidator()


def errors_with_code(result, code: str) -> list[ValidationError]:
    return [e for e in result.errors if e.code == code]

def warnings_with_code(result, code: str) -> list[ValidationError]:
    return [w for w in result.warnings if w.code == code]


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_spec_passes():
    spec = _make_valid_spec()
    result = validator.validate(spec)
    assert result.passed, f"Expected pass, got:\n{result}"


# ─────────────────────────────────────────────────────────────────────────────
# Tier A: Presence checks
# ─────────────────────────────────────────────────────────────────────────────

def test_A_missing_name():
    spec = _make_valid_spec(name="")
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_field"), result


def test_A_missing_description():
    spec = _make_valid_spec(description="")
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_field"), result


def test_A_empty_variables():
    spec = _make_valid_spec(variables=[])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_field"), result


def test_A_variable_missing_default():
    v = Variable(name="x", description="position", kind=VariableKind.FLOAT, default=None)
    spec = _make_valid_spec(variables=[v])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_variable_default"), result


def test_A_missing_check_expr():
    sc = StoppingCondition(
        kind="success", name="done", description="done",
        check_expr="",        # MISSING
        reason_expr="'done'",
    )
    sc2 = StoppingCondition(
        kind="failure", name="fail", description="fail",
        check_expr="False", reason_expr="'fail'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc, sc2])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_check_expr"), result


def test_A_missing_reason_expr():
    sc = StoppingCondition(
        kind="success", name="done", description="done",
        check_expr="True",
        reason_expr="",      # MISSING
    )
    sc2 = StoppingCondition(
        kind="failure", name="fail", description="fail",
        check_expr="False", reason_expr="'fail'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc, sc2])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_reason_expr"), result


# ─────────────────────────────────────────────────────────────────────────────
# Tier B: Syntax checks
# ─────────────────────────────────────────────────────────────────────────────

def test_B_step_code_syntax_error():
    spec = _make_valid_spec(
        step_code=(
            "    new = copy.copy(state)\n"
            "    if True\n"          # <- missing colon
            "        pass\n"
            "    return new\n"
        )
    )
    result = validator.validate(spec)
    assert not result.passed
    errs = errors_with_code(result, "syntax_error")
    assert errs, f"Expected syntax_error, got: {result.errors}"
    assert any("step_code" in e.field for e in errs)


def test_B_precompute_syntax_error():
    spec = _make_valid_spec(precompute_code="    return {{{bad\n")
    result = validator.validate(spec)
    assert not result.passed
    errs = errors_with_code(result, "syntax_error")
    assert any("precompute_code" in e.field for e in errs)


def test_B_check_expr_syntax_error():
    sc = StoppingCondition(
        kind="success", name="done", description="desc",
        check_expr="state.x >=",   # incomplete expression
        reason_expr="'done'",
    )
    sc2 = StoppingCondition(
        kind="failure", name="fail", description="fail",
        check_expr="False", reason_expr="'fail'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc, sc2])
    result = validator.validate(spec)
    assert not result.passed
    errs = errors_with_code(result, "syntax_error")
    assert any("check_expr" in e.field for e in errs), f"check_expr error not found: {result.errors}"


def test_B_setup_code_valid_module():
    # setup_code may contain function definitions — should compile as module
    spec = _make_valid_spec(
        setup_code=(
            "import math\n"
            "import random\n\n"
            "def helper(x):\n"
            "    return math.sqrt(abs(x))\n"
        )
    )
    result = validator.validate(spec)
    assert result.passed, f"Expected pass, got:\n{result}"


# ─────────────────────────────────────────────────────────────────────────────
# Tier C: Semantic checks
# ─────────────────────────────────────────────────────────────────────────────

def test_C_precompute_missing_return():
    spec = _make_valid_spec(
        precompute_code="    x = 1\n    y = 2\n"  # no return
    )
    result = validator.validate(spec)
    assert not result.passed
    errs = errors_with_code(result, "missing_return")
    assert any("precompute_code" in e.field for e in errs)


def test_C_step_code_missing_return():
    spec = _make_valid_spec(
        step_code="    new = copy.copy(state)\n"  # no return
    )
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "missing_return")


def test_C_undefined_output_variable():
    spec = _make_valid_spec(output_variables=["x", "nonexistent_field"])
    result = validator.validate(spec)
    assert not result.passed
    errs = errors_with_code(result, "undefined_output_variable")
    assert errs, f"Expected undefined_output_variable error"
    assert any("nonexistent_field" in e.field for e in errs)


def test_C_undefined_data_log_variable():
    spec = _make_valid_spec(data_log_variables=["x", "ghost_field"])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "undefined_data_log_variable")


def test_C_step_and_sim_time_are_allowed_in_output():
    """step and sim_time are built-in state fields and always valid."""
    spec = _make_valid_spec(output_variables=["step", "sim_time", "x"])
    result = validator.validate(spec)
    assert result.passed, f"step and sim_time should be allowed. Got:\n{result}"


def test_C_no_success_condition():
    sc = StoppingCondition(
        kind="failure", name="fail", description="fail",
        check_expr="False", reason_expr="'fail'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "no_success_condition")


def test_C_no_failure_condition():
    sc = StoppingCondition(
        kind="success", name="done", description="done",
        check_expr="True", reason_expr="'done'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "no_failure_condition")


def test_C_invalid_variable_name():
    v = Variable(
        name="my var",   # space — invalid identifier
        description="test", kind=VariableKind.FLOAT, default=1.0,
    )
    spec = _make_valid_spec(variables=[v])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "invalid_variable_name")


def test_C_keyword_variable_name():
    v = Variable(
        name="for",   # Python keyword
        description="test", kind=VariableKind.FLOAT, default=1.0,
    )
    spec = _make_valid_spec(variables=[v])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "invalid_variable_name")


def test_C_assignment_in_check_expr():
    sc_bad = StoppingCondition(
        kind="success", name="done", description="done",
        check_expr="state.x = 5",   # = instead of ==
        reason_expr="'done'",
    )
    sc2 = StoppingCondition(
        kind="failure", name="fail", description="fail",
        check_expr="False", reason_expr="'fail'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc_bad, sc2])
    result = validator.validate(spec)
    # May fail B (syntax) or C (assignment) — at minimum not pass
    assert not result.passed


def test_C_walrus_in_check_expr():
    sc_bad = StoppingCondition(
        kind="success", name="done", description="done",
        check_expr="(y := state.x) > 5",  # walrus operator — side effect
        reason_expr="'done'",
    )
    sc2 = StoppingCondition(
        kind="failure", name="fail", description="fail",
        check_expr="False", reason_expr="'fail'",
    )
    spec = _make_valid_spec(stopping_conditions=[sc_bad, sc2])
    result = validator.validate(spec)
    assert not result.passed
    assert errors_with_code(result, "assignment_in_check_expr")


# ─────────────────────────────────────────────────────────────────────────────
# Tier D: Quality warnings (non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

def test_D_bare_print_is_warning_not_error():
    spec = _make_valid_spec(
        step_code=(
            "    new = copy.copy(state)\n"
            "    print(f'step {state.step}')\n"   # should use algo_log
            "    return new\n"
        )
    )
    result = validator.validate(spec)
    # Should PASS (warning only)
    assert result.passed, f"print() should be a warning, not a blocking error. Got:\n{result}"
    assert warnings_with_code(result, "bare_print_found"), "Expected bare_print_found warning"


def test_D_sim_tool_import_is_warning():
    spec = _make_valid_spec(
        setup_code="import math\nimport sim_tool\n"
    )
    result = validator.validate(spec)
    assert result.passed, "sim_tool import should warn, not block"
    assert warnings_with_code(result, "sim_tool_import_in_code")


def test_D_trivial_assert_is_warning():
    spec = _make_valid_spec(config_assert_code="    pass\n")
    result = validator.validate(spec)
    assert result.passed, "trivial assert should warn not block"
    assert warnings_with_code(result, "trivial_assert_block")


def test_D_empty_data_log_variables_is_warning():
    spec = _make_valid_spec(data_log_variables=[])
    result = validator.validate(spec)
    assert result.passed
    assert warnings_with_code(result, "empty_data_log_variables")


def test_D_multiple_warnings_still_passes():
    spec = _make_valid_spec(
        config_assert_code="    pass\n",
        state_assert_code="    pass\n",
        data_log_variables=[],
        output_variables=[],
    )
    result = validator.validate(spec)
    assert result.passed, f"Multiple warnings should still pass. Got:\n{result}"
    assert len(result.warnings) >= 4


# ─────────────────────────────────────────────────────────────────────────────
# Error summary format (used to feed back to LLM)
# ─────────────────────────────────────────────────────────────────────────────

def test_error_summary_is_self_contained():
    """The error summary must make sense without access to the source code."""
    spec = _make_valid_spec(
        step_code="    pass\n",                   # missing return
        output_variables=["x", "nonexistent"],    # bad reference
        stopping_conditions=[
            StoppingCondition(
                kind="success", name="done", description="done",
                check_expr="True", reason_expr="'done'",
            )
        # no failure condition
        ],
    )
    result = validator.validate(spec)
    assert not result.passed

    summary = result.error_summary()
    assert "VALIDATION FAILED" in summary
    assert "missing_return" in summary or "return" in summary.lower()
    assert "nonexistent" in summary
    assert "failure" in summary.lower()
    # Must not reference line numbers of the outer script
    assert "designer.py" not in summary
    assert "contract_validator.py" not in summary
    print("\nError summary (shown to LLM for correction):")
    print(summary)


# ─────────────────────────────────────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  ✗ {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ! {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed+failed} tests.")
    sys.exit(0 if failed == 0 else 1)
