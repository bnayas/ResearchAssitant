"""
test_bug_ticket_validator.py
─────────────────────────────
Unit tests for BugTicketValidator — the deterministic 8-check gate
that validates BugTicket objects before they are returned to the Director.

Mirrors the structure of test_contract_validator.py.

Run:  python test_bug_ticket_validator.py
      or: python -m pytest test_bug_ticket_validator.py -v
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.bug_ticket_validator import BugTicketValidator, TicketValidationError, TicketValidationResult
except ImportError:
    from bug_ticket_validator import BugTicketValidator, TicketValidationError, TicketValidationResult
try:
    from sim_tool.contract import BugKind, BugTicket
except ImportError:
    from contract import BugKind, BugTicket


validator = BugTicketValidator()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_valid_ticket(tmp_script: Path, **overrides) -> BugTicket:
    """Build the minimal BugTicket that passes all 8 checks."""
    defaults = dict(
        kind=BugKind.RUNTIME_ERROR,
        session_id="sess-abc123",
        script_path=str(tmp_script),
        exit_code=1,
        reproducible_command=f"python {tmp_script} --temperature 2.5",
        stdout_tail="",
        stderr_tail="Traceback (most recent call last):\n  RuntimeError: bad state",
        failure_pattern=r"(Error|Exception|Traceback).*\n.*\n.*",
        error_line="RuntimeError: bad state at step 100",
        suggested_fix="Check the step_code for out-of-bounds access.",
        is_retryable=False,
        wall_time_seconds=3.5,
    )
    defaults.update(overrides)
    return BugTicket(**defaults)


def errors_with_code(result: TicketValidationResult, code: str) -> list:
    return [e for e in result.errors if e.code == code]


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_ticket_passes(tmp_path):
    script = tmp_path / "sim.py"
    script.write_text("# sim")
    ticket = _make_valid_ticket(script)
    result = validator.validate(ticket)
    assert result.passed, f"Expected pass, got:\n{result.summary()}"


# ─────────────────────────────────────────────────────────────────────────────
# V1: Required fields
# ─────────────────────────────────────────────────────────────────────────────

def test_V1_missing_session_id(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, session_id="")
    result = validator.validate(ticket)
    assert not result.passed
    assert errors_with_code(result, "missing_required_field")


def test_V1_missing_script_path(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, script_path="")
    result = validator.validate(ticket)
    assert not result.passed
    assert errors_with_code(result, "missing_required_field")


def test_V1_missing_reproducible_command(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command="")
    result = validator.validate(ticket)
    assert not result.passed
    assert errors_with_code(result, "missing_required_field")


# ─────────────────────────────────────────────────────────────────────────────
# V2: kind is valid BugKind
# ─────────────────────────────────────────────────────────────────────────────

def test_V2_all_valid_bugkinds_pass(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    retryable_map = {
        BugKind.ENV_SETUP: True,
        BugKind.TIMEOUT: True,
        BugKind.OOM: True,
        BugKind.DISK_FULL: True,
        BugKind.RUNTIME_ERROR: False,
        BugKind.SYNTAX_ERROR: False,
        BugKind.ASSERTION_ERROR: False,
        BugKind.SIGNAL: False,
        BugKind.IMPORT_ERROR: False,
        BugKind.UNKNOWN: False,
    }
    for kind, retryable in retryable_map.items():
        # UNKNOWN uses sentinel exit code
        ec = -1 if kind == BugKind.UNKNOWN else 1
        ticket = _make_valid_ticket(
            script, kind=kind, is_retryable=retryable, exit_code=ec
        )
        result = validator.validate(ticket)
        assert result.passed, f"kind={kind.value} should pass. Got:\n{result.summary()}"


# ─────────────────────────────────────────────────────────────────────────────
# V3: exit_code sentinel check
# ─────────────────────────────────────────────────────────────────────────────

def test_V3_sentinel_minus1_for_unknown_is_ok(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, kind=BugKind.UNKNOWN, exit_code=-1, is_retryable=False)
    result = validator.validate(ticket)
    assert result.passed, result.summary()


def test_V3_sentinel_minus1_for_runtime_error_fails(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, kind=BugKind.RUNTIME_ERROR, exit_code=-1, is_retryable=False)
    result = validator.validate(ticket)
    assert not result.passed
    assert errors_with_code(result, "sentinel_exit_code")


def test_V3_sentinel_minus1_for_env_setup_fails(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, kind=BugKind.ENV_SETUP, exit_code=-1, is_retryable=True)
    result = validator.validate(ticket)
    assert not result.passed
    assert errors_with_code(result, "sentinel_exit_code")


# ─────────────────────────────────────────────────────────────────────────────
# V4: reproducible_command starts with recognised executable
# ─────────────────────────────────────────────────────────────────────────────

def test_V4_python_prefix_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command=f"python {script} --N 32")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "unrecognised_executable")


def test_V4_python3_prefix_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command=f"python3 {script}")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "unrecognised_executable")


def test_V4_uv_prefix_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command=f"uv run python {script}")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "unrecognised_executable")


def test_V4_bash_prefix_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command=f"bash run_pipeline.sh")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "unrecognised_executable")


def test_V4_unknown_prefix_fails(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command=f"node run.js {script}")
    result = validator.validate(ticket)
    assert errors_with_code(result, "unrecognised_executable")


def test_V4_absolute_python_path_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, reproducible_command=f"/usr/bin/python3 {script}")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "unrecognised_executable")


# ─────────────────────────────────────────────────────────────────────────────
# V5: stdout_tail and stderr_tail are strings
# ─────────────────────────────────────────────────────────────────────────────

def test_V5_empty_tails_are_valid(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, stdout_tail="", stderr_tail="")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "tail_not_string")


def test_V5_none_stdout_tail_fails(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script)
    ticket.stdout_tail = None  # inject invalid type
    result = validator.validate(ticket)
    assert errors_with_code(result, "tail_not_string")


# ─────────────────────────────────────────────────────────────────────────────
# V6: script_path exists on disk
# ─────────────────────────────────────────────────────────────────────────────

def test_V6_existing_script_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("# hello")
    ticket = _make_valid_ticket(script)
    result = validator.validate(ticket)
    assert not errors_with_code(result, "script_not_found")


def test_V6_nonexistent_script_fails(tmp_path):
    script = tmp_path / "does_not_exist.py"  # not created
    ticket = _make_valid_ticket(script)
    result = validator.validate(ticket)
    assert errors_with_code(result, "script_not_found")


# ─────────────────────────────────────────────────────────────────────────────
# V7: suggested_fix is non-empty
# ─────────────────────────────────────────────────────────────────────────────

def test_V7_empty_fix_fails(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, suggested_fix="")
    result = validator.validate(ticket)
    assert not result.passed
    assert errors_with_code(result, "empty_suggested_fix")


def test_V7_whitespace_fix_fails(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, suggested_fix="   ")
    result = validator.validate(ticket)
    assert errors_with_code(result, "empty_suggested_fix")


def test_V7_nonempty_fix_passes(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, suggested_fix="Inspect the stderr_tail.")
    result = validator.validate(ticket)
    assert not errors_with_code(result, "empty_suggested_fix")


# ─────────────────────────────────────────────────────────────────────────────
# V8: is_retryable consistent with kind
# ─────────────────────────────────────────────────────────────────────────────

def test_V8_retryable_kinds_must_be_true(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    for kind in (BugKind.ENV_SETUP, BugKind.TIMEOUT, BugKind.OOM, BugKind.DISK_FULL):
        ticket = _make_valid_ticket(script, kind=kind, is_retryable=False, exit_code=1)
        result = validator.validate(ticket)
        assert errors_with_code(result, "retryable_mismatch"), \
            f"{kind.value} with is_retryable=False should fail V8"


def test_V8_nonretryable_kinds_must_be_false(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    for kind in (BugKind.RUNTIME_ERROR, BugKind.SYNTAX_ERROR, BugKind.ASSERTION_ERROR):
        ticket = _make_valid_ticket(script, kind=kind, is_retryable=True, exit_code=1)
        result = validator.validate(ticket)
        assert errors_with_code(result, "retryable_mismatch"), \
            f"{kind.value} with is_retryable=True should fail V8"


# ─────────────────────────────────────────────────────────────────────────────
# Summary format
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_passed(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script)
    result = validator.validate(ticket)
    assert "passed" in result.summary().lower()


def test_summary_failed_contains_error_count(tmp_path):
    script = tmp_path / "sim.py"; script.write_text("")
    ticket = _make_valid_ticket(script, suggested_fix="", session_id="")
    result = validator.validate(ticket)
    summary = result.summary()
    assert "FAILED" in summary or "failed" in summary.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback, inspect

    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"  ✓ {name}")
                passed += 1
            except Exception as exc:
                print(f"  ✗ {name}: {exc}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
