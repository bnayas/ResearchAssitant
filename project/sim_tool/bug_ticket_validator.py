"""
sim_tool.bug_ticket_validator
─────────────────────────────
Deterministic validation of BugTicket before it is returned to the director.
Mirrors the role of SpecValidator for the coding agent's output.

Rules
─────
  V1  All required fields are present and non-None.
  V2  kind is a known BugKind enum value.
  V3  exit_code is an integer (not the sentinel -1 unless kind==UNKNOWN).
  V4  reproducible_command is non-empty and starts with a recognisable
      executable (uv, python, python3, bash, sh, /path/to/...).
  V5  stdout_tail and stderr_tail are strings (empty is fine).
  V6  script_path points to a file that exists on disk.
  V7  suggested_fix is non-empty (the launcher must always try).
  V8  is_retryable is consistent with kind:
        retryable kinds:    ENV_SETUP, TIMEOUT, OOM, DISK_FULL
        non-retryable kinds: SYNTAX_ERROR, ASSERTION_ERROR, RUNTIME_ERROR,
                             SIGNAL, IMPORT_ERROR, UNKNOWN

Usage
─────
  from sim_tool.bug_ticket_validator import BugTicketValidator

  validator = BugTicketValidator()
  result = validator.validate(ticket)
  if not result.passed:
      # fix the ticket before returning to director
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .contract import BugKind, BugTicket

log = logging.getLogger("sim_tool.bug_ticket_validator")

# Kinds that mean "retry after fixing the environment"
_RETRYABLE_KINDS = {BugKind.ENV_SETUP, BugKind.TIMEOUT, BugKind.OOM, BugKind.DISK_FULL}
_NONRETRYABLE_KINDS = {
    BugKind.SYNTAX_ERROR, BugKind.ASSERTION_ERROR, BugKind.RUNTIME_ERROR,
    BugKind.SIGNAL, BugKind.IMPORT_ERROR, BugKind.UNKNOWN,
}

# Executables that are acceptable as the first token of reproducible_command
_VALID_EXECUTABLES = {"uv", "python", "python3", "bash", "sh"}


@dataclass(frozen=True)
class TicketValidationError:
    code:    str
    field:   str
    message: str

    def __str__(self) -> str:
        return f"  [{self.code}] field={self.field}\n    {self.message}"


@dataclass
class TicketValidationResult:
    errors: list[TicketValidationError] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        if self.passed:
            return "BugTicket validation passed."
        lines = [f"BugTicket validation FAILED ({len(self.errors)} error(s)):"]
        for e in self.errors:
            lines.append(str(e))
        return "\n".join(lines)


class BugTicketValidator:

    def validate(self, ticket: BugTicket) -> TicketValidationResult:
        r = TicketValidationResult()

        # V1: required fields present
        for f_name in ("kind", "session_id", "script_path",
                        "exit_code", "reproducible_command"):
            val = getattr(ticket, f_name, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                r.errors.append(TicketValidationError(
                    code="missing_required_field", field=f_name,
                    message=(
                        f"BugTicket.{f_name} is required but is empty or None. "
                        f"Every ticket must have a complete identity."
                    ),
                ))

        # V2: kind is a known value (already enforced by enum, but be explicit)
        if not isinstance(ticket.kind, BugKind):
            r.errors.append(TicketValidationError(
                code="invalid_kind", field="kind",
                message=f"kind must be a BugKind enum value, got {ticket.kind!r}.",
            ))

        # V3: exit_code
        if ticket.exit_code == -1 and ticket.kind != BugKind.UNKNOWN:
            r.errors.append(TicketValidationError(
                code="sentinel_exit_code", field="exit_code",
                message=(
                    f"exit_code is -1 (sentinel) but kind is {ticket.kind.value!r}. "
                    f"The sentinel -1 is only valid for kind=UNKNOWN when the process "
                    f"exit code could not be determined."
                ),
            ))

        # V4: reproducible_command starts with a known executable or absolute path
        cmd = ticket.reproducible_command.strip() if ticket.reproducible_command else ""
        if cmd:
            first_token = cmd.split()[0] if cmd.split() else ""
            first_name = Path(first_token).name  # handles /usr/bin/python3 → python3
            if first_name not in _VALID_EXECUTABLES:
                r.errors.append(TicketValidationError(
                    code="unrecognised_executable", field="reproducible_command",
                    message=(
                        f"reproducible_command starts with {first_token!r} "
                        f"which is not a recognised executable. "
                        f"Valid first tokens: {sorted(_VALID_EXECUTABLES)}. "
                        f"Use absolute paths for non-standard interpreters."
                    ),
                ))

        # V5: tail fields are strings
        for f_name in ("stdout_tail", "stderr_tail"):
            if not isinstance(getattr(ticket, f_name, None), str):
                r.errors.append(TicketValidationError(
                    code="tail_not_string", field=f_name,
                    message=f"{f_name} must be a str (empty string is fine).",
                ))

        # V6: script_path must exist on disk
        if ticket.script_path and not Path(ticket.script_path).exists():
            r.errors.append(TicketValidationError(
                code="script_not_found", field="script_path",
                message=(
                    f"script_path {ticket.script_path!r} does not exist on disk. "
                    f"The ticket must reference the actual script that was run."
                ),
            ))

        # V7: suggested_fix is non-empty
        if not ticket.suggested_fix.strip():
            r.errors.append(TicketValidationError(
                code="empty_suggested_fix", field="suggested_fix",
                message=(
                    "suggested_fix must be non-empty. The launcher must always "
                    "provide at least a deterministic hint, even if it's "
                    "'Inspect the stderr_tail above for the root cause.'"
                ),
            ))

        # V8: is_retryable consistent with kind
        if isinstance(ticket.kind, BugKind):
            expected_retryable = ticket.kind in _RETRYABLE_KINDS
            if ticket.is_retryable != expected_retryable:
                r.errors.append(TicketValidationError(
                    code="retryable_mismatch", field="is_retryable",
                    message=(
                        f"is_retryable={ticket.is_retryable} is inconsistent with "
                        f"kind={ticket.kind.value!r}. "
                        f"Retryable kinds: {[k.value for k in _RETRYABLE_KINDS]}. "
                        f"Non-retryable: {[k.value for k in _NONRETRYABLE_KINDS]}. "
                        f"Expected is_retryable={expected_retryable}."
                    ),
                ))

        if not r.passed:
            log.warning(
                f"BugTicket for session={ticket.session_id!r} "
                f"failed validation: {len(r.errors)} error(s)"
            )
        return r
