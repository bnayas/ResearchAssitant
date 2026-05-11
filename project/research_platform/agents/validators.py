"""
research_platform.agents.validators
───────────────────────────────────
Reusable validator factories for ToolRequirement.

Each factory returns a ``Callable[[Any], bool]`` suitable for the
``validator`` field of ``ToolRequirement``.  Validators are pure
functions — no side effects, no I/O.
"""
from __future__ import annotations

from typing import Any, Callable

from ..contracts import ArtifactRef


# ---------------------------------------------------------------------------
# String validators
# ---------------------------------------------------------------------------

def is_non_empty_string(value: Any) -> bool:
    """Value must be a non-empty string after stripping whitespace."""
    return isinstance(value, str) and bool(value.strip())


def string_max_length(max_len: int) -> Callable[[Any], bool]:
    """Value must be a string no longer than ``max_len``."""
    def _check(value: Any) -> bool:
        return isinstance(value, str) and len(value) <= max_len
    return _check


def string_matches_any(*allowed: str) -> Callable[[Any], bool]:
    """Value must be one of the allowed strings (case-insensitive)."""
    lowered = {s.lower() for s in allowed}
    def _check(value: Any) -> bool:
        return isinstance(value, str) and value.strip().lower() in lowered
    return _check


# ---------------------------------------------------------------------------
# Numeric validators
# ---------------------------------------------------------------------------

def is_positive_number(value: Any) -> bool:
    """Value must be a positive int or float."""
    return isinstance(value, (int, float)) and value > 0


def number_in_range(lo: float, hi: float) -> Callable[[Any], bool]:
    """Value must be a number in ``[lo, hi]``."""
    def _check(value: Any) -> bool:
        return isinstance(value, (int, float)) and lo <= value <= hi
    return _check


# ---------------------------------------------------------------------------
# ArtifactRef validators
# ---------------------------------------------------------------------------

def artifact_kind_is(kind: str) -> Callable[[Any], bool]:
    """ArtifactRef must have the specified ``kind``."""
    def _check(value: Any) -> bool:
        return isinstance(value, ArtifactRef) and value.kind == kind
    return _check


def artifact_has_metadata_key(key: str) -> Callable[[Any], bool]:
    """ArtifactRef must have ``key`` present in its ``metadata``."""
    def _check(value: Any) -> bool:
        return isinstance(value, ArtifactRef) and key in value.metadata
    return _check


def artifact_has_content(value: Any) -> bool:
    """ArtifactRef must have non-empty ``content`` or a valid ``path``."""
    if not isinstance(value, ArtifactRef):
        return False
    if value.content and value.content.strip():
        return True
    if value.path:
        from pathlib import Path
        return Path(value.path).exists()
    return False


# ---------------------------------------------------------------------------
# Collection validators
# ---------------------------------------------------------------------------

def list_non_empty(value: Any) -> bool:
    """Value must be a non-empty list."""
    return isinstance(value, list) and len(value) > 0


def list_min_length(min_len: int) -> Callable[[Any], bool]:
    """List must have at least ``min_len`` elements."""
    def _check(value: Any) -> bool:
        return isinstance(value, list) and len(value) >= min_len
    return _check


def list_max_length(max_len: int) -> Callable[[Any], bool]:
    """List must have at most ``max_len`` elements."""
    def _check(value: Any) -> bool:
        return isinstance(value, list) and len(value) <= max_len
    return _check


# ---------------------------------------------------------------------------
# Dict validators
# ---------------------------------------------------------------------------

def dict_has_keys(*keys: str) -> Callable[[Any], bool]:
    """Dict must contain all specified keys."""
    required = set(keys)
    def _check(value: Any) -> bool:
        return isinstance(value, dict) and required.issubset(value.keys())
    return _check


def dict_non_empty(value: Any) -> bool:
    """Value must be a non-empty dict."""
    return isinstance(value, dict) and len(value) > 0


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------

def all_of(*validators: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """All validators must pass."""
    def _check(value: Any) -> bool:
        return all(v(value) for v in validators)
    return _check


def any_of(*validators: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """At least one validator must pass."""
    def _check(value: Any) -> bool:
        return any(v(value) for v in validators)
    return _check
