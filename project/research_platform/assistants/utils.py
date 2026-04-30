from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def truncate(text: str, *, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonify(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: jsonify(getattr(value, field.name))
            for field in fields(value)
        }
    return str(value)
