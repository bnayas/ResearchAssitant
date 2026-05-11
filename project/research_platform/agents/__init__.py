"""
research_platform.agents
────────────────────────
Tool-first agent architecture.

Every agent exposes its capabilities as typed ToolDescriptors with
machine-validated requirements. The orchestrator discovers, plans,
validates, and dispatches tools generically.
"""

from .base import (
    BaseAgent,
    ToolContext,
    ToolDescriptor,
    ToolRequirement,
    ToolResult,
)
from .validators import (
    artifact_has_content,
    artifact_has_metadata_key,
    artifact_kind_is,
    is_non_empty_string,
    is_positive_number,
)

__all__ = [
    "BaseAgent",
    "ToolContext",
    "ToolDescriptor",
    "ToolRequirement",
    "ToolResult",
    "artifact_has_content",
    "artifact_has_metadata_key",
    "artifact_kind_is",
    "is_non_empty_string",
    "is_positive_number",
]
