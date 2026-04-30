"""
research_platform.assistants
────────────────────────────
Local assistant adapters used by the professor workflow.
"""

from .backends import build_literature_review_bridge_from_env, make_service_backend
from .coding import LocalCodingAssistant
from .errors import AssistantExecutionError
from .literature import LocalLiteratureAssistant
from .writer import LocalWriterAssistant

__all__ = [
    "AssistantExecutionError",
    "LocalCodingAssistant",
    "LocalLiteratureAssistant",
    "LocalWriterAssistant",
    "build_literature_review_bridge_from_env",
    "make_service_backend",
]
