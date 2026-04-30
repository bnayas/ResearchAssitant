"""
sim_tool.symbolic_agent
────────────────────────
SymPy-backed symbolic formalization agent.
Converts a simulation description into a SymbolicSpec via LLM + CAS validation.
Stub — full implementation requires SymPy and LLM backend.
"""
from __future__ import annotations
from typing import Optional
from .symbolic_types import SymbolicResult, SymbolicSpec
from .symbolic_validator import SymbolicValidator
from .llm import LLMBackend, make_backend


class SymbolicAgent:
    """Formalizes simulation goals into mathematical specifications."""

    def __init__(self, backend: Optional[LLMBackend] = None) -> None:
        self._backend = backend or make_backend(service="symbolic")
        self._validator = SymbolicValidator()

    def formalize(self, description: str, session_id: str = "") -> SymbolicResult:
        """
        Convert a natural-language simulation description into a SymbolicSpec.
        Returns SymbolicResult with validated spec or error information.
        """
        raise NotImplementedError(
            "SymbolicAgent.formalize() requires full SymPy implementation. "
            "See symbolic_types.py and symbolic_validator.py for the contract."
        )
