"""
sim_tool.symbolic_types
────────────────────────
Typed schemas for the SymbolicAgent output.
Stub — full implementation uses SymPy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StatementKind(str, Enum):
    EQUATION   = "equation"
    INEQUALITY = "inequality"
    DEFINITION = "definition"


class GapSeverity(str, Enum):
    BLOCKING  = "blocking"
    ADVISORY  = "advisory"


@dataclass
class SymbolDef:
    name: str
    description: str
    latex: str = ""
    unit: str = ""


@dataclass
class Equation:
    latex: str
    description: str
    kind: StatementKind = StatementKind.EQUATION


@dataclass
class Assumption:
    text: str
    justification: str = ""


@dataclass
class UnresolvedGap:
    description: str
    severity: GapSeverity = GapSeverity.ADVISORY


@dataclass
class SymbolicSpec:
    symbols: list[SymbolDef] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    gaps: list[UnresolvedGap] = field(default_factory=list)
    downstream_instruction: str = ""


@dataclass
class SymbolicResult:
    session_id: str
    spec: Optional[SymbolicSpec] = None
    raw_output: str = ""


@dataclass
class SymbolicValidationError:
    code: str
    field: str
    message: str


@dataclass
class SymbolicValidationResult:
    errors: list[SymbolicValidationError] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0
