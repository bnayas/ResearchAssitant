"""
sim_tool
════════
"""
from .contract import (
    ClarificationRequest, ClarificationQuestion, QuestionTopic,
    SpecApproval, OutputContract,
    GeneratedArtifacts,
    RunResult, RunSummary, RunOutcome,
    AnalysisResult, SpecPatch, PatchChange, Verdict,
    Flag,
)
from .llm import (
    LLMBackend, LLMError, LLMServiceConfig,
    AnthropicBackend, OpenAICompatibleBackend, make_backend, resolve_service_config,
)
from .memory import AgentMemory
from .models import SimulationSpec, Variable, VariableKind, StoppingCondition
from .designer import SimulationDesigner
from .codegen import generate_script, generate_notebook
from .runner import run_script, build_sweep_configs
from .analyst import RunAnalyst, RunMetrics
from .symbolic_types import (
    SymbolDef, Equation, Assumption, UnresolvedGap, SymbolicSpec,
    StatementKind, GapSeverity, SymbolicResult,
    SymbolicValidationResult, SymbolicValidationError,
)
from .symbolic_validator import SymbolicValidator
from .symbolic_agent import SymbolicAgent
from .bug_ticket_validator import BugTicketValidator, TicketValidationResult
from .launcher_log import LauncherLog
from .launcher import Launcher, LaunchRuntimeConfig, LaunchResult, write_pipeline_script
from .contract import BugTicket, BugKind, EnvironmentInfo, LaunchResult
from .contract_validator import SpecValidator, ValidationResult, ValidationError
from .tool import SimulationTool
from .orchestrator import ResearchOrchestrator, OrchestratorUpdate, OrchestratorState, SweepProgress

__all__ = [
    "SimulationTool",
    "ClarificationRequest", "ClarificationQuestion", "QuestionTopic",
    "SpecApproval", "OutputContract",
    "GeneratedArtifacts",
    "RunResult", "RunSummary", "RunOutcome",
    "AnalysisResult", "SpecPatch", "PatchChange", "Verdict", "Flag",
    "LLMBackend", "LLMError", "LLMServiceConfig",
    "AnthropicBackend", "OpenAICompatibleBackend", "make_backend", "resolve_service_config",
    "AgentMemory",
    "SimulationSpec", "Variable", "VariableKind", "StoppingCondition",
    "SimulationDesigner", "generate_script", "generate_notebook",
    "run_script", "build_sweep_configs",
    "RunAnalyst", "RunMetrics",
    # Validator
    "SpecValidator", "ValidationResult", "ValidationError",
    # Launcher
    "Launcher", "LaunchRuntimeConfig", "LaunchResult", "BugTicket", "BugKind", "EnvironmentInfo",
    "BugTicketValidator", "TicketValidationResult", "LauncherLog",
    # Symbolic agent
    "SymbolicAgent", "SymbolicSpec", "SymbolicResult", "SymbolicValidator",
    "SymbolDef", "Equation", "Assumption", "UnresolvedGap",
    "StatementKind", "GapSeverity",
    "SymbolicValidationResult", "SymbolicValidationError",
    # Orchestrator
    "ResearchOrchestrator", "OrchestratorUpdate", "OrchestratorState", "SweepProgress",
]
