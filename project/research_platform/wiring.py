"""
research_platform.wiring
────────────────────────
Wiring layer for the professor workflow and literature-review bridge helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .assistants import build_literature_review_bridge_from_env, make_service_backend
from .flow_management import FlowManagementService
from .math_agent import MathAgentService
from .mediated_services import (
    AcademicWriterService,
    JournalReviewerService,
    LiteratureAgentService,
    SimulationDesignerService,
)
from .mediated_workflow import MediatorWorkflowRunner
from .registry import ArtifactRegistry
from .runtime_assemble import RuntimeAssembleService
from .service_contracts import AgentRuntimeProfile, MixedServiceRuntimeConfig
from .workflow import ProfessorWorkflowRunner


def build_professor_workflow_runner(
    directive: Any,
    mailbox: Any,
    *,
    artifact_registry: Optional[ArtifactRegistry] = None,
    steering_control: Optional[Any] = None,
    flow_management: Optional[FlowManagementService] = None,
) -> Any:
    registry = artifact_registry or ArtifactRegistry(root_dir=Path(directive.output_dir))
    profiles = getattr(directive, "agent_profiles", {}) or {}
    literature_profile = profiles.get("literature_agent") or getattr(directive, "llm", None)
    simulation_profile = profiles.get("simulation_designer") or getattr(directive, "llm", None)
    writer_profile = profiles.get("academic_writer") or getattr(directive, "llm", None)
    reviewer_profile = profiles.get("journal_reviewer") or getattr(directive, "llm", None)
    math_profile = profiles.get("math_agent")
    if isinstance(math_profile, MixedServiceRuntimeConfig):
        math_config = math_profile
    elif math_profile is None:
        math_config = MixedServiceRuntimeConfig()
    else:
        math_config = MixedServiceRuntimeConfig(
            nl_profile=AgentRuntimeProfile(
                provider=getattr(math_profile, "provider", None),
                model=getattr(math_profile, "model", None),
                base_url=getattr(math_profile, "base_url", None),
                api_key=getattr(math_profile, "api_key", None),
                timeout_seconds=getattr(math_profile, "timeout_seconds", None),
                enabled=getattr(math_profile, "enabled", True),
            )
        )

    flow = flow_management or FlowManagementService()
    literature_agent = LiteratureAgentService(registry, profile=literature_profile)
    simulation_designer = SimulationDesignerService(registry, profile=simulation_profile)
    runtime_assemble = RuntimeAssembleService(
        registry,
        output_root=Path(directive.output_dir) / "simulation_results",
    )
    math_agent = MathAgentService(config=math_config)
    writer_agent = AcademicWriterService(registry, profile=writer_profile)
    reviewer_agent = JournalReviewerService(profile=reviewer_profile)
    return MediatorWorkflowRunner(
        directive,
        mailbox,
        artifact_registry=registry,
        flow_management=flow,
        literature_agent=literature_agent,
        simulation_designer=simulation_designer,
        runtime_assemble=runtime_assemble,
        math_agent=math_agent,
        writer_agent=writer_agent,
        reviewer_agent=reviewer_agent,
    )


def build_literature_bridge_from_env(
    *,
    llm_sync_backend: Optional[Any] = None,
    gui_event_sink: Optional[Any] = None,
) -> Any:
    backend = llm_sync_backend or make_service_backend("literature_review", None)
    return build_literature_review_bridge_from_env(
        llm_sync_backend=backend,
        gui_event_sink=gui_event_sink,
    )
