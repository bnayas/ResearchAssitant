"""
sim_tool.research_directive
───────────────────────────
Compatibility seam for the professor workflow runner.

The actual workflow implementation lives in `research_platform.workflow` and
`research_platform.wiring`; this module keeps the historical import path stable
for the CLI and API entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from comms.pi_email import PIMailbox
from research_platform.flow_management import FlowManagementService
from research_platform.registry import ArtifactRegistry
from research_platform.service_contracts import AgentRuntimeProfile, MixedServiceRuntimeConfig
from research_platform.steering import SteeringController
from research_platform.wiring import build_professor_workflow_runner


@dataclass
class SimulationBlueprint:
    model_description: str = ""
    key_parameters: dict[str, str] = field(default_factory=dict)
    procedure: str = ""
    expected_figures: list[str] = field(default_factory=list)


@dataclass
class AgentLLMConfig:
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: Optional[float] = None
    enabled: bool = True


@dataclass
class PIDirective:
    directive_id: str
    pi_name: str
    instruction: str
    topic_hint: str
    phases: list[str]
    output_dir: str
    llm: Optional[AgentLLMConfig] = None
    agent_profiles: dict[str, AgentRuntimeProfile | MixedServiceRuntimeConfig | AgentLLMConfig] = field(default_factory=dict)


class DirectiveRunner:
    def __init__(
        self,
        directive: PIDirective,
        mailbox: PIMailbox,
        *,
        literature_agent=None,
        coding_agent=None,
        writer_agent=None,
        artifact_registry: Optional[ArtifactRegistry] = None,
        steering_control: Optional[SteeringController] = None,
        flow_management: Optional[FlowManagementService] = None,
    ) -> None:
        self.directive = directive
        self.mailbox = mailbox
        self.output_dir = Path(directive.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_registry = artifact_registry or ArtifactRegistry(root_dir=self.output_dir)

        if literature_agent or coding_agent or writer_agent:
            from research_platform.workflow import ProfessorWorkflowRunner

            if not (literature_agent and coding_agent and writer_agent):
                raise ValueError(
                    "DirectiveRunner injection requires literature_agent, coding_agent, "
                    "and writer_agent together."
                )
            self._runner = ProfessorWorkflowRunner(
                directive,
                mailbox,
                literature_agent=literature_agent,
                coding_agent=coding_agent,
                writer_agent=writer_agent,
                artifact_registry=self.artifact_registry,
                steering_control=steering_control,
            )
        else:
            self._runner = build_professor_workflow_runner(
                directive,
                mailbox,
                artifact_registry=self.artifact_registry,
                steering_control=steering_control,
                flow_management=flow_management,
            )

    def run_all_phases(self) -> None:
        self._runner.run_all_phases()
