"""Shared platform contracts, registries, and assistant wiring."""

from .contracts import (
    ArtifactCatalog,
    ArtifactRef,
    AssistantId,
    AttachmentRef,
    CodingAssistantPort,
    LiteratureAssistantPort,
    Mailbox,
    MailboxMessage,
    TaskEnvelope,
    WriterAssistantPort,
)
from .flow_management import FlowManagementService
from .math_agent import MathAgentService
from .runtime_assemble import RuntimeAssembleService
from .registry import ArtifactRegistry
from .service_contracts import (
    AgentRuntimeProfile,
    ContextInjection,
    MediatedResponse,
    MixedServiceRuntimeConfig,
    OrchestrationRequestEnvelope,
    RuntimeDiagnostics,
    ServiceRuntimeConfig,
)

__all__ = [
    "AgentRuntimeProfile",
    "ArtifactCatalog",
    "ArtifactRef",
    "ArtifactRegistry",
    "AssistantId",
    "AttachmentRef",
    "CodingAssistantPort",
    "ContextInjection",
    "FlowManagementService",
    "LiteratureAssistantPort",
    "Mailbox",
    "MailboxMessage",
    "MathAgentService",
    "MediatedResponse",
    "MixedServiceRuntimeConfig",
    "OrchestrationRequestEnvelope",
    "RuntimeAssembleService",
    "RuntimeDiagnostics",
    "ServiceRuntimeConfig",
    "TaskEnvelope",
    "WriterAssistantPort",
]
