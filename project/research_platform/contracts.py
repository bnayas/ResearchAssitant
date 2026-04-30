"""
research_platform.contracts
───────────────────────────
Shared contracts for the professor workflow platform.

Only the orchestrator should know about multiple assistants. All cross-assistant
handoffs should use these generic envelopes and references.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable


class AssistantId(str, Enum):
    LITERATURE_REVIEWER = "literature_reviewer"
    CODING_AGENT = "coding_agent"
    WRITER = "writer"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AttachmentRef:
    name: str
    path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    mime_type: str = "text/plain"
    artifact_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRef:
    artifact_id: str
    assistant: str
    kind: str
    title: str
    summary: str = ""
    path: Optional[str] = None
    url: Optional[str] = None
    content: Optional[str] = None
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_attachment(self, *, name: Optional[str] = None) -> AttachmentRef:
        return AttachmentRef(
            name=name or self.title,
            path=self.path,
            url=self.url,
            content=self.content,
            mime_type=self.mime_type,
            artifact_id=self.artifact_id,
            metadata=dict(self.metadata),
        )


@dataclass
class MailboxMessage:
    directive_id: str
    assistant: str
    assistant_addr: str
    to_name: str
    subject: str
    body: str
    attachments: list[AttachmentRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attachments"] = [a.to_dict() for a in self.attachments]
        return payload


@dataclass
class TaskEnvelope:
    task_id: str
    directive_id: str
    assistant: str
    requestor: str = AssistantId.ORCHESTRATOR.value
    subject: str = ""
    instructions: str = ""
    output_dir: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[AttachmentRef] = field(default_factory=list)

    @property
    def output_path(self) -> Optional[Path]:
        if not self.output_dir:
            return None
        return Path(self.output_dir)


@runtime_checkable
class ArtifactCatalog(Protocol):
    def register(self, artifact: ArtifactRef) -> ArtifactRef: ...

    def exists(self, artifact_id: str) -> bool: ...

    def get_agent(self, artifact_id: str) -> Optional[str]: ...

    def get_summary(self, artifact_id: str) -> Optional[str]: ...

    def list_by_assistant(self, assistant: str) -> list[ArtifactRef]: ...


@runtime_checkable
class Mailbox(Protocol):
    def append(self, message: Any) -> None: ...


@runtime_checkable
class LiteratureAssistantPort(Protocol):
    def prepare_article_lookup_spec(self, task: TaskEnvelope) -> dict[str, Any]: ...

    def find_primary_article(self, task: TaskEnvelope) -> ArtifactRef: ...

    def build_article_brief(self, task: TaskEnvelope, article: ArtifactRef) -> ArtifactRef: ...

    def review_related_literature(
        self,
        task: TaskEnvelope,
        article: ArtifactRef,
        brief: ArtifactRef,
    ) -> ArtifactRef: ...


@runtime_checkable
class CodingAssistantPort(Protocol):
    def execute_task(
        self,
        task: TaskEnvelope,
        *,
        sources: list[ArtifactRef],
        update_callback: Optional[Callable[[str, list[ArtifactRef], str], None]] = None,
        review_callback: Optional[Callable[..., Any]] = None,
        cancel_callback: Optional[Callable[..., Any]] = None,
    ) -> list[ArtifactRef]: ...


@runtime_checkable
class WriterAssistantPort(Protocol):
    def draft_review(
        self,
        task: TaskEnvelope,
        *,
        sources: list[ArtifactRef],
    ) -> ArtifactRef: ...
