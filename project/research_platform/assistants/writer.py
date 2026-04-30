from __future__ import annotations

from typing import Any

from ..contracts import ArtifactRef, TaskEnvelope
from ..registry import ArtifactRegistry
from ..writer_runtime import run_writer_pipeline
from .errors import AssistantExecutionError


class LocalWriterAssistant:
    def __init__(self, registry: ArtifactRegistry, llm_backend: Any) -> None:
        self._registry = registry
        self._llm = llm_backend

    def draft_review(
        self,
        task: TaskEnvelope,
        *,
        sources: list[ArtifactRef],
    ) -> ArtifactRef:
        artifact_catalog = [
            {
                "artifact_id": source.artifact_id,
                "assistant": source.assistant,
                "kind": source.kind,
                "title": source.title,
                "summary": source.summary,
                "path": source.path,
                "url": source.url,
                "content": source.content,
                "mime_type": source.mime_type,
                "metadata": dict(source.metadata),
            }
            for source in sources
        ]
        result = run_writer_pipeline(
            llm_backend=self._llm,
            registry=self._registry,
            description=task.instructions,
            venue=str(task.metadata.get("venue") or "arXiv preprint"),
            artifact_catalog=artifact_catalog,
            artifact_context=str(task.metadata.get("artifact_context") or ""),
            draft_filename="writer/mini_review.md",
        )
        draft_ref = self._registry.get(result["draft_artifact"]["artifact_id"])
        if draft_ref is None:
            raise AssistantExecutionError("Writer completed without registering a draft artifact")
        return draft_ref
