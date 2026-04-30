"""
research_platform.writer_runtime
────────────────────────────────
Real writer integration backed by the shared artifact registry.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from writer.academic_writer import AcademicWriter
from writer.contract import StreamChunk, WritingTask
from writer.tools import AgentQueryResult, AgentQueryTool, ArtifactInfo, TaskRequestResult, TaskRequestTool, WriterToolbox

from .contracts import ArtifactRef, AssistantId
from .registry import ArtifactRegistry


class WriterLLMAdapter:
    """Adapts the project's sync LLM backend to AcademicWriter's interface."""

    def __init__(self, backend: Any):
        self._backend = backend
        self.model_name = getattr(backend, "model_name", "writer-adapter")

    def complete(self, system: str, prompt: str) -> str:
        return self._backend.complete(
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )


def build_registry_toolbox(
    registry: ArtifactRegistry,
    request_fn: Optional[Callable[[str, dict[str, Any]], TaskRequestResult]] = None,
) -> WriterToolbox:
    async def _agents_fn() -> list[str]:
        return registry.assistants()

    async def _query_fn(mode: str, agent: str, query: str) -> AgentQueryResult:
        refs = registry.list_by_assistant(agent)
        if query:
            needle = query.lower()
            refs = [
                ref for ref in refs
                if needle in ref.title.lower()
                or needle in ref.summary.lower()
                or needle in (ref.content or "").lower()
            ]
        artifacts = [
            ArtifactInfo(
                artifact_id=ref.artifact_id,
                agent=ref.assistant,
                summary=ref.summary or ref.title,
            )
            for ref in refs
        ]
        return AgentQueryResult(success=True, artifacts=artifacts)

    async def _request_fn(task_type: str, params: dict[str, Any]) -> TaskRequestResult:
        if request_fn is None:
            return TaskRequestResult(success=True)
        return await asyncio.to_thread(request_fn, task_type, params)

    return WriterToolbox(
        query=AgentQueryTool(_query_fn, _agents_fn),
        request=TaskRequestTool(_request_fn),
    )


def seed_registry_from_catalog(
    registry: ArtifactRegistry,
    *,
    artifact_catalog: Optional[list[dict[str, Any]]] = None,
    artifact_context: str = "",
) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for index, raw in enumerate(artifact_catalog or [], start=1):
        assistant = (
            raw.get("assistant")
            or raw.get("agent")
            or raw.get("source_agent")
            or AssistantId.ORCHESTRATOR.value
        )
        title = raw.get("title") or raw.get("name") or raw.get("label") or f"Source {index}"
        summary = raw.get("summary") or raw.get("abstract") or title
        ref = registry.create(
            artifact_id=raw.get("artifact_id"),
            assistant=assistant,
            kind=raw.get("kind") or raw.get("type") or "source_artifact",
            title=title,
            summary=summary,
            path=raw.get("path"),
            url=raw.get("url"),
            content=raw.get("content"),
            mime_type=raw.get("mime_type", "text/plain"),
            metadata=dict(raw.get("metadata") or {}),
        )
        refs.append(ref)

    if artifact_context.strip():
        refs.append(
            registry.create(
                assistant=AssistantId.ORCHESTRATOR.value,
                kind="context_note",
                title="Upstream Artifact Context",
                summary=_summarize(artifact_context),
                content=artifact_context,
                metadata={"source": "artifactContext"},
            )
        )

    return refs


def run_writer_pipeline(
    *,
    llm_backend: Any,
    registry: ArtifactRegistry,
    description: str,
    venue: str,
    artifact_catalog: Optional[list[dict[str, Any]]] = None,
    artifact_context: str = "",
    stream_callback: Optional[Callable[[StreamChunk], None]] = None,
    draft_filename: str = "writer/mini_review.md",
) -> dict[str, Any]:
    seed_registry_from_catalog(
        registry,
        artifact_catalog=artifact_catalog,
        artifact_context=artifact_context,
    )
    toolbox = build_registry_toolbox(registry)
    writer = AcademicWriter(
        llm=WriterLLMAdapter(llm_backend),
        toolbox=toolbox,
        store=registry,
    )
    task = WritingTask.new(description=description, venue=venue)

    async def _run() -> dict[str, Any]:
        async for chunk in writer.plan(task):
            if stream_callback:
                stream_callback(chunk)
        async for chunk in writer.draft():
            if stream_callback:
                stream_callback(chunk)
        assembled = writer.materialize_paper()
        draft_ref = registry.save_text(
            assistant=AssistantId.WRITER.value,
            kind="review_draft",
            title="Mini-review Draft",
            filename=draft_filename,
            text=assembled,
            summary=_summarize(assembled),
            metadata={"paper_id": writer.artifact.paper_id if writer.artifact else ""},
        )
        return {
            "assembled_paper": assembled,
            "artifact": _jsonify(writer.artifact),
            "draft_artifact": _jsonify(draft_ref),
        }

    return asyncio.run(_run())


def _summarize(text: str, *, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if is_dataclass(value):
        return {
            field.name: _jsonify(getattr(value, field.name))
            for field in fields(value)
        }
    return str(value)
