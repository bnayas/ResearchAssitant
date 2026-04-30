"""
research_platform.registry
──────────────────────────
Concrete artifact registry used by the professor workflow and writer service.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .contracts import ArtifactRef


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "artifact"


@dataclass
class StoredArtifact:
    ref: ArtifactRef


class ArtifactRegistry:
    """In-memory catalog with optional file-backed artifact persistence."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self._items: dict[str, StoredArtifact] = {}

    def register(self, artifact: ArtifactRef) -> ArtifactRef:
        self._items[artifact.artifact_id] = StoredArtifact(ref=artifact)
        return artifact

    def put_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        return self.register(artifact)

    def create(
        self,
        *,
        assistant: str,
        kind: str,
        title: str,
        summary: str = "",
        path: Optional[str] = None,
        url: Optional[str] = None,
        content: Optional[str] = None,
        mime_type: str = "text/plain",
        metadata: Optional[dict[str, Any]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactRef:
        ref = ArtifactRef(
            artifact_id=artifact_id or f"art-{uuid.uuid4().hex[:10]}",
            assistant=assistant,
            kind=kind,
            title=title,
            summary=summary,
            path=path,
            url=url,
            content=content,
            mime_type=mime_type,
            metadata=metadata or {},
        )
        return self.register(ref)

    def save_text(
        self,
        *,
        assistant: str,
        kind: str,
        title: str,
        filename: str,
        text: str,
        summary: str = "",
        metadata: Optional[dict[str, Any]] = None,
        mime_type: str = "text/plain",
        artifact_id: Optional[str] = None,
    ) -> ArtifactRef:
        path = self._ensure_path(filename)
        path.write_text(text, encoding="utf-8")
        return self.create(
            assistant=assistant,
            kind=kind,
            title=title,
            summary=summary or title,
            path=str(path.resolve()),
            content=text,
            mime_type=mime_type,
            metadata=metadata,
            artifact_id=artifact_id,
        )

    def save_json(
        self,
        *,
        assistant: str,
        kind: str,
        title: str,
        filename: str,
        payload: Any,
        summary: str = "",
        metadata: Optional[dict[str, Any]] = None,
        artifact_id: Optional[str] = None,
    ) -> ArtifactRef:
        path = self._ensure_path(filename)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        return self.create(
            assistant=assistant,
            kind=kind,
            title=title,
            summary=summary or title,
            path=str(path.resolve()),
            content=text,
            mime_type="application/json",
            metadata=metadata,
            artifact_id=artifact_id,
        )

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._items

    def get(self, artifact_id: str) -> Optional[ArtifactRef]:
        stored = self._items.get(artifact_id)
        return stored.ref if stored else None

    def get_agent(self, artifact_id: str) -> Optional[str]:
        ref = self.get(artifact_id)
        return ref.assistant if ref else None

    def get_summary(self, artifact_id: str) -> Optional[str]:
        ref = self.get(artifact_id)
        return ref.summary if ref else None

    def get_content(self, artifact_id: str) -> Optional[str]:
        ref = self.get(artifact_id)
        return ref.content if ref else None

    def read_artifact_text(self, artifact_id: str) -> Optional[str]:
        ref = self.get(artifact_id)
        if ref is None:
            return None
        if ref.content is not None:
            return ref.content
        if ref.path:
            path = Path(ref.path)
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def list_by_assistant(self, assistant: str) -> list[ArtifactRef]:
        return [
            stored.ref
            for stored in self._items.values()
            if stored.ref.assistant == assistant
        ]

    def list_artifacts(self, assistant: Optional[str] = None) -> list[ArtifactRef]:
        if assistant is None:
            return self.all()
        return self.list_by_assistant(assistant)

    def search_artifacts(self, query: str, *, assistant: Optional[str] = None) -> list[ArtifactRef]:
        needle = query.strip().lower()
        refs = self.list_artifacts(assistant=assistant)
        if not needle:
            return refs
        matched: list[ArtifactRef] = []
        for ref in refs:
            haystacks = [
                ref.title,
                ref.summary,
                ref.content or "",
                json.dumps(ref.metadata, ensure_ascii=False),
            ]
            if any(needle in field.lower() for field in haystacks):
                matched.append(ref)
        return matched

    def assistants(self) -> list[str]:
        return sorted({stored.ref.assistant for stored in self._items.values()})

    def all(self) -> list[ArtifactRef]:
        return [stored.ref for stored in self._items.values()]

    def _ensure_path(self, filename: str) -> Path:
        if self.root_dir is None:
            raise ValueError("ArtifactRegistry.root_dir is required to persist files")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def next_filename(self, stem: str, suffix: str) -> str:
        return f"{_slugify(stem)}-{uuid.uuid4().hex[:8]}{suffix}"
