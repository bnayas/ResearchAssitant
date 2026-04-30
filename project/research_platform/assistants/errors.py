from __future__ import annotations

from typing import Optional

from ..contracts import ArtifactRef


class AssistantExecutionError(RuntimeError):
    def __init__(self, message: str, *, artifacts: Optional[list[ArtifactRef]] = None):
        super().__init__(message)
        self.artifacts = artifacts or []
