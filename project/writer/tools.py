"""
writer/tools.py
================
Tool interfaces used by AcademicWriter to query agent artifacts
and request new agent work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ArtifactInfo:
    artifact_id: str
    agent: str
    summary: str


@dataclass
class AgentQueryResult:
    success: bool
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    error: str = ""

    def artifact_ids(self) -> List[str]:
        return [a.artifact_id for a in self.artifacts]


@dataclass
class TaskRequestResult:
    success: bool
    artifact: Optional[Dict[str, Any]] = None
    error: str = ""


class AgentQueryTool:
    """Queries an agent for its available artifacts."""

    def __init__(
        self,
        query_fn: Optional[Callable] = None,
        agents_fn: Optional[Callable] = None,
    ):
        self._fn = query_fn
        self._agents_fn = agents_fn

    async def available_agents(self) -> list[str]:
        if self._agents_fn:
            return list(await self._agents_fn())
        return []

    async def list_artifacts(self, agent: str) -> AgentQueryResult:
        if self._fn:
            return await self._fn("list", agent, "")
        return AgentQueryResult(success=True, artifacts=[])

    async def call(self, agent: str, query: str) -> AgentQueryResult:
        if self._fn:
            return await self._fn("query", agent, query)
        return AgentQueryResult(success=True, artifacts=[])


class TaskRequestTool:
    """Requests a new task from an agent."""

    def __init__(self, request_fn: Optional[Callable] = None):
        self._fn = request_fn

    async def call(self, task_type: str, params: Dict[str, Any]) -> TaskRequestResult:
        if self._fn:
            return await self._fn(task_type, params)
        return TaskRequestResult(success=True)


@dataclass
class WriterToolbox:
    query: AgentQueryTool
    request: TaskRequestTool


def make_mock_toolbox(agents: Optional[Dict[str, List[Dict]]] = None) -> WriterToolbox:
    """
    Build a WriterToolbox backed by in-memory mock data.
    agents: {"agent_name": [{"artifact_id": "...", "agent": "...", "summary": "..."}]}
    """
    _agents = agents or {}

    async def _query_fn(mode: str, agent: str, query: str) -> AgentQueryResult:
        raw = _agents.get(agent, [])
        artifacts = [
            ArtifactInfo(
                artifact_id=a.get("artifact_id", f"art-{i}"),
                agent=a.get("agent", agent),
                summary=a.get("summary", ""),
            )
            for i, a in enumerate(raw)
        ]
        return AgentQueryResult(success=True, artifacts=artifacts)

    async def _agents_fn() -> list[str]:
        return sorted(_agents) if _agents else ["analyst", "simulator", "literature", "coder"]

    async def _request_fn(task_type: str, params: Dict) -> TaskRequestResult:
        return TaskRequestResult(success=True)

    return WriterToolbox(
        query=AgentQueryTool(_query_fn, _agents_fn),
        request=TaskRequestTool(_request_fn),
    )
