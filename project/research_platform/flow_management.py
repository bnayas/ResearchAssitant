"""
research_platform.flow_management
─────────────────────────────────
Session registry and immediate context injection for mediated services.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .service_contracts import ContextInjection


@dataclass
class SessionRecord:
    workflow_id: str
    session_id: str
    service_id: str
    state: str = "idle"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class WorkflowRecord:
    workflow_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stop_requested: bool = False
    stop_reason: str = ""
    latest_session_id: str = ""
    sessions: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PendingQueue:
    injections: list[ContextInjection] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)


class FlowManagementService:
    def __init__(self, event_sink: Optional[Callable[[dict[str, Any]], None]] = None) -> None:
        self._event_sink = event_sink
        self._workflows: dict[str, WorkflowRecord] = {}
        self._sessions: dict[str, SessionRecord] = {}
        self._pending: dict[str, _PendingQueue] = {}
        self._lock = threading.Lock()

    def open_workflow(self, workflow_id: str, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        with self._lock:
            record = self._workflows.get(workflow_id)
            if record is None:
                record = WorkflowRecord(workflow_id=workflow_id, metadata=dict(metadata or {}))
                self._workflows[workflow_id] = record
            else:
                record.metadata.update(metadata or {})
                record.updated_at = time.time()
        self._emit(workflow_id, "workflow_opened", {"metadata": dict(metadata or {})})
        return self.get_workflow_snapshot(workflow_id)

    def register_session(
        self,
        workflow_id: str,
        service_id: str,
        session_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            workflow = self._workflows.setdefault(workflow_id, WorkflowRecord(workflow_id=workflow_id))
            if session_id not in workflow.sessions:
                workflow.sessions.append(session_id)
            workflow.latest_session_id = session_id
            workflow.updated_at = time.time()
            self._sessions[session_id] = SessionRecord(
                workflow_id=workflow_id,
                session_id=session_id,
                service_id=service_id,
                metadata=dict(metadata or {}),
            )
            self._pending.setdefault(session_id, _PendingQueue())
        self._emit(
            workflow_id,
            "session_registered",
            {"service_id": service_id, "session_id": session_id, "metadata": dict(metadata or {})},
        )
        return self.get_workflow_snapshot(workflow_id)

    def update_session_state(
        self,
        workflow_id: str,
        session_id: str,
        state: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._sessions[session_id]
            session.state = state
            if metadata:
                session.metadata.update(metadata)
            session.updated_at = time.time()
            workflow = self._workflows[workflow_id]
            workflow.latest_session_id = session_id
            workflow.updated_at = time.time()
        self._emit(
            workflow_id,
            "session_updated",
            {"session_id": session_id, "state": state, "metadata": dict(metadata or {})},
        )
        return self.get_workflow_snapshot(workflow_id)

    def inject_context(
        self,
        target_session_id: str,
        *,
        payload: dict[str, Any],
        reason: str = "",
        mode: str = "soft",
        injected_by: str = "system",
    ) -> ContextInjection:
        injection = ContextInjection(
            target_session_id=target_session_id,
            mode=mode,
            payload=dict(payload),
            reason=reason,
            injected_by=injected_by,
        )
        pending = self._pending[target_session_id]
        with pending.condition:
            pending.injections.append(injection)
            pending.condition.notify_all()
        workflow_id = self._sessions[target_session_id].workflow_id
        self._emit(
            workflow_id,
            "context_injected",
            {
                "target_session_id": target_session_id,
                "mode": mode,
                "payload": dict(payload),
                "reason": reason,
                "injected_by": injected_by,
            },
        )
        return injection

    def poll_injected_context(self, target_session_id: str) -> list[ContextInjection]:
        pending = self._pending[target_session_id]
        with pending.condition:
            items = list(pending.injections)
            pending.injections.clear()
        return items

    def wait_for_injection(self, target_session_id: str, timeout: Optional[float] = None) -> ContextInjection:
        pending = self._pending[target_session_id]
        with pending.condition:
            if pending.injections:
                return pending.injections.pop(0)
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                workflow_id = self._sessions[target_session_id].workflow_id
                workflow = self._workflows[workflow_id]
                if workflow.stop_requested:
                    raise RuntimeError(workflow.stop_reason or "Workflow stop requested")
                if pending.injections:
                    return pending.injections.pop(0)
                if deadline is None:
                    pending.condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timed out waiting for context injection for session {target_session_id}")
                    pending.condition.wait(timeout=remaining)

    def interrupt_session(
        self,
        target_session_id: str,
        *,
        payload: dict[str, Any],
        reason: str = "",
        injected_by: str = "system",
    ) -> ContextInjection:
        return self.inject_context(
            target_session_id,
            payload=payload,
            reason=reason,
            mode="hard",
            injected_by=injected_by,
        )

    def request_stop(self, workflow_id: str, reason: str = "") -> dict[str, Any]:
        with self._lock:
            workflow = self._workflows[workflow_id]
            workflow.stop_requested = True
            workflow.stop_reason = reason.strip()
            workflow.updated_at = time.time()
            sessions = list(workflow.sessions)
        for session_id in sessions:
            pending = self._pending.get(session_id)
            if pending is None:
                continue
            with pending.condition:
                pending.condition.notify_all()
        self._emit(workflow_id, "stop_requested", {"reason": workflow.stop_reason})
        return self.get_workflow_snapshot(workflow_id)

    def resume_workflow(self, workflow_id: str) -> dict[str, Any]:
        with self._lock:
            workflow = self._workflows[workflow_id]
            workflow.stop_requested = False
            workflow.stop_reason = ""
            workflow.updated_at = time.time()
        self._emit(workflow_id, "workflow_resumed", {})
        return self.get_workflow_snapshot(workflow_id)

    def latest_session_id(self, workflow_id: str) -> Optional[str]:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                return None
            return workflow.latest_session_id or None

    def get_workflow_snapshot(self, workflow_id: str) -> dict[str, Any]:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise KeyError(f"Unknown workflow: {workflow_id}")
            sessions = [
                {
                    "workflow_id": session.workflow_id,
                    "session_id": session.session_id,
                    "service_id": session.service_id,
                    "state": session.state,
                    "metadata": dict(session.metadata),
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "pending_injections": len(self._pending.get(session.session_id, _PendingQueue()).injections),
                }
                for session in (
                    self._sessions[sid] for sid in workflow.sessions
                    if sid in self._sessions
                )
            ]
            events = list(workflow.events)
            latest_session_id = workflow.latest_session_id
            stop_requested = workflow.stop_requested
            stop_reason = workflow.stop_reason
            metadata = dict(workflow.metadata)
        return {
            "workflow_id": workflow_id,
            "metadata": metadata,
            "latest_session_id": latest_session_id,
            "stop_requested": stop_requested,
            "stop_reason": stop_reason,
            "sessions": sessions,
            "events": events,
        }

    def stream_events(self, workflow_id: str) -> list[dict[str, Any]]:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                return []
            return list(workflow.events)

    def _emit(self, workflow_id: str, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "workflow_id": workflow_id,
            "payload": dict(payload),
            "timestamp": time.time(),
        }
        with self._lock:
            workflow = self._workflows.setdefault(workflow_id, WorkflowRecord(workflow_id=workflow_id))
            workflow.events.append(event)
            workflow.updated_at = time.time()
        if self._event_sink is not None:
            self._event_sink(event)
