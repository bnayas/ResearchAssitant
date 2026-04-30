"""
research_platform.steering
──────────────────────────
Interactive PI steering checkpoints for the professor workflow.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .contracts import AttachmentRef


class WorkflowStopRequested(RuntimeError):
    def __init__(self, reason: str = "") -> None:
        self.reason = reason.strip()
        super().__init__(self.reason or "Workflow stopped by PI")


@dataclass
class SteeringCheckpoint:
    checkpoint_id: str
    directive_id: str
    phase: str
    assistant: str
    title: str
    prompt: str
    attachments: list[AttachmentRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attachments"] = [attachment.to_dict() for attachment in self.attachments]
        return payload


@dataclass
class SteeringDecision:
    checkpoint_id: str
    action: str = "continue"
    feedback: str = ""
    submitted_at: float = field(default_factory=time.time)

    def normalized_action(self) -> str:
        action = str(self.action or "continue").strip().lower()
        return action if action in {"continue", "revise", "stop"} else "continue"


class SteeringController:
    """
    Single-pending-checkpoint controller used by the streamed directive flow.

    The professor workflow opens one checkpoint at a time, waits for the PI to
    continue or request a revision, and then proceeds.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: Optional[SteeringCheckpoint] = None
        self._decision: Optional[SteeringDecision] = None
        self._stop_requested = False
        self._stop_reason = ""

    def open(self, checkpoint: SteeringCheckpoint) -> None:
        with self._condition:
            if self._pending is not None:
                raise RuntimeError("A steering checkpoint is already pending")
            self._pending = checkpoint
            self._decision = None
            self._condition.notify_all()

    def pending(self) -> Optional[SteeringCheckpoint]:
        with self._condition:
            return self._pending

    def request_stop(self, reason: str = "") -> None:
        with self._condition:
            self._stop_requested = True
            self._stop_reason = reason.strip()
            if self._pending is not None:
                self._decision = SteeringDecision(
                    checkpoint_id=self._pending.checkpoint_id,
                    action="stop",
                    feedback=self._stop_reason,
                )
            self._condition.notify_all()

    def stop_requested(self) -> bool:
        with self._condition:
            return self._stop_requested

    def stop_reason(self) -> str:
        with self._condition:
            return self._stop_reason

    def wait(self, checkpoint_id: str, timeout: Optional[float] = None) -> SteeringDecision:
        with self._condition:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if self._stop_requested:
                    self._pending = None
                    self._decision = None
                    raise WorkflowStopRequested(self._stop_reason)
                decision = self._decision
                pending = self._pending
                if decision is not None and decision.checkpoint_id == checkpoint_id:
                    self._pending = None
                    self._decision = None
                    return decision
                if pending is None:
                    raise RuntimeError("Steering checkpoint cleared before a decision was received")
                if pending.checkpoint_id != checkpoint_id:
                    raise RuntimeError("Waiting for a checkpoint that is no longer pending")
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timed out waiting for steering on checkpoint {checkpoint_id}")
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()

    def submit(self, checkpoint_id: str, *, action: str, feedback: str = "") -> SteeringDecision:
        with self._condition:
            pending = self._pending
            if pending is None:
                raise KeyError("No steering checkpoint is currently awaiting PI input")
            if pending.checkpoint_id != checkpoint_id:
                raise KeyError(
                    f"Checkpoint {checkpoint_id!r} is not pending; current checkpoint is {pending.checkpoint_id!r}"
                )
            decision = SteeringDecision(
                checkpoint_id=checkpoint_id,
                action=action,
                feedback=feedback,
            )
            if decision.normalized_action() == "revise" and not decision.feedback.strip():
                raise ValueError("Revision requests must include feedback for the assistants")
            self._decision = decision
            self._condition.notify_all()
            return decision
