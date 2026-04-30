"""
writer/promise_registry.py
===========================
Tracks all cross-section forward/backward references (Promises).

Promises survive context-window compression because they are embedded in
section text as opaque tags:  <<promise:prom-abc12345>>

During materialisation (final paper assembly) the registry replaces every
tag with the appropriate resolved text, placeholder, or broken-promise marker.

Thread-safety: not required.  The writer is single-threaded per paper.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List, Optional

from .contract import Promise, PromiseStatus

PROMISE_TAG_RE = re.compile(r"<<promise:([A-Za-z0-9\-]+)>>")


class PromiseRegistry:
    """
    Central store for all cross-section references in a paper draft.

    Lifecycle of a Promise
    ----------------------
    1. Created  – when a section's LLM output declares a new_promise
                  (or from initial_promises in the TOCPlan).
    2. Pending  – tag <<promise:id>> embedded in origin section text.
    3. Resolved – target section explicitly lists promise_id as resolved,
                  and provides the resolved text fragment.
    4. Broken   – target section was drafted but did not resolve it.
    5. Waived   – explicitly removed (e.g. section dropped from outline).

    After any phase the registry can be serialised to dict for checkpointing.
    """

    def __init__(self) -> None:
        self._promises: Dict[str, Promise] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, promise: Promise) -> Promise:
        """Add an already-constructed Promise to the registry."""
        self._promises[promise.promise_id] = promise
        return promise

    def create(
        self,
        origin: str,
        target: str,
        description: str,
        placeholder: str,
    ) -> Promise:
        """Build a new Promise and register it in one call."""
        p = Promise.new(origin, target, description, placeholder)
        return self.register(p)

    # ── Resolution ────────────────────────────────────────────────────────────

    def resolve(self, promise_id: str, resolved_text: str) -> None:
        """Mark a promise fulfilled.  resolved_text replaces the tag in the final paper."""
        if promise_id not in self._promises:
            raise KeyError(f"Unknown promise: {promise_id}")
        self._promises[promise_id].resolve(resolved_text)

    def break_promise(self, promise_id: str, reason: str) -> None:
        """Mark a promise broken (target section drafted without fulfilling it)."""
        if promise_id not in self._promises:
            raise KeyError(f"Unknown promise: {promise_id}")
        self._promises[promise_id].break_(reason)

    def waive(self, promise_id: str) -> None:
        """Drop a promise without error (e.g. section removed from outline)."""
        if promise_id in self._promises:
            self._promises[promise_id].waive()

    # ── Queries ───────────────────────────────────────────────────────────────

    def pending_for_section(self, section_id: str) -> List[Promise]:
        """Return all pending promises whose target is section_id."""
        return [
            p for p in self._promises.values()
            if p.target_section == section_id and p.status == "pending"
        ]

    def made_by_section(self, section_id: str) -> List[Promise]:
        """Return all promises created by a given section (any status)."""
        return [p for p in self._promises.values() if p.origin_section == section_id]

    def broken(self) -> List[Promise]:
        return [p for p in self._promises.values() if p.status == "broken"]

    def pending(self) -> List[Promise]:
        return [p for p in self._promises.values() if p.status == "pending"]

    def all_promises(self) -> List[Promise]:
        return list(self._promises.values())

    def get(self, promise_id: str) -> Optional[Promise]:
        return self._promises.get(promise_id)

    # ── Text operations ───────────────────────────────────────────────────────

    def embed_tag(self, text: str, promise: Promise) -> str:
        """
        Replace the promise's placeholder text in `text` with its tag.
        If placeholder not found, appends the tag at the end.
        """
        if promise.placeholder_text and promise.placeholder_text in text:
            return text.replace(promise.placeholder_text, promise.tag, 1)
        return text + f" {promise.tag}"

    def materialize(self, text: str) -> str:
        """
        Replace all <<promise:id>> tags in `text` with their final text.

        resolved  → resolved_text
        broken    → [BROKEN PROMISE: reason]
        waived    → (omitted)
        pending   → placeholder_text  (still a draft warning marker)
        """
        def replacer(m: re.Match) -> str:
            pid = m.group(1)
            p = self._promises.get(pid)
            if p is None:
                return f"[UNKNOWN PROMISE {pid}]"
            if p.status == "resolved" and p.resolved_text:
                return p.resolved_text
            if p.status == "broken":
                return p.resolved_text or "[BROKEN PROMISE]"
            if p.status == "waived":
                return ""
            # pending
            return f"[PENDING: {p.placeholder_text}]"

        return PROMISE_TAG_RE.sub(replacer, text)

    def extract_ids_from_text(self, text: str) -> List[str]:
        """Return all promise IDs embedded in a block of text."""
        return PROMISE_TAG_RE.findall(text)

    # ── Auto-audit: detect broken promises after a section is finalised ───────

    def audit_section_completion(self, section_id: str, section_text: str) -> List[str]:
        """
        Call after a section is finalised.
        Any promise whose target is section_id and whose tag does NOT appear
        in section_text as resolved → mark broken.
        Returns list of promise_ids that were broken.
        """
        broken_ids: List[str] = []
        for p in self.pending_for_section(section_id):
            # A promise is fulfilled if it appears in the resolved list OR
            # its tag no longer appears as <<promise:...>> in the text
            # (meaning the writer materialised it inline).
            if p.promise_id not in section_text:
                self.break_promise(
                    p.promise_id,
                    f"§{section_id} completed without resolving promise '{p.description}'",
                )
                broken_ids.append(p.promise_id)
        return broken_ids

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, dict]:
        return {pid: asdict(p) for pid, p in self._promises.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, dict]) -> "PromiseRegistry":
        registry = cls()
        for pid, pd in data.items():
            registry._promises[pid] = Promise(**pd)
        return registry

    def snapshot(self) -> Dict[str, Promise]:
        """Shallow copy keyed by promise_id — safe to iterate while registry mutates."""
        return dict(self._promises)

    # ── Summaries ─────────────────────────────────────────────────────────────

    def status_summary(self) -> str:
        total = len(self._promises)
        by_status: Dict[str, int] = {}
        for p in self._promises.values():
            by_status[p.status] = by_status.get(p.status, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(by_status.items())]
        return f"{total} promise(s): " + ", ".join(parts) if parts else "no promises"

    def __len__(self) -> int:
        return len(self._promises)

    def __repr__(self) -> str:
        return f"PromiseRegistry({self.status_summary()})"
