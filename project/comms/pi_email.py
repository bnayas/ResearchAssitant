"""
comms.pi_email
──────────────
Defines the structures for the Research Assistant to report back to the PI
at various milestones of a research directive.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from research_platform.contracts import AttachmentRef, MailboxMessage


@dataclass
class Attachment:
    name: str
    path: Optional[str] = None      # Local file path, relative to the output_dir
    url: Optional[str] = None       # Web link (e.g. arXiv URL)
    content: Optional[str] = None   # Inline text for small artifacts
    mime_type: str = "text/plain"


@dataclass
class PIEmail:
    directive_id: str
    from_agent: str        # e.g., "Literature Agent", "Simulation Agent"
    from_addr: str         # e.g., "lit@research.local"
    to_name: str           # The PI's name
    subject: str
    body: str
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class PIMailbox:
    """
    A thread-safe, ordered list of emails sent to the PI, with persistence support.
    """
    def __init__(self, output_dir: Optional[Path] = None):
        self._emails: list[PIEmail] = []
        self._output_dir = output_dir

    def append(self, email: PIEmail | MailboxMessage) -> None:
        if isinstance(email, MailboxMessage):
            email = self._coerce_message(email)
        self._emails.append(email)
        if self._output_dir:
            self._save_email(email)

    def _save_email(self, email: PIEmail) -> None:
        emails_dir = self._output_dir / "emails"
        emails_dir.mkdir(parents=True, exist_ok=True)
        idx = len(self._emails)
        safe_subject = "".join(c if c.isalnum() else "_" for c in email.subject)
        safe_subject = safe_subject.strip("_")[:50]
        filename = f"{idx:03d}_{safe_subject}.json"
        
        with open(emails_dir / filename, "w", encoding="utf-8") as f:
            json.dump(email.to_dict(), f, indent=2)

    def all_emails(self) -> list[PIEmail]:
        return list(self._emails)

    def _coerce_message(self, message: MailboxMessage) -> PIEmail:
        attachments = [self._coerce_attachment(a) for a in message.attachments]
        return PIEmail(
            directive_id=message.directive_id,
            from_agent=message.assistant,
            from_addr=message.assistant_addr,
            to_name=message.to_name,
            subject=message.subject,
            body=message.body,
            attachments=attachments,
            metadata=dict(message.metadata),
            timestamp=message.timestamp,
        )

    @staticmethod
    def _coerce_attachment(attachment: Attachment | AttachmentRef) -> Attachment:
        if isinstance(attachment, Attachment):
            return attachment
        return Attachment(
            name=attachment.name,
            path=attachment.path,
            url=attachment.url,
            content=attachment.content,
            mime_type=attachment.mime_type,
        )
