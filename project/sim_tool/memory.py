"""
sim_tool.memory
───────────────
Agent memory: each agent maintains its own AGENTS.md file that it reads
at the start of every session and appends to when it learns something.

Purpose
───────
Prevent agents from repeating the same mistakes across sessions.
When the designer writes bad code that fails at runtime, or the analyst
makes a wrong call, the lesson is written to AGENTS.md and injected
into the system prompt of all future sessions.

Memory files
────────────
  ~/.sim_tool/designer_memory.md   — rules the designer has learned
  ~/.sim_tool/analyst_memory.md    — rules the analyst has learned

Both are plain Markdown — human-readable and hand-editable.
The agent appends; the human can edit or delete entries.

Format of a memory entry
────────────────────────
  ## [2024-01-15 14:32] Session abc123 — 2D Ising Model
  ### Simulation type: Monte Carlo
  ### What failed:
  - Generated state_assert checked `state.acceptance_rate <= 1.0` but
    omitted the lower bound — missed a bug when acceptance_rate went negative
    due to integer overflow in accepted/n2.
  ### Rule added:
  - ALWAYS assert both bounds for rate and probability variables.
    Use `0.0 <= value <= 1.0`, never just `value <= 1.0`.

Usage
─────
    from sim_tool.memory import AgentMemory

    mem = AgentMemory("designer")
    prompt_prefix = mem.as_prompt_prefix()    # injected before system prompt

    # After a session with a known failure:
    mem.append_entry(
        session_id="abc123",
        sim_name="2D Ising Model",
        sim_type="Monte Carlo",
        what_failed=["state_assert missed lower bound on acceptance_rate"],
        rule_added="ALWAYS assert both bounds for rate variables.",
    )

    # Ask the LLM to consolidate what it learned:
    lesson = mem.consolidate_session(session_id, run_summary, backend)
    # lesson is written to the file automatically
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("sim_tool.memory")

# Where memory files live — user-level, survives across projects
_DEFAULT_MEMORY_DIR = Path.home() / ".sim_tool"

# Max characters to inject into the system prompt (to stay within context limits)
_MAX_PROMPT_CHARS = 3000

# Prompt for asking the LLM to extract lessons from a session
_CONSOLIDATION_PROMPT = """
You are reviewing a completed simulation session to extract lessons for future use.

You will receive:
  - The session's simulation name and type
  - The run outcomes (what statuses were returned)
  - The flags that were raised
  - The patch that was applied (if any)
  - Any assertion errors that occurred

Your task: identify at most 3 concrete, specific rules that should be added
to your AGENTS.md to prevent the same mistakes in future sessions.

Rules must be:
  - Specific (mention the exact field, condition, or pattern)
  - Actionable (tell the future agent exactly what to do differently)
  - Falsifiable (possible to verify whether the rule was followed)

NOT useful: "Be careful with edge cases"
USEFUL: "When sweep includes temperature values below 1.0 J/kB in the Ising model,
         add a failure condition checking acceptance_rate < 1e-6 — the lattice
         freezes at low T and runs forever without it."

Output ONLY a JSON object:
{
  "what_failed": ["<concise description of what went wrong>", ...],
  "rules": ["<specific rule for AGENTS.md>", ...]
}

If nothing went wrong and no lessons were learned, return:
{"what_failed": [], "rules": []}
""".strip()


class AgentMemory:
    """
    Read/write interface for one agent's AGENTS.md file.

    Args:
        agent:      "designer" or "analyst"
        memory_dir: Override the default ~/.sim_tool directory.
    """

    def __init__(
        self,
        agent: str,
        memory_dir: Optional[Path] = None,
    ) -> None:
        self.agent = agent
        self._dir = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{agent}_memory.md"

        if not self._path.exists():
            self._path.write_text(
                f"# {agent.capitalize()} AGENTS.md\n\n"
                f"Lessons learned by the {agent} agent across sessions.\n"
                f"Appended automatically. Human-editable.\n\n"
            )
            log.info(f"Created new memory file: {self._path}")
        else:
            n = self._count_entries()
            log.debug(f"Loaded {agent} memory: {n} entries from {self._path}")

    # ── Reading ───────────────────────────────────────────────────────────────

    def as_prompt_prefix(self) -> str:
        """
        Return the memory content formatted for injection into a system prompt.
        Truncated to _MAX_PROMPT_CHARS to respect context limits.
        Empty string if no entries have been written yet.
        """
        content = self._path.read_text(encoding="utf-8")
        # Strip the file header, keep only the entries
        entries_start = content.find("\n## [")
        if entries_start == -1:
            return ""  # no entries yet

        entries = content[entries_start:].strip()
        if not entries:
            return ""

        # Truncate from the END (most recent entries are most relevant)
        # Keep the tail, not the head
        if len(entries) > _MAX_PROMPT_CHARS:
            entries = "…[older entries truncated]\n\n" + entries[-_MAX_PROMPT_CHARS:]

        return (
            "═" * 60 + "\n"
            f"LESSONS LEARNED (from previous {self.agent} sessions)\n"
            "Read these carefully. Do NOT repeat these mistakes.\n"
            + "─" * 60 + "\n"
            + entries + "\n"
            + "═" * 60 + "\n\n"
        )

    def read_all(self) -> str:
        """Return the raw file content."""
        return self._path.read_text(encoding="utf-8")

    # ── Writing ───────────────────────────────────────────────────────────────

    def append_entry(
        self,
        session_id:    str,
        sim_name:      str,
        sim_type:      str,
        what_failed:   list[str],
        rules:         list[str],
    ) -> None:
        """
        Append a lessons-learned entry to the memory file.
        Called automatically by consolidate_session().
        Can also be called manually by the coder to add domain knowledge.
        """
        if not what_failed and not rules:
            log.debug(f"[{session_id}] No lessons to record.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"\n## [{timestamp}] Session {session_id} — {sim_name}",
            f"### Simulation type: {sim_type}",
        ]
        if what_failed:
            lines.append("### What went wrong:")
            for item in what_failed:
                lines.append(f"- {item}")
        if rules:
            lines.append("### Rules added:")
            for rule in rules:
                lines.append(f"- {rule}")

        entry = "\n".join(lines) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(entry)

        log.info(
            f"[{session_id}] Wrote {len(rules)} rule(s) to {self._path.name}"
        )

    def consolidate_session(
        self,
        session_id:   str,
        sim_name:     str,
        sim_type:     str,
        session_facts: dict,   # structured summary: outcomes, flags, patch
        backend,               # LLMBackend — may be None for offline use
    ) -> list[str]:
        """
        Ask the LLM to extract lessons from a completed session and write them.
        Returns the list of rules that were added (empty if backend is None).

        session_facts keys expected:
          outcomes: list of {status, reason}
          flags: list of {kind, severity, message}
          patch_applied: bool
          patch_changes: list of {field, why}
          assertion_errors: list of str
        """
        if backend is None:
            log.debug(f"[{session_id}] No LLM backend — skipping session consolidation")
            return []

        # Build a compact summary for the LLM
        facts_lines = [f"Simulation: {sim_name} ({sim_type})", ""]
        outcomes = session_facts.get("outcomes", [])
        if outcomes:
            facts_lines.append("Run outcomes:")
            for o in outcomes:
                facts_lines.append(f"  [{o.get('status','?')}] {o.get('reason','')[:80]}")
        flags = session_facts.get("flags", [])
        if flags:
            facts_lines.append("Diagnostic flags:")
            for f in flags:
                facts_lines.append(f"  [{f.get('severity','?')}] {f.get('kind','?')}: {f.get('message','')[:80]}")
        patch_changes = session_facts.get("patch_changes", [])
        if patch_changes:
            facts_lines.append("Patch applied:")
            for c in patch_changes:
                facts_lines.append(f"  {c.get('field','?')} — {c.get('why','')[:80]}")
        assertion_errors = session_facts.get("assertion_errors", [])
        if assertion_errors:
            facts_lines.append("Assertion errors caught:")
            for e in assertion_errors:
                facts_lines.append(f"  {e[:100]}")

        facts_str = "\n".join(facts_lines)
        messages = [{"role": "user", "content": facts_str}]

        log.info(f"[{session_id}] Asking {backend.model_name} to consolidate lessons...")
        try:
            raw = backend.complete(
                system=_CONSOLIDATION_PROMPT,
                messages=messages,
                max_tokens=1024,
            )
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                log.warning(f"[{session_id}] LLM returned no JSON for consolidation")
                return []
            data = json.loads(m.group())
            what_failed = data.get("what_failed", [])
            rules = data.get("rules", [])
            self.append_entry(session_id, sim_name, sim_type, what_failed, rules)
            return rules
        except Exception as exc:
            log.warning(f"[{session_id}] Memory consolidation failed: {exc}")
            return []

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def _count_entries(self) -> int:
        content = self._path.read_text(encoding="utf-8")
        return content.count("\n## [")

    def clear(self) -> None:
        """Reset to empty (keeps the file header). Irreversible."""
        self._path.write_text(
            f"# {self.agent.capitalize()} AGENTS.md\n\n"
            f"Lessons learned by the {self.agent} agent across sessions.\n"
            f"Appended automatically. Human-editable.\n\n"
        )
        log.info(f"Cleared {self._path}")

    def show(self) -> str:
        """Return the full memory file for display."""
        return self.read_all()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entry_count(self) -> int:
        return self._count_entries()
