"""
sim_tool.description_builder
─────────────────────────────
Builds the compact description string that is sent to SimulationDesigner.start().

Why this exists
───────────────
mediated_services._compose_description used to concatenate all source
artifact text verbatim: article brief body + procedure + literature
synthesis + every source summary.  That routinely produced 8k-15k char
descriptions which overflowed the LLM context window and caused the
designer to return prose instead of JSON.

This module replaces that logic with a precision-ordered builder that:
  1. Puts the most decisive information first (model description, params).
  2. Hard-caps each section and the overall total.
  3. Drops low-value tail content (extended literature refs, extra sources)
     when the budget is exhausted.

The staged designer in staged_designer.py can then use the full original
description for its per-function code passes, because those passes send
only structure context (field names and types), not prose.

Usage
─────
    from sim_tool.description_builder import build_designer_description

    description = build_designer_description(
        task_instructions=task.instructions,
        sources=sources,              # list[ArtifactRef]
        task_metadata=task.metadata,
    )
    result = self._designer.start(description)
"""
from __future__ import annotations

import json
from typing import Any, Optional

# ── Tuneable limits ──────────────────────────────────────────────────────────

# Total chars sent to SimulationDesigner.start().
# ~6000 chars ≈ 1500 tokens — comfortably fits with a 4k system prompt.
MAX_DESCRIPTION_CHARS = 6_000

# Max chars for the procedure section (tends to be very long)
MAX_PROCEDURE_CHARS = 800

# Max chars for the literature synthesis excerpt
MAX_SYNTHESIS_CHARS = 600

# Max chars per extra source summary
MAX_SOURCE_SUMMARY_CHARS = 300

# How many extra (non-brief, non-literature) sources to include
MAX_EXTRA_SOURCES = 2


# ── Public entry point ────────────────────────────────────────────────────────

def build_designer_description(
    *,
    task_instructions: str,
    sources: list,          # list[ArtifactRef] — typed loosely to avoid circular imports
    task_metadata: dict,
) -> str:
    """
    Return a description string suitable for SimulationDesigner.start().

    Precision order (most decisive first):
      1. PI instruction / task
      2. Selected simulation targets (if PI narrowed scope)
      3. Model description (extracted from article brief metadata)
      4. Key parameters (as compact JSON)
      5. Simulation procedure (truncated)
      6. Expected figures (short list)
      7. Literature synthesis excerpt
      8. Other source summaries (capped)
      9. PI steering notes

    The total is capped at MAX_DESCRIPTION_CHARS by trimming from the tail.
    """
    parts: list[str] = []

    # ── 1. PI instruction ────────────────────────────────────────────────────
    instruction = str(task_instructions or "").strip()
    if instruction:
        parts.append(instruction)

    # ── 2. Selected simulation targets ───────────────────────────────────────
    selected_targets = task_metadata.get("selected_simulation_targets")
    if isinstance(selected_targets, list) and selected_targets:
        lines: list[str] = []
        for i, target in enumerate(selected_targets, 1):
            if isinstance(target, dict):
                name = str(target.get("name") or f"Target {i}")
                desc = str(target.get("description") or "").strip()
            else:
                name = f"Target {i}"
                desc = str(target or "").strip()
            lines.append(f"{i}. {name}" + (f": {desc}" if desc else ""))
        parts.append("Selected simulation target:\n" + "\n".join(lines))

    # ── 3–6. Article brief ───────────────────────────────────────────────────
    brief = _find_source(sources, artifact_id="article-brief", kind="article_brief")
    if brief:
        meta = brief.metadata or {}

        model_desc = str(meta.get("model_description") or "").strip()
        if model_desc:
            parts.append(f"Model to reproduce:\n{model_desc}")

        params = meta.get("key_parameters")
        if isinstance(params, dict) and params:
            parts.append(f"Key parameters:\n{json.dumps(params, indent=2)}")
        elif isinstance(params, str) and params.strip():
            parts.append(f"Key parameters:\n{params.strip()}")

        procedure = str(meta.get("procedure") or "").strip()
        if procedure:
            if len(procedure) > MAX_PROCEDURE_CHARS:
                procedure = procedure[:MAX_PROCEDURE_CHARS - 20].rstrip() + "\n[truncated]"
            parts.append(f"Simulation procedure:\n{procedure}")

        figures = meta.get("expected_figures")
        if isinstance(figures, list) and figures:
            parts.append(
                "Expected figures:\n" + "\n".join(f"- {f}" for f in figures[:5])
            )

    # ── 7. Literature synthesis ───────────────────────────────────────────────
    lit = _find_source(
        sources, artifact_id="literature-review", kind="literature_review"
    )
    if lit:
        synthesis = str(
            (lit.metadata or {}).get("synthesis") or lit.summary or ""
        ).strip()
        if synthesis:
            if len(synthesis) > MAX_SYNTHESIS_CHARS:
                synthesis = synthesis[:MAX_SYNTHESIS_CHARS - 1].rstrip() + "…"
            parts.append(f"Related literature context:\n{synthesis}")

    # ── 8. Other source summaries ─────────────────────────────────────────────
    used: set[str] = {
        brief.artifact_id if brief else "",
        lit.artifact_id if lit else "",
        "",
    }
    extra = 0
    for src in sources:
        if extra >= MAX_EXTRA_SOURCES:
            break
        if src.artifact_id in used:
            continue
        used.add(src.artifact_id)
        summary = str(src.summary or "").strip()
        if not summary:
            continue
        if len(summary) > MAX_SOURCE_SUMMARY_CHARS:
            summary = summary[:MAX_SOURCE_SUMMARY_CHARS - 1].rstrip() + "…"
        parts.append(f"{src.title}:\n{summary}")
        extra += 1

    # ── 9. Steering notes ────────────────────────────────────────────────────
    notes = [
        str(n).strip()
        for n in (task_metadata.get("steering_notes") or [])
        if str(n).strip()
    ]
    if notes:
        parts.append("PI steering notes:\n" + "\n".join(f"- {n}" for n in notes))

    # ── Assembly and hard cap ─────────────────────────────────────────────────
    description = "\n\n".join(p for p in parts if p.strip())

    if len(description) > MAX_DESCRIPTION_CHARS:
        cut = MAX_DESCRIPTION_CHARS - 80
        description = (
            description[:cut].rstrip()
            + "\n\n[additional context trimmed to fit context window]"
        )

    return description


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_source(sources: list, *, artifact_id: str, kind: str):
    """Return the first source matching artifact_id, then kind, else None."""
    for s in sources:
        if getattr(s, "artifact_id", None) == artifact_id:
            return s
    for s in sources:
        if getattr(s, "kind", None) == kind:
            return s
    return None
