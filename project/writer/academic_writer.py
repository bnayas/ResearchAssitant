"""
writer/academic_writer.py
==========================
AcademicWriter agent — multi-phase academic paper authoring.

Phases
------
1. PLAN    produce TOCPlan: sections + figures + initial cross-section promises
2. DRAFT   section-by-section in dependency order; each section streams live
3. AUDIT   run GroundingValidator; auto-repair up to max_repair_rounds times
4. REVIEW  accept ReviewerDemand, revise targeted sections, stream changes

Streaming
---------
Every public method is an async generator yielding StreamChunk objects.
The UI/Director consumes chunks in real time and may steer by:
  - closing the generator (GeneratorExit) to abort
  - calling respond_to_review() after AUDIT phase completes

Promise passing
---------------
When §Introduction says "as we show in §Results", the LLM outputs a
new_promise block.  The PromiseRegistry stores it as <<promise:prom-xxxx>>.
When §Results is drafted the LLM lists the promise_id in resolved_promise_ids.
After context compression the promise tag survives in the text; materialise()
replaces it with the resolved text fragment for the final paper.

Grounding
---------
Every section draft must include a ---GROUNDING--- JSON block listing claims.
The validator rejects sections with missing, low-confidence, or unverifiable
claims.  Auto-repair injects validation issues back into the prompt.

Integration
-----------
    from writer import AcademicWriter, WriterToolbox, WritingTask
    from writer.tools import make_mock_toolbox

    toolbox = WriterToolbox(query_fn=my_fn, request_fn=my_req_fn)
    writer  = AcademicWriter(llm=my_llm, toolbox=toolbox, store=my_store)

    async def run():
        task = WritingTask.new("SIR epidemic model study", venue="NeurIPS")
        async for chunk in writer.plan(task):
            handle_chunk(chunk)
        async for chunk in writer.draft():
            handle_chunk(chunk)
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from .contract import (
    FigureSpec,
    GroundingClaim,
    Promise,
    ReviewCycle,
    ReviewerDemand,
    RevisionResponse,
    SectionDraft,
    SectionSpec,
    StreamChunk,
    TOCPlan,
    WriterArtifact,
    WritingTask,
)
from .promise_registry import PromiseRegistry
from .tools import AgentQueryResult, WriterToolbox
from .validator import ArtifactStore, GroundingValidator

# ── LLM backend duck-type ─────────────────────────────────────────────────────
# We depend only on:
#   llm.complete(system, prompt) → str          (sync fallback)
#   llm.stream(system, prompt)  → AsyncIter[str] (preferred)
#   llm.stream_with_thinking(s, p) → AsyncIter[(kind, text)]  (extended)


# ── System prompts ────────────────────────────────────────────────────────────

_PLAN_SYSTEM = """
You are the AcademicWriter in PLANNING mode.
Given a research description and a list of available agent artifacts, produce a
structured paper plan as a single JSON object with this exact shape:

{
  "title": "...",
  "venue": "...",
  "abstract_outline": "One paragraph.",
  "sections": [
    {
      "section_id": "s1",
      "number": "1",
      "title": "Introduction",
      "level": 1,
      "estimated_words": 500,
      "key_points": ["..."],
      "required_agents": ["literature"],
      "depends_on": [],
      "figures": []
    }
  ],
  "figures": [
    {
      "figure_id": "f1",
      "number": 1,
      "caption": "...",
      "source_agent": "simulator",
      "artifact_id": "art-xxxx",
      "section_placement": "s4",
      "figure_type": "plot"
    }
  ],
  "initial_promises": [
    {
      "origin_section": "s1",
      "target_section": "s4",
      "description": "We will show that β=0.3 yields peak infection at day 12.",
      "placeholder_text": "(see §Results)"
    }
  ]
}

Rules:
- section_id values must be stable identifiers (no spaces).
- depends_on lists section_ids that must be drafted BEFORE this section.
- initial_promises captures cross-section forward references identified NOW.
- Only reference artifact_ids that exist in the provided agent artifact list.
- Reply ONLY with valid JSON.  No prose, no markdown fences.
""".strip()

_DRAFT_SYSTEM = """
You are the AcademicWriter in DRAFTING mode.
Write the assigned section in polished academic prose (LaTeX-ready markdown).

After the section text output a separator line and a JSON block:

---GROUNDING---
{
  "grounding_claims": [
    {
      "claim_text": "The epidemic peaked at day 12 with I_max = 0.34.",
      "source_agent": "simulator",
      "artifact_id": "art-xxxx",
      "artifact_excerpt": "peak_day=12, I_max=0.3412",
      "confidence": 0.95
    }
  ],
  "resolved_promise_ids": ["prom-abc123"],
  "new_promises": [
    {
      "target_section": "s5",
      "description": "Appendix will list all ODE solver hyperparameters.",
      "placeholder_text": "(see Appendix)"
    }
  ],
  "summary": "One paragraph compressed summary for passing to later sections."
}
---END---

Critical rules:
1. EVERY factual assertion must have a grounding_claim entry.
2. Use <<promise:prom-xxxx>> tags in the text for cross-section references.
3. resolved_promise_ids lists any pending promises THIS section fulfills.
4. new_promises lists forward references THIS section creates.
5. artifact_id values must come from the provided agent artifact list.
6. confidence < 0.5 will cause a hard validation error — do not guess.
""".strip()

_REVIEW_SYSTEM = """
You are the AcademicWriter in REVIEW mode.
Revise the section to address the listed reviewer comments.

Output the revised section text, then:

---GROUNDING---
{ ... same structure as DRAFTING ... }
---END---

---RESPONSES---
{
  "responses": [
    {
      "comment_id": "rc-001",
      "action": "addressed",
      "explanation": "Added experiment with γ=0.1 as requested."
    }
  ]
}
---END---

Rules:
1. Every response must have action: addressed | declined | deferred.
2. Re-establish grounding for any NEW claims introduced during revision.
3. Do not drop existing grounding claims unless the claim itself is removed.
""".strip()


# ── Main agent ────────────────────────────────────────────────────────────────


class AcademicWriter:
    """
    Multi-phase academic writing agent.

    Parameters
    ----------
    llm             LLM backend (must support .stream or .complete)
    toolbox         AgentQueryTool + TaskRequestTool wrappers
    store           ArtifactStore for grounding validation
    max_repair_rounds  auto-repair iterations before giving up on a section
    """

    def __init__(
        self,
        llm,
        toolbox: WriterToolbox,
        store: ArtifactStore,
        max_repair_rounds: int = 3,
    ) -> None:
        self._llm = llm
        self._toolbox = toolbox
        self._store = store
        self._max_repair = max_repair_rounds
        self._validator = GroundingValidator(store)
        self._registry = PromiseRegistry()
        self._artifact: Optional[WriterArtifact] = None

    # ── Phase 1: Plan ─────────────────────────────────────────────────────────

    async def plan(
        self, task: WritingTask
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Produce the TOCPlan: sections, figures, and initial promises.
        Streams reasoning + JSON plan.
        """
        paper_id = f"paper-{uuid.uuid4().hex[:8]}"
        self._artifact = WriterArtifact(
            paper_id=paper_id,
            toc_plan=None,
            sections={},
            promise_registry={},
            review_cycles=[],
            status="planning",
        )
        self._registry = PromiseRegistry()

        yield StreamChunk(kind="phase_change", text="[PLAN] Starting — querying agents for existing work")

        # Discover available artifacts from all known agents supplied by the orchestrator/toolbox.
        context_lines: List[str] = []
        agent_names = await self._toolbox.query.available_agents()
        if not agent_names:
            agent_names = ["analyst", "simulator", "literature", "coder"]
        for agent in agent_names:
            yield StreamChunk(
                kind="tool_call",
                text=f"Listing artifacts from '{agent}'",
            )
            result = await self._toolbox.query.list_artifacts(agent)
            if result.success and result.artifacts:
                block_lines = [f"## {agent} artifacts"]
                for a in result.artifacts[:12]:
                    block_lines.append(f"  [{a.artifact_id}] {a.summary}")
                context_lines.extend(block_lines)
                yield StreamChunk(
                    kind="tool_result",
                    text=f"  {agent}: {len(result.artifacts)} artifact(s)",
                    payload={"agent": agent, "ids": result.artifact_ids()},
                )
            else:
                yield StreamChunk(
                    kind="tool_result",
                    text=f"  {agent}: no artifacts (or unavailable)",
                )

        agent_context = "\n".join(context_lines) or "(no agent artifacts found)"
        prompt = (
            f"Research task:\n{task.description}\n\n"
            f"Target venue: {task.venue}\n\n"
            f"Available agent artifacts:\n{agent_context}\n\n"
            "Produce the paper plan JSON."
        )

        # Stream LLM → collect full output
        raw_json = ""
        yield StreamChunk(kind="reasoning", text="[PLAN] Generating table of contents and figures …")
        async for chunk in self._stream_llm(_PLAN_SYSTEM, prompt):
            yield chunk
            if chunk.kind == "text_delta":
                raw_json += chunk.text

        # Parse and register
        toc = self._parse_toc(paper_id, task.venue, raw_json)
        self._artifact.toc_plan = toc
        self._artifact.status = "drafting"

        for p in toc.initial_promises:
            self._registry.register(p)
            yield StreamChunk(
                kind="promise_created",
                text=f"Promise [{p.promise_id}]: §{p.origin_section} → §{p.target_section}: {p.description}",
                payload={"promise_id": p.promise_id},
            )

        self._artifact.promise_registry = self._registry.to_dict()

        yield StreamChunk(
            kind="phase_change",
            text=(
                f"[PLAN] Done — {len(toc.sections)} sections, "
                f"{len(toc.figures)} figures, "
                f"{len(toc.initial_promises)} initial promises"
            ),
            payload={
                "paper_id": paper_id,
                "sections": [
                    {"id": s.section_id, "title": s.title}
                    for s in toc.sections
                ],
            },
        )

    # ── Phase 2: Draft ────────────────────────────────────────────────────────

    async def draft(self) -> AsyncGenerator[StreamChunk, None]:
        """
        Draft all sections in dependency order.
        Each section is streamed live; validated after completion.
        If validation fails, auto-repair loops up to max_repair_rounds.
        Paper-level audit runs after all sections.
        """
        if not self._artifact or not self._artifact.toc_plan:
            yield StreamChunk(kind="error", text="Must call plan() before draft()")
            return

        yield StreamChunk(kind="phase_change", text="[DRAFT] Beginning section drafting")

        order = self._topo_sort(self._artifact.toc_plan.sections)
        summaries: Dict[str, str] = {}  # section_id → compressed summary

        for spec in order:
            async for chunk in self._draft_section(spec, summaries):
                yield chunk
            draft = self._artifact.sections.get(spec.section_id)
            if draft:
                summaries[spec.section_id] = draft.summary

        # Paper-level audit
        self._artifact.status = "auditing"
        yield StreamChunk(kind="phase_change", text="[AUDIT] Running full grounding validation")
        async for chunk in self._audit_paper():
            yield chunk

    # ── Phase 4: Respond to review ────────────────────────────────────────────

    async def respond_to_review(
        self, demand: ReviewerDemand
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Receive reviewer comments, optionally commission new agent work,
        revise targeted sections, stream all changes.
        """
        if not self._artifact:
            yield StreamChunk(kind="error", text="No active artifact")
            return

        cycle = ReviewCycle(
            cycle_id=f"rev-{uuid.uuid4().hex[:8]}",
            demand=demand,
        )
        self._artifact.review_cycles.append(cycle)
        self._artifact.status = "revising"

        yield StreamChunk(
            kind="phase_change",
            text=f"[REVIEW] Processing {len(demand.comments)} comment(s) from {demand.reviewer_id}",
        )

        # Group by section
        by_section: Dict[Optional[str], list] = {}
        for c in demand.comments:
            by_section.setdefault(c.section_id, []).append(c)

        for section_id, comments in by_section.items():
            if section_id is None:
                section_id = self._intro_section_id()

            if section_id not in (self._artifact.sections or {}):
                yield StreamChunk(
                    kind="error",
                    text=f"[REVIEW] Section '{section_id}' not found — skipping",
                )
                continue

            # Commission new agent work where needed
            for c in [x for x in comments if x.requires_new_agent_work]:
                yield StreamChunk(
                    kind="reasoning",
                    text=f"[REVIEW] Comment {c.comment_id} requires new agent work — dispatching",
                )
                yield StreamChunk(kind="tool_call", text=f"Requesting task for comment {c.comment_id}")
                result = await self._toolbox.request.call(
                    "research_task",
                    {"comment": c.text, "section": section_id},
                )
                yield StreamChunk(
                    kind="tool_result",
                    text=f"  Task result: {'OK' if result.success else result.error}",
                    payload=result.artifact,
                    section_id=section_id,
                )

            # Revise section
            async for chunk in self._revise_section(section_id, comments, cycle):
                yield chunk

        cycle.status = "addressed"
        self._artifact.status = "review"
        yield StreamChunk(
            kind="phase_change",
            text=f"[REVIEW] Cycle {cycle.cycle_id} complete — all comments addressed",
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def artifact(self) -> Optional[WriterArtifact]:
        return self._artifact

    def materialize_section(self, section_id: str) -> Optional[str]:
        """Return the final section text with all promise tags resolved."""
        draft = (self._artifact.sections or {}).get(section_id)
        if not draft:
            return None
        return self._registry.materialize(draft.content)

    def materialize_paper(self) -> str:
        """Assemble the full paper as a single markdown string."""
        if not self._artifact:
            return ""
        parts: List[str] = []
        if self._artifact.toc_plan:
            parts.append(f"# {self._artifact.toc_plan.title}\n")
            parts.append(f"**Venue:** {self._artifact.toc_plan.venue}\n")
            parts.append(f"**Abstract outline:** {self._artifact.toc_plan.abstract_outline}\n")

        order = self._topo_sort(self._artifact.toc_plan.sections) if self._artifact.toc_plan else []
        for spec in order:
            draft = self._artifact.sections.get(spec.section_id)
            if draft:
                heading = "#" * (spec.level + 1)
                parts.append(f"\n{heading} {spec.number}. {spec.title}\n")
                parts.append(self._registry.materialize(draft.content))

        return "\n".join(parts)

    # ── Internal: section drafting ────────────────────────────────────────────

    async def _draft_section(
        self, spec: SectionSpec, summaries: Dict[str, str]
    ) -> AsyncGenerator[StreamChunk, None]:
        sid = spec.section_id

        yield StreamChunk(
            kind="phase_change",
            text=f"[DRAFT §{spec.number}] {spec.title}",
            section_id=sid,
        )

        # --- Gather agent artifacts for this section ---
        agent_context_parts: List[str] = []
        for agent in spec.required_agents:
            yield StreamChunk(kind="tool_call", text=f"Querying {agent} for §{spec.number}", section_id=sid)
            result = await self._toolbox.query.call(agent, f"artifacts for: {spec.title}")
            if result.success:
                part = f"### {agent}\n" + "\n".join(
                    f"  [{a.artifact_id}] {a.summary}"
                    for a in result.artifacts[:6]
                )
                agent_context_parts.append(part)
                yield StreamChunk(
                    kind="tool_result",
                    text=f"  {agent}: {len(result.artifacts)} artifact(s)",
                    section_id=sid,
                )
            else:
                yield StreamChunk(kind="tool_result", text=f"  {agent}: {result.error}", section_id=sid)

        agent_context = "\n\n".join(agent_context_parts) or "(no artifacts retrieved)"

        # --- Build prior section context (compressed summaries) ---
        prior_parts = []
        for dep_id in spec.depends_on:
            if dep_id in summaries:
                prior_parts.append(f"### Summary of §{dep_id}\n{summaries[dep_id]}")
        prior_context = "\n\n".join(prior_parts) if prior_parts else ""

        # --- Pending promises to fulfill ---
        pending = self._registry.pending_for_section(sid)
        promise_ctx = ""
        if pending:
            lines = [f"  [{p.promise_id}] {p.description}" for p in pending]
            promise_ctx = "\n\nPending promises to fulfill in this section:\n" + "\n".join(lines)

        prompt = (
            f"## Section {spec.number}: {spec.title}\n\n"
            f"Key points:\n" + "\n".join(f"- {kp}" for kp in spec.key_points) + "\n\n" +
            (f"## Prior section summaries\n{prior_context}\n\n" if prior_context else "") +
            f"## Available agent artifacts\n{agent_context}" +
            promise_ctx
        )

        # --- Auto-repair loop ---
        final_draft: Optional[SectionDraft] = None
        grounding_data: dict = {}

        for round_idx in range(self._max_repair + 1):
            if round_idx > 0:
                yield StreamChunk(
                    kind="reasoning",
                    text=f"[REPAIR {round_idx}/{self._max_repair}] Retrying §{spec.number} …",
                    section_id=sid,
                )

            raw_full = ""
            async for chunk in self._stream_llm(_DRAFT_SYSTEM, prompt, section_id=sid):
                yield chunk
                if chunk.kind == "text_delta":
                    raw_full += chunk.text

            section_text, grounding_data = _split_grounding_block(raw_full)
            draft = self._build_section_draft(spec, section_text, grounding_data)

            # Validate immediately
            report = self._validator.validate_section(draft)
            draft.status = "validated" if report.passed else "flagged"
            draft.validation_issues = [i.message for i in report.issues]

            for issue in report.issues:
                yield StreamChunk(
                    kind="validation_issue",
                    text=f"[{issue.check_id}] {issue.message}",
                    payload={"check_id": issue.check_id, "severity": issue.severity},
                    section_id=sid,
                )

            if report.passed:
                final_draft = draft
                break

            if round_idx < self._max_repair:
                # Inject issues for the next round
                error_ctx = "\n## Validation errors to fix\n" + "\n".join(
                    f"- [{i.check_id}] {i.message}" for i in report.issues
                )
                prompt = prompt + error_ctx
            else:
                yield StreamChunk(
                    kind="error",
                    text=f"§{sid} failed validation after {self._max_repair} repair round(s)",
                    section_id=sid,
                )
                final_draft = draft  # accept flagged; Director decides next step

        # --- Register promises ---
        self._register_section_promises(sid, grounding_data, final_draft)

        # --- Audit broken promises ---
        broken_ids = self._registry.audit_section_completion(sid, final_draft.content)
        for pid in broken_ids:
            yield StreamChunk(
                kind="validation_issue",
                text=f"[P01] Promise {pid} was not resolved by §{sid}",
                payload={"promise_id": pid, "severity": "error"},
                section_id=sid,
            )

        # --- Emit new promises ---
        for pid in final_draft.pending_promise_ids:
            p = self._registry.get(pid)
            if p:
                yield StreamChunk(
                    kind="promise_created",
                    text=f"Promise [{pid}]: §{p.origin_section} → §{p.target_section}: {p.description}",
                    payload={"promise_id": pid},
                    section_id=sid,
                )

        # --- Emit resolved promises ---
        for pid in final_draft.resolved_promise_ids:
            yield StreamChunk(
                kind="promise_resolved",
                text=f"Promise [{pid}] resolved by §{sid}",
                payload={"promise_id": pid},
                section_id=sid,
            )

        # --- Emit grounding claims ---
        for claim in final_draft.grounding_claims:
            yield StreamChunk(
                kind="grounding_claim",
                text=f"[{claim.artifact_id}] {claim.claim_text[:80]}",
                payload={
                    "claim_id": claim.claim_id,
                    "source_agent": claim.source_agent,
                    "artifact_id": claim.artifact_id,
                    "confidence": claim.confidence,
                },
                section_id=sid,
            )

        self._artifact.sections[sid] = final_draft
        self._artifact.touch()

        yield StreamChunk(
            kind="phase_change",
            text=(
                f"[DRAFT §{spec.number}] ✓ {final_draft.word_count} words, "
                f"{len(final_draft.grounding_claims)} claims, "
                f"status={final_draft.status}"
            ),
            section_id=sid,
        )

    # ── Internal: section revision ────────────────────────────────────────────

    async def _revise_section(
        self, section_id: str, comments: list, cycle: ReviewCycle
    ) -> AsyncGenerator[StreamChunk, None]:
        draft = self._artifact.sections[section_id]
        spec = self._artifact.toc_plan.section_by_id(section_id) if self._artifact.toc_plan else None

        comment_text = "\n".join(
            f"[{c.comment_id} | {c.severity}] {c.text}" for c in comments
        )
        prompt = (
            f"## Current section\n{draft.content}\n\n"
            f"## Reviewer comments\n{comment_text}\n\n"
            "Revise to address all comments."
        )

        yield StreamChunk(
            kind="phase_change",
            text=f"[REVIEW §{section_id}] Revising …",
            section_id=section_id,
        )

        raw_full = ""
        async for chunk in self._stream_llm(_REVIEW_SYSTEM, prompt, section_id=section_id):
            yield chunk
            if chunk.kind == "text_delta":
                raw_full += chunk.text

        section_text, grounding_data = _split_grounding_block(raw_full)
        responses_list = _extract_responses_block(raw_full)

        if spec is None:
            # Build a minimal spec for the builder
            from .contract import SectionSpec
            spec = SectionSpec(
                section_id=section_id,
                number=section_id,
                title=draft.title,
                level=1,
                estimated_words=draft.word_count,
                key_points=[],
                required_agents=[],
                depends_on=[],
                figures=[],
            )

        revised = self._build_section_draft(spec, section_text, grounding_data)
        revised.status = "draft"

        # Validate revised
        report = self._validator.validate_section(revised)
        revised.status = "validated" if report.passed else "flagged"
        revised.validation_issues = [i.message for i in report.issues]

        for issue in report.issues:
            yield StreamChunk(
                kind="validation_issue",
                text=f"[{issue.check_id}] {issue.message}",
                payload={"severity": issue.severity},
                section_id=section_id,
            )

        self._artifact.sections[section_id] = revised

        for r in responses_list:
            cycle.responses.append(RevisionResponse(
                comment_id=r.get("comment_id", "?"),
                action=r.get("action", "addressed"),
                explanation=r.get("explanation", ""),
                changed_sections=[section_id],
            ))

        yield StreamChunk(
            kind="phase_change",
            text=f"[REVIEW §{section_id}] Revised — status={revised.status}",
            section_id=section_id,
        )

    # ── Internal: paper audit ─────────────────────────────────────────────────

    async def _audit_paper(self) -> AsyncGenerator[StreamChunk, None]:
        report = self._validator.validate_paper(
            self._artifact,
            self._registry.snapshot(),
        )
        for issue in report.all_issues:
            yield StreamChunk(
                kind="validation_issue",
                text=f"[{issue.check_id}] {issue.message}",
                payload={"severity": issue.severity},
                section_id=issue.section_id,
            )

        registry_summary = self._registry.status_summary()
        if report.overall_passed:
            self._artifact.status = "review"
            yield StreamChunk(
                kind="phase_change",
                text=(
                    f"[AUDIT] ✓ Passed — "
                    f"{report.error_count()} errors, {report.warning_count()} warnings. "
                    f"{registry_summary}"
                ),
            )
        else:
            self._artifact.status = "drafting"  # needs repair
            yield StreamChunk(
                kind="error",
                text=(
                    f"[AUDIT] ✗ Failed — "
                    f"{report.error_count()} error(s), {report.warning_count()} warning(s). "
                    f"{registry_summary}"
                ),
            )

    # ── Internal: LLM streaming ───────────────────────────────────────────────

    async def _stream_llm(
        self,
        system: str,
        prompt: str,
        section_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Thin adapter over the project LLM backend.

        Supports three backend styles:
          1. stream_with_thinking(system, prompt) → AsyncIter[(kind, text)]
             where kind ∈ {"thinking", "text"}
          2. stream(system, prompt) → AsyncIter[str]
          3. complete(system, prompt) → str  (fallback, non-streaming)
        """
        try:
            if hasattr(self._llm, "stream_with_thinking"):
                async for kind, text in self._llm.stream_with_thinking(system, prompt):
                    yield StreamChunk(
                        kind="reasoning" if kind == "thinking" else "text_delta",
                        text=text,
                        section_id=section_id,
                    )
            elif hasattr(self._llm, "stream"):
                async for text in self._llm.stream(system, prompt):
                    yield StreamChunk(kind="text_delta", text=text, section_id=section_id)
            else:
                # Sync fallback: complete then yield as single chunk
                text = self._llm.complete(system, prompt)
                yield StreamChunk(kind="text_delta", text=text, section_id=section_id)
        except Exception as exc:
            yield StreamChunk(kind="error", text=str(exc), section_id=section_id)

    # ── Internal: parse helpers ───────────────────────────────────────────────

    def _parse_toc(self, paper_id: str, venue: str, raw: str) -> TOCPlan:
        data = _json_safe(raw)
        sections = [
            SectionSpec(
                section_id=s.get("section_id", f"s{i}"),
                number=s.get("number", str(i + 1)),
                title=s.get("title", ""),
                level=s.get("level", 1),
                estimated_words=s.get("estimated_words", 400),
                key_points=s.get("key_points", []),
                required_agents=s.get("required_agents", []),
                depends_on=s.get("depends_on", []),
                figures=s.get("figures", []),
            )
            for i, s in enumerate(data.get("sections", []))
        ]
        figures = [
            FigureSpec(
                figure_id=f.get("figure_id", f"f{i}"),
                number=f.get("number", i + 1),
                caption=f.get("caption", ""),
                source_agent=f.get("source_agent", ""),
                artifact_id=f.get("artifact_id", ""),
                section_placement=f.get("section_placement", ""),
                figure_type=f.get("figure_type", "plot"),
            )
            for i, f in enumerate(data.get("figures", []))
        ]
        initial_promises = [
            Promise.new(
                origin=p.get("origin_section", ""),
                target=p.get("target_section", ""),
                description=p.get("description", ""),
                placeholder=p.get("placeholder_text", ""),
            )
            for p in data.get("initial_promises", [])
        ]
        return TOCPlan(
            paper_id=paper_id,
            title=data.get("title", "Untitled"),
            venue=data.get("venue", venue),
            abstract_outline=data.get("abstract_outline", ""),
            sections=sections,
            figures=figures,
            initial_promises=initial_promises,
        )

    def _build_section_draft(
        self, spec: SectionSpec, text: str, grounding_data: dict
    ) -> SectionDraft:
        claims = [
            GroundingClaim.new(
                claim_text=c.get("claim_text", ""),
                source_agent=c.get("source_agent", ""),
                artifact_id=c.get("artifact_id", ""),
                artifact_excerpt=c.get("artifact_excerpt", ""),
                confidence=float(c.get("confidence", 0.8)),
            )
            for c in grounding_data.get("grounding_claims", [])
        ]
        return SectionDraft(
            section_id=spec.section_id,
            title=spec.title,
            content=text,
            grounding_claims=claims,
            resolved_promise_ids=grounding_data.get("resolved_promise_ids", []),
            pending_promise_ids=[],  # filled in _register_section_promises
            summary=grounding_data.get(
                "summary", text[:400] if len(text) > 400 else text
            ),
            word_count=len(text.split()),
        )

    def _register_section_promises(
        self, section_id: str, grounding_data: dict, draft: SectionDraft
    ) -> None:
        # Resolve promises this section fulfilled
        for pid in draft.resolved_promise_ids:
            p = self._registry.get(pid)
            if p and p.status == "pending":
                self._registry.resolve(pid, f"(resolved in §{section_id})")

        # Register new promises this section makes
        for np in grounding_data.get("new_promises", []):
            p = self._registry.create(
                origin=section_id,
                target=np.get("target_section", ""),
                description=np.get("description", ""),
                placeholder=np.get("placeholder_text", ""),
            )
            draft.pending_promise_ids.append(p.promise_id)

        self._artifact.promise_registry = self._registry.to_dict()

    # ── Internal: ordering + context ─────────────────────────────────────────

    def _topo_sort(self, sections: List[SectionSpec]) -> List[SectionSpec]:
        """Topological sort respecting depends_on; cycles → append remainder."""
        order: List[SectionSpec] = []
        done: set = set()
        remaining = list(sections)
        max_iters = len(sections) ** 2 + 1
        iters = 0
        while remaining:
            iters += 1
            if iters > max_iters:
                order.extend(remaining)
                break
            progressed = False
            for spec in list(remaining):
                if all(d in done for d in spec.depends_on):
                    order.append(spec)
                    done.add(spec.section_id)
                    remaining.remove(spec)
                    progressed = True
            if not progressed:
                order.extend(remaining)
                break
        return order

    def _intro_section_id(self) -> Optional[str]:
        """Return the first top-level section as a fallback for paper-wide comments."""
        if not self._artifact or not self._artifact.toc_plan:
            return None
        for s in self._artifact.toc_plan.sections:
            if s.level == 1:
                return s.section_id
        return None


# ── Parse utilities ───────────────────────────────────────────────────────────


def _json_safe(raw: str) -> dict:
    """Best-effort JSON parse; strips markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object anywhere in the string
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def _split_grounding_block(raw: str):
    """Split section text from the ---GROUNDING--- ... ---END--- block."""
    marker = "---GROUNDING---"
    end_marker = "---END---"
    if marker in raw:
        parts = raw.split(marker, 1)
        section_text = parts[0].strip()
        rest = parts[1] if len(parts) > 1 else ""
        grounding_raw = rest.split(end_marker, 1)[0].strip()
        grounding_data = _json_safe(grounding_raw)
    else:
        section_text = raw.strip()
        grounding_data = {}
    return section_text, grounding_data


def _extract_responses_block(raw: str) -> list:
    """Extract the ---RESPONSES--- ... ---END--- block from review output."""
    marker = "---RESPONSES---"
    end_marker = "---END---"
    if marker in raw:
        parts = raw.split(marker, 1)
        rest = parts[1] if len(parts) > 1 else ""
        # Skip past the first ---END--- (which closes GROUNDING)
        responses_raw = rest.split(end_marker, 1)[0].strip()
        data = _json_safe(responses_raw)
        return data.get("responses", [])
    return []
