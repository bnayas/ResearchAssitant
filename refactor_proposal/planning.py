"""
research_platform.planning
──────────────────────────
Two-tier planning system for the research orchestrator.

Tier 1 — DeepPlanner
  Before committing to an execution plan the planner runs a structured
  chain-of-thought loop:

    Step 0  : generate_agenda  — "what do I need to figure out?"
              Returns a ThinkingAgenda (ordered list of ThinkingSteps).
    Step 1-N: resolve_step     — answer each ThinkingStep in sequence,
              accumulating constraints and assumptions.
    Final   : synthesize_plan  — given all answers + tool catalog,
              produce the ExecutionPlan.

  The whole trace (agenda + results) is kept in DeepPlanResult so it
  can be surfaced in the UI and stored for auditability.

Tier 2 — SteeringRouter
  Incoming PI steering during execution is matched against a
  ShortcutCatalog.  A high-confidence match returns a ShortcutMatch
  that tells the orchestrator how to mutate the live plan without a
  full replan (skip, inject, jump, abort-and-replan).

  Low-confidence / unrecognized steering falls back to full replan.

Design constraints
  - Every LLM call returns a well-typed dataclass; JSON is the wire format.
  - DeepPlanner is stateless between calls; state lives in DeepPlanResult.
  - SteeringRouter is a pure function of (text, plan, step_idx).
  - Both degrade gracefully when the planner backend is None.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class ShortcutAction(str, Enum):
    SKIP_TO       = "skip_to"       # jump forward to the first step matching `target`
    INJECT_BEFORE = "inject_before" # insert new steps before current step
    INJECT_AFTER  = "inject_after"  # insert new steps after current step
    REPLACE_TAIL  = "replace_tail"  # discard remaining steps and replace with new ones
    ABORT_REPLAN  = "abort_replan"  # discard whole plan, replan with steering baked in
    ABORT_STOP    = "abort_stop"    # stop the workflow entirely


# ---------------------------------------------------------------------------
# Deep planner data structures
# ---------------------------------------------------------------------------

@dataclass
class ThinkingStep:
    """One question the planner must answer before it can commit to a plan."""
    step_id: str
    question: str
    rationale: str = ""             # why this question matters
    context_keys: list[str] = field(default_factory=list)  # tool names / artifact kinds relevant
    priority: int = 0               # lower = resolve earlier

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "question": self.question,
            "rationale": self.rationale,
            "context_keys": self.context_keys,
            "priority": self.priority,
        }


@dataclass
class ThinkingAgenda:
    """
    Ordered list of ThinkingSteps produced in planning phase 0.
    The agenda itself is a planning artefact: it tells the planner (and the PI)
    *what the orchestrator thought it needed to figure out* before acting.
    """
    directive_summary: str
    steps: list[ThinkingStep] = field(default_factory=list)
    tool_catalog_digest: str = ""   # short human-readable summary of available tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "directive_summary": self.directive_summary,
            "steps": [s.to_dict() for s in self.steps],
            "tool_catalog_digest": self.tool_catalog_digest,
        }


@dataclass
class ThinkingResult:
    """The planner's answer to one ThinkingStep."""
    step_id: str
    question: str
    answer: str
    confidence: float = 1.0         # 0–1; low confidence surfaces a warning
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)  # guide plan synthesis
    flags: list[str] = field(default_factory=list)         # anything unusual

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "constraints": self.constraints,
            "flags": self.flags,
        }


@dataclass
class DeepPlanResult:
    """
    Full output of one DeepPlanner.plan() call.

    The ExecutionPlan is the actionable part.
    agenda + thinking_results are the audit trail.
    """
    plan: Any                        # ExecutionPlan (avoid circular import)
    agenda: ThinkingAgenda
    thinking_results: list[ThinkingResult] = field(default_factory=list)
    aggregated_constraints: list[str] = field(default_factory=list)
    aggregated_assumptions: list[str] = field(default_factory=list)
    planning_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict() if hasattr(self.plan, "to_dict") else {},
            "agenda": self.agenda.to_dict(),
            "thinking_results": [r.to_dict() for r in self.thinking_results],
            "aggregated_constraints": self.aggregated_constraints,
            "aggregated_assumptions": self.aggregated_assumptions,
            "planning_warnings": self.planning_warnings,
        }


# ---------------------------------------------------------------------------
# Planner backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PlannerBackend(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> str: ...


# ---------------------------------------------------------------------------
# DeepPlanner
# ---------------------------------------------------------------------------

_AGENDA_SYSTEM = """\
You are a general-purpose workflow orchestration planner.
You have a PI directive and a catalog of available agent tools.
Your task is to produce a structured list of questions you must answer before
you can commit to an execution plan.

The questions must be derived entirely from the directive text and the tool
catalog.  Do NOT assume the workflow involves papers, articles, simulations, or
any particular domain — the directive alone determines what is needed.

Return ONLY valid JSON matching this schema:
{
  "directive_summary": "one-sentence summary of what the PI wants",
  "tool_catalog_digest": "comma-separated list of top-level tool capabilities drawn from the catalog",
  "steps": [
    {
      "step_id": "q1",
      "question": "Exactly what output does the PI need — a report, a dataset, a running model?",
      "rationale": "Determines which output tools to select and what counts as done.",
      "context_keys": ["<tool_name_from_catalog>"],
      "priority": 0
    }
  ]
}

Rules:
- 3 to 8 questions maximum.
- Every question must target a decision that materially affects which tools are
  chosen or in what order — skip questions whose answers are obvious from the
  directive text.
- context_keys must reference ONLY tool names that appear in the provided
  catalog; do not invent tool names.
- Order steps by dependency (answer earlier questions before later ones).
- Do NOT start planning yet — only generate the agenda.
- Do NOT hard-code domain knowledge; reason only from the directive and catalog.
"""

_RESOLVE_SYSTEM = """\
You are a research orchestration planner resolving one planning question.
Use the directive, available tools, and previous answers to answer the question.

Return ONLY valid JSON matching this schema:
{
  "step_id": "q1",
  "answer": "concise answer",
  "confidence": 0.9,
  "assumptions": ["assumption A", "assumption B"],
  "constraints": ["use tool X before Y", "skip simulation if no model found"],
  "flags": ["potential ambiguity in directive"]
}
"""

_SYNTHESIZE_SYSTEM = """\
You are a general-purpose workflow orchestration planner.
Given a PI directive, a catalog of available agent tools, and the answers to
your planning questions, produce a concrete execution plan.

Return ONLY valid JSON:
{
  "reasoning": "why these steps in this order, derived from directive + thinking answers",
  "steps": [
    {
      "tool": "tool_name",
      "inputs": {"key": "value"},
      "artifact_inputs": {"param_name": "artifact_kind"},
      "label": "Human description",
      "condition": ""
    }
  ]
}

Rules:
- Use ONLY tools that appear in the provided catalog; do not invent tool names.
- Wire artifact_inputs to reference artifact kinds produced by earlier steps.
- Apply every constraint produced during the thinking phase.
- Prefer depth over breadth: sequential steps when later ones depend on earlier.
- The plan must be directly executable; no placeholder steps.
- Do NOT assume a fixed workflow shape (e.g. article → literature → simulation).
  The steps must follow solely from the directive and the thinking answers.
"""


class DeepPlanner:
    """
    Multi-step planner that reasons before acting.

    Phase 0: generate_agenda  — "what do I need to figure out?"
    Phase 1-N: resolve_step   — answer each question in order
    Final: synthesize_plan    — produce the ExecutionPlan

    Falls back to single-shot planning when no backend is available.
    """

    def __init__(
        self,
        backend: Optional[PlannerBackend] = None,
        *,
        max_thinking_steps: int = 8,
        confidence_warn_threshold: float = 0.5,
    ) -> None:
        self._backend = backend
        self._max_thinking_steps = max_thinking_steps
        self._confidence_warn = confidence_warn_threshold

    # -- public API ----------------------------------------------------------

    def plan(
        self,
        directive: Any,
        tool_catalog: list[dict[str, Any]],
        *,
        steering_context: Optional[str] = None,
    ) -> DeepPlanResult:
        """
        Full deep-planning pipeline.

        Parameters
        ----------
        directive:
            The PI directive.  Only `.instruction` and (optionally) `.topic_hint`
            are read.  Phase lists or domain-specific attributes are ignored —
            the plan is derived from the directive text and the tool catalog alone.
        tool_catalog:
            JSON-safe tool catalog from Orchestrator.tool_catalog().
        steering_context:
            Optional accumulated PI steering text to fold into planning.
        """
        instruction = getattr(directive, "instruction", str(directive))
        topic_hint  = getattr(directive, "topic_hint", "")

        if self._backend is None:
            return self._fallback_result(directive, tool_catalog)

        catalog_json = json.dumps(tool_catalog, indent=2)
        steering_block = (
            f"\n\nPI steering context (fold into plan):\n{steering_context}"
            if steering_context else ""
        )
        topic_block = f"\nAdditional context: {topic_hint}" if topic_hint.strip() else ""
        directive_block = (
            f"Instruction: {instruction}"
            f"{topic_block}"
            f"{steering_block}"
        )

        # Phase 0: agenda
        agenda = self._generate_agenda(directive_block, catalog_json)
        log.info("Deep planner: agenda with %d thinking steps", len(agenda.steps))

        # Phase 1-N: resolve each thinking step
        thinking_results: list[ThinkingResult] = []
        for step in agenda.steps[: self._max_thinking_steps]:
            result = self._resolve_step(step, directive_block, catalog_json, thinking_results)
            thinking_results.append(result)
            if result.confidence < self._confidence_warn:
                log.warning(
                    "Low confidence (%.2f) on thinking step %s: %s",
                    result.confidence, step.step_id, step.question,
                )

        # Aggregate constraints + assumptions
        constraints: list[str] = []
        assumptions: list[str] = []
        warnings:    list[str] = []
        for r in thinking_results:
            constraints.extend(r.constraints)
            assumptions.extend(r.assumptions)
            if r.confidence < self._confidence_warn:
                warnings.append(
                    f"Low confidence ({r.confidence:.0%}) on '{r.question}': {r.answer}"
                )
            warnings.extend(r.flags)

        # Final: synthesize plan
        plan = self._synthesize_plan(
            directive_block, catalog_json, agenda, thinking_results, constraints
        )

        return DeepPlanResult(
            plan=plan,
            agenda=agenda,
            thinking_results=thinking_results,
            aggregated_constraints=_dedupe(constraints),
            aggregated_assumptions=_dedupe(assumptions),
            planning_warnings=_dedupe(warnings),
        )

    # -- internals -----------------------------------------------------------

    def _generate_agenda(
        self, directive_block: str, catalog_json: str
    ) -> ThinkingAgenda:
        prompt = (
            f"## PI Directive\n{directive_block}\n\n"
            f"## Available Tools\n{catalog_json}\n\n"
            "Generate the thinking agenda."
        )
        try:
            raw = self._backend.complete(  # type: ignore[union-attr]
                system=_AGENDA_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            payload = _parse_json(raw)
            steps = [
                ThinkingStep(
                    step_id=str(item.get("step_id") or f"q{i+1}"),
                    question=str(item.get("question") or ""),
                    rationale=str(item.get("rationale") or ""),
                    context_keys=list(item.get("context_keys") or []),
                    priority=int(item.get("priority") or i),
                )
                for i, item in enumerate(payload.get("steps") or [])
                if str(item.get("question") or "").strip()
            ]
            steps.sort(key=lambda s: s.priority)
            return ThinkingAgenda(
                directive_summary=str(payload.get("directive_summary") or ""),
                steps=steps,
                tool_catalog_digest=str(payload.get("tool_catalog_digest") or ""),
            )
        except Exception as exc:
            log.warning("Agenda generation failed (%s), using empty agenda", exc)
            return ThinkingAgenda(directive_summary="", steps=[])

    def _resolve_step(
        self,
        step: ThinkingStep,
        directive_block: str,
        catalog_json: str,
        previous: list[ThinkingResult],
    ) -> ThinkingResult:
        prev_block = ""
        if previous:
            prev_block = "\n\n## Previous answers\n" + "\n".join(
                f"[{r.step_id}] {r.question}\n→ {r.answer}" for r in previous
            )
        prompt = (
            f"## PI Directive\n{directive_block}\n\n"
            f"## Available Tools (digest)\n{catalog_json[:4000]}\n"
            f"{prev_block}\n\n"
            f"## Question to answer ({step.step_id})\n{step.question}\n"
            f"Rationale: {step.rationale}"
        )
        try:
            raw = self._backend.complete(  # type: ignore[union-attr]
                system=_RESOLVE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            payload = _parse_json(raw)
            return ThinkingResult(
                step_id=str(payload.get("step_id") or step.step_id),
                question=step.question,
                answer=str(payload.get("answer") or ""),
                confidence=float(payload.get("confidence") or 1.0),
                assumptions=list(payload.get("assumptions") or []),
                constraints=list(payload.get("constraints") or []),
                flags=list(payload.get("flags") or []),
            )
        except Exception as exc:
            log.warning("Step resolve failed (%s): %s", step.step_id, exc)
            return ThinkingResult(
                step_id=step.step_id,
                question=step.question,
                answer="[resolve failed]",
                confidence=0.0,
                flags=[str(exc)],
            )

    def _synthesize_plan(
        self,
        directive_block: str,
        catalog_json: str,
        agenda: ThinkingAgenda,
        results: list[ThinkingResult],
        constraints: list[str],
    ) -> Any:
        """Returns an ExecutionPlan (imported lazily to avoid circular import)."""
        from .orchestrator import ExecutionPlan, PlanStep  # local import

        answers_block = "\n".join(
            f"[{r.step_id}] {r.question}\n→ {r.answer}" for r in results
        )
        constraint_block = "\n".join(f"- {c}" for c in constraints) or "(none)"
        prompt = (
            f"## PI Directive\n{directive_block}\n\n"
            f"## Available Tools\n{catalog_json}\n\n"
            f"## Thinking answers\n{answers_block}\n\n"
            f"## Plan constraints\n{constraint_block}\n\n"
            "Produce the execution plan."
        )
        try:
            raw = self._backend.complete(  # type: ignore[union-attr]
                system=_SYNTHESIZE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            payload = _parse_json(raw)
            steps = [
                PlanStep(
                    tool=str(item.get("tool") or ""),
                    inputs=dict(item.get("inputs") or {}),
                    artifact_inputs=dict(item.get("artifact_inputs") or {}),
                    label=str(item.get("label") or ""),
                    condition=str(item.get("condition") or ""),
                )
                for item in (payload.get("steps") or [])
                if str(item.get("tool") or "").strip()
            ]
            return ExecutionPlan(
                steps=steps,
                reasoning=str(payload.get("reasoning") or ""),
            )
        except Exception as exc:
            log.warning("Plan synthesis failed (%s), returning empty plan", exc)
            from .orchestrator import ExecutionPlan
            return ExecutionPlan(steps=[], reasoning=f"synthesis failed: {exc}")

    def _fallback_result(self, directive: Any, tool_catalog: list[dict[str, Any]]) -> DeepPlanResult:
        from .orchestrator import ExecutionPlan
        agenda = ThinkingAgenda(
            directive_summary=getattr(directive, "instruction", ""),
            steps=[],
        )
        return DeepPlanResult(
            plan=ExecutionPlan(steps=[], reasoning="no backend"),
            agenda=agenda,
        )


# ---------------------------------------------------------------------------
# Steering router
# ---------------------------------------------------------------------------

@dataclass
class SteeringShortcut:
    """
    A reusable shortcut pattern that maps a class of PI steering text to a
    concrete plan mutation.

    pattern_keywords: any of these words / phrases in the steering text
                      triggers a candidate match.
    action:           what to do to the live plan.
    target:           tool name or step-label substring used by SKIP_TO.
    description:      shown to the PI in confirmation messages.
    """
    name: str
    description: str
    pattern_keywords: list[str]
    action: ShortcutAction
    target: str = ""
    min_confidence: float = 0.75


@dataclass
class ShortcutMatch:
    """Result of routing PI steering through the shortcut catalog."""
    shortcut: SteeringShortcut
    confidence: float
    # Concrete plan mutations:
    steps_to_skip_until: int = -1        # skip from current step to this index (exclusive)
    steps_to_inject: list[Any] = field(default_factory=list)   # PlanStep objects to insert
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.confidence >= self.shortcut.min_confidence


# Built-in shortcut catalog
_BUILTIN_SHORTCUTS: list[SteeringShortcut] = [
    SteeringShortcut(
        name="skip_to_write",
        description="Jump directly to the writing phase, skipping remaining upstream steps.",
        pattern_keywords=["skip to write", "just write", "write now", "go to write", "jump to write"],
        action=ShortcutAction.SKIP_TO,
        target="assemble_review",
    ),
    SteeringShortcut(
        name="skip_to_simulation",
        description="Jump directly to simulation, skipping remaining literature steps.",
        pattern_keywords=["skip to sim", "just simulate", "start simulation", "go to simulation"],
        action=ShortcutAction.SKIP_TO,
        target="design_simulation_spec",
    ),
    SteeringShortcut(
        name="use_defaults",
        description="Accept defaults for all pending clarifications and continue.",
        pattern_keywords=["use defaults", "accept defaults", "continue with defaults", "default settings", "use default"],
        action=ShortcutAction.REPLACE_TAIL,
        target="",
    ),
    SteeringShortcut(
        name="abort_and_stop",
        description="Stop the workflow immediately.",
        pattern_keywords=["stop", "abort", "cancel", "halt", "quit", "terminate"],
        action=ShortcutAction.ABORT_STOP,
        target="",
        min_confidence=0.85,
    ),
    SteeringShortcut(
        name="replan",
        description="Discard the current plan and replan with the steering baked in.",
        pattern_keywords=["replan", "start over", "redo", "rethink", "new plan", "revise plan"],
        action=ShortcutAction.ABORT_REPLAN,
        target="",
    ),
]


class SteeringRouter:
    """
    Routes PI steering text to a ShortcutMatch (fast path) or signals
    that a full replan is needed (slow path).

    Fast-path matching uses keyword heuristics + optional LLM scoring.
    Slow-path is signalled by returning None.

    Usage::

        router = SteeringRouter(backend=llm, shortcuts=_BUILTIN_SHORTCUTS)
        match = router.route(steering_text, plan, current_step_idx, completed_artifacts)
        if match and match.is_actionable:
            # apply match.shortcut.action
        else:
            # full replan
    """

    def __init__(
        self,
        backend: Optional[PlannerBackend] = None,
        *,
        shortcuts: Optional[list[SteeringShortcut]] = None,
        extra_shortcuts: Optional[list[SteeringShortcut]] = None,
    ) -> None:
        self._backend = backend
        self._shortcuts: list[SteeringShortcut] = list(shortcuts or _BUILTIN_SHORTCUTS)
        if extra_shortcuts:
            self._shortcuts.extend(extra_shortcuts)

    def add_shortcut(self, shortcut: SteeringShortcut) -> None:
        self._shortcuts.append(shortcut)

    def route(
        self,
        steering_text: str,
        plan: Any,                         # ExecutionPlan
        current_step_idx: int,
        completed_artifacts: dict[str, list[Any]],
        *,
        inject_factory: Optional[Any] = None,  # callable(steering_text) → list[PlanStep]
    ) -> Optional[ShortcutMatch]:
        """
        Match steering_text against the shortcut catalog.

        Returns ShortcutMatch if a confident match is found, else None.
        Caller should replan when None is returned.
        """
        text_lower = steering_text.lower().strip()
        candidates: list[tuple[SteeringShortcut, float]] = []

        for shortcut in self._shortcuts:
            score = _keyword_score(text_lower, shortcut.pattern_keywords)
            if score > 0:
                candidates.append((shortcut, score))

        if not candidates:
            log.debug("SteeringRouter: no keyword candidates for %r", steering_text[:80])
            return None

        # Sort by score descending
        candidates.sort(key=lambda t: t[1], reverse=True)
        best_shortcut, best_score = candidates[0]

        # Optionally boost with LLM scoring when the heuristic is ambiguous
        if self._backend is not None and 0.3 < best_score < 0.8:
            best_score = self._llm_score(steering_text, best_shortcut)

        if best_score < best_shortcut.min_confidence:
            log.debug(
                "SteeringRouter: best candidate %r score %.2f below threshold %.2f",
                best_shortcut.name, best_score, best_shortcut.min_confidence,
            )
            return None

        return self._build_match(
            best_shortcut, best_score, steering_text, plan, current_step_idx, inject_factory
        )

    # -- internals -----------------------------------------------------------

    def _build_match(
        self,
        shortcut: SteeringShortcut,
        confidence: float,
        steering_text: str,
        plan: Any,
        current_step_idx: int,
        inject_factory: Optional[Any],
    ) -> ShortcutMatch:
        steps = list(getattr(plan, "steps", []))

        if shortcut.action == ShortcutAction.SKIP_TO:
            # Find the first step at or after current whose tool or label matches target
            target = shortcut.target.lower()
            skip_until = len(steps)  # default: skip to end
            for i in range(current_step_idx, len(steps)):
                step = steps[i]
                if target in step.tool.lower() or target in step.label.lower():
                    skip_until = i
                    break
            return ShortcutMatch(
                shortcut=shortcut,
                confidence=confidence,
                steps_to_skip_until=skip_until,
                notes=f"Skipping to step {skip_until} ({shortcut.target})",
            )

        if shortcut.action == ShortcutAction.INJECT_BEFORE:
            injected = inject_factory(steering_text) if callable(inject_factory) else []
            return ShortcutMatch(
                shortcut=shortcut,
                confidence=confidence,
                steps_to_inject=injected,
                notes=f"Injecting {len(injected)} step(s) before step {current_step_idx}",
            )

        if shortcut.action == ShortcutAction.INJECT_AFTER:
            injected = inject_factory(steering_text) if callable(inject_factory) else []
            return ShortcutMatch(
                shortcut=shortcut,
                confidence=confidence,
                steps_to_inject=injected,
                notes=f"Injecting {len(injected)} step(s) after step {current_step_idx}",
            )

        # REPLACE_TAIL, ABORT_REPLAN, ABORT_STOP — no step mutations needed here
        return ShortcutMatch(
            shortcut=shortcut,
            confidence=confidence,
            notes=f"Action: {shortcut.action.value}",
        )

    def _llm_score(self, steering_text: str, shortcut: SteeringShortcut) -> float:
        prompt = (
            f"Rate how well this PI steering message matches the shortcut described below.\n\n"
            f"Steering: \"{steering_text[:400]}\"\n"
            f"Shortcut: {shortcut.name} — {shortcut.description}\n\n"
            f"Return ONLY a JSON object: {{\"score\": 0.85}}\n"
            f"Score is 0.0 (no match) to 1.0 (perfect match)."
        )
        try:
            raw = self._backend.complete(  # type: ignore[union-attr]
                system="You are a steering classifier. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            payload = _parse_json(raw)
            return float(payload.get("score") or 0.0)
        except Exception as exc:
            log.debug("LLM steering score failed: %s", exc)
            return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_score(text: str, keywords: list[str]) -> float:
    """Heuristic keyword match score in [0, 1]."""
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return min(1.0, matches / max(1, min(len(keywords), 3)) * 0.9 + (0.1 if matches else 0.0))


def _parse_json(raw: Any) -> dict[str, Any]:
    """Parse a raw LLM response as JSON, stripping markdown fences."""
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = item.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
