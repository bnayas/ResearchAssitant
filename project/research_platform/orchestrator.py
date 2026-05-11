"""
research_platform.orchestrator
──────────────────────────────
LLM-planned, tool-driven orchestrator — upgraded with deep planning and
steering shortcuts.

Planning tiers
──────────────
1. Deep planning (DeepPlanner)
   Before committing to an execution plan the orchestrator runs a
   structured planning loop:

     Phase 0  — generate_agenda: "what needs to be resolved?"
                Produces a ThinkingAgenda (3-8 ThinkingSteps).
     Phase 1-N — resolve_step: answer each question in turn,
                accumulating constraints and assumptions.
     Final     — synthesize_plan: build the ExecutionPlan from answers
                and constraints.

   The full trace is emitted as a planning message so the PI can inspect
   the decisions before execution begins.

   Use plan_deep() for the first plan.  Subsequent replans triggered by
   steering use the lighter plan() (single LLM call), with the
   accumulated steering text folded into the prompt.

2. Steering shortcuts (SteeringRouter)
   When PI steering arrives mid-execution the router first checks the
   shortcut catalog for a high-confidence match:

     SKIP_TO       — jump forward to a named step (e.g. "skip to write")
     INJECT_BEFORE — insert steps before the current step
     REPLACE_TAIL  — replace remaining steps (e.g. "use defaults")
     ABORT_REPLAN  — discard plan, trigger a new deep plan with steering
     ABORT_STOP    — stop the workflow immediately

   If no shortcut matches with sufficient confidence the orchestrator
   falls back to a full replan with the steering text baked in.

Backward compatibility
──────────────────────
run() signature is unchanged; plan_deep() is a new entry point.
Existing callers that pass plan() a directive still work.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .agents.base import BaseAgent, ToolContext, ToolDescriptor, ToolResult
from .contracts import ArtifactRef, AssistantId, AttachmentRef, MailboxMessage
from .flow_management import FlowManagementService
from .planning import (
    DeepPlanResult,
    DeepPlanner,
    ShortcutAction,
    ShortcutMatch,
    SteeringRouter,
    SteeringShortcut,
)
from .service_contracts import OrchestrationRequestEnvelope

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """One step in the execution plan produced by the LLM planner."""
    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    artifact_inputs: dict[str, str] = field(default_factory=dict)
    label: str = ""
    condition: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "inputs": self.inputs,
            "artifact_inputs": self.artifact_inputs,
            "label": self.label,
            "condition": self.condition,
        }


@dataclass
class ExecutionPlan:
    """Ordered list of PlanSteps produced by the planner."""
    steps: list[PlanStep] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
        }


@dataclass
class StepResult:
    """Outcome of executing one plan step."""
    step: PlanStep
    result: ToolResult
    step_index: int = 0


# ---------------------------------------------------------------------------
# Internal run state
# ---------------------------------------------------------------------------

@dataclass
class _RunState:
    """Mutable state shared across the execution loop of one run() call."""
    directive_id: str
    instruction: str
    out: Path
    steps: list[PlanStep]
    completed_artifacts: dict[str, list[ArtifactRef]] = field(default_factory=dict)
    all_artifacts: list[ArtifactRef] = field(default_factory=list)
    inquiry_answers: dict[str, Any] = field(default_factory=dict)
    step_results: list[StepResult] = field(default_factory=list)
    steering_log: list[str] = field(default_factory=list)   # accumulated PI steering text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    LLM-planned, tool-driven research workflow orchestrator.

    Usage (deep planning, recommended)::

        result = orchestrator.run(
            directive,
            mailbox,
            deep=True,      # ← enables multi-step planning
        )

    Usage (quick single-shot planning)::

        result = orchestrator.run(directive, mailbox)

    Steering mid-run::

        orchestrator.inject_steering(directive_id, "skip to write")
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        *,
        planner_backend: Any = None,
        flow_management: Optional[FlowManagementService] = None,
        on_message: Optional[Callable[[MailboxMessage], None]] = None,
        extra_shortcuts: Optional[list[SteeringShortcut]] = None,
    ) -> None:
        # Agent + tool registry
        self._agents: dict[str, BaseAgent] = {}
        self._tool_registry: dict[str, tuple[BaseAgent, ToolDescriptor]] = {}
        for agent in agents:
            agent_id = agent.agent_id()
            if agent_id in self._agents:
                raise ValueError(f"Duplicate agent_id registered: {agent_id}")
            agent.validate_catalog()
            self._agents[agent_id] = agent
            for tool in agent.tools():
                if tool.name in self._tool_registry:
                    owner = self._tool_registry[tool.name][0].agent_id()
                    raise ValueError(
                        f"Duplicate tool name: {tool.name} (agents: {owner}, {agent_id})"
                    )
                self._tool_registry[tool.name] = (agent, tool)

        self._planner_backend = planner_backend
        self._flow = flow_management
        self._on_message = on_message

        # Planning helpers
        self._deep_planner = DeepPlanner(backend=planner_backend)
        self._steering_router = SteeringRouter(
            backend=planner_backend,
            extra_shortcuts=extra_shortcuts,
        )

        # Per-directive pending steering injections
        # directive_id → list of steering text blocks
        self._pending_steering: dict[str, list[str]] = {}

    # ── public properties ────────────────────────────────────────────────────

    @property
    def tool_count(self) -> int:
        return len(self._tool_registry)

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    def tool_catalog(self) -> list[dict[str, Any]]:
        """JSON-safe tool catalog for the LLM planner."""
        return [desc.to_dict() for _, desc in self._tool_registry.values()]

    def tool_catalog_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.tool_catalog(), indent=indent)

    # ── steering injection ───────────────────────────────────────────────────

    def inject_steering(self, directive_id: str, steering_text: str) -> None:
        """
        Thread-safe-ish steering injection (called from the API layer while
        run() is blocking).  The text is queued and consumed at the next
        step boundary.
        """
        self._pending_steering.setdefault(directive_id, []).append(steering_text.strip())
        log.info("Steering queued for %s: %r", directive_id, steering_text[:80])

    def _pop_pending_steering(self, directive_id: str) -> Optional[str]:
        queue = self._pending_steering.get(directive_id)
        if not queue:
            return None
        text = queue.pop(0)
        if not queue:
            del self._pending_steering[directive_id]
        return text

    # ── planning ─────────────────────────────────────────────────────────────

    def plan(self, directive: Any, *, steering_context: str = "") -> ExecutionPlan:
        """
        Single-shot LLM planning (fast path, used for replans).
        Folds steering_context into the prompt when provided.

        The planner receives only the directive text and the tool catalog.
        No phase names or domain examples are injected — the plan is derived
        solely from what the PI said and which tools exist.
        """
        catalog = self.tool_catalog_json()
        instruction = getattr(directive, "instruction", str(directive))
        topic_hint  = getattr(directive, "topic_hint", "")

        steering_block = (
            f"\n\nPI steering (integrate into plan):\n{steering_context}"
            if steering_context else ""
        )
        topic_block = f"\nAdditional context: {topic_hint}" if topic_hint.strip() else ""

        prompt = (
            f"You are a general-purpose workflow orchestration planner.\n"
            f"Produce an execution plan for this directive using ONLY the tools in the catalog.\n\n"
            f"## Directive\n{instruction}"
            f"{topic_block}"
            f"{steering_block}\n\n"
            f"## Available Tools\n{catalog}\n\n"
            f"## Output Format\n"
            f"Return ONLY valid JSON — no commentary:\n"
            f'{{"reasoning": "why these steps", '
            f'"steps": [{{"tool": "tool_name", "inputs": {{}}, '
            f'"artifact_inputs": {{}}, "label": "description"}}]}}\n\n'
            f"Rules:\n"
            f"- Use ONLY tool names from the catalog above.\n"
            f"- Do NOT assume a fixed workflow shape; derive steps from the directive.\n"
            f"- Wire artifact_inputs using artifact kinds produced by earlier steps."
        )

        if self._planner_backend is None:
            return self._default_plan(directive)

        try:
            raw = self._planner_backend.complete(
                system="You are a general-purpose workflow planner. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            payload = _parse_json_payload(raw)
            return self._parse_plan(payload)
        except Exception as exc:
            log.warning("LLM planning failed (%s), using default plan", exc)
            return self._default_plan(directive)

    def plan_deep(
        self,
        directive: Any,
        *,
        mailbox: Optional[list] = None,
        steering_context: str = "",
    ) -> DeepPlanResult:
        """
        Multi-step deep planning.

        Phase 0 — generate planning agenda ("what needs to be resolved?")
        Phase 1-N — resolve each planning step
        Final — synthesize the ExecutionPlan from the results

        Emits a mailbox message with the full planning trace so the PI can
        inspect the planner decisions before execution begins.
        """
        directive_id = getattr(directive, "directive_id", "default")

        result = self._deep_planner.plan(
            directive,
            self.tool_catalog(),
            steering_context=steering_context or None,
        )

        # Build a human-readable planning trace for the mailbox
        agenda_lines = "\n".join(
            f"  {s.step_id}. {s.question}" for s in result.agenda.steps
        ) or "  (no planning steps generated)"

        answer_lines = "\n".join(
            f"  [{r.step_id}] {r.question}\n"
            f"    → {r.answer}"
            + (f"\n    ⚠ {'; '.join(r.flags)}" if r.flags else "")
            for r in result.thinking_results
        ) or "  (none)"

        constraint_lines = (
            "\n".join(f"  • {c}" for c in result.aggregated_constraints)
            or "  (none)"
        )
        warning_lines = (
            "\n".join(f"  ⚠ {w}" for w in result.planning_warnings)
            or ""
        )

        body = (
            f"Summary: {result.agenda.directive_summary}\n\n"
            f"Planning agenda ({len(result.agenda.steps)} questions):\n{agenda_lines}\n\n"
            f"Resolved answers:\n{answer_lines}\n\n"
            f"Constraints applied to plan:\n{constraint_lines}\n\n"
            f"Plan: {len(result.plan.steps)} steps\n{result.plan.reasoning}"
            + (f"\n\nWarnings:\n{warning_lines}" if warning_lines else "")
        )

        self._emit_message(
            mailbox, directive_id, AssistantId.ORCHESTRATOR.value,
            "Planning complete",
            body,
            metadata={"kind": "deep_plan_trace", "plan_result": result.to_dict()},
        )

        if result.planning_warnings:
            log.warning(
                "Deep planner produced %d warning(s) for %s",
                len(result.planning_warnings), directive_id,
            )

        return result

    # ── execution ────────────────────────────────────────────────────────────

    def run(
        self,
        directive: Any,
        mailbox: Optional[list] = None,
        *,
        output_dir: Optional[Path] = None,
        deep: bool = True,
    ) -> list[ArtifactRef]:
        """
        Plan and execute the full workflow.

        Parameters
        ----------
        directive:
            The PI directive.
        mailbox:
            Optional list to append MailboxMessage objects to.
        output_dir:
            Override for the output directory.
        deep:
            When True (default), use multi-step deep planning for the first
            plan.  Replans triggered by steering always use the fast path.
        """
        directive_id = getattr(directive, "directive_id", "default")
        instruction  = getattr(directive, "instruction", str(directive))
        out = Path(output_dir or getattr(directive, "output_dir", "."))
        out.mkdir(parents=True, exist_ok=True)

        if self._flow:
            self._flow.open_workflow(directive_id, metadata={"instruction": instruction})
            self._flow.register_session(
                directive_id,
                AssistantId.ORCHESTRATOR.value,
                f"{directive_id}:orchestrator",
                metadata={"role": "orchestrator"},
            )

        # ── Planning ────────────────────────────────────────────────────────
        if deep and self._planner_backend is not None:
            deep_result = self.plan_deep(directive, mailbox=mailbox)
            execution_plan = deep_result.plan
        else:
            execution_plan = self.plan(directive)
            self._emit_message(
                mailbox, directive_id, AssistantId.ORCHESTRATOR.value,
                "Workflow started",
                f"Plan: {len(execution_plan.steps)} steps\n{execution_plan.reasoning}",
            )

        # ── Execution state ──────────────────────────────────────────────────
        state = _RunState(
            directive_id=directive_id,
            instruction=instruction,
            out=out,
            steps=list(execution_plan.steps),
        )

        idx = 0
        while idx < len(state.steps):
            step = state.steps[idx]

            # ── Flow management hooks ────────────────────────────────────────
            if self._flow:
                self._check_stop(directive_id)
                self._flow.update_session_state(
                    directive_id, f"{directive_id}:orchestrator",
                    "running",
                    metadata={"step": idx, "tool": step.tool, "label": step.label},
                )

            # ── Consume pending steering ─────────────────────────────────────
            steering_text = self._pop_pending_steering(directive_id)
            if steering_text:
                state.steering_log.append(steering_text)
                mutated = self._apply_steering(
                    steering_text=steering_text,
                    state=state,
                    directive=directive,
                    current_step_idx=idx,
                    mailbox=mailbox,
                )
                if mutated == "stop":
                    break
                if mutated == "replan":
                    # Full replan — rebuild steps from current position
                    new_plan = self.plan(
                        directive,
                        steering_context="\n".join(state.steering_log),
                    )
                    # Keep already-completed artifacts; replace remaining steps
                    state.steps = state.steps[:idx] + new_plan.steps
                    self._emit_message(
                        mailbox, directive_id, AssistantId.ORCHESTRATOR.value,
                        "♻ Replanned",
                        f"Steering applied. New plan: {len(new_plan.steps)} steps.\n{new_plan.reasoning}",
                        metadata={"kind": "replan", "new_plan": new_plan.to_dict()},
                    )
                    # Don't advance idx — re-evaluate at the same position
                    continue
                # else: "continue" — steering was handled via shortcut, steps already mutated

            log.info(
                "Step %d/%d: %s (%s)",
                idx + 1, len(state.steps), step.label, step.tool,
            )

            # ── Skip if condition not met ────────────────────────────────────
            if step.condition and not self._evaluate_condition(
                step.condition, state.completed_artifacts
            ):
                log.info("Skipping step %d: condition not met", idx)
                idx += 1
                continue

            # ── Resolve tool ─────────────────────────────────────────────────
            entry = self._tool_registry.get(step.tool)
            if entry is None:
                log.error("Unknown tool in plan: %s", step.tool)
                idx += 1
                continue
            agent, descriptor = entry

            # ── Resolve artifact inputs ──────────────────────────────────────
            artifacts = self._resolve_artifact_inputs(
                step, state.completed_artifacts, descriptor
            )

            # ── Build tool context ───────────────────────────────────────────
            context = ToolContext(
                tool_name=step.tool,
                directive_id=directive_id,
                instruction=instruction,
                inputs=dict(step.inputs),
                artifacts=artifacts,
                output_dir=out,
                metadata={
                    "step_index": idx,
                    "label": step.label,
                    "inquiry_answers": dict(state.inquiry_answers),
                    "steering_log": list(state.steering_log),
                },
            )

            # ── Validate requirements ────────────────────────────────────────
            validation = descriptor.validate_inputs(
                context.inputs, context.artifacts, context.metadata
            )
            failures = [r for r in validation if not r.ok]
            if failures:
                msgs = "; ".join(r.message for r in failures)
                log.warning("Step %d validation failed: %s", idx, msgs)
                state.step_results.append(StepResult(
                    step=step, step_index=idx,
                    result=ToolResult(status="failed", message=f"Validation failed: {msgs}"),
                ))
                idx += 1
                continue

            # ── Invoke ───────────────────────────────────────────────────────
            try:
                result = agent.invoke(step.tool, context)
            except Exception as exc:
                log.error("Step %d failed: %s\n%s", idx, exc, traceback.format_exc())
                result = ToolResult(status="failed", message=str(exc))

            # ── Handle needs_input ───────────────────────────────────────────
            if result.needs_input and result.request:
                result = self._handle_needs_input(
                    directive_id, mailbox, agent, step, context, result
                )

            if result.inquiries:
                answers = self._handle_inquiries(
                    directive_id, mailbox, descriptor.agent_id, result.inquiries
                )
                if answers:
                    state.inquiry_answers.update(answers)
                    result.metadata.setdefault("inquiry_answers", {}).update(answers)

            # ── Register produced artifacts ──────────────────────────────────
            for art in result.artifacts:
                state.completed_artifacts.setdefault(art.kind, []).append(art)
                state.all_artifacts.append(art)

            state.step_results.append(StepResult(step=step, result=result, step_index=idx))

            self._emit_message(
                mailbox, directive_id, descriptor.agent_id,
                f"{'✓' if result.ok else '✗'} {step.label}",
                result.message,
                attachments=[a.as_attachment() for a in result.artifacts],
            )

            # ── Inject follow-up steps requested by the tool ─────────────────
            if result.follow_up_steps:
                generated = self._parse_follow_up_steps(result.follow_up_steps)
                if generated:
                    state.steps[idx + 1:idx + 1] = generated

            idx += 1

        # ── Finalise ─────────────────────────────────────────────────────────
        if self._flow:
            self._flow.update_session_state(
                directive_id, f"{directive_id}:orchestrator",
                "complete",
                metadata={"steps_completed": len(state.step_results)},
            )

        self._emit_message(
            mailbox, directive_id, AssistantId.ORCHESTRATOR.value,
            "Workflow complete",
            (
                f"Completed {len(state.step_results)} steps, "
                f"produced {len(state.all_artifacts)} artifacts."
                + (
                    f"\nSteering applied {len(state.steering_log)} time(s)."
                    if state.steering_log else ""
                )
            ),
            attachments=[a.as_attachment() for a in state.all_artifacts[:10]],
        )

        return state.all_artifacts

    # ── steering application ─────────────────────────────────────────────────

    def _apply_steering(
        self,
        *,
        steering_text: str,
        state: _RunState,
        directive: Any,
        current_step_idx: int,
        mailbox: Optional[list],
    ) -> str:
        """
        Try to apply a steering shortcut.  Mutates state.steps in place.

        Returns:
            "continue"  — shortcut applied, keep executing from current idx
            "replan"    — no shortcut matched or action == ABORT_REPLAN
            "stop"      — action == ABORT_STOP
        """
        match: Optional[ShortcutMatch] = self._steering_router.route(
            steering_text=steering_text,
            plan=ExecutionPlan(steps=state.steps),
            current_step_idx=current_step_idx,
            completed_artifacts=state.completed_artifacts,
        )

        if match is None or not match.is_actionable:
            # No confident shortcut → full replan
            self._emit_message(
                mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                "⚡ Steering received — replanning",
                f"No shortcut matched for: \"{steering_text[:200]}\"\n\nReplanning…",
                metadata={"kind": "steering_replan", "steering": steering_text},
            )
            return "replan"

        action = match.shortcut.action

        if action == ShortcutAction.ABORT_STOP:
            self._emit_message(
                mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                "⛔ Workflow stopped by PI steering",
                f"Reason: {steering_text}",
                metadata={"kind": "steering_stop"},
            )
            return "stop"

        if action == ShortcutAction.ABORT_REPLAN:
            self._emit_message(
                mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                "⚡ Steering received — replanning",
                f"Shortcut '{match.shortcut.name}' matched. {match.notes}\n\nReplanning…",
                metadata={"kind": "steering_replan", "shortcut": match.shortcut.name},
            )
            return "replan"

        if action == ShortcutAction.SKIP_TO:
            skip_until = match.steps_to_skip_until
            if 0 <= skip_until < len(state.steps) and skip_until > current_step_idx:
                skipped = [s.label or s.tool for s in state.steps[current_step_idx:skip_until]]
                state.steps[current_step_idx:skip_until] = []  # delete skipped steps
                self._emit_message(
                    mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                    f"⚡ Shortcut: {match.shortcut.name}",
                    f"Skipped {len(skipped)} step(s): {', '.join(skipped[:5])}\n{match.notes}",
                    metadata={"kind": "steering_skip", "shortcut": match.shortcut.name},
                )
            return "continue"

        if action in (ShortcutAction.INJECT_BEFORE, ShortcutAction.INJECT_AFTER):
            inject_at = current_step_idx if action == ShortcutAction.INJECT_BEFORE else current_step_idx + 1
            for i, new_step in enumerate(match.steps_to_inject):
                state.steps.insert(inject_at + i, new_step)
            self._emit_message(
                mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                f"⚡ Shortcut: {match.shortcut.name}",
                f"Injected {len(match.steps_to_inject)} step(s). {match.notes}",
                metadata={"kind": "steering_inject", "shortcut": match.shortcut.name},
            )
            return "continue"

        if action == ShortcutAction.REPLACE_TAIL:
            if match.steps_to_inject:
                state.steps[current_step_idx:] = match.steps_to_inject
                self._emit_message(
                    mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                    f"⚡ Shortcut: {match.shortcut.name}",
                    f"Replaced remaining steps with {len(match.steps_to_inject)} new step(s). {match.notes}",
                    metadata={"kind": "steering_replace", "shortcut": match.shortcut.name},
                )
            else:
                # "use defaults" without explicit steps → skip clarification steps
                non_clarification = [
                    s for s in state.steps[current_step_idx:]
                    if "clarif" not in s.tool.lower() and "clarif" not in s.label.lower()
                ]
                state.steps[current_step_idx:] = non_clarification
                self._emit_message(
                    mailbox, state.directive_id, AssistantId.ORCHESTRATOR.value,
                    f"⚡ Shortcut: {match.shortcut.name}",
                    f"Removed clarification steps; continuing with {len(non_clarification)} step(s). {match.notes}",
                    metadata={"kind": "steering_defaults", "shortcut": match.shortcut.name},
                )
            return "continue"

        log.warning("Unhandled shortcut action: %s", action)
        return "replan"

    # ── planning helpers ─────────────────────────────────────────────────────

    def _parse_plan(self, payload: dict[str, Any]) -> ExecutionPlan:
        steps = []
        for item in payload.get("steps", []):
            tool = str(item.get("tool", "")).strip()
            if not tool:
                continue
            if tool not in self._tool_registry:
                log.warning("Planner proposed unknown tool %r; dropping step", tool)
                continue
            steps.append(PlanStep(
                tool=tool,
                inputs=dict(item.get("inputs", {})),
                artifact_inputs=dict(item.get("artifact_inputs", {})),
                label=str(item.get("label", "")),
                condition=str(item.get("condition", "")),
            ))
        return ExecutionPlan(
            steps=steps,
            reasoning=str(payload.get("reasoning", "")),
        )

    def _default_plan(self, directive: Any) -> ExecutionPlan:
        """
        Fallback invoked when the planner backend is unavailable.

        Rather than silently returning the old hardcoded research sequence, this
        returns an empty plan with an explicit warning in the reasoning field.
        The orchestrator will log and emit a mailbox message indicating that
        planning could not proceed without a backend.

        Callers that genuinely need a no-backend offline mode should subclass
        Orchestrator and override this method with a domain-specific fallback
        appropriate to their tool catalog — not a generic hardcoded sequence.
        """
        instruction = getattr(directive, "instruction", str(directive))
        log.error(
            "Orchestrator.plan() called with no planner backend for directive: %r — "
            "returning empty plan.  Wire a planner_backend to enable LLM planning.",
            instruction[:120],
        )
        return ExecutionPlan(
            steps=[],
            reasoning=(
                "ERROR: no planner backend is configured.  "
                "The orchestrator cannot derive an execution plan from the directive "
                "without an LLM backend.  "
                "Set planner_backend= when constructing the Orchestrator."
            ),
        )

    # ── execution helpers ────────────────────────────────────────────────────

    def _resolve_artifact_inputs(
        self,
        step: PlanStep,
        completed: dict[str, list[ArtifactRef]],
        descriptor: Optional[ToolDescriptor] = None,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        for param_name, artifact_kind in step.artifact_inputs.items():
            if artifact_kind == "_all_artifacts":
                artifacts[param_name] = [
                    art
                    for arts in completed.values()
                    for art in arts
                ]
            elif artifact_kind.startswith("$step["):
                pass  # future: resolve from specific step output
            else:
                refs = completed.get(artifact_kind) or []
                if refs:
                    requirement = (
                        descriptor.requirement_by_name(param_name) if descriptor else None
                    )
                    if requirement and requirement.type == "list[ArtifactRef]":
                        artifacts[param_name] = list(refs)
                    else:
                        artifacts[param_name] = refs[-1]
        return artifacts

    def _evaluate_condition(
        self, condition: str, completed: dict[str, list[ArtifactRef]]
    ) -> bool:
        if not condition.strip():
            return True
        if condition.startswith("has:"):
            kind = condition[4:].strip()
            return bool(completed.get(kind))
        return True

    def _parse_follow_up_steps(
        self, items: list[dict[str, Any]]
    ) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for item in items:
            try:
                tool_name = str(item.get("tool", "")).strip()
                if not tool_name:
                    continue
                steps.append(PlanStep(
                    tool=tool_name,
                    inputs=dict(item.get("inputs", {})),
                    artifact_inputs=dict(item.get("artifact_inputs", {})),
                    label=str(item.get("label", "")),
                    condition=str(item.get("condition", "")),
                ))
            except Exception as exc:
                log.warning("Ignoring malformed follow-up step %r: %s", item, exc)
        return steps

    def _handle_needs_input(
        self,
        directive_id: str,
        mailbox: Optional[list],
        agent: BaseAgent,
        step: PlanStep,
        context: ToolContext,
        result: ToolResult,
    ) -> ToolResult:
        request = result.request
        if request is None:
            return result

        cap = request.capability_hint or "user"

        if cap == "math" and "math" in self._agents:
            math_agent = self._agents["math"]
            math_ctx = ToolContext(
                tool_name="evaluate_expression",
                directive_id=directive_id,
                instruction=request.question,
                inputs={"expression": request.question},
            )
            math_result = math_agent.invoke("evaluate_expression", math_ctx)
            return ToolResult(
                status="completed" if math_result.ok else "failed",
                message=math_result.message,
                metadata=math_result.metadata,
            )

        if cap == "user":
            self._emit_message(
                mailbox, directive_id, agent.agent_id(),
                "Clarification needed",
                request.question,
                metadata={
                    "request_id": request.request_id,
                    "request": request.to_dict(),
                },
            )
            if self._flow:
                injection = self._flow.wait_for_injection(
                    f"{directive_id}:orchestrator"
                )
                context.inputs.update(injection.payload)
                try:
                    return agent.invoke(step.tool, context)
                except Exception as exc:
                    return ToolResult(status="failed", message=f"Resume failed: {exc}")

        log.warning("Unhandled needs_input capability: %s", cap)
        return result

    def _handle_inquiries(
        self,
        directive_id: str,
        mailbox: Optional[list],
        assistant: str,
        inquiries: list[OrchestrationRequestEnvelope],
    ) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for request in inquiries:
            inquiry = dict(request.metadata.get("inquiry") or {})
            inquiry_id = str(inquiry.get("id") or request.request_id)
            self._emit_message(
                mailbox, directive_id, assistant,
                "Inquiry",
                request.question,
                metadata={
                    "request_id": request.request_id,
                    "request": request.to_dict(),
                },
            )
            if self._flow and request.capability_hint == "user" and request.blocking:
                self._flow.register_session(
                    directive_id, assistant, request.resume_token,
                    metadata={
                        "request_kind": request.request_kind,
                        "request": request.to_dict(),
                    },
                )
                self._flow.update_session_state(
                    directive_id, request.resume_token, "waiting",
                    metadata={
                        "request_kind": request.request_kind,
                        "request": request.to_dict(),
                    },
                )
                injection = self._flow.wait_for_injection(request.resume_token)
                answers[inquiry_id] = {
                    "request_id": request.request_id,
                    "question": request.question,
                    "payload": dict(injection.payload),
                    "inquiry": inquiry,
                }
                self._flow.update_session_state(
                    directive_id, request.resume_token, "complete",
                    metadata={"answer": answers[inquiry_id]},
                )
            else:
                answers[inquiry_id] = self._default_inquiry_answer(request)
        return answers

    @staticmethod
    def _default_inquiry_answer(
        request: OrchestrationRequestEnvelope,
    ) -> dict[str, Any]:
        inquiry = dict(request.metadata.get("inquiry") or {})
        options = list(inquiry.get("options") or [])
        default_policy = str(inquiry.get("default_policy") or "ask").lower()
        selected = []
        if options and default_policy in {"all", "both", "everything"}:
            selected = options
        elif options and default_policy in {"first", "default"}:
            selected = options[:1]
        return {
            "request_id": request.request_id,
            "question": request.question,
            "payload": {
                "default_policy": default_policy,
                "selected_options": selected,
            },
            "inquiry": inquiry,
            "defaulted": True,
        }

    def _check_stop(self, directive_id: str) -> None:
        if self._flow is None:
            return
        snapshot = self._flow.get_workflow_snapshot(directive_id)
        if snapshot["stop_requested"]:
            raise RuntimeError(snapshot["stop_reason"] or "Workflow stop requested")

    def _emit_message(
        self,
        mailbox: Optional[list],
        directive_id: str,
        assistant: str,
        subject: str,
        body: str,
        attachments: Optional[list[AttachmentRef]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if mailbox is None and self._on_message is None:
            return
        msg = MailboxMessage(
            directive_id=directive_id,
            assistant=assistant,
            assistant_addr=f"{assistant}@research.local",
            to_name="PI",
            subject=subject,
            body=body,
            attachments=list(attachments or []),
            metadata=dict(metadata or {}),
        )
        if mailbox is not None:
            mailbox.append(msg)
        if self._on_message is not None:
            self._on_message(msg)


def _parse_json_payload(raw: Any) -> dict[str, Any]:
    """Parse LLM JSON, accepting fenced or prefixed text."""
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise
