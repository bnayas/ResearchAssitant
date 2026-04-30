from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from sim_tool.contract import ClarificationRequest, QuestionTopic
from sim_tool.orchestrator import OrchestratorState, ResearchOrchestrator
from sim_tool.tool import SimulationTool

from ..contracts import ArtifactRef, AssistantId, TaskEnvelope
from ..registry import ArtifactRegistry
from ..steering import SteeringDecision
from .backends import make_service_backend
from .errors import AssistantExecutionError
from .utils import jsonify, truncate


class LocalCodingAssistant:
    def __init__(self, registry: ArtifactRegistry, llm_config: Optional[Any]) -> None:
        self._registry = registry
        self._llm_config = llm_config

    def execute_task(
        self,
        task: TaskEnvelope,
        *,
        sources: list[ArtifactRef],
        update_callback: Optional[Callable[[str, list[ArtifactRef], str], None]] = None,
        review_callback: Optional[Callable[[str, list[ArtifactRef], str], SteeringDecision]] = None,
        cancel_callback: Optional[Callable[[], None]] = None,
    ) -> list[ArtifactRef]:
        brief = self._require_source(
            sources,
            artifact_id="article-brief",
            kind="article_brief",
            label="article brief",
        )
        literature = self._require_source(
            sources,
            artifact_id="literature-review",
            kind="literature_review",
            label="literature review",
        )
        output_root = task.output_path / "simulation_results" if task.output_path else Path("./simulation_results")
        tool = SimulationTool(
            output_root=output_root,
            designer_backend=make_service_backend("designer", self._llm_config),
            analyst_backend=make_service_backend("analyst", self._llm_config),
        )
        orchestrator = ResearchOrchestrator(
            tool,
            sample_steps=int(task.metadata.get("sample_steps", 200)),
        )
        research_goal = self._compose_research_goal(task, brief, literature)
        self._check_cancel(cancel_callback)
        update = orchestrator.start(research_goal)
        update = self._resolve_clarifications(orchestrator, update, brief, literature)
        while True:
            self._check_cancel(cancel_callback)
            if update.state != OrchestratorState.AWAITING_SPEC_APPROVAL:
                raise AssistantExecutionError(
                    f"Expected spec approval gate, got {update.state.value}"
                )

            spec_artifact = self._registry.save_text(
                assistant=AssistantId.CODING_AGENT.value,
                kind="spec_summary",
                title="Simulation Spec Summary",
                filename="simulation/spec_summary.txt",
                text=str(update.spec_approval or update),
                summary=truncate(update.message),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-spec-summary",
            )
            if update_callback and review_callback is None:
                update_callback(
                    "simulation_spec_ready",
                    [spec_artifact],
                    "A simulation specification is ready for PI review before code generation.",
                )
            decision = self._review(
                review_callback,
                "simulation_spec_ready",
                [spec_artifact],
                (
                    "The coding agent prepared a simulation specification. Review the spec summary "
                    "before any code is generated. Continue to generate code, or request a revision "
                    "with concrete feedback."
                ),
            )
            if self._is_revision(decision):
                update = orchestrator.reject(update.session_id, decision.feedback.strip())
                update = self._resolve_clarifications(orchestrator, update, brief, literature)
                continue

            self._check_cancel(cancel_callback)
            generated = orchestrator.approve(update.session_id)
            if generated.state != OrchestratorState.READY_TO_SAMPLE or generated.artifacts is None:
                raise AssistantExecutionError(
                    f"Expected generated artifacts, got {generated.state.value}"
                )
            code_payload = jsonify(generated.artifacts)
            code_artifact = self._registry.save_json(
                assistant=AssistantId.CODING_AGENT.value,
                kind="generated_code",
                title="Generated Simulation Artifacts",
                filename="simulation/generated_artifacts.json",
                payload=code_payload,
                summary=str(generated.artifacts.script_path),
                metadata=code_payload,
                artifact_id="simulation-generated-artifacts",
            )
            script_ref = self._registry.create(
                assistant=AssistantId.CODING_AGENT.value,
                kind="simulation_script",
                title="Simulation Script",
                summary=str(Path(generated.artifacts.script_path).resolve()),
                path=str(Path(generated.artifacts.script_path).resolve()),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-script",
            )
            if update_callback:
                update_callback(
                    "simulation_launched",
                    [spec_artifact, code_artifact, script_ref],
                    "Simulation specification approved and execution artifacts generated.",
                )
            decision = self._review(
                review_callback,
                "simulation_script_ready",
                [spec_artifact, code_artifact, script_ref],
                (
                    "The generated simulation script is attached. Inspect the script or generated "
                    "artifacts now. Continue to run the sample, or request a revision with specific feedback."
                ),
            )
            if self._is_revision(decision):
                update = orchestrator.reject(update.session_id, decision.feedback.strip())
                update = self._resolve_clarifications(orchestrator, update, brief, literature)
                continue

            self._check_cancel(cancel_callback)
            sample = orchestrator.run_sample(update.session_id)
            if sample.state != OrchestratorState.AWAITING_SAMPLE_REVIEW:
                raise self._coding_failure("sample_run", sample)
            sample_summary = self._registry.save_json(
                assistant=AssistantId.CODING_AGENT.value,
                kind="sample_run_summary",
                title="Sample Run Summary",
                filename="simulation/sample_run_summary.json",
                payload=jsonify(sample.sample_summary),
                summary=truncate(sample.message),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-sample-summary",
            )
            sample_analysis = self._registry.save_json(
                assistant=AssistantId.CODING_AGENT.value,
                kind="sample_run_analysis",
                title="Sample Run Analysis",
                filename="simulation/sample_run_analysis.json",
                payload=jsonify(sample.sample_analysis),
                summary=truncate(sample.message),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-sample-analysis",
            )
            if update_callback and review_callback is None:
                update_callback(
                    "sample_reviewed",
                    [sample_summary, sample_analysis],
                    sample.message,
                )
            decision = self._review(
                review_callback,
                "sample_reviewed",
                [sample_summary, sample_analysis],
                (
                    "The sample run completed. Review the sample summary and analysis before the "
                    "full sweep launches. Continue to the full sweep, or request a revision with feedback."
                ),
            )
            if self._is_revision(decision):
                update = orchestrator.reject(update.session_id, decision.feedback.strip())
                update = self._resolve_clarifications(orchestrator, update, brief, literature)
                continue

            self._check_cancel(cancel_callback)
            launched = orchestrator.approve_sample(update.session_id)
            if launched.state != OrchestratorState.FULL_RUNNING:
                raise self._coding_failure("full_launch", launched)
            if update_callback:
                update_callback(
                    "full_sweep_launched",
                    [],
                    launched.message,
                )

            timeout_seconds = float(task.metadata.get("simulation_timeout_seconds", 600.0))
            elapsed = 0.0
            final = orchestrator.get_status(update.session_id)
            while final.state == OrchestratorState.FULL_RUNNING:
                self._check_cancel(cancel_callback)
                slice_timeout = min(2.0, max(timeout_seconds - elapsed, 0.0))
                if slice_timeout <= 0.0:
                    final = orchestrator.wait(update.session_id, timeout=0.0)
                    break
                final = orchestrator.wait(update.session_id, timeout=slice_timeout)
                elapsed += slice_timeout
            if final.state != OrchestratorState.AWAITING_RESULTS_REVIEW:
                raise self._coding_failure("full_results", final)
            full_summary = self._registry.save_json(
                assistant=AssistantId.CODING_AGENT.value,
                kind="full_run_summary",
                title="Full Run Summary",
                filename="simulation/full_run_summary.json",
                payload=jsonify(final.full_summary),
                summary=truncate(final.message),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-full-summary",
            )
            full_analysis = self._registry.save_json(
                assistant=AssistantId.CODING_AGENT.value,
                kind="full_run_analysis",
                title="Full Run Analysis",
                filename="simulation/full_run_analysis.json",
                payload=jsonify(final.full_analysis),
                summary=truncate(final.message),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-full-analysis",
            )
            results_ref = self._registry.create(
                assistant=AssistantId.CODING_AGENT.value,
                kind="simulation_results_directory",
                title="Simulation Results Directory",
                summary=str((output_root / update.session_id).resolve()),
                path=str((output_root / update.session_id).resolve()),
                metadata={"session_id": update.session_id},
                artifact_id="simulation-results-dir",
            )
            if update_callback and review_callback is None:
                update_callback(
                    "full_results_ready",
                    [full_summary, full_analysis, results_ref],
                    final.message,
                )
            decision = self._review(
                review_callback,
                "full_results_ready",
                [full_summary, full_analysis, results_ref],
                (
                    "The full sweep and analysis are ready. Continue to hand the results to the writer, "
                    "or request a revision if the parameterization or run behavior was wrong."
                ),
            )
            if self._is_revision(decision):
                update = orchestrator.reject_results(update.session_id, decision.feedback.strip())
                update = self._resolve_clarifications(orchestrator, update, brief, literature)
                continue

            orchestrator.approve_results(update.session_id)
            return [
                spec_artifact,
                code_artifact,
                script_ref,
                sample_summary,
                sample_analysis,
                full_summary,
                full_analysis,
                results_ref,
            ]

    def _resolve_clarifications(
        self,
        orchestrator: ResearchOrchestrator,
        update: Any,
        brief: ArtifactRef,
        literature: ArtifactRef,
    ) -> Any:
        rounds = 0
        while update.state == OrchestratorState.CLARIFYING:
            clarification = update.clarification
            if not isinstance(clarification, ClarificationRequest):
                raise AssistantExecutionError("Missing clarification payload from coding assistant")
            answers = self._auto_answer_questions(clarification, brief, literature)
            update = orchestrator.answer(update.session_id, answers)
            rounds += 1
            if rounds >= ClarificationRequest.MAX_ROUNDS:
                break
        return update

    def _auto_answer_questions(
        self,
        clarification: ClarificationRequest,
        brief: ArtifactRef,
        literature: ArtifactRef,
    ) -> dict[str, str]:
        params = brief.metadata.get("key_parameters") or {}
        model_description = str(brief.metadata.get("model_description") or "")
        literature_summary = str(literature.metadata.get("synthesis") or literature.summary or "")
        answers: dict[str, str] = {}
        for question in clarification.questions:
            if question.topic == QuestionTopic.VARIABLE_RANGE:
                text = (
                    "Use the parameter values reported in the article. "
                    "Treat the environmental noise amplitude and system size as the main sweep targets. "
                    f"Parsed parameters: {json.dumps(params)}"
                )
            elif question.topic == QuestionTopic.STOPPING_THRESHOLD:
                text = (
                    "Use conservative stopping criteria suitable for reproduction: "
                    "treat stable summary statistics over repeated intervals as success and obvious numerical divergence as failure."
                )
            elif question.topic == QuestionTopic.STEP_BUDGET:
                text = (
                    "Use a moderate sample budget and a larger full-sweep budget. "
                    "Prefer enough steps to reproduce the paper's headline figures while keeping checkpoints and data logging enabled."
                )
            elif question.topic == QuestionTopic.PHYSICAL_UNITS:
                text = (
                    "Use the units and conventions stated in the article. "
                    "If the paper uses reduced or dimensionless units, keep those conventions unchanged."
                )
            else:
                text = (
                    "Sweep the environmental stochasticity controls and any reported population-size variable first. "
                    f"Model context: {model_description or literature_summary}"
                )
            answers[str(question.index)] = text
        return answers

    def _compose_research_goal(
        self,
        task: TaskEnvelope,
        brief: ArtifactRef,
        literature: ArtifactRef,
    ) -> str:
        params = brief.metadata.get("key_parameters") or {}
        figures = brief.metadata.get("expected_figures") or []
        literature_summary = str(literature.metadata.get("synthesis") or literature.summary or "")
        steering_notes = [
            str(item).strip()
            for item in (task.metadata.get("steering_notes") or [])
            if str(item).strip()
        ]
        steering_block = (
            "\n\nPI steering notes:\n" + "\n".join(f"- {note}" for note in steering_notes)
            if steering_notes
            else ""
        )
        return (
            f"{task.instructions}\n\n"
            f"Model description:\n{brief.metadata.get('model_description', '')}\n\n"
            f"Procedure:\n{brief.metadata.get('procedure', '')}\n\n"
            f"Key parameters:\n{json.dumps(params, indent=2)}\n\n"
            f"Expected figures:\n" + "\n".join(f"- {figure}" for figure in figures) + "\n\n"
            f"Related literature context:\n{literature_summary}"
            f"{steering_block}"
        )

    @staticmethod
    def _review(
        review_callback: Optional[Callable[[str, list[ArtifactRef], str], SteeringDecision]],
        stage: str,
        artifacts: list[ArtifactRef],
        message: str,
    ) -> SteeringDecision:
        if review_callback is None:
            return SteeringDecision(checkpoint_id=stage, action="continue")
        decision = review_callback(stage, artifacts, message)
        return decision or SteeringDecision(checkpoint_id=stage, action="continue")

    @staticmethod
    def _is_revision(decision: SteeringDecision) -> bool:
        return decision.normalized_action() == "revise"

    @staticmethod
    def _check_cancel(cancel_callback: Optional[Callable[[], None]]) -> None:
        if cancel_callback is not None:
            cancel_callback()

    @staticmethod
    def _require_source(
        sources: list[ArtifactRef],
        *,
        artifact_id: str,
        kind: str,
        label: str,
    ) -> ArtifactRef:
        for source in sources:
            if source.artifact_id == artifact_id:
                return source
        for source in sources:
            if source.kind == kind:
                return source
        raise AssistantExecutionError(
            f"Missing required source artifact for coding task: {label}"
        )

    def _coding_failure(self, stage: str, update: Any) -> AssistantExecutionError:
        artifact = self._registry.save_json(
            assistant=AssistantId.CODING_AGENT.value,
            kind="coding_failure",
            title=f"Simulation Failure ({stage})",
            filename=f"simulation/failure_{stage}.json",
            payload=jsonify(update),
            summary=truncate(getattr(update, "message", stage)),
            metadata={"stage": stage},
            artifact_id=f"simulation-failure-{stage}",
        )
        return AssistantExecutionError(
            f"Simulation phase '{stage}' failed: {getattr(update, 'message', update)}",
            artifacts=[artifact],
        )
