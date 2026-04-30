"""
research_platform.mediated_services
───────────────────────────────────
Isolated service wrappers used by the mediator-first workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

from literature_review.article_parser import ArticleParser
from literature_review.search_backends.arxiv_backend import ArXivBackend
from literature_review.search_backends.semantic_scholar_backend import SemanticScholarBackend
from reviewer.journal_reviewer import JournalReviewer
from reviewer.contract import ReviewTask
from sim_tool.analyst import RunAnalyst
from sim_tool.contract import AnalysisResult, ClarificationRequest, SpecApproval, Verdict, SpecPatch
from sim_tool.designer import SimulationDesigner
from sim_tool.models import SimulationSpec

from .assistants.backends import build_literature_review_bridge_from_env, make_service_backend
from .assistants.literature import LocalLiteratureAssistant
from .contracts import ArtifactRef, AssistantId, TaskEnvelope
from .math_agent import MathAgentService
from .registry import ArtifactRegistry
from .runtime_assemble import _jsonify_spec
from .service_contracts import (
    AgentRuntimeProfile,
    MediatedResponse,
    OrchestrationRequestEnvelope,
)
from .writer_runtime import run_writer_pipeline


@dataclass
class LiteratureTaskState:
    session_id: str
    task_kind: str
    task: Optional[TaskEnvelope] = None
    article_handle: str = ""
    question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationTaskState:
    session_id: str
    task: TaskEnvelope
    description: str
    stage: str = "started"
    spec: Optional[SimulationSpec] = None
    spec_payload: Optional[dict[str, Any]] = None
    bundle: Optional[dict[str, Any]] = None
    artifact_ids: list[str] = field(default_factory=list)


@dataclass
class ReviewerTaskState:
    session_id: str
    task: ReviewTask
    manuscript_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LiteratureAgentService:
    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        profile: Optional[AgentRuntimeProfile] = None,
    ) -> None:
        self._registry = registry
        self._profile = profile or AgentRuntimeProfile()
        self._backend = make_service_backend("literature_review", self._profile)
        self._local = LocalLiteratureAssistant(registry=registry, llm_backend=self._backend)
        self._parser = ArticleParser(
            ArXivBackend(sort_by="relevance"),
            fallback_backends=[SemanticScholarBackend()],
        )
        self._sessions: dict[str, LiteratureTaskState] = {}
        self._article_cache: dict[str, dict[str, Any]] = {}

    def prepare_article_lookup_spec(self, task: TaskEnvelope) -> MediatedResponse:
        session_id = task.task_id
        state = LiteratureTaskState(session_id=session_id, task_kind="prepare_lookup", task=task)
        self._sessions[session_id] = state
        spec = self._local.prepare_article_lookup_spec(task)
        if spec.get("needs_clarification"):
            question = str(spec.get("clarification_question") or "Clarify the target article.")
            return MediatedResponse(
                status="needs_request",
                session_id=session_id,
                payload={"lookup_spec": spec},
                request=OrchestrationRequestEnvelope.new(
                    resume_token=session_id,
                    request_kind="clarification",
                    question=question,
                    expected_schema={"answers": {"clarification": "string"}},
                    capability_hint="user",
                    metadata={"task_kind": "prepare_lookup", "original_spec": spec},
                ),
            )
        return MediatedResponse(status="completed", session_id=session_id, payload=spec)

    def resume_literature_task(self, session_id: str, payload: dict[str, Any]) -> MediatedResponse:
        state = self._sessions[session_id]
        if state.task_kind == "prepare_lookup":
            original_task = state.task
            amended_task = TaskEnvelope(
                task_id=original_task.task_id,
                directive_id=original_task.directive_id,
                assistant=original_task.assistant,
                requestor=original_task.requestor,
                subject=original_task.subject,
                instructions=original_task.instructions + "\n\nClarification:\n" + str(payload.get("clarification") or payload.get("feedback") or ""),
                output_dir=original_task.output_dir,
                metadata=dict(original_task.metadata),
                attachments=list(original_task.attachments),
            )
            return self.prepare_article_lookup_spec(amended_task)
        if state.task_kind == "article_question":
            question = state.question
            text = self._article_cache[state.article_handle]["text"]
            external_answer = str(payload.get("answer") or payload.get("text") or "").strip()
            answer = self._answer_question(text, f"{question}\n\nExternal analysis:\n{external_answer}")
            return MediatedResponse(status="completed", session_id=session_id, payload={"answer": answer, "article_handle": state.article_handle})
        if state.task_kind == "review_question":
            review_ref = self._registry.get(str(state.metadata.get("review_artifact_id")))
            review_text = self._registry.read_artifact_text(review_ref.artifact_id) if review_ref else ""
            question = state.question
            external_answer = str(payload.get("answer") or payload.get("text") or "").strip()
            answer = self._answer_question(review_text or "", f"{question}\n\nExternal analysis:\n{external_answer}")
            return MediatedResponse(status="completed", session_id=session_id, payload={"answer": answer})
        return MediatedResponse(status="failed", session_id=session_id, error=f"Unsupported literature task kind: {state.task_kind}")

    def find_primary_article(self, task: TaskEnvelope) -> MediatedResponse:
        article = self._local.find_primary_article(task)
        return MediatedResponse(
            status="completed",
            session_id=task.task_id,
            payload={"artifact_id": article.artifact_id},
            artifact_ids=[article.artifact_id],
        )

    def fetch_article(
        self,
        *,
        article: ArtifactRef,
        session_id: str,
    ) -> MediatedResponse:
        handle = f"article:{article.artifact_id}"
        cached = self._article_cache.get(handle)
        if cached is None:
            text = self._parser.fetch_full_text(
                str(article.metadata.get("arxiv_id") or ""),
                fallback_abstract=str(article.metadata.get("abstract") or ""),
                paper_url=str(article.metadata.get("url") or ""),
            )
            cached = {
                "text": text,
                "title": str(article.metadata.get("title") or article.title),
                "artifact_id": article.artifact_id,
            }
            self._article_cache[handle] = cached
        return MediatedResponse(
            status="completed",
            session_id=session_id,
            payload={
                "article_handle": handle,
                "title": cached["title"],
                "article_artifact_id": article.artifact_id,
            },
        )

    def answer_from_article(
        self,
        *,
        article_handle: str,
        question: str,
        session_id: str,
    ) -> MediatedResponse:
        cached = self._article_cache[article_handle]
        if _looks_like_math_or_proof(question):
            self._sessions[session_id] = LiteratureTaskState(
                session_id=session_id,
                task_kind="article_question",
                article_handle=article_handle,
                question=question,
            )
            return MediatedResponse(
                status="needs_request",
                session_id=session_id,
                payload={"article_handle": article_handle, "question": question},
                request=OrchestrationRequestEnvelope.new(
                    resume_token=session_id,
                    request_kind="analysis",
                    question=question,
                    capability_hint="math",
                    expected_schema={"answer": "string"},
                    metadata={"article_handle": article_handle},
                ),
            )
        answer = self._answer_question(cached["text"], question)
        return MediatedResponse(
            status="completed",
            session_id=session_id,
            payload={"article_handle": article_handle, "answer": answer},
        )

    def build_article_brief(self, task: TaskEnvelope, article: ArtifactRef) -> MediatedResponse:
        brief = self._local.build_article_brief(task, article)
        return MediatedResponse(
            status="completed",
            session_id=task.task_id,
            payload={"artifact_id": brief.artifact_id},
            artifact_ids=[brief.artifact_id],
        )

    def review_related_literature(
        self,
        task: TaskEnvelope,
        article: ArtifactRef,
        brief: ArtifactRef,
    ) -> MediatedResponse:
        review = self._local.review_related_literature(task, article, brief)
        return MediatedResponse(
            status="completed",
            session_id=task.task_id,
            payload={"artifact_id": review.artifact_id},
            artifact_ids=[review.artifact_id],
        )

    def answer_from_literature_review(
        self,
        *,
        review_artifact_id: str,
        question: str,
        session_id: str,
    ) -> MediatedResponse:
        text = self._registry.read_artifact_text(review_artifact_id) or ""
        if _looks_like_math_or_proof(question):
            self._sessions[session_id] = LiteratureTaskState(
                session_id=session_id,
                task_kind="review_question",
                question=question,
                metadata={"review_artifact_id": review_artifact_id},
            )
            return MediatedResponse(
                status="needs_request",
                session_id=session_id,
                request=OrchestrationRequestEnvelope.new(
                    resume_token=session_id,
                    request_kind="analysis",
                    question=question,
                    capability_hint="math",
                    expected_schema={"answer": "string"},
                    metadata={"review_artifact_id": review_artifact_id},
                ),
            )
        answer = self._answer_question(text, question)
        return MediatedResponse(status="completed", session_id=session_id, payload={"answer": answer})

    def export_literature_artifact(self, artifact_id: str) -> MediatedResponse:
        artifact = self._registry.get(artifact_id)
        if artifact is None:
            return MediatedResponse(status="failed", error=f"Unknown literature artifact: {artifact_id}")
        return MediatedResponse(status="completed", payload={"artifact_id": artifact.artifact_id}, artifact_ids=[artifact.artifact_id])

    def _answer_question(self, text: str, question: str) -> str:
        answers = self._parser.extract_answers(text, [question], self._backend)
        answer = str(answers.get(question) or "").strip()
        if answer:
            return answer
        return "NOT_FOUND"


class SimulationDesignerService:
    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        profile: Optional[AgentRuntimeProfile] = None,
    ) -> None:
        self._registry = registry
        self._profile = profile or AgentRuntimeProfile()
        self._backend = make_service_backend("designer", self._profile)
        self._designer = SimulationDesigner(backend=self._backend)
        self._analyst = RunAnalyst(backend=self._backend)
        self._sessions: dict[str, SimulationTaskState] = {}

    def start_simulation(self, task: TaskEnvelope, *, sources: list[ArtifactRef]) -> MediatedResponse:
        description = self._compose_description(task, sources)
        result = self._designer.start(description)
        session_id = result.session_id
        state = SimulationTaskState(session_id=session_id, task=task, description=description)
        self._sessions[session_id] = state
        return self._from_designer_result(state, result)

    def resume_simulation(self, session_id: str, payload: dict[str, Any]) -> MediatedResponse:
        state = self._sessions[session_id]
        kind = str(payload.get("kind") or "")
        if kind == "clarification_answers":
            answers = dict(payload.get("answers") or {})
            result = self._designer.answer(session_id, answers)
            return self._from_designer_result(state, result)

        if kind == "runtime_response":
            action = str(payload.get("action") or "")
            if action == "materialize_runtime":
                state.bundle = dict(payload.get("bundle") or {})
                state.artifact_ids.extend(
                    [
                        aid
                        for aid in (
                            state.bundle.get("spec_artifact_id"),
                            state.bundle.get("script_artifact_id"),
                            state.bundle.get("notebook_artifact_id"),
                        )
                        if aid
                    ]
                )
                state.stage = "runtime_materialized"
                return MediatedResponse(
                    status="needs_request",
                    session_id=session_id,
                    payload={"bundle": dict(state.bundle)},
                    request=OrchestrationRequestEnvelope.new(
                        resume_token=session_id,
                        request_kind="runtime_action",
                        question="Run a sample execution for the generated simulation.",
                        capability_hint="runtime",
                        metadata={
                            "action": "run_sample",
                            "spec_payload": dict(state.spec_payload or {}),
                            "bundle": dict(state.bundle),
                            "sample_steps": int(state.task.metadata.get("sample_steps", 200)),
                        },
                    ),
                )
            if action == "run_sample":
                diagnostics = payload.get("diagnostics") or {}
                if diagnostics.get("errors") or diagnostics.get("status") == "failed":
                    feedback = self._diagnostics_feedback("sample run failed", diagnostics)
                    result = self._designer.reject(session_id, feedback)
                    return self._from_designer_result(state, result)
                if diagnostics.get("warnings"):
                    feedback = self._diagnostics_feedback("sample run warnings", diagnostics)
                    result = self._designer.reject(session_id, feedback)
                    return self._from_designer_result(state, result)
                state.stage = "sample_ok"
                return MediatedResponse(
                    status="needs_request",
                    session_id=session_id,
                    payload={"bundle": dict(state.bundle)},
                    request=OrchestrationRequestEnvelope.new(
                        resume_token=session_id,
                        request_kind="runtime_action",
                        question="Run the full simulation sweep and return parsed diagnostics.",
                        capability_hint="runtime",
                        metadata={
                            "action": "run_sweep",
                            "spec_payload": dict(state.spec_payload or {}),
                            "bundle": dict(state.bundle),
                        },
                    ),
                )
            if action == "run_sweep":
                diagnostics = payload.get("diagnostics") or {}
                if diagnostics.get("errors") or diagnostics.get("status") == "failed":
                    feedback = self._diagnostics_feedback("full sweep failed", diagnostics)
                    result = self._designer.reject(session_id, feedback)
                    return self._from_designer_result(state, result)
                analysis = self._analyse_runtime_results(session_id, diagnostics)
                if analysis.verdict == Verdict.OK:
                    analysis_ref = self._registry.save_json(
                        assistant=AssistantId.CODING_AGENT.value,
                        kind="simulation_analysis",
                        title="Simulation Analysis",
                        filename=f"{session_id}/simulation/analysis.json",
                        payload=_analysis_to_dict(analysis),
                        summary=f"Analysis verdict: {analysis.verdict.value}",
                        metadata={"session_id": session_id},
                        artifact_id=f"{session_id}-simulation-analysis",
                    )
                    state.artifact_ids.append(analysis_ref.artifact_id)
                    state.stage = "complete"
                    return MediatedResponse(
                        status="completed",
                        session_id=session_id,
                        payload={
                            "bundle": dict(state.bundle or {}),
                            "analysis": _analysis_to_dict(analysis),
                        },
                        artifact_ids=list(dict.fromkeys(state.artifact_ids)),
                    )
                if analysis.patch is not None and analysis.verdict in {Verdict.MINOR_FIX, Verdict.MAJOR_FIX}:
                    if state.spec is None:
                        return MediatedResponse(status="failed", session_id=session_id, error="No simulation spec available for patching")
                    self._apply_patch(state.spec, analysis.patch)
                    state.spec_payload = _jsonify_spec(state.spec)
                    spec_ref = self._persist_spec(state)
                    state.artifact_ids.append(spec_ref.artifact_id)
                    state.stage = "patched_from_results"
                    return MediatedResponse(
                        status="needs_request",
                        session_id=session_id,
                        payload={"analysis": _analysis_to_dict(analysis)},
                        request=OrchestrationRequestEnvelope.new(
                            resume_token=session_id,
                            request_kind="runtime_action",
                            question="Re-materialize the runtime after applying result-driven fixes.",
                            capability_hint="runtime",
                            metadata={
                                "action": "materialize_runtime",
                                "spec_payload": dict(state.spec_payload),
                                "output_dir": str((Path(state.task.output_dir or ".") / "simulation_results" / session_id).resolve()),
                            },
                        ),
                        artifact_ids=[spec_ref.artifact_id],
                    )
                feedback = analysis.patch.reason if analysis.patch else "Results did not match the expected behavior; revise the simulation design."
                result = self._designer.reject(session_id, feedback)
                return self._from_designer_result(state, result)

        if kind == "steering_feedback":
            result = self._designer.reject(session_id, str(payload.get("feedback") or "Revise the simulation."))
            return self._from_designer_result(state, result)

        return MediatedResponse(status="failed", session_id=session_id, error=f"Unsupported simulation payload kind: {kind}")

    def get_simulation_status(self, session_id: str) -> MediatedResponse:
        state = self._sessions[session_id]
        return MediatedResponse(
            status="completed",
            session_id=session_id,
            payload={
                "stage": state.stage,
                "artifact_ids": list(dict.fromkeys(state.artifact_ids)),
                "bundle": dict(state.bundle or {}),
            },
            artifact_ids=list(dict.fromkeys(state.artifact_ids)),
        )

    def _from_designer_result(
        self,
        state: SimulationTaskState,
        result: ClarificationRequest | SpecApproval,
    ) -> MediatedResponse:
        if isinstance(result, ClarificationRequest):
            state.stage = "clarifying"
            question_text = "\n".join(f"{question.index}. {question.text}" for question in result.questions)
            return MediatedResponse(
                status="needs_request",
                session_id=result.session_id,
                payload={"clarification": _clarification_to_dict(result)},
                request=OrchestrationRequestEnvelope.new(
                    resume_token=result.session_id,
                    request_kind="clarification",
                    question=question_text,
                    expected_schema={"answers": {str(question.index): "string" for question in result.questions}},
                    capability_hint="user",
                    metadata={"kind": "simulation_clarification"},
                ),
            )

        state.stage = "spec_ready"
        state.spec = self._designer.get_spec(result.session_id)
        state.spec_payload = _jsonify_spec(state.spec)
        summary_ref = self._registry.save_text(
            assistant=AssistantId.CODING_AGENT.value,
            kind="simulation_spec_summary",
            title="Simulation Spec Summary",
            filename=f"{result.session_id}/simulation/spec_summary.txt",
            text=str(result),
            summary=result.time_estimate,
            metadata={"session_id": result.session_id},
            artifact_id=f"{result.session_id}-simulation-spec-summary",
        )
        spec_ref = self._persist_spec(state)
        state.artifact_ids.extend([summary_ref.artifact_id, spec_ref.artifact_id])
        return MediatedResponse(
            status="needs_request",
            session_id=result.session_id,
            payload={"spec_approval": _spec_approval_to_dict(result)},
            request=OrchestrationRequestEnvelope.new(
                resume_token=result.session_id,
                request_kind="runtime_action",
                question="Materialize the runtime for the approved simulation specification.",
                capability_hint="runtime",
                metadata={
                    "action": "materialize_runtime",
                    "spec_payload": dict(state.spec_payload),
                    "output_dir": str((Path(state.task.output_dir or ".") / "simulation_results" / result.session_id).resolve()),
                },
            ),
            artifact_ids=[summary_ref.artifact_id, spec_ref.artifact_id],
        )

    def _persist_spec(self, state: SimulationTaskState) -> ArtifactRef:
        return self._registry.save_json(
            assistant=AssistantId.CODING_AGENT.value,
            kind="simulation_spec",
            title="Simulation Spec",
            filename=f"{state.session_id}/simulation/spec.json",
            payload=dict(state.spec_payload or {}),
            summary=state.spec.description[:160] if state.spec and state.spec.description else state.description[:160],
            metadata={"session_id": state.session_id},
            artifact_id=f"{state.session_id}-simulation-spec",
        )

    def _analyse_runtime_results(self, session_id: str, diagnostics: dict[str, Any]) -> AnalysisResult:
        sweep_dir = str(((diagnostics.get("metadata") or {}).get("sweep_dir") or "")).strip()
        if not sweep_dir:
            raise ValueError("Runtime diagnostics did not include a sweep_dir for analysis")
        return self._analyst.analyse_sweep(Path(sweep_dir), session_id=session_id)

    def _compose_description(self, task: TaskEnvelope, sources: list[ArtifactRef]) -> str:
        parts = [task.instructions.strip()]
        context_bits = []
        for source in sources:
            if source.kind == "article_brief":
                text = self._registry.read_artifact_text(source.artifact_id) or json.dumps(source.metadata, indent=2)
                context_bits.append(f"Article brief:\n{text}")
            elif source.kind == "literature_review":
                context_bits.append(f"Literature synthesis summary:\n{source.summary}")
            elif source.summary:
                context_bits.append(f"{source.title}:\n{source.summary}")
        if context_bits:
            parts.append("Context artifacts:\n" + "\n\n".join(context_bits))
        return "\n\n".join(part for part in parts if part.strip())

    def _diagnostics_feedback(self, prefix: str, diagnostics: dict[str, Any]) -> str:
        warnings = "\n".join(f"- {item}" for item in diagnostics.get("warnings") or [])
        errors = "\n".join(f"- {item}" for item in diagnostics.get("errors") or [])
        failure_class = str(diagnostics.get("failure_class") or "").strip()
        chunks = [prefix]
        if failure_class:
            chunks.append(f"Failure class: {failure_class}")
        if warnings:
            chunks.append(f"Warnings:\n{warnings}")
        if errors:
            chunks.append(f"Errors:\n{errors}")
        return "\n\n".join(chunks)

    def _apply_patch(self, spec: SimulationSpec, patch: SpecPatch) -> None:
        for change in patch.changes:
            parts = change.field.split(".")
            top = parts[0]
            if top == "max_steps":
                spec.max_steps = int(change.new_value)
                continue
            if top == "checkpoint_interval":
                spec.checkpoint_interval = int(change.new_value)
                continue
            if top == "progress_interval":
                spec.progress_interval = int(change.new_value)
                continue
            if top == "data_log_interval":
                spec.data_log_interval = int(change.new_value)
                continue
            if top == "variables" and len(parts) == 3:
                variable_name, field_name = parts[1], parts[2]
                for variable in spec.variables:
                    if variable.name == variable_name:
                        setattr(variable, field_name, change.new_value)
                        break
                continue
            if top == "stopping_conditions" and len(parts) == 3:
                condition_name, field_name = parts[1], parts[2]
                for condition in spec.stopping_conditions:
                    if condition.name == condition_name:
                        setattr(condition, field_name, change.new_value)
                        break


class AcademicWriterService:
    def __init__(
        self,
        registry: ArtifactRegistry,
        *,
        profile: Optional[AgentRuntimeProfile] = None,
    ) -> None:
        self._registry = registry
        self._profile = profile or AgentRuntimeProfile()
        self._backend = make_service_backend("writer", self._profile)

    def draft_review(self, task: TaskEnvelope, *, sources: list[ArtifactRef]) -> MediatedResponse:
        artifact_catalog = [
            {
                "artifact_id": source.artifact_id,
                "assistant": source.assistant,
                "kind": source.kind,
                "title": source.title,
                "summary": source.summary,
                "path": source.path,
                "url": source.url,
                "content": source.content,
                "mime_type": source.mime_type,
                "metadata": dict(source.metadata),
            }
            for source in sources
        ]
        result = run_writer_pipeline(
            llm_backend=self._backend,
            registry=self._registry,
            description=task.instructions,
            venue=str(task.metadata.get("venue") or "arXiv preprint"),
            artifact_catalog=artifact_catalog,
            artifact_context=str(task.metadata.get("artifact_context") or ""),
            draft_filename="writer/mini_review.md",
        )
        draft_artifact_id = str(result["draft_artifact"]["artifact_id"])
        return MediatedResponse(
            status="completed",
            session_id=task.task_id,
            payload=result,
            artifact_ids=[draft_artifact_id],
        )


class JournalReviewerService:
    def __init__(
        self,
        *,
        profile: Optional[AgentRuntimeProfile] = None,
    ) -> None:
        self._profile = profile or AgentRuntimeProfile()
        self._backend = make_service_backend("journal_reviewer", self._profile)
        self._sessions: dict[str, ReviewerTaskState] = {}

    def review_manuscript(self, *, session_id: str, article_text: str, venue: str = "journal") -> MediatedResponse:
        task = ReviewTask.new(article_text=article_text, venue=venue)
        self._sessions[session_id] = ReviewerTaskState(session_id=session_id, task=task, manuscript_text=article_text)
        return MediatedResponse(
            status="needs_request",
            session_id=session_id,
            request=OrchestrationRequestEnvelope.new(
                resume_token=session_id,
                request_kind="analysis",
                question="Verify mathematical claims in the manuscript before writing the review.",
                capability_hint="math",
                expected_schema={"math_results": []},
                metadata={"manuscript_excerpt": article_text[:4000]},
            ),
        )

    def resume_review(self, session_id: str, payload: dict[str, Any]) -> MediatedResponse:
        state = self._sessions[session_id]
        math_results = payload.get("math_results") or payload.get("answer") or []
        prompt = (
            "Write a concise journal review based only on the manuscript and the provided math-check results.\n\n"
            f"Math results:\n{json.dumps(math_results, indent=2)}\n\n"
            f"Manuscript excerpt:\n{state.manuscript_text[:8000]}"
        )
        review_text = self._backend.complete(
            system="You are a rigorous journal reviewer. Return plain text only.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        return MediatedResponse(status="completed", session_id=session_id, payload={"review_text": review_text})


def _looks_like_math_or_proof(question: str) -> bool:
    lower = question.lower()
    return any(token in lower for token in ["proof", "derive", "equation", "integral", "differentiate", "calculate", "solve"])


def _clarification_to_dict(clarification: ClarificationRequest) -> dict[str, Any]:
    return {
        "session_id": clarification.session_id,
        "iteration": clarification.iteration,
        "questions": [
            {
                "index": question.index,
                "text": question.text,
                "topic": question.topic.value,
                "required": question.required,
            }
            for question in clarification.questions
        ],
    }


def _spec_approval_to_dict(approval: SpecApproval) -> dict[str, Any]:
    return {
        "session_id": approval.session_id,
        "spec_card": approval.spec_card,
        "time_estimate": approval.time_estimate,
        "output_contract": {
            "data_log_fields": list(approval.output_contract.data_log_fields),
            "results_fields": list(approval.output_contract.results_fields),
            "checkpoint_format": approval.output_contract.checkpoint_format,
        },
        "variable_count": approval.variable_count,
        "stop_cond_count": approval.stop_cond_count,
    }


def _analysis_to_dict(analysis: AnalysisResult) -> dict[str, Any]:
    return {
        "session_id": analysis.session_id,
        "verdict": analysis.verdict.value,
        "flags": [
            {
                "kind": flag.kind,
                "severity": flag.severity,
                "message": flag.message,
                "config_hint": flag.config_hint,
                "affects": list(flag.affects),
            }
            for flag in analysis.flags
        ],
        "patch": {
            "verdict": analysis.patch.verdict.value,
            "reason": analysis.patch.reason,
            "changes": [
                {
                    "field": change.field,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "why": change.why,
                }
                for change in analysis.patch.changes
            ],
        } if analysis.patch else None,
        "llm_reasoning": analysis.llm_reasoning,
        "data_files": [str(path) for path in analysis.data_files],
    }
