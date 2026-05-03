"""
research_platform.mediated_workflow
───────────────────────────────────
Mediator-first workflow orchestrator and default directive runner.
"""
from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from .contracts import ArtifactRef, AssistantId, AttachmentRef, MailboxMessage, TaskEnvelope
from .flow_management import FlowManagementService
from .math_agent import MathAgentService
from .mediated_services import AcademicWriterService, JournalReviewerService, LiteratureAgentService, SimulationDesignerService
from .registry import ArtifactRegistry
from .runtime_assemble import RuntimeAssembleService
from .service_contracts import MediatedResponse, OrchestrationRequestEnvelope

log = logging.getLogger(__name__)


class MediatorWorkflowRunner:
    def __init__(
        self,
        directive: Any,
        mailbox: Any,
        *,
        artifact_registry: ArtifactRegistry,
        flow_management: FlowManagementService,
        literature_agent: LiteratureAgentService,
        simulation_designer: SimulationDesignerService,
        runtime_assemble: RuntimeAssembleService,
        math_agent: MathAgentService,
        writer_agent: AcademicWriterService,
        reviewer_agent: Optional[JournalReviewerService] = None,
    ) -> None:
        self.directive = directive
        self.mailbox = mailbox
        self.artifact_registry = artifact_registry
        self.flow_management = flow_management
        self.literature_agent = literature_agent
        self.simulation_designer = simulation_designer
        self.runtime_assemble = runtime_assemble
        self.math_agent = math_agent
        self.writer_agent = writer_agent
        self.reviewer_agent = reviewer_agent
        self.output_dir = Path(directive.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all_phases(self) -> None:
        directive_id = self.directive.directive_id
        self.flow_management.open_workflow(
            directive_id,
            metadata={
                "instruction": self.directive.instruction,
                "phases": list(self.directive.phases),
            },
        )
        self.flow_management.register_session(
            directive_id,
            AssistantId.ORCHESTRATOR.value,
            f"{directive_id}:orchestrator",
            metadata={"phase": "bootstrap"},
        )
        self._send_message(
            assistant=AssistantId.ORCHESTRATOR.value,
            subject="Instruction accepted — workflow initiated",
            body=(
                f"I have accepted the directive and opened a mediated research workflow.\n\n"
                f"Instruction:\n{self.directive.instruction}\n\n"
                f"Planned phases: {', '.join(self.directive.phases)}"
            ),
        )

        article: Optional[ArtifactRef] = None
        brief: Optional[ArtifactRef] = None
        literature: Optional[ArtifactRef] = None
        simulation_artifacts: list[ArtifactRef] = []

        try:
            for phase in self.directive.phases:
                self._raise_if_stopped()
                self.flow_management.update_session_state(
                    directive_id,
                    f"{directive_id}:orchestrator",
                    "running",
                    metadata={"phase": phase},
                )
                if phase == "find_article":
                    article = self._phase_find_article()
                elif phase == "parse_article":
                    if article is None:
                        article = self._phase_find_article()
                    brief = self._phase_parse_article(article)
                elif phase == "literature":
                    if article is None:
                        article = self._phase_find_article()
                    if brief is None:
                        brief = self._phase_parse_article(article)
                    literature = self._phase_literature(article, brief)
                elif phase == "simulation":
                    if article is None:
                        article = self._phase_find_article()
                    if brief is None:
                        brief = self._phase_parse_article(article)
                    if literature is None:
                        literature = self._phase_literature(article, brief)
                    simulation_artifacts = self._phase_simulation(article, brief, literature)
                elif phase == "write":
                    sources = [item for item in [article, brief, literature] if item is not None]
                    sources.extend(simulation_artifacts)
                    self._phase_write(sources)
            self.flow_management.update_session_state(
                directive_id,
                f"{directive_id}:orchestrator",
                "complete",
                metadata={"phase": "done"},
            )
        except RuntimeError as exc:
            if "stop requested" in str(exc).lower():
                self._send_message(
                    assistant=AssistantId.ORCHESTRATOR.value,
                    subject="Workflow stopped by PI",
                    body=str(exc),
                )
                return
            raise
        except Exception:
            trace = traceback.format_exc()
            log.error("Mediated workflow failed:\n%s", trace)
            self._send_message(
                assistant=AssistantId.ORCHESTRATOR.value,
                subject="Workflow failed",
                body=trace,
            )
            raise

    def _phase_find_article(self) -> ArtifactRef:
        task = self._task(
            AssistantId.LITERATURE_REVIEWER.value,
            task_id=f"{self.directive.directive_id}-article-lookup",
            subject="Prepare and run article lookup",
            instructions=self.directive.instruction,
            metadata={"topic_hint": self.directive.topic_hint},
        )
        spec_response = self.literature_agent.prepare_article_lookup_spec(task)
        spec_response = self._resolve_response(
            service_id=AssistantId.LITERATURE_REVIEWER.value,
            response=spec_response,
            resume_fn=self.literature_agent.resume_literature_task,
        )
        lookup_spec = dict(spec_response.payload or {})
        lookup_task = self._task(
            AssistantId.LITERATURE_REVIEWER.value,
            task_id=f"{self.directive.directive_id}-find-article",
            subject="Find primary article",
            instructions=self.directive.instruction,
            metadata={
                "topic_hint": self.directive.topic_hint,
                "article_lookup_spec": lookup_spec,
                "article_lookup_query": lookup_spec.get("query_string", ""),
            },
        )
        article_response = self._resolve_response(
            service_id=AssistantId.LITERATURE_REVIEWER.value,
            response=self.literature_agent.find_primary_article(lookup_task),
            resume_fn=self.literature_agent.resume_literature_task,
        )
        article = self._artifact(article_response.payload["artifact_id"])
        authors = ", ".join(article.metadata.get("authors") or [])
        year = article.metadata.get("year") or "?"
        self._send_message(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            subject="Found article — ready to proceed",
            body=(
                f"I found the target article:\n\n"
                f"\"{article.metadata.get('title', article.title)}\"\n"
                f"{authors} ({year})\n\n"
                f"Query:\n{lookup_spec.get('query_string', '')}"
            ),
            attachments=[article.as_attachment(name="article_match.json")],
        )
        return article

    def _phase_parse_article(self, article: ArtifactRef) -> ArtifactRef:
        task = self._task(
            AssistantId.LITERATURE_REVIEWER.value,
            task_id=f"{self.directive.directive_id}-article-brief",
            subject="Build article brief",
            instructions=self.directive.instruction,
            metadata={"topic_hint": self.directive.topic_hint},
        )
        response = self._resolve_response(
            service_id=AssistantId.LITERATURE_REVIEWER.value,
            response=self.literature_agent.build_article_brief(task, article),
            resume_fn=self.literature_agent.resume_literature_task,
        )
        brief = self._artifact(response.payload["artifact_id"])
        self._send_message(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            subject="Article brief ready — parameters confirmed",
            body="The structured article brief is ready for downstream simulation work.",
            attachments=[brief.as_attachment(name="article_brief.json")],
        )
        return brief

    def _phase_literature(self, article: ArtifactRef, brief: ArtifactRef) -> ArtifactRef:
        task = self._task(
            AssistantId.LITERATURE_REVIEWER.value,
            task_id=f"{self.directive.directive_id}-literature-review",
            subject="Review related literature",
            instructions=self.directive.instruction,
            metadata={"topic_hint": self.directive.topic_hint},
        )
        response = self._resolve_response(
            service_id=AssistantId.LITERATURE_REVIEWER.value,
            response=self.literature_agent.review_related_literature(task, article, brief),
            resume_fn=self.literature_agent.resume_literature_task,
        )
        review = self._artifact(response.payload["artifact_id"])
        count = int(review.metadata.get("accepted_count", 0))
        self._send_message(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            subject=f"{count} related papers found",
            body=review.summary or "Related literature synthesis is ready.",
            attachments=[review.as_attachment(name="literature_review.json")],
        )
        return review

    def _phase_simulation(
        self,
        article: ArtifactRef,
        brief: ArtifactRef,
        literature: ArtifactRef,
    ) -> list[ArtifactRef]:
        selected_targets = self._resolve_simulation_target_scope(brief)
        metadata: dict[str, Any] = {"sample_steps": 200}
        if selected_targets:
            metadata["selected_simulation_targets"] = selected_targets
        task = self._task(
            AssistantId.CODING_AGENT.value,
            task_id=f"{self.directive.directive_id}-simulation",
            subject="Design and run simulation",
            instructions=self.directive.instruction,
            metadata=metadata,
            output_dir=str((self.output_dir / "simulation_results").resolve()),
        )
        response = self.simulation_designer.start_simulation(task, sources=[article, brief, literature])
        response = self._resolve_response(
            service_id=AssistantId.CODING_AGENT.value,
            response=response,
            resume_fn=self.simulation_designer.resume_simulation,
        )
        artifact_ids = response.artifact_ids or []
        artifacts = [self._artifact(artifact_id) for artifact_id in artifact_ids if self.artifact_registry.get(artifact_id)]
        self._send_message(
            assistant=AssistantId.CODING_AGENT.value,
            subject="Full results ready — analysis attached",
            body="The mediated simulation workflow completed and attached its runtime artifacts and analysis.",
            attachments=[artifact.as_attachment(name=artifact.title) for artifact in artifacts],
        )
        return artifacts

    def _resolve_simulation_target_scope(self, brief: ArtifactRef) -> list[dict[str, Any]]:
        targets = self._simulation_targets_from_brief(brief)
        if len(targets) <= 1:
            return targets

        session_id = f"{self.directive.directive_id}-simulation-scope"
        options_text = "\n".join(
            f"{index}. {target.get('name') or 'Simulation target'}"
            + (f" — {target.get('description')}" if target.get("description") else "")
            for index, target in enumerate(targets, 1)
        )
        request = OrchestrationRequestEnvelope.new(
            resume_token=session_id,
            request_kind="selection",
            question=(
                "The article brief contains multiple simulation targets.\n\n"
                f"{options_text}\n\n"
                "Which target(s) should be simulated? You can answer with option numbers, names, or 'all'."
            ),
            expected_schema={"answers": {"selection": "string"}},
            capability_hint="user",
            metadata={"options": targets, "default_policy": "all"},
        )

        def _resume_selection(_session_id: str, payload: dict[str, Any]) -> MediatedResponse:
            return MediatedResponse(
                status="completed",
                session_id=session_id,
                payload={"selected_targets": self._select_targets(targets, payload)},
            )

        resolved = self._resolve_response(
            service_id=AssistantId.ORCHESTRATOR.value,
            response=MediatedResponse(status="needs_request", session_id=session_id, request=request),
            resume_fn=_resume_selection,
        )
        selected = resolved.payload.get("selected_targets") if isinstance(resolved.payload, dict) else None
        return list(selected or targets)

    def _phase_write(self, sources: list[ArtifactRef]) -> ArtifactRef:
        task = self._task(
            AssistantId.WRITER.value,
            task_id=f"{self.directive.directive_id}-write",
            subject="Draft review",
            instructions=self.directive.instruction,
            metadata={"venue": "arXiv preprint"},
        )
        response = self._resolve_response(
            service_id=AssistantId.WRITER.value,
            response=self.writer_agent.draft_review(task, sources=sources),
            resume_fn=lambda _session_id, _payload: MediatedResponse(status="failed", error="Writer does not support resume"),
        )
        draft = self._artifact(response.artifact_ids[0])
        self._send_message(
            assistant=AssistantId.WRITER.value,
            subject="Mini-review draft ready",
            body="The grounded draft review is ready.",
            attachments=[draft.as_attachment(name="mini_review.md")],
        )
        return draft

    def _resolve_response(
        self,
        *,
        service_id: str,
        response: MediatedResponse,
        resume_fn: Callable[[str, dict[str, Any]], MediatedResponse],
    ) -> MediatedResponse:
        directive_id = self.directive.directive_id
        if response.session_id:
            self.flow_management.register_session(
                directive_id,
                service_id,
                response.session_id,
                metadata={"service_id": service_id},
            )
        while response.request is not None:
            self._raise_if_stopped()
            request = response.request
            steering_metadata = self._request_steering_metadata(
                service_id=service_id,
                session_id=response.session_id,
                request=request,
            )
            self.flow_management.update_session_state(
                directive_id,
                response.session_id,
                "waiting",
                metadata={
                    "request_kind": request.request_kind,
                    "capability_hint": request.capability_hint or "",
                    "request": request.to_dict(),
                    "steering": steering_metadata,
                },
            )
            if request.capability_hint == "user":
                self._send_message(
                    assistant=service_id,
                    subject="Clarification requested",
                    body=(
                        f"{request.question}\n\n"
                        "Reply in the response panel so I can resume this same agent session with your answer."
                    ),
                    metadata={
                        "session_id": response.session_id,
                        "request_id": request.request_id,
                        "request": request.to_dict(),
                        "steering": steering_metadata,
                    },
                )
                injection = self.flow_management.wait_for_injection(response.session_id)
                response = resume_fn(response.session_id, self._normalize_injection_payload(injection.payload))
            elif request.capability_hint == "runtime":
                runtime_payload = self._fulfill_runtime_request(request)
                response = resume_fn(
                    response.session_id,
                    {"kind": "runtime_response", "action": request.metadata.get("action"), **runtime_payload},
                )
            elif request.capability_hint == "math":
                math_payload = self._fulfill_math_request(request)
                response = resume_fn(response.session_id, math_payload)
            elif request.capability_hint == "literature":
                lit_payload = self._fulfill_literature_request(request)
                response = resume_fn(response.session_id, lit_payload)
            else:
                raise ValueError(f"Unsupported capability hint: {request.capability_hint}")
        self.flow_management.update_session_state(
            directive_id,
            response.session_id,
            "complete",
            metadata={"artifact_ids": list(response.artifact_ids)},
        )
        return response

    def _fulfill_runtime_request(self, request: OrchestrationRequestEnvelope) -> dict[str, Any]:
        action = str(request.metadata.get("action") or "")
        spec_payload = dict(request.metadata.get("spec_payload") or {})
        if action == "materialize_runtime":
            bundle = self.runtime_assemble.materialize_runtime(
                session_id=request.resume_token,
                spec_payload=spec_payload,
                output_dir=request.metadata.get("output_dir"),
            )
            return {"bundle": bundle}
        if action == "run_sample":
            diagnostics = self.runtime_assemble.run_sample(
                session_id=request.resume_token,
                spec_payload=spec_payload,
                bundle=dict(request.metadata.get("bundle") or {}),
                sample_steps=int(request.metadata.get("sample_steps", 200)),
            )
            return {"diagnostics": diagnostics.to_dict()}
        if action == "run_sweep":
            diagnostics = self.runtime_assemble.run_sweep(
                session_id=request.resume_token,
                spec_payload=spec_payload,
                bundle=dict(request.metadata.get("bundle") or {}),
            )
            return {"diagnostics": diagnostics.to_dict()}
        raise ValueError(f"Unsupported runtime action: {action}")

    def _fulfill_math_request(self, request: OrchestrationRequestEnvelope) -> dict[str, Any]:
        question = request.question
        if request.request_kind == "analysis" or "verify" in question.lower():
            answer = self.math_agent.verify_from_text(question)
            return {"kind": "analysis_response", "answer": json.dumps(answer)}
        answer = self.math_agent.solve_from_text(question)
        return {"kind": "analysis_response", "answer": json.dumps(answer)}

    def _fulfill_literature_request(self, request: OrchestrationRequestEnvelope) -> dict[str, Any]:
        return {"kind": "analysis_response", "answer": "NOT_IMPLEMENTED"}

    def _normalize_injection_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind") or "").strip()
        if kind:
            normalized = dict(payload)
            if kind == "clarification_answers" and "answers" not in normalized:
                normalized["answers"] = {}
            return normalized
        if "answers" in payload:
            return {"kind": "clarification_answers", "answers": dict(payload.get("answers") or {})}
        feedback = str(payload.get("feedback") or "")
        if feedback:
            return {"kind": "steering_feedback", "feedback": feedback}
        if "clarification" in payload:
            return {"kind": "clarification_answers", "answers": {"clarification": str(payload["clarification"])}}
        return {"kind": "steering_feedback", "feedback": json.dumps(payload)}

    def _request_steering_metadata(
        self,
        *,
        service_id: str,
        session_id: str,
        request: OrchestrationRequestEnvelope,
    ) -> dict[str, Any]:
        title = {
            "clarification": "Answer agent clarification",
            "selection": "Choose simulation scope",
        }.get(request.request_kind, "Respond to agent request")
        return {
            "checkpoint_id": request.request_id,
            "state": "awaiting_pi",
            "kind": "agent_request",
            "assistant": service_id,
            "session_id": session_id,
            "request_id": request.request_id,
            "request_kind": request.request_kind,
            "title": title,
            "prompt": request.question,
            "expected_schema": dict(request.expected_schema),
            "request": request.to_dict(),
        }

    @staticmethod
    def _simulation_targets_from_brief(brief: ArtifactRef) -> list[dict[str, Any]]:
        raw_targets = brief.metadata.get("simulation_targets")
        if not isinstance(raw_targets, list):
            return []
        targets: list[dict[str, Any]] = []
        for index, item in enumerate(raw_targets, 1):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("title") or f"Target {index}").strip()
                description = str(item.get("description") or item.get("summary") or "").strip()
                metadata = {str(key): value for key, value in item.items() if key not in {"name", "title", "description", "summary"}}
            else:
                name = f"Target {index}"
                description = str(item or "").strip()
                metadata = {}
            if not name and not description:
                continue
            targets.append({"name": name or f"Target {index}", "description": description, "metadata": metadata})
        return targets

    @staticmethod
    def _select_targets(targets: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
        answers = payload.get("answers") if isinstance(payload.get("answers"), dict) else {}
        text = " ".join(
            str(value)
            for value in [
                payload.get("selection"),
                payload.get("feedback"),
                payload.get("clarification"),
                *(answers.values() if isinstance(answers, dict) else []),
            ]
            if str(value or "").strip()
        ).strip()
        if not text or text.lower() in {"all", "both", "everything"}:
            return targets

        selected: list[dict[str, Any]] = []
        lowered = text.lower()
        for index, target in enumerate(targets, 1):
            name = str(target.get("name") or "").lower()
            if str(index) in lowered.split() or f"{index}," in lowered or f"{index}." in lowered:
                selected.append(target)
                continue
            if name and name in lowered:
                selected.append(target)
        return selected or targets

    def _artifact(self, artifact_id: str) -> ArtifactRef:
        artifact = self.artifact_registry.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return artifact

    def _raise_if_stopped(self) -> None:
        snapshot = self.flow_management.get_workflow_snapshot(self.directive.directive_id)
        if snapshot["stop_requested"]:
            raise RuntimeError(snapshot["stop_reason"] or "Workflow stop requested")

    def _task(
        self,
        assistant: str,
        *,
        task_id: str,
        subject: str,
        instructions: str,
        metadata: Optional[dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> TaskEnvelope:
        return TaskEnvelope(
            task_id=task_id,
            directive_id=self.directive.directive_id,
            assistant=assistant,
            subject=subject,
            instructions=instructions,
            output_dir=output_dir,
            metadata=dict(metadata or {}),
        )

    def _send_message(
        self,
        *,
        assistant: str,
        subject: str,
        body: str,
        attachments: Optional[list[AttachmentRef]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.mailbox.append(
            MailboxMessage(
                directive_id=self.directive.directive_id,
                assistant=assistant,
                assistant_addr=f"{assistant}@research.local",
                to_name=getattr(self.directive, "pi_name", "PI"),
                subject=subject,
                body=body,
                attachments=list(attachments or []),
                metadata=dict(metadata or {}),
            )
        )
