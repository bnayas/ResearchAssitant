"""
research_platform.mediated_workflow
───────────────────────────────────
Mediator-first workflow orchestrator and default directive runner.
"""
from __future__ import annotations

import json
import logging
import inspect
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
        import uuid
        instructions = self.directive.instruction
        while True:
            task = self._task(
                AssistantId.LITERATURE_REVIEWER.value,
                task_id=f"{self.directive.directive_id}-article-lookup",
                subject="Prepare and run article lookup",
                instructions=instructions,
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
                instructions=instructions,
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

            if not self._should_confirm_primary_article():
                return article

            request = OrchestrationRequestEnvelope.new(
                resume_token=f"{self.directive.directive_id}-article-review-{uuid.uuid4().hex[:8]}",
                request_kind="review",
                question="I found the target article. Proceed with parsing, or revise the lookup query?",
                expected_schema={},
                capability_hint="user",
            )

            def _resume_review(session_id: str, payload: dict[str, Any]) -> MediatedResponse:
                return MediatedResponse(status="completed", session_id=session_id, payload=payload)

            review_response = self._resolve_response(
                service_id=AssistantId.ORCHESTRATOR.value,
                response=MediatedResponse(status="needs_request", session_id=request.resume_token, request=request),
                resume_fn=_resume_review,
            )

            payload = review_response.payload if isinstance(review_response.payload, dict) else {}
            action = payload.get("action", "continue")
            feedback = payload.get("feedback", "")

            if action == "continue":
                return article
            else:
                instructions += f"\n\n[PI Feedback on previous article lookup:\n{feedback}]"

    def _should_confirm_primary_article(self) -> bool:
        """Return whether article lookup should block for PI confirmation."""
        direct = getattr(self.directive, "confirm_primary_article", None)
        if direct is not None:
            return bool(direct)
        metadata = getattr(self.directive, "metadata", {}) or {}
        if isinstance(metadata, dict):
            for key in ("confirm_primary_article", "confirm_article", "review_primary_article"):
                if key in metadata:
                    return bool(metadata[key])
        return False

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
        def _on_literature_event(event: Any) -> None:
            payload = dict(getattr(event, "payload", {}) or {})
            event_type = str(getattr(event, "event_type", ""))
            if event_type == "title_found":
                title = str(payload.get("title") or "Accepted paper")
                year = payload.get("year") or "?"
                source = payload.get("source") or "unknown"
                authors = ", ".join(str(author) for author in (payload.get("authors") or [])[:4])
                url = str(payload.get("url") or "")
                body = f"{title} ({year})\nSource: {source}"
                if authors:
                    body += f"\nAuthors: {authors}"
                if url:
                    body += f"\nURL: {url}"
                self._send_message(
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject=f"Accepted related paper — {title[:80]}",
                    body=body,
                )
            elif event_type == "search_round_start":
                self._send_message(
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject=f"Literature search round {payload.get('round', '?')} started",
                    body=(
                        f"Keywords: {payload.get('keywords')}\n"
                        f"Authors: {payload.get('authors') or []}\n"
                        f"Year range: {payload.get('year_min')}–{payload.get('year_max')}\n"
                        f"Relaxed: {bool(payload.get('relaxed'))}"
                    ),
                )
            elif event_type == "round_summary":
                self._send_message(
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject=f"Literature search round {payload.get('round', '?')} summary",
                    body=(
                        f"Raw candidates: {payload.get('raw', 0)}\n"
                        f"Validated: {payload.get('validated', 0)}\n"
                        f"Skipped by prefilter: {payload.get('skipped_prefilter', 0)}\n"
                        f"Accepted in scope: {payload.get('in_scope', 0)}\n"
                        f"Sources: {payload.get('sources') or []}"
                    ),
                )
            elif event_type == "keyword_refined":
                self._send_message(
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject=f"Literature keywords refined for round {payload.get('round', '?')}",
                    body=f"Next keyword sets: {payload.get('keywords')}",
                )

        review_fn = self.literature_agent.review_related_literature
        review_kwargs: dict[str, Any] = {}
        signature = inspect.signature(review_fn)
        if "stream_callback" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            review_kwargs["stream_callback"] = _on_literature_event
        response = self._resolve_response(
            service_id=AssistantId.LITERATURE_REVIEWER.value,
            response=review_fn(task, article, brief, **review_kwargs),
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
        metadata: dict[str, Any] = {"sample_steps": 200}
        resolved_inquiries = self._resolve_phase_inquiries(brief, phase="simulation")
        if resolved_inquiries:
            metadata["resolved_inquiries"] = resolved_inquiries
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

    def _resolve_phase_inquiries(self, brief: ArtifactRef, *, phase: str) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for inquiry in self._inquiries_from_brief(brief, phase=phase):
            if not inquiry.get("blocking", True):
                continue
            resolved.append(self._resolve_inquiry(inquiry))
        return resolved

    def _resolve_inquiry(self, inquiry: dict[str, Any]) -> dict[str, Any]:
        inquiry_id = str(inquiry.get("id") or "inquiry").strip() or "inquiry"
        session_id = f"{self.directive.directive_id}-inquiry-{inquiry_id}"
        options = list(inquiry.get("options") or [])
        options_text = "\n".join(
            f"{index}. {option.get('label') or option.get('name') or 'Option'}"
            + (f" — {option.get('description')}" if option.get("description") else "")
            for index, option in enumerate(options, 1)
        )
        prompt = str(inquiry.get("question") or "").strip()
        if options_text:
            prompt = f"{prompt}\n\n{options_text}\n\nAnswer with option numbers, names, or 'all'."
        request = OrchestrationRequestEnvelope.new(
            resume_token=session_id,
            request_kind="inquiry",
            question=prompt,
            expected_schema=inquiry.get("expected_schema") or (
                {"answers": {"selection": "string"}} if options else {"answers": {"answer": "string"}}
            ),
            capability_hint="user",
            metadata={"inquiry": inquiry, "options": options},
        )

        def _resume_inquiry(_session_id: str, payload: dict[str, Any]) -> MediatedResponse:
            return MediatedResponse(
                status="completed",
                session_id=session_id,
                payload={
                    "inquiry_id": inquiry_id,
                    "answer": payload,
                    "selected_options": self._select_options(options, payload) if options else [],
                },
            )

        resolved = self._resolve_response(
            service_id=AssistantId.ORCHESTRATOR.value,
            response=MediatedResponse(status="needs_request", session_id=session_id, request=request),
            resume_fn=_resume_inquiry,
        )
        payload = resolved.payload if isinstance(resolved.payload, dict) else {}
        return {
            "id": inquiry_id,
            "kind": inquiry.get("kind") or "inquiry",
            "question": inquiry.get("question") or "",
            "answer": payload.get("answer") or {},
            "selected_options": payload.get("selected_options") or [],
            "inquiry": inquiry,
        }

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
        if "action" in payload or "feedback" in payload:
            return {
                "kind": "steering_feedback",
                "action": payload.get("action", "continue"),
                "feedback": payload.get("feedback", "")
            }
        if "clarification" in payload:
            return {"kind": "clarification_answers", "answers": {"clarification": str(payload["clarification"])}}
        return {"kind": "steering_feedback", "feedback": json.dumps(payload), "action": "continue"}

    def _request_steering_metadata(
        self,
        *,
        service_id: str,
        session_id: str,
        request: OrchestrationRequestEnvelope,
    ) -> dict[str, Any]:
        title = {
            "clarification": "Answer agent clarification",
            "inquiry": "Resolve agent inquiry",
        }.get(request.request_kind, "Respond to agent request")
        return {
            "checkpoint_id": request.request_id,
            "state": "awaiting_pi",
            "kind": "agent_request" if request.request_kind in ("inquiry", "clarification") else "checkpoint",
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
    def _inquiries_from_brief(brief: ArtifactRef, *, phase: str) -> list[dict[str, Any]]:
        raw_inquiries = brief.metadata.get("inquiries")
        if not isinstance(raw_inquiries, list):
            return []
        inquiries: list[dict[str, Any]] = []
        for index, item in enumerate(raw_inquiries, 1):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or item.get("prompt") or "").strip()
            if not question:
                continue
            applies_to = item.get("applies_to")
            if isinstance(applies_to, list):
                scopes = {str(value).strip().lower() for value in applies_to if str(value).strip()}
                if scopes and phase.lower() not in scopes:
                    continue
            inquiry = dict(item)
            inquiry["id"] = str(item.get("id") or item.get("inquiry_id") or f"inquiry_{index}").strip()
            inquiry["question"] = question
            inquiry["kind"] = str(item.get("kind") or "inquiry").strip()
            inquiry["options"] = MediatorWorkflowRunner._normalize_options(item.get("options"))
            inquiries.append(inquiry)
        return inquiries

    @staticmethod
    def _normalize_options(raw_options: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_options, list):
            return []
        options: list[dict[str, Any]] = []
        for index, item in enumerate(raw_options, 1):
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("name") or item.get("title") or f"Option {index}").strip()
                description = str(item.get("description") or item.get("summary") or "").strip()
                value = item.get("value", item)
            else:
                label = str(item or "").strip()
                description = ""
                value = item
            if label or description:
                options.append({"label": label or f"Option {index}", "description": description, "value": value})
        return options

    @staticmethod
    def _select_options(options: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
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
            return options

        selected: list[dict[str, Any]] = []
        lowered = text.lower()
        for index, option in enumerate(options, 1):
            label = str(option.get("label") or "").lower()
            if str(index) in lowered.split() or f"{index}," in lowered or f"{index}." in lowered:
                selected.append(option)
                continue
            if label and label in lowered:
                selected.append(option)
        return selected or options

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
