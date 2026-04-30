"""
research_platform.workflow
──────────────────────────
Professor-oriented orchestration built on shared assistant ports.
"""
from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Optional

from .assistants import AssistantExecutionError
from .article_lookup import build_article_lookup_query
from .contracts import (
    ArtifactRef,
    AssistantId,
    AttachmentRef,
    CodingAssistantPort,
    LiteratureAssistantPort,
    MailboxMessage,
    TaskEnvelope,
    WriterAssistantPort,
)
from .registry import ArtifactRegistry
from .steering import (
    SteeringCheckpoint,
    SteeringController,
    SteeringDecision,
    WorkflowStopRequested,
)

log = logging.getLogger(__name__)


class ProfessorWorkflowRunner:
    def __init__(
        self,
        directive: Any,
        mailbox: Any,
        *,
        literature_agent: LiteratureAssistantPort,
        coding_agent: CodingAssistantPort,
        writer_agent: WriterAssistantPort,
        artifact_registry: ArtifactRegistry,
        steering_control: Optional[SteeringController] = None,
    ) -> None:
        self.directive = directive
        self.mailbox = mailbox
        self.literature_agent = literature_agent
        self.coding_agent = coding_agent
        self.writer_agent = writer_agent
        self.artifact_registry = artifact_registry
        self.steering_control = steering_control
        self._steering_notes: dict[str, list[str]] = {}
        self.output_dir = Path(directive.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all_phases(self) -> None:
        self._send_message(
            assistant=AssistantId.ORCHESTRATOR.value,
            subject="Instruction accepted — workflow initiated",
            body=(
                f"I have accepted the directive and opened a new research workflow.\n\n"
                f"Instruction:\n{self.directive.instruction}\n\n"
                f"Planned phases: {', '.join(self.directive.phases)}"
            ),
        )

        article: Optional[ArtifactRef] = None
        brief: Optional[ArtifactRef] = None
        literature: Optional[ArtifactRef] = None
        coding_artifacts: list[ArtifactRef] = []

        for phase in self.directive.phases:
            try:
                self._raise_if_stopped()
                log.info("[%s] Running phase: %s", self.directive.directive_id, phase)
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
                    coding_artifacts = self._phase_simulation(article, brief, literature)
                elif phase == "write":
                    if not coding_artifacts:
                        if article is None:
                            article = self._phase_find_article()
                        if brief is None:
                            brief = self._phase_parse_article(article)
                        if literature is None:
                            literature = self._phase_literature(article, brief)
                        coding_artifacts = self._phase_simulation(article, brief, literature)
                    self._phase_write([article, brief, literature] + coding_artifacts)
                else:
                    log.warning("Unknown phase: %s", phase)
            except WorkflowStopRequested as exc:
                self._send_message(
                    assistant=AssistantId.ORCHESTRATOR.value,
                    subject="Workflow stopped by PI",
                    body=(
                        "I stopped the professor workflow.\n\n"
                        + (f"Reason:\n{exc.reason}" if exc.reason else "No reason was provided.")
                    ),
                )
                break
            except AssistantExecutionError as exc:
                self._send_error(phase, str(exc), attachments=[artifact.as_attachment() for artifact in exc.artifacts])
                break
            except Exception:
                error_trace = traceback.format_exc()
                log.error("Error in phase %s:\n%s", phase, error_trace)
                self._send_error(phase, error_trace)
                break

    def _phase_find_article(self) -> ArtifactRef:
        while True:
            query_task = self._task(
                AssistantId.LITERATURE_REVIEWER.value,
                task_id=f"{self.directive.directive_id}-prepare-article-query",
                subject="Prepare article lookup query",
                instructions=self._instruction_with_notes("find_article"),
                metadata={"steering_notes": self._phase_notes("find_article")},
            )
            lookup_spec = self._prepare_article_lookup_spec(query_task)
            article_query = str(lookup_spec.get("query_string") or "")
            lookup_attachments = [
                AttachmentRef(
                    name="article_lookup_spec.json",
                    content=json.dumps(lookup_spec, indent=2),
                    mime_type="application/json",
                    metadata=lookup_spec,
                ),
            ]
            if self.steering_control is not None and lookup_spec.get("needs_clarification"):
                question = str(lookup_spec.get("clarification_question") or "").strip()
                decision = self._checkpoint(
                    phase="find_article",
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject="Article search needs clarification",
                    body=(
                        "I need one clarification before I search for the target paper.\n\n"
                        f"Question:\n{question}\n\n"
                        "If you want me to refine the search before I proceed, request a revision and answer the question. "
                        "If you are comfortable with a broad search, continue."
                    ),
                    attachments=lookup_attachments,
                    title="Clarify article search scope",
                    prompt=(
                        "If you want to answer the clarification question, request a revision and provide the answer. "
                        "If you prefer a broad search, continue."
                    ),
                    checkpoint_key="find_article:clarify",
                )
                if decision.normalized_action() == "revise":
                    self._record_steering("find_article", decision.feedback)
                    self._send_message(
                        assistant=AssistantId.ORCHESTRATOR.value,
                        subject="PI clarification received — revising article search",
                        body=f"I will rerun article lookup preparation with this clarification:\n\n{decision.feedback.strip()}",
                    )
                    continue
            task = self._task(
                AssistantId.LITERATURE_REVIEWER.value,
                task_id=f"{self.directive.directive_id}-find-article",
                subject="Find primary article",
                instructions=(
                    "Locate the primary article described by the directive using the prepared lookup query.\n\n"
                    f"{self._notes_block('find_article')}"
                ).strip(),
                metadata={
                    "article_lookup_query": article_query,
                    "article_lookup_spec": lookup_spec,
                    "steering_notes": self._phase_notes("find_article"),
                },
            )
            article = self.literature_agent.find_primary_article(task)
            authors = ", ".join(article.metadata.get("authors") or [])
            year = article.metadata.get("year") or "?"
            short_id = article.metadata.get("short_id") or article.metadata.get("url") or article.artifact_id
            body = (
                "I searched for the requested paper and found the following match:\n\n"
                    f"  \"{article.metadata.get('title', article.title)}\"\n"
                    f"  {authors} ({year}) — {short_id}\n\n"
                    f"Article lookup query:\n{article_query}\n\n"
                    f"Abstract:\n{article.metadata.get('abstract', '')}\n\n"
                    "Please confirm that this is the correct target article, or request a revised search."
            )
            attachments = [
                article.as_attachment(name="article_match.json"),
                *lookup_attachments,
            ]
            if article.url:
                attachments.append(
                    AttachmentRef(name="Original paper", url=article.url, mime_type="text/plain")
                )
            if self.steering_control is None:
                self._send_message(
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject="Found article — ready to proceed",
                    body=(
                        "I searched for the requested paper and found the following match:\n\n"
                            f"  \"{article.metadata.get('title', article.title)}\"\n"
                            f"  {authors} ({year}) — {short_id}\n\n"
                            f"Article lookup query:\n{article_query}\n\n"
                            f"Abstract:\n{article.metadata.get('abstract', '')}\n\n"
                            "I will now extract the reproduction brief."
                    ),
                    attachments=attachments,
                )
                return article
            decision = self._checkpoint(
                phase="find_article",
                assistant=AssistantId.LITERATURE_REVIEWER.value,
                subject="Found article — awaiting PI confirmation",
                body=body,
                attachments=attachments,
                title="Review the primary article match",
                prompt=(
                    "Continue if the located paper is correct. If the wrong paper was matched, "
                    "request a revision and say what authors, year, title terms, or model details should be used instead."
                ),
            )
            if decision.normalized_action() != "revise":
                return article
            self._record_steering("find_article", decision.feedback)
            self._send_message(
                assistant=AssistantId.ORCHESTRATOR.value,
                subject="PI steering received — revising article search",
                body=f"I will rerun article lookup with this correction:\n\n{decision.feedback.strip()}",
            )

    def _prepare_article_lookup_spec(self, task: TaskEnvelope) -> dict[str, Any]:
        prepare_spec = getattr(self.literature_agent, "prepare_article_lookup_spec", None)
        if callable(prepare_spec):
            spec = prepare_spec(task)
            if isinstance(spec, dict) and (
                str(spec.get("query_string") or "").strip() or spec.get("needs_clarification")
            ):
                return spec
        prepare = getattr(self.literature_agent, "prepare_article_lookup_query", None)
        if callable(prepare):
            query = str(prepare(task) or "").strip()
            if query:
                return {"query_string": query}
        return {
            "query_string": build_article_lookup_query(
                str(task.metadata.get("topic_hint") or ""),
                task.instructions,
            )
        }

    def _phase_parse_article(self, article: ArtifactRef) -> ArtifactRef:
        while True:
            task = self._task(
                AssistantId.LITERATURE_REVIEWER.value,
                task_id=f"{self.directive.directive_id}-article-brief",
                subject="Build article brief",
                metadata={"steering_notes": self._phase_notes("parse_article")},
                attachments=[article.as_attachment()],
            )
            brief = self.literature_agent.build_article_brief(task, article)
            params = brief.metadata.get("key_parameters") or {}
            params_text = "\n".join(f"    - {key}: {value}" for key, value in params.items()) or "    - (none parsed)"
            body = (
                "Here is the extracted reproduction brief:\n\n"
                f"  MODEL: {brief.metadata.get('model_description', '')}\n"
                f"  KEY PARAMETERS:\n{params_text}\n"
                f"  PROCEDURE: {brief.metadata.get('procedure', '')}\n\n"
                "Please confirm that this is the correct model and procedure interpretation, or request a revision."
            )
            attachments = [brief.as_attachment(name="article_brief.json")]
            brief_md = self.artifact_registry.get("article-brief-md")
            if brief_md is not None:
                attachments.append(brief_md.as_attachment(name="article_brief.md"))
            if self.steering_control is None:
                self._send_message(
                    assistant=AssistantId.LITERATURE_REVIEWER.value,
                    subject="Article brief ready — parameters confirmed",
                    body=(
                        "Here is the extracted reproduction brief:\n\n"
                        f"  MODEL: {brief.metadata.get('model_description', '')}\n"
                        f"  KEY PARAMETERS:\n{params_text}\n"
                        f"  PROCEDURE: {brief.metadata.get('procedure', '')}\n\n"
                        "I will now gather the surrounding literature."
                    ),
                    attachments=attachments,
                )
                return brief
            decision = self._checkpoint(
                phase="parse_article",
                assistant=AssistantId.LITERATURE_REVIEWER.value,
                subject="Article brief ready — awaiting PI confirmation",
                body=body,
                attachments=attachments,
                title="Review the article brief",
                prompt=(
                    "Continue if the extracted model, parameters, and procedure match the paper. "
                    "If the brief focused on the wrong model or missed critical details, request a revision with specific guidance."
                ),
            )
            if decision.normalized_action() != "revise":
                return brief
            self._record_steering("parse_article", decision.feedback)
            self._send_message(
                assistant=AssistantId.ORCHESTRATOR.value,
                subject="PI steering received — revising article brief",
                body=f"I will rebuild the article brief with this guidance:\n\n{decision.feedback.strip()}",
            )

    def _phase_literature(self, article: ArtifactRef, brief: ArtifactRef) -> ArtifactRef:
        task = self._task(
            AssistantId.LITERATURE_REVIEWER.value,
            task_id=f"{self.directive.directive_id}-literature",
            subject="Review related literature",
            metadata={"steering_notes": self._phase_notes("literature")},
            attachments=[article.as_attachment(), brief.as_attachment()],
        )
        literature = self.literature_agent.review_related_literature(task, article, brief)
        accepted_count = int(literature.metadata.get("accepted_count", 0))
        synthesis = str(literature.metadata.get("synthesis", literature.summary))
        attachments = [literature.as_attachment(name="literature_review.json")]
        synthesis_md = self.artifact_registry.get("literature-synthesis-md")
        if synthesis_md is not None:
            attachments.append(synthesis_md.as_attachment(name="literature_synthesis.md"))
        self._send_message(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            subject=f"{accepted_count} related papers found",
            body=(
                f"I completed the related-literature review and found {accepted_count} in-scope paper(s).\n\n"
                f"Synthesis:\n{synthesis}\n\n"
                "The synthesis and structured review are attached."
            ),
            attachments=attachments,
        )
        return literature

    def _phase_simulation(
        self,
        article: ArtifactRef,
        brief: ArtifactRef,
        literature: ArtifactRef,
    ) -> list[ArtifactRef]:
        task = self._task(
            AssistantId.CODING_AGENT.value,
            task_id=f"{self.directive.directive_id}-simulation",
            subject="Reproduce the model",
            attachments=[article.as_attachment(), brief.as_attachment(), literature.as_attachment()],
            metadata={"simulation_timeout_seconds": 600.0},
        )
        return self.coding_agent.execute_task(
            task,
            sources=[article, brief, literature],
            update_callback=self._on_coding_update,
            review_callback=self._on_coding_review if self.steering_control is not None else None,
            cancel_callback=self._raise_if_stopped if self.steering_control is not None else None,
        )

    def _phase_write(self, sources: list[ArtifactRef]) -> ArtifactRef:
        self._send_message(
            assistant=AssistantId.WRITER.value,
            subject="Mini-review draft in progress",
            body=(
                "I have begun drafting the review article based on the article brief, "
                "related literature, and simulation outputs."
            ),
        )
        task = self._task(
            AssistantId.WRITER.value,
            task_id=f"{self.directive.directive_id}-write",
            subject="Write review draft",
            instructions=(
                "Write a grounded review draft for the PI based on the primary article, the related "
                "literature, and the simulation outputs. Summarize the article's model, the related "
                "work, the implemented reproduction, the outcome, and the main limitations."
            ),
            metadata={
                "venue": "Research note",
                "directive_instruction": getattr(self.directive, "instruction", ""),
            },
            attachments=[source.as_attachment() for source in sources if source is not None],
        )
        draft = self.writer_agent.draft_review(task, sources=[source for source in sources if source is not None])
        self._send_message(
            assistant=AssistantId.WRITER.value,
            subject="Mini-review draft ready",
            body="The grounded review draft is ready and attached.",
            attachments=[draft.as_attachment(name="mini_review.md")],
        )
        return draft

    def _on_coding_update(self, stage: str, artifacts: list[ArtifactRef], message: str) -> None:
        subject_map = {
            "simulation_launched": "Simulation launched — code and spec attached",
            "sample_reviewed": "Sample reviewed — ready for full sweep",
            "full_sweep_launched": "Full sweep launched",
            "full_results_ready": "Full results ready — analysis attached",
        }
        assistant = AssistantId.CODING_AGENT.value
        attachments = [artifact.as_attachment() for artifact in artifacts]
        self._send_message(
            assistant=assistant,
            subject=subject_map.get(stage, stage.replace("_", " ").title()),
            body=message,
            attachments=attachments,
        )

    def _task(
        self,
        assistant: str,
        *,
        task_id: str,
        subject: str,
        instructions: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        attachments: Optional[list[AttachmentRef]] = None,
    ) -> TaskEnvelope:
        merged_metadata = {
            "topic_hint": getattr(self.directive, "topic_hint", ""),
            "pi_name": getattr(self.directive, "pi_name", "PI"),
        }
        if metadata:
            merged_metadata.update(metadata)
        return TaskEnvelope(
            task_id=task_id,
            directive_id=self.directive.directive_id,
            assistant=assistant,
            subject=subject,
            instructions=instructions or self.directive.instruction,
            output_dir=str(self.output_dir),
            metadata=merged_metadata,
            attachments=attachments or [],
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
                assistant=_display_name(assistant),
                assistant_addr=_assistant_addr(assistant),
                to_name=self.directive.pi_name,
                subject=subject,
                body=body,
                attachments=attachments or [],
                metadata=metadata or {},
            )
        )

    def _send_error(
        self,
        phase: str,
        details: str,
        *,
        attachments: Optional[list[AttachmentRef]] = None,
    ) -> None:
        self._send_message(
            assistant=AssistantId.ORCHESTRATOR.value,
            subject=f"Error in phase: {phase}",
            body=(
                f"An error occurred while executing phase '{phase}'.\n\n"
                f"Error details:\n{details}"
            ),
            attachments=attachments,
        )

    def _on_coding_review(
        self,
        stage: str,
        artifacts: list[ArtifactRef],
        message: str,
    ) -> SteeringDecision:
        titles = {
            "simulation_spec_ready": "Review the simulation specification",
            "simulation_script_ready": "Review the generated simulation script",
            "sample_reviewed": "Review the sample run before the full sweep",
            "full_results_ready": "Review the simulation results before writing",
        }
        prompts = {
            "simulation_spec_ready": (
                "Continue if the planned simulation model is correct. "
                "If the wrong model or parameterization was chosen, request a revision with concrete corrections."
            ),
            "simulation_script_ready": (
                "Continue if the generated code matches the intended model. "
                "If the script reflects the wrong assumptions or setup, request a revision."
            ),
            "sample_reviewed": (
                "Continue if the sample run looks credible. "
                "If the sample shows the wrong behavior, request a revision before the full sweep."
            ),
            "full_results_ready": (
                "Continue if the simulation output is good enough to summarize. "
                "If the run needs to be re-parameterized or rerun, request a revision with specific feedback."
            ),
        }
        subjects = {
            "simulation_spec_ready": "Simulation spec ready — awaiting PI confirmation",
            "simulation_script_ready": "Simulation script ready — awaiting PI confirmation",
            "sample_reviewed": "Sample reviewed — awaiting PI confirmation",
            "full_results_ready": "Full results ready — awaiting PI confirmation",
        }
        return self._checkpoint(
            phase="simulation",
            assistant=AssistantId.CODING_AGENT.value,
            subject=subjects.get(stage, "Coding review — awaiting PI confirmation"),
            body=message,
            attachments=[artifact.as_attachment() for artifact in artifacts],
            title=titles.get(stage, "Review coding output"),
            prompt=prompts.get(stage, "Continue or request a revision with specific guidance."),
            checkpoint_key=f"simulation:{stage}",
        )

    def _checkpoint(
        self,
        *,
        phase: str,
        assistant: str,
        subject: str,
        body: str,
        attachments: list[AttachmentRef],
        title: str,
        prompt: str,
        checkpoint_key: Optional[str] = None,
    ) -> SteeringDecision:
        checkpoint = self._build_checkpoint(
            phase=phase,
            assistant=assistant,
            title=title,
            prompt=prompt,
            attachments=attachments,
            checkpoint_key=checkpoint_key or phase,
        )
        if checkpoint is not None:
            assert self.steering_control is not None
            self.steering_control.open(checkpoint)
        metadata = self._checkpoint_metadata(checkpoint) if checkpoint else None
        self._send_message(
            assistant=assistant,
            subject=subject,
            body=body,
            attachments=attachments,
            metadata=metadata,
        )
        if checkpoint is None:
            return SteeringDecision(checkpoint_id=checkpoint_key or phase, action="continue")
        return self._await_checkpoint(checkpoint)

    def _build_checkpoint(
        self,
        *,
        phase: str,
        assistant: str,
        title: str,
        prompt: str,
        attachments: list[AttachmentRef],
        checkpoint_key: str,
    ) -> Optional[SteeringCheckpoint]:
        if self.steering_control is None:
            return None
        return SteeringCheckpoint(
            checkpoint_id=f"{self.directive.directive_id}:{checkpoint_key}",
            directive_id=self.directive.directive_id,
            phase=phase,
            assistant=assistant,
            title=title,
            prompt=prompt,
            attachments=attachments,
        )

    def _await_checkpoint(self, checkpoint: SteeringCheckpoint) -> SteeringDecision:
        assert self.steering_control is not None
        decision = self.steering_control.wait(checkpoint.checkpoint_id)
        if decision.normalized_action() == "stop":
            raise WorkflowStopRequested(decision.feedback)
        if decision.normalized_action() == "continue":
            self._send_message(
                assistant=AssistantId.ORCHESTRATOR.value,
                subject="PI confirmed — proceeding",
                body=f"PI confirmed checkpoint '{checkpoint.title}'. Proceeding to the next step.",
            )
        return decision

    @staticmethod
    def _checkpoint_metadata(checkpoint: SteeringCheckpoint) -> dict[str, Any]:
        payload = checkpoint.to_dict()
        payload["state"] = "awaiting_pi"
        return {"steering": payload}

    def _record_steering(self, phase: str, feedback: str) -> None:
        note = feedback.strip()
        if not note:
            return
        self._steering_notes.setdefault(phase, []).append(note)

    def _phase_notes(self, phase: str) -> list[str]:
        return list(self._steering_notes.get(phase, []))

    def _notes_block(self, phase: str) -> str:
        notes = self._phase_notes(phase)
        if not notes:
            return ""
        return "PI steering notes:\n" + "\n".join(f"- {note}" for note in notes)

    def _instruction_with_notes(self, phase: str) -> str:
        notes = self._phase_notes(phase)
        if not notes:
            return getattr(self.directive, "instruction", "")
        return (
            f"{getattr(self.directive, 'instruction', '')}\n\n"
            "PI steering notes:\n"
            + "\n".join(f"- {note}" for note in notes)
        )

    def _raise_if_stopped(self) -> None:
        if self.steering_control is None:
            return
        if self.steering_control.stop_requested():
            raise WorkflowStopRequested(self.steering_control.stop_reason())


def _assistant_addr(assistant: str) -> str:
    return f"{assistant.lower().replace(' ', '.')}@research.local"


def _display_name(assistant: str) -> str:
    mapping = {
        AssistantId.LITERATURE_REVIEWER.value: "Literature Reviewer",
        AssistantId.CODING_AGENT.value: "Coding Agent",
        AssistantId.WRITER.value: "Writer Agent",
        AssistantId.ORCHESTRATOR.value: "Research Orchestrator",
    }
    return mapping.get(assistant, assistant.replace("_", " ").title())
