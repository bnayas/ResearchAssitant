from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from comms.pi_email import PIMailbox
from research_platform import ArtifactRef, AssistantId, AttachmentRef, MailboxMessage
from research_platform.article_lookup import build_article_lookup_query
from research_platform.assistants.literature import LocalLiteratureAssistant
from research_platform.registry import ArtifactRegistry
from research_platform.steering import SteeringController
from research_platform.contracts import TaskEnvelope
from sim_tool.research_directive import DirectiveRunner, PIDirective


def test_mailbox_message_serialization_and_persistence(tmp_path):
    mailbox = PIMailbox(output_dir=tmp_path)
    message = MailboxMessage(
        directive_id="dir-1",
        assistant="Literature Reviewer",
        assistant_addr="literature_reviewer@research.local",
        to_name="Prof. Shnerb",
        subject="Article brief ready",
        body="The article brief is attached.",
        attachments=[
            AttachmentRef(
                name="article_brief.json",
                path=str((tmp_path / "article_brief.json").resolve()),
                mime_type="application/json",
            )
        ],
    )
    mailbox.append(message)

    emails = mailbox.all_emails()
    assert len(emails) == 1
    assert emails[0].subject == "Article brief ready"
    assert emails[0].attachments[0].name == "article_brief.json"

    saved = sorted((tmp_path / "emails").glob("*.json"))
    assert saved
    payload = json.loads(saved[0].read_text(encoding="utf-8"))
    assert payload["subject"] == "Article brief ready"
    assert payload["attachments"][0]["path"].endswith("article_brief.json")


class _FakeLiteratureAssistant:
    def __init__(self, registry: ArtifactRegistry):
        self.registry = registry
        self.find_task = None

    def prepare_article_lookup_spec(self, task):
        return {
            "query_string": "environmental stochasticity, neutral model",
            "query_terms": ["environmental stochasticity", "neutral model"],
            "required_authors": ["Danino", "Shnerb"],
            "preferred_year": None,
            "year_min": None,
            "year_max": None,
            "title_phrases": [],
            "needs_clarification": False,
            "clarification_question": "",
        }

    def find_primary_article(self, task):
        self.find_task = task
        return self.registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="primary_article",
            title="Primary Article Match",
            filename="article/article_match.json",
            payload={
                "title": "Theory of time-averaged neutral dynamics with environmental stochasticity",
                "authors": ["Matan Danino", "Nadav M. Shnerb"],
                "abstract": "Neutral model with environmental noise.",
                "url": "https://arxiv.org/abs/1711.11332v3",
                "year": 2017,
                "short_id": "arXiv:1711.11332v3",
            },
            summary="Primary article located",
            metadata={
                "title": "Theory of time-averaged neutral dynamics with environmental stochasticity",
                "authors": ["Matan Danino", "Nadav M. Shnerb"],
                "abstract": "Neutral model with environmental noise.",
                "url": "https://arxiv.org/abs/1711.11332v3",
                "year": 2017,
                "short_id": "arXiv:1711.11332v3",
            },
            artifact_id="article-match",
        )

    def build_article_brief(self, task, article):
        self.registry.save_text(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="article_brief_markdown",
            title="Article Brief",
            filename="article/article_brief.md",
            text="# Article Brief\n\nModel B neutral dynamics.",
            summary="Model B brief",
            artifact_id="article-brief-md",
        )
        return self.registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="article_brief",
            title="Article Brief",
            filename="article/article_brief.json",
            payload={
                "model_description": "Model B neutral dynamics with environmental stochasticity.",
                "key_parameters": {"delta": "0.1", "N": "1000"},
                "procedure": "Monte Carlo simulation.",
                "expected_figures": ["Species abundance distribution"],
            },
            summary="Model B brief",
            metadata={
                "model_description": "Model B neutral dynamics with environmental stochasticity.",
                "key_parameters": {"delta": "0.1", "N": "1000"},
                "procedure": "Monte Carlo simulation.",
                "expected_figures": ["Species abundance distribution"],
            },
            artifact_id="article-brief",
        )

    def review_related_literature(self, task, article, brief):
        self.registry.save_text(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="literature_synthesis_markdown",
            title="Literature Synthesis",
            filename="literature/literature_synthesis.md",
            text="# Literature Synthesis\n\nRelated neutral theory papers.",
            summary="Related neutral theory papers.",
            artifact_id="literature-synthesis-md",
        )
        return self.registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="literature_review",
            title="Related Literature Review",
            filename="literature/literature_review.json",
            payload={"artifact": {"status": "complete"}, "audit": {"passed": True}},
            summary="2 accepted paper(s)",
            metadata={
                "accepted_count": 2,
                "synthesis": "Related neutral theory papers.",
            },
            artifact_id="literature-review",
        )


class _FakeCodingAssistant:
    def __init__(self, registry: ArtifactRegistry):
        self.registry = registry

    def execute_task(self, task, *, sources, update_callback=None, review_callback=None, cancel_callback=None):
        launched = self.registry.save_text(
            assistant=AssistantId.CODING_AGENT.value,
            kind="simulation_script",
            title="Simulation Script",
            filename="simulation/sim.py",
            text="print('simulation')\n",
            summary="Generated simulation script",
            artifact_id="simulation-script",
        )
        if update_callback:
            update_callback("simulation_launched", [launched], "Simulation launched.")

        sample = self.registry.save_json(
            assistant=AssistantId.CODING_AGENT.value,
            kind="sample_run_summary",
            title="Sample Run Summary",
            filename="simulation/sample_run_summary.json",
            payload={"status": "ok"},
            summary="Sample run ok",
            artifact_id="simulation-sample-summary",
        )
        if update_callback:
            update_callback("sample_reviewed", [sample], "Sample reviewed.")

        full = self.registry.save_json(
            assistant=AssistantId.CODING_AGENT.value,
            kind="full_run_analysis",
            title="Full Run Analysis",
            filename="simulation/full_run_analysis.json",
            payload={"verdict": "ok"},
            summary="Full results ready",
            artifact_id="simulation-full-analysis",
        )
        if update_callback:
            update_callback("full_results_ready", [full], "Full results ready.")
        return [launched, sample, full]


class _FakeWriterAssistant:
    def __init__(self, registry: ArtifactRegistry):
        self.registry = registry

    def draft_review(self, task, *, sources):
        return self.registry.save_text(
            assistant=AssistantId.WRITER.value,
            kind="review_draft",
            title="Mini-review Draft",
            filename="writer/mini_review.md",
            text="# Mini-review\n\nGrounded draft.",
            summary="Grounded draft.",
            artifact_id="mini-review",
        )


class _SteerableLiteratureAssistant(_FakeLiteratureAssistant):
    def __init__(self, registry: ArtifactRegistry):
        super().__init__(registry)
        self.calls = 0
        self.tasks = []

    def prepare_article_lookup_spec(self, task):
        notes = "\n".join(str(item) for item in task.metadata.get("steering_notes", []))
        preferred_year = 2017 if "2017" in notes else None
        return {
            "query_string": "environmental stochasticity, neutral model",
            "query_terms": ["environmental stochasticity", "neutral model"],
            "required_authors": ["Danino", "Shnerb"],
            "preferred_year": preferred_year,
            "year_min": preferred_year,
            "year_max": preferred_year,
            "title_phrases": [],
            "needs_clarification": False,
            "clarification_question": "",
        }

    def find_primary_article(self, task):
        self.find_task = task
        self.tasks.append(task)
        self.calls += 1
        if self.calls == 1:
            title = "Unrelated neutral theory overview"
            authors = ["Someone Else"]
            short_id = "arXiv:0000.00001"
        else:
            title = "Theory of time-averaged neutral dynamics with environmental stochasticity"
            authors = ["Matan Danino", "Nadav M. Shnerb"]
            short_id = "arXiv:1711.11332v3"
        return self.registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="primary_article",
            title="Primary Article Match",
            filename="article/article_match.json",
            payload={
                "title": title,
                "authors": authors,
                "abstract": "Neutral model with environmental noise.",
                "url": f"https://arxiv.org/abs/{short_id}",
                "year": 2017,
                "short_id": short_id,
            },
            summary="Primary article located",
            metadata={
                "title": title,
                "authors": authors,
                "abstract": "Neutral model with environmental noise.",
                "url": f"https://arxiv.org/abs/{short_id}",
                "year": 2017,
                "short_id": short_id,
            },
            artifact_id="article-match",
        )


def test_directive_runner_uses_injected_assistants_and_persists_mailbox(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    directive = PIDirective(
        directive_id="danino-repro-001",
        pi_name="Prof. Shnerb",
        instruction="Read the paper, reproduce the results, and write a review.",
        topic_hint="neutral model, environmental stochasticity, Danino, Shnerb",
        phases=["find_article", "parse_article", "literature", "simulation", "write"],
        output_dir=str(tmp_path),
    )
    runner = DirectiveRunner(
        directive,
        mailbox,
        literature_agent=_FakeLiteratureAssistant(registry),
        coding_agent=_FakeCodingAssistant(registry),
        writer_agent=_FakeWriterAssistant(registry),
        artifact_registry=registry,
    )
    runner.run_all_phases()

    subjects = [email.subject for email in mailbox.all_emails()]
    assert "Instruction accepted — workflow initiated" in subjects
    assert "Found article — ready to proceed" in subjects
    assert "Article brief ready — parameters confirmed" in subjects
    assert "2 related papers found" in subjects
    assert "Simulation launched — code and spec attached" in subjects
    assert "Sample reviewed — ready for full sweep" in subjects
    assert "Full results ready — analysis attached" in subjects
    assert "Mini-review draft ready" in subjects

    saved = sorted((tmp_path / "emails").glob("*.json"))
    assert len(saved) >= 8
    assert (tmp_path / "writer" / "mini_review.md").exists()


def test_article_lookup_query_builder_extracts_structured_terms_from_instruction():
    query = build_article_lookup_query(
        "",
        (
            "Read Danino and Shnerb article about adding Environmental "
            "Stochasticity to the Neutral Model. Reproduce the results from this paper."
        ),
    )
    assert "Danino" in query
    assert "Shnerb" in query
    assert "environmental stochasticity" in query.lower()
    assert "neutral model" in query.lower()


def test_article_lookup_query_builder_combines_topic_hint_with_instruction_terms():
    query = build_article_lookup_query(
        "neutral model, environmental noise",
        "Use the 2017 Danino and Shnerb time-averaged neutral dynamics paper.",
    )
    assert "neutral model" in query.lower()
    assert "environmental noise" in query.lower()
    assert "Danino" in query
    assert "Shnerb" in query
    assert "2017" in query


def test_article_lookup_query_builder_is_not_domain_hardcoded():
    query = build_article_lookup_query(
        "",
        "Read the 2014 Smith and Welling paper on stochastic gradient Langevin dynamics.",
    )
    assert "Smith" in query
    assert "Welling" in query
    assert "2014" in query
    assert "stochastic gradient langevin dynamics" in query.lower()
    assert "neutral" not in query.lower()
    assert "environmental" not in query.lower()


def test_local_literature_assistant_prepares_article_query_with_llm(tmp_path):
    class _LookupLLM:
        def complete(self, system, messages, temperature=0.0):
            assert "structured article lookup requests" in system.lower()
            return json.dumps(
                {
                    "query_string": "stochastic gradient Langevin dynamics",
                    "query_terms": ["stochastic gradient Langevin dynamics"],
                    "required_authors": ["Smith", "Welling"],
                    "preferred_year": 2014,
                    "year_min": 2014,
                    "year_max": 2014,
                    "title_phrases": ["stochastic gradient Langevin dynamics"],
                    "needs_clarification": False,
                    "clarification_question": "",
                }
            )

    assistant = LocalLiteratureAssistant(ArtifactRegistry(root_dir=tmp_path), _LookupLLM())
    task = TaskEnvelope(
        task_id="lookup-1",
        directive_id="dir-1",
        assistant=AssistantId.LITERATURE_REVIEWER.value,
        instructions="Read the 2014 Smith and Welling paper on stochastic gradient Langevin dynamics.",
        metadata={"topic_hint": ""},
    )

    spec = assistant.prepare_article_lookup_spec(task)

    assert spec["query_string"] == "stochastic gradient Langevin dynamics"
    assert spec["query_terms"] == ["stochastic gradient Langevin dynamics"]
    assert spec["required_authors"] == ["Smith", "Welling"]
    assert spec["preferred_year"] == 2014
    assert spec["year_min"] == 2014
    assert spec["year_max"] == 2014
    assert "stochastic gradient langevin dynamics" in spec["title_phrases"][0].lower()


def test_local_literature_assistant_ignores_unrelated_lookup_json(tmp_path):
    class _BadLookupLLM:
        def complete(self, system, messages, temperature=0.0):
            return json.dumps(
                {
                    "complete": True,
                    "payload": {"name": "unrelated structured response"},
                }
            )

    assistant = LocalLiteratureAssistant(ArtifactRegistry(root_dir=tmp_path), _BadLookupLLM())
    task = TaskEnvelope(
        task_id="lookup-bad-json",
        directive_id="dir-bad-json",
        assistant=AssistantId.LITERATURE_REVIEWER.value,
        instructions="Read the 2014 Smith and Welling paper on stochastic gradient Langevin dynamics.",
        metadata={"topic_hint": ""},
    )

    spec = assistant.prepare_article_lookup_spec(task)

    assert spec["query_string"] == "stochastic gradient Langevin dynamics"
    assert spec["required_authors"] == ["Smith", "Welling"]
    assert spec["preferred_year"] == 2014


def test_directive_runner_uses_literature_query_preparation_when_available(tmp_path):
    class _PreparedQueryLiteratureAssistant(_FakeLiteratureAssistant):
        def prepare_article_lookup_spec(self, task):
            return {
                "query_string": "stochastic gradient Langevin dynamics",
                "query_terms": ["stochastic gradient Langevin dynamics"],
                "required_authors": ["Smith", "Welling"],
                "preferred_year": 2014,
                "year_min": 2014,
                "year_max": 2014,
                "title_phrases": ["stochastic gradient Langevin dynamics"],
                "needs_clarification": False,
                "clarification_question": "",
            }

    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    literature = _PreparedQueryLiteratureAssistant(registry)
    directive = PIDirective(
        directive_id="dir-prepared-query",
        pi_name="Professor",
        instruction="Read the 2014 Smith and Welling paper on stochastic gradient Langevin dynamics.",
        topic_hint="",
        phases=["find_article"],
        output_dir=str(tmp_path),
    )
    runner = DirectiveRunner(
        directive,
        mailbox,
        literature_agent=literature,
        coding_agent=_FakeCodingAssistant(registry),
        writer_agent=_FakeWriterAssistant(registry),
        artifact_registry=registry,
    )

    runner.run_all_phases()

    assert literature.find_task is not None
    assert literature.find_task.metadata["article_lookup_query"] == "stochastic gradient Langevin dynamics"
    assert literature.find_task.metadata["article_lookup_spec"]["required_authors"] == ["Smith", "Welling"]
    assert literature.find_task.metadata["article_lookup_spec"]["preferred_year"] == 2014
    assert literature.find_task.metadata["article_lookup_spec"]["year_min"] == 2014
    assert literature.find_task.metadata["article_lookup_spec"]["year_max"] == 2014


def test_directive_runner_can_pause_for_article_search_clarification(tmp_path):
    class _ClarifyingLiteratureAssistant(_FakeLiteratureAssistant):
        def prepare_article_lookup_spec(self, task):
            notes = "\n".join(str(item) for item in task.metadata.get("steering_notes", []))
            if "last 20 years" not in notes:
                return {
                    "query_string": "environmental fluctuations, neutral model",
                    "query_terms": ["environmental fluctuations", "neutral model"],
                    "required_authors": ["Danino", "Shnerb"],
                    "preferred_year": None,
                    "year_min": None,
                    "year_max": None,
                    "title_phrases": [],
                    "needs_clarification": True,
                    "clarification_question": "Should I search all years, or restrict the paper search to a specific time window?",
                }
            return {
                "query_string": "environmental fluctuations, neutral model",
                "query_terms": ["environmental fluctuations", "neutral model"],
                "required_authors": ["Danino", "Shnerb"],
                "preferred_year": None,
                "year_min": 2004,
                "year_max": 2024,
                "title_phrases": [],
                "needs_clarification": False,
                "clarification_question": "",
            }

    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    steering = SteeringController()
    literature = _ClarifyingLiteratureAssistant(registry)
    directive = PIDirective(
        directive_id="dir-clarify-search",
        pi_name="Professor",
        instruction="Read Danino and Shnerb paper on adding environmental fluctuations to the neutral model.",
        topic_hint="",
        phases=["find_article"],
        output_dir=str(tmp_path),
    )
    runner = DirectiveRunner(
        directive,
        mailbox,
        literature_agent=literature,
        coding_agent=_FakeCodingAssistant(registry),
        writer_agent=_FakeWriterAssistant(registry),
        artifact_registry=registry,
        steering_control=steering,
    )

    worker = threading.Thread(target=runner.run_all_phases, daemon=True)
    worker.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if steering.pending() is not None:
            break
        time.sleep(0.02)
    checkpoint = steering.pending()
    assert checkpoint is not None
    assert checkpoint.title == "Clarify article search scope"
    steering.submit(
        checkpoint.checkpoint_id,
        action="revise",
        feedback="Search the last 20 years only.",
    )

    deadline = time.time() + 5
    while time.time() < deadline:
        if literature.find_task is not None and steering.pending() is not None:
            break
        time.sleep(0.02)
    checkpoint = steering.pending()
    assert checkpoint is not None
    steering.submit(checkpoint.checkpoint_id, action="continue", feedback="")

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert literature.find_task is not None
    assert literature.find_task.metadata["article_lookup_spec"]["year_min"] == 2004
    assert literature.find_task.metadata["article_lookup_spec"]["year_max"] == 2024


def test_directive_runner_prepares_structured_article_lookup_query(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    literature = _FakeLiteratureAssistant(registry)
    directive = PIDirective(
        directive_id="dir-article-lookup",
        pi_name="Professor",
        instruction=(
            "Read Danino and Shnerb article about adding Environmental "
            "Stochasticity to the Neutral Model. Reproduce the results from this paper."
        ),
        topic_hint="",
        phases=["find_article"],
        output_dir=str(tmp_path),
    )
    runner = DirectiveRunner(
        directive,
        mailbox,
        literature_agent=literature,
        coding_agent=_FakeCodingAssistant(registry),
        writer_agent=_FakeWriterAssistant(registry),
        artifact_registry=registry,
    )

    runner.run_all_phases()

    assert literature.find_task is not None
    assert literature.find_task.instructions.startswith("Locate the primary article")
    article_query = literature.find_task.metadata.get("article_lookup_query", "")
    assert "environmental stochasticity" in article_query.lower()
    assert "neutral model" in article_query.lower()
    article_spec = literature.find_task.metadata.get("article_lookup_spec", {})
    assert article_spec["required_authors"] == ["Danino", "Shnerb"]


def test_directive_runner_allows_pi_to_revise_article_selection(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    steering = SteeringController()
    literature = _SteerableLiteratureAssistant(registry)
    directive = PIDirective(
        directive_id="dir-steering",
        pi_name="Professor",
        instruction="Read the Danino and Shnerb environmental stochasticity neutral model paper.",
        topic_hint="neutral model, environmental stochasticity",
        phases=["find_article"],
        output_dir=str(tmp_path),
    )
    runner = DirectiveRunner(
        directive,
        mailbox,
        literature_agent=literature,
        coding_agent=_FakeCodingAssistant(registry),
        writer_agent=_FakeWriterAssistant(registry),
        artifact_registry=registry,
        steering_control=steering,
    )

    worker = threading.Thread(target=runner.run_all_phases, daemon=True)
    worker.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if steering.pending() is not None:
            break
        time.sleep(0.02)
    checkpoint = steering.pending()
    assert checkpoint is not None
    steering.submit(
        checkpoint.checkpoint_id,
        action="revise",
        feedback="Use the 2017 Danino and Shnerb article on environmental stochasticity in the neutral model.",
    )

    deadline = time.time() + 5
    while time.time() < deadline:
        matches = [
            email
            for email in mailbox.all_emails()
            if email.subject == "Found article — awaiting PI confirmation"
        ]
        if literature.calls >= 2 and len(matches) >= 2 and steering.pending() is not None:
            break
        time.sleep(0.02)
    checkpoint = steering.pending()
    assert checkpoint is not None
    steering.submit(checkpoint.checkpoint_id, action="continue", feedback="")

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert literature.calls == 2
    assert len(literature.tasks) == 2
    assert literature.tasks[0].metadata["article_lookup_spec"] != literature.tasks[1].metadata["article_lookup_spec"]
    assert literature.tasks[1].metadata["article_lookup_spec"]["preferred_year"] == 2017
    assert literature.tasks[1].metadata["article_lookup_spec"]["required_authors"] == ["Danino", "Shnerb"]
    article = registry.get("article-match")
    assert article is not None
    assert article.metadata["title"] == "Theory of time-averaged neutral dynamics with environmental stochasticity"


def test_directive_runner_can_stop_at_pending_checkpoint(tmp_path):
    registry = ArtifactRegistry(root_dir=tmp_path)
    mailbox = PIMailbox(output_dir=tmp_path)
    steering = SteeringController()
    literature = _SteerableLiteratureAssistant(registry)
    directive = PIDirective(
        directive_id="dir-stop",
        pi_name="Professor",
        instruction="Find the target paper.",
        topic_hint="neutral model",
        phases=["find_article", "parse_article"],
        output_dir=str(tmp_path),
    )
    runner = DirectiveRunner(
        directive,
        mailbox,
        literature_agent=literature,
        coding_agent=_FakeCodingAssistant(registry),
        writer_agent=_FakeWriterAssistant(registry),
        artifact_registry=registry,
        steering_control=steering,
    )

    worker = threading.Thread(target=runner.run_all_phases, daemon=True)
    worker.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if steering.pending() is not None:
            break
        time.sleep(0.02)
    assert steering.pending() is not None
    steering.request_stop("User decided to stop.")

    worker.join(timeout=5)
    assert not worker.is_alive()
    subjects = [email.subject for email in mailbox.all_emails()]
    assert "Workflow stopped by PI" in subjects
