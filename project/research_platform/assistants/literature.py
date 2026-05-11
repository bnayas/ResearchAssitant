from __future__ import annotations

import json
import re
from typing import Any

from literature_review.article_parser import ArticleParser
from literature_review.contract import LiteratureReviewTask, ScopeConstraint, SearchDepthConfig
from literature_review.search_backends.arxiv_backend import ArXivBackend
from literature_review.search_backends.semantic_scholar_backend import SemanticScholarBackend

from ..article_lookup import article_query_candidates, build_article_lookup_query
from ..contracts import ArtifactRef, AssistantId, TaskEnvelope
from ..registry import ArtifactRegistry
from .backends import build_literature_review_bridge_from_env
from .errors import AssistantExecutionError
from .utils import jsonify, truncate


class LocalLiteratureAssistant:
    def __init__(self, registry: ArtifactRegistry, llm_backend: Any) -> None:
        self._registry = registry
        self._llm = llm_backend

    def prepare_article_lookup_spec(self, task: TaskEnvelope) -> dict[str, Any]:
        fallback = self._fallback_lookup_spec(task)
        try:
            response = self._llm.complete(
                system=(
                    "You prepare structured article lookup requests for academic search backends.\n"
                    "Return ONLY a raw JSON object. Do not use markdown.\n"
                    "Extract specific search constraints from the directive.\n"
                    "Important usage rules:\n"
                    "- query_terms are broad free-text retrieval terms sent to search backends.\n"
                    "- required_authors are post-retrieval constraints and should usually NOT appear in query_terms.\n"
                    "- preferred_year, year_min, and year_max are post-retrieval time constraints and should usually NOT appear in query_terms.\n"
                    "- title_phrases should contain only literal title fragments if the directive clearly implies them; otherwise use an empty list.\n"
                    "- Do not invent title phrases.\n"
                    "- Focus query_terms on topic phrases that are likely to appear in the paper title or abstract.\n"
                    "- If the request is materially underspecified and you need one professor answer before searching well, set needs_clarification to true and ask one concise question.\n"
                    "- Example: if no publication year or search window is given and the topic spans many years, ask whether to search all years or constrain the time window."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Directive instruction:\n{task.instructions}\n\n"
                            f"Topic hint:\n{task.metadata.get('topic_hint', '')}\n\n"
                            f"Steering notes:\n{self._steering_text(task)}\n\n"
                            "Return a JSON object with exactly these keys:\n"
                            "{\n"
                            '  "query_terms": ["<topic term or phrase used for retrieval>"],\n'
                            '  "required_authors": ["<surname>"],\n'
                            '  "preferred_year": 2017,\n'
                            '  "year_min": 2000,\n'
                            '  "year_max": 2020,\n'
                            '  "title_phrases": ["<literal title fragment if clearly implied>"],\n'
                            '  "query_string": "<compact comma-separated form of query_terms only>",\n'
                            '  "needs_clarification": false,\n'
                            '  "clarification_question": ""\n'
                            "}\n"
                            "Use null for preferred_year/year_min/year_max if unspecified."
                        ),
                    }
                ],
                temperature=0.0,
            )
        except Exception:
            return fallback
        return self._with_lookup_clarification_if_needed(
            self._coerce_lookup_spec(response, fallback=fallback),
            task,
        )

    def prepare_article_lookup_query(self, task: TaskEnvelope) -> str:
        return self.prepare_article_lookup_spec(task)["query_string"]

    def find_primary_article(self, task: TaskEnvelope) -> ArtifactRef:
        parser = ArticleParser(
            ArXivBackend(sort_by="relevance"),
            fallback_backends=[SemanticScholarBackend()],
        )
        lookup_spec = self._lookup_spec_from_task(task)
        preferred_query = str(lookup_spec.get("query_string") or "")
        attempted_queries = self._lookup_query_candidates(
            lookup_spec,
            task,
        )

        paper = None
        selected_query = preferred_query
        for candidate in attempted_queries:
            candidate_spec = dict(lookup_spec)
            candidate_spec["query_string"] = candidate
            candidate_spec["query_terms"] = [
                part.strip() for part in candidate.split(",") if part.strip()
            ]
            paper = parser.find_article(candidate, lookup_spec=candidate_spec)
            if paper is not None:
                selected_query = candidate
                break
            diagnostics = getattr(parser, "last_search_diagnostics", {})
            if diagnostics.get("all_rate_limited") or diagnostics.get("all_temporarily_blocked"):
                rate_limited = ", ".join(diagnostics.get("rate_limited_backends") or []) or "none"
                unavailable = ", ".join(diagnostics.get("unavailable_backends") or []) or "none"
                raise AssistantExecutionError(
                    "Primary article lookup is temporarily blocked across all configured backends. "
                    "The workflow stopped retrying to avoid flooding them.\n"
                    f"Rate-limited backends: {rate_limited}\n"
                    f"Temporarily unavailable backends: {unavailable}"
                )

        if paper is None:
            attempted = "\n- ".join(attempted_queries) if attempted_queries else "(no query prepared)"
            raise AssistantExecutionError(
                "Could not find any article matching the prepared article lookup query.\n"
                f"Attempted queries:\n- {attempted}"
            )
        payload = {
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "url": paper.url,
            "year": paper.year,
            "arxiv_id": paper.arxiv_id,
            "doi": paper.doi,
            "source": paper.source,
            "short_id": paper.short_id,
            "lookup_query": selected_query,
            "lookup_spec": {
                **lookup_spec,
                "query_string": selected_query,
            },
        }
        return self._registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="primary_article",
            title="Primary Article Match",
            filename="article/article_match.json",
            payload=payload,
            summary=f"{paper.title} ({paper.year or '?'})",
            metadata=payload,
            artifact_id="article-match",
        )

    def build_article_brief(self, task: TaskEnvelope, article: ArtifactRef) -> ArtifactRef:
        parser = ArticleParser(ArXivBackend())
        arxiv_id = str(article.metadata.get("arxiv_id") or "")
        abstract = str(article.metadata.get("abstract") or "")
        paper_url = str(article.metadata.get("url") or "")
        text = parser.fetch_full_text(arxiv_id, fallback_abstract=abstract, paper_url=paper_url)
        guidance = "\n".join(
            str(item).strip()
            for item in (task.metadata.get("steering_notes") or [])
            if str(item).strip()
        )
        questions = [
            "What is the detailed description of the primary simulation model (equations, dynamics)? If there are multiple distinct models, list them and state 'MULTIPLE_MODELS'.",
            (
                "List unresolved downstream inquiries that materially affect reproduction or interpretation. "
                "Return a JSON array of objects with id, kind, question, reason, applies_to, blocking, and options fields. "
                "Use options with label, description, and value for selection inquiries. Return [] if no explicit inquiry is needed."
            ),
            "What are the key parameters and their values? Provide a JSON object mapping parameter names to values.",
            "What is the step-by-step simulation algorithm procedure (e.g., Gillespie, Euler)?",
            "What are the expected figures?",
        ]
        answers = parser.extract_answers(text, questions, self._llm, guidance=guidance)
        model_description = self._normalize_answer(
            answers.get(questions[0], ""),
            fallback=self._fallback_model_description(article, text),
        )
        params = self._coerce_parameters(
            answers.get(questions[2], ""),
            text=text,
        )
        procedure = self._normalize_answer(
            answers.get(questions[3], ""),
            fallback="Use the simulation procedure reported in the article.",
        )
        expected_figures = self._coerce_figures(
            answers.get(questions[4], ""),
            fallback=[
                "Primary reproduction figure from the article",
                "Diagnostic comparison between reproduced and reported results",
            ],
        )
        inquiries = self._coerce_inquiries(answers.get(questions[1], ""))
        payload = {
            "article_artifact_id": article.artifact_id,
            "article_title": article.metadata.get("title", ""),
            "model_description": model_description,
            "inquiries": inquiries,
            "key_parameters": params,
            "procedure": procedure,
            "expected_figures": expected_figures,
        }
        summary_md = (
            f"# Article Brief\n\n"
            f"## Model\n{model_description}\n\n"
            f"## Inquiries\n"
            + "\n".join(
                f"- **{inquiry.get('id', 'inquiry')}**: {inquiry.get('question', '')}"
                for inquiry in inquiries
            )
            + "\n\n"
            f"## Procedure\n{procedure}\n\n"
            f"## Key Parameters\n"
            + "\n".join(f"- **{key}**: {value}" for key, value in params.items())
            + "\n\n## Expected Figures\n"
            + "\n".join(f"- {item}" for item in expected_figures)
        )
        self._registry.save_text(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="article_brief_markdown",
            title="Article Brief",
            filename="article/article_brief.md",
            text=summary_md,
            summary=truncate(summary_md),
            metadata=payload,
            artifact_id="article-brief-md",
        )
        return self._registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="article_brief",
            title="Article Brief",
            filename="article/article_brief.json",
            payload=payload,
            summary=truncate(model_description),
            metadata=payload,
            artifact_id="article-brief",
        )

    def review_related_literature(
        self,
        task: TaskEnvelope,
        article: ArtifactRef,
        brief: ArtifactRef,
        stream_callback: Any = None,
    ) -> ArtifactRef:
        bridge = build_literature_review_bridge_from_env(
            llm_sync_backend=self._llm,
            gui_event_sink=stream_callback,
        )
        query = str(brief.metadata.get("model_description") or article.metadata.get("title") or task.instructions)
        include_topics = [
            topic.strip()
            for topic in str(task.metadata.get("topic_hint") or "").split(",")
            if topic.strip()
        ]
        include_topics.extend(self._compact_literature_topics(query, article))
        include_topics = list(dict.fromkeys(topic for topic in include_topics if topic))[:10]
        authors = self._author_surnames(article.metadata.get("authors") or [])
        anchor_year = self._coerce_int(article.metadata.get("year"))
        review_task = LiteratureReviewTask(
            task_id=f"{task.task_id}-lit",
            branch_id=task.directive_id,
            query=self._with_steering_note(query, task),
            scope=ScopeConstraint(
                include_topics=include_topics,
                year_min=task.metadata.get("year_min"),
                year_max=task.metadata.get("year_max"),
                anchor_year=anchor_year,
                required_authors=authors[:3],
                max_papers=int(task.metadata.get("max_papers", 5)),
            ),
            depth=SearchDepthConfig(
                max_rounds=int(task.metadata.get("max_rounds", 4)),
                max_term_variations=int(task.metadata.get("max_term_variations", 2)),
                min_papers_threshold=int(task.metadata.get("min_papers_threshold", 2)),
                papers_per_query=int(task.metadata.get("papers_per_query", 5)),
            ),
            requestor_agent=AssistantId.ORCHESTRATOR.value,
        )
        result = bridge.run_sync(review_task)
        artifact_payload = {
            "artifact": jsonify(result.artifact),
            "audit": jsonify(result.audit),
        }
        synthesis = result.artifact.synthesis
        search_summary = self._format_literature_search_summary(result.artifact)
        self._registry.save_text(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="literature_synthesis_markdown",
            title="Literature Synthesis",
            filename="literature/literature_synthesis.md",
            text=f"# Literature Synthesis\n\n{search_summary}\n\n{synthesis}\n",
            summary=truncate(search_summary),
            metadata={
                "accepted_count": result.artifact.accepted_count,
                "audit_passed": result.audit.passed,
            },
            artifact_id="literature-synthesis-md",
        )
        return self._registry.save_json(
            assistant=AssistantId.LITERATURE_REVIEWER.value,
            kind="literature_review",
            title="Related Literature Review",
            filename="literature/literature_review.json",
            payload=artifact_payload,
            summary=search_summary,
            metadata={
                "accepted_count": result.artifact.accepted_count,
                "removed_count": result.artifact.removed_count,
                "status": result.artifact.status,
                "audit_passed": result.audit.passed,
                "synthesis": synthesis,
                "search_log": jsonify(result.artifact.search_log),
                "include_topics": include_topics,
                "required_authors": authors[:3],
                "anchor_year": anchor_year,
            },
            artifact_id="literature-review",
        )

    @staticmethod
    def _with_steering_note(query: str, task: TaskEnvelope) -> str:
        notes = [
            str(item).strip()
            for item in (task.metadata.get("steering_notes") or [])
            if str(item).strip()
        ]
        if not notes:
            return query
        return f"{query}\n\nPI steering notes:\n" + "\n".join(f"- {note}" for note in notes)

    @staticmethod
    def _steering_text(task: TaskEnvelope) -> str:
        notes = [
            str(item).strip()
            for item in (task.metadata.get("steering_notes") or [])
            if str(item).strip()
        ]
        if not notes:
            return "(none)"
        return "\n".join(f"- {note}" for note in notes)

    @staticmethod
    def _compact_literature_topics(query: str, article: ArtifactRef) -> list[str]:
        text = " ".join([
            str(article.metadata.get("title") or article.title or ""),
            str(article.metadata.get("abstract") or ""),
            query,
        ]).lower()
        candidates = [
            "time-averaged neutral dynamics",
            "time-average neutral model",
            "time averaged neutral model",
            "environmental stochasticity",
            "environmental noise",
            "neutral theory of biodiversity",
            "neutral dynamics",
            "neutral model",
            "demographic noise",
            "storage effect",
            "species abundance distribution",
            "species richness",
            "biodiversity",
        ]
        topics = [candidate for candidate in candidates if candidate in text]
        if "neutral" in text and not any("neutral" in topic for topic in topics):
            topics.append("neutral model")
        if "stochastic" in text and not any("stochastic" in topic for topic in topics):
            topics.append("environmental stochasticity")
        return topics or [str(article.metadata.get("title") or article.title or query)[:80]]

    @staticmethod
    def _author_surnames(raw_authors: Any) -> list[str]:
        if isinstance(raw_authors, str):
            raw_items = [item.strip() for item in raw_authors.split(",") if item.strip()]
        elif isinstance(raw_authors, list):
            raw_items = [str(item).strip() for item in raw_authors if str(item).strip()]
        else:
            raw_items = []
        surnames: list[str] = []
        for author in raw_items:
            surname = author.split(",", 1)[0].strip() if "," in author else author.split()[-1].strip()
            if surname and surname.lower() not in {item.lower() for item in surnames}:
                surnames.append(surname)
        return surnames

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None and str(value).strip() else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_literature_search_summary(artifact: Any) -> str:
        lines = [
            f"{artifact.accepted_count} accepted paper(s); {artifact.removed_count} removed; status={artifact.status}."
        ]
        for round_info in artifact.search_log:
            lines.append(
                "Round "
                f"{round_info.round_number}: raw={round_info.papers_found_raw}, "
                f"in_scope={round_info.papers_in_scope}, "
                f"sources={', '.join(round_info.sources_queried) or 'none'}, "
                f"keywords={round_info.keyword_sets}"
            )
        if artifact.removed_papers:
            lines.append("First removed candidates:")
            for paper in artifact.removed_papers[:5]:
                lines.append(
                    f"- {paper.title} ({paper.year or '?'}) [{paper.source}]: "
                    f"{paper.scope_violation_reason or 'out of scope'}"
                )
        return "\n".join(lines)

    @staticmethod
    def _coerce_lookup_query(response: str, *, fallback: str) -> str:
        text = str(response or "").strip()
        if not text:
            return fallback
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                value = payload.get("query") or payload.get("lookup_query") or payload.get("search_query")
                if isinstance(value, list):
                    value = ", ".join(str(item).strip() for item in value if str(item).strip())
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                else:
                    return fallback

        text = re.sub(r"^(?:query|lookup query|search query)\s*:\s*", "", text, flags=re.IGNORECASE)
        parts = [
            re.sub(r"\s+", " ", part.strip(" -•\t"))
            for part in re.split(r"[,\n;]+", text)
            if part.strip(" -•\t")
        ]
        if not parts:
            return fallback
        if len(parts) == 1 and len(parts[0].split()) > 8:
            return fallback
        return ", ".join(parts[:8])

    def _lookup_spec_from_task(self, task: TaskEnvelope) -> dict[str, Any]:
        raw = task.metadata.get("article_lookup_spec")
        if isinstance(raw, dict):
            return self._normalize_lookup_spec(raw, fallback=self._fallback_lookup_spec(task))
        query = str(task.metadata.get("article_lookup_query") or "").strip()
        if query:
            spec = self._fallback_lookup_spec(task)
            spec["query_string"] = query
            spec["query_terms"] = [part.strip() for part in query.split(",") if part.strip()]
            return spec
        return self._fallback_lookup_spec(task)

    def _lookup_query_candidates(self, lookup_spec: dict[str, Any], task: TaskEnvelope) -> list[str]:
        title_phrases = [str(item).strip() for item in lookup_spec.get("title_phrases", []) if str(item).strip()]
        preferred_year = lookup_spec.get("preferred_year")
        year_token = str(preferred_year).strip() if preferred_year is not None else ""
        focused_title_query = ", ".join([*title_phrases[:1], year_token]).strip(", ")
        fallback_query = build_article_lookup_query("", task.instructions)
        return article_query_candidates(
            str(lookup_spec.get("query_string") or ""),
            focused_title_query,
            str(task.metadata.get("topic_hint") or ""),
            fallback_query,
        )

    def _fallback_lookup_spec(self, task: TaskEnvelope) -> dict[str, Any]:
        query_string = build_article_lookup_query(
            str(task.metadata.get("topic_hint") or ""),
            task.instructions,
        )
        combined_text = f"{task.instructions}\n{self._steering_text(task)}"
        required_authors = self._extract_author_surnames(combined_text)
        preferred_year = self._extract_preferred_year(combined_text)
        spec = self._normalize_lookup_spec(
            {
                "query_string": query_string,
                "query_terms": [part.strip() for part in query_string.split(",") if part.strip()],
                "required_authors": required_authors,
                "preferred_year": preferred_year,
                "year_min": None,
                "year_max": None,
                "title_phrases": [],
                "needs_clarification": False,
                "clarification_question": "",
            },
            fallback={
                "query_string": query_string,
                "query_terms": [part.strip() for part in query_string.split(",") if part.strip()],
                "required_authors": required_authors,
                "preferred_year": preferred_year,
                "year_min": None,
                "year_max": None,
                "title_phrases": [],
                "needs_clarification": False,
                "clarification_question": "",
            },
        )
        return self._with_lookup_clarification_if_needed(spec, task)

    @staticmethod
    def _normalize_lookup_spec(spec: dict[str, Any], *, fallback: dict[str, Any]) -> dict[str, Any]:
        query_terms = [
            str(item).strip()
            for item in (spec.get("query_terms") or [])
            if str(item).strip()
        ]
        query_string = str(spec.get("query_string") or "").strip()
        if not query_terms and query_string:
            query_terms = [part.strip() for part in query_string.split(",") if part.strip()]

        required_authors = [
            str(item).strip()
            for item in (spec.get("required_authors") or [])
            if str(item).strip()
        ]
        if not required_authors:
            required_authors = list(fallback.get("required_authors") or [])

        preferred_year = spec.get("preferred_year")
        if preferred_year is None:
            preferred_year = fallback.get("preferred_year")
        try:
            preferred_year = int(preferred_year) if preferred_year is not None else None
        except (TypeError, ValueError):
            preferred_year = fallback.get("preferred_year")

        year_min = spec.get("year_min")
        if year_min is None:
            year_min = fallback.get("year_min")
        try:
            year_min = int(year_min) if year_min is not None else None
        except (TypeError, ValueError):
            year_min = fallback.get("year_min")

        year_max = spec.get("year_max")
        if year_max is None:
            year_max = fallback.get("year_max")
        try:
            year_max = int(year_max) if year_max is not None else None
        except (TypeError, ValueError):
            year_max = fallback.get("year_max")

        if preferred_year is not None:
            year_min = preferred_year if year_min is None else year_min
            year_max = preferred_year if year_max is None else year_max

        title_phrases = [
            str(item).strip()
            for item in (spec.get("title_phrases") or [])
            if str(item).strip()
        ]

        needs_clarification = bool(spec.get("needs_clarification"))
        clarification_question = str(spec.get("clarification_question") or "").strip()

        author_tokens = {item.lower() for item in required_authors}
        filtered_terms = [
            term for term in query_terms
            if term.lower() not in author_tokens
            and (preferred_year is None or term != str(preferred_year))
            and (year_min is None or term != str(year_min))
            and (year_max is None or term != str(year_max))
        ]
        query_terms = filtered_terms
        if not query_terms and title_phrases:
            query_terms = list(title_phrases)
        if not query_terms:
            query_terms = [
                str(item).strip()
                for item in (fallback.get("query_terms") or [])
                if str(item).strip() and str(item).strip().lower() not in author_tokens
            ]
        if not query_terms and query_string:
            query_terms = [part.strip() for part in query_string.split(",") if part.strip()]

        query_string = ", ".join(query_terms[:8])

        return {
            "query_string": query_string,
            "query_terms": query_terms[:8],
            "required_authors": required_authors[:4],
            "preferred_year": preferred_year,
            "year_min": year_min,
            "year_max": year_max,
            "title_phrases": title_phrases[:3],
            "needs_clarification": needs_clarification and bool(clarification_question),
            "clarification_question": clarification_question,
        }

    def _coerce_lookup_spec(self, response: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
        text = str(response or "").strip()
        if not text:
            return fallback
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            return fallback
        if not payload.get("query_string"):
            query_terms = [
                str(item).strip()
                for item in (payload.get("query_terms") or [])
                if str(item).strip()
            ]
            if query_terms:
                payload["query_string"] = ", ".join(query_terms[:8])
            else:
                payload["query_string"] = self._coerce_lookup_query(text, fallback=fallback["query_string"])
        return self._normalize_lookup_spec(payload, fallback=fallback)

    @staticmethod
    def _with_lookup_clarification_if_needed(spec: dict[str, Any], task: TaskEnvelope) -> dict[str, Any]:
        if spec.get("needs_clarification"):
            return spec
        query_terms = [str(item).strip() for item in spec.get("query_terms", []) if str(item).strip()]
        has_author = bool(spec.get("required_authors"))
        has_time = spec.get("preferred_year") is not None or spec.get("year_min") is not None or spec.get("year_max") is not None
        has_title = bool(spec.get("title_phrases"))
        has_multiword_topic = any(len(term.split()) >= 2 for term in query_terms)
        has_topic_hint = bool(str(task.metadata.get("topic_hint") or "").strip())
        if has_author or has_time or has_title or has_multiword_topic or has_topic_hint:
            return spec

        amended = dict(spec)
        amended["needs_clarification"] = True
        amended["clarification_question"] = (
            "I do not yet have enough specific detail to search confidently. "
            "Please provide any known title words, authors, year or year range, venue, or distinctive model details."
        )
        return amended

    @staticmethod
    def _extract_author_surnames(text: str) -> list[str]:
        values: list[str] = []
        for match in re.finditer(r"\b([A-Z][A-Za-z'`.-]{2,})\s+(?:and|&)\s+([A-Z][A-Za-z'`.-]{2,})\b", text):
            values.extend([match.group(1), match.group(2)])
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    @staticmethod
    def _extract_preferred_year(text: str) -> int | None:
        matches = re.findall(r"\b(19\d{2}|20\d{2})\b", str(text or ""))
        if not matches:
            return None
        try:
            return int(matches[0])
        except ValueError:
            return None

    def _fallback_model_description(self, article: ArtifactRef, text: str) -> str:
        abstract = str(article.metadata.get("abstract") or "").strip()
        title = str(article.metadata.get("title") or "").strip()
        source = abstract or text[:1200].strip() or title
        if not source:
            return "Reproduction target from the selected article."
        return source

    @staticmethod
    def _normalize_answer(answer: str, *, fallback: str) -> str:
        normalized = (answer or "").strip()
        if not normalized or normalized.upper() == "NOT_FOUND":
            return fallback.strip()
        return normalized

    @staticmethod
    def _coerce_parameters(raw: Any, *, text: str) -> dict[str, str]:
        if isinstance(raw, dict):
            return {str(key): str(value) for key, value in raw.items()}
        candidate = (raw or "").strip()
        if candidate and candidate.upper() != "NOT_FOUND":
            try:
                if candidate.startswith("{"):
                    data = json.loads(candidate)
                    return {str(key): str(value) for key, value in data.items()}
            except json.JSONDecodeError:
                pass
            return {"parameters": candidate}

        found: dict[str, str] = {}
        for key, value in re.findall(r"\b([A-Za-z][A-Za-z0-9_\-]{0,20})\s*=\s*([0-9.eE+\-]+)", text):
            if key.lower() in {"http", "https"}:
                continue
            found.setdefault(key, value)
        if found:
            return found
        return {"parameters": "Use the parameter values reported in the article text."}

    @staticmethod
    def _coerce_figures(raw: str, *, fallback: list[str]) -> list[str]:
        text = (raw or "").strip()
        if not text or text.upper() == "NOT_FOUND":
            return fallback
        if text.startswith("["):
            try:
                data = json.loads(text)
                return [str(item) for item in data if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip(" -") for item in re.split(r"[\n;]+", text) if item.strip()]

    @staticmethod
    def _coerce_inquiries(raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            inquiries: list[dict[str, Any]] = []
            for index, item in enumerate(raw, 1):
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question") or item.get("prompt") or "").strip()
                if not question:
                    continue
                options = LocalLiteratureAssistant._coerce_inquiry_options(item.get("options"))
                inquiries.append({
                    "id": str(item.get("id") or item.get("inquiry_id") or f"inquiry_{index}").strip(),
                    "kind": str(item.get("kind") or ("selection" if options else "clarification")).strip(),
                    "question": question,
                    "reason": str(item.get("reason") or "").strip(),
                    "applies_to": [
                        str(value).strip()
                        for value in (item.get("applies_to") if isinstance(item.get("applies_to"), list) else [])
                        if str(value).strip()
                    ],
                    "blocking": bool(item.get("blocking", True)),
                    "options": options,
                    "expected_schema": item.get("expected_schema") if isinstance(item.get("expected_schema"), dict) else {},
                    "default_policy": str(item.get("default_policy") or "ask").strip(),
                })
            return inquiries[:12]
        text = str(raw or "").strip()
        if not text or text.upper() == "NOT_FOUND":
            return []
        if text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, list):
                return LocalLiteratureAssistant._coerce_inquiries(data)
        return []

    @staticmethod
    def _coerce_inquiry_options(raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        options: list[dict[str, Any]] = []
        for index, item in enumerate(raw, 1):
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
        return options[:20]
