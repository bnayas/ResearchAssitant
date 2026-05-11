"""
Tests for LiteratureAgent tool descriptors, validation, and dispatch.

These tests verify the agent's *structure* — tool registration, requirement
validation, and dispatch routing — without hitting real LLM or search backends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_platform.agents.base import ToolContext, ToolResult
from research_platform.agents.literature import LiteratureAgent, AGENT_ID
from research_platform.contracts import ArtifactRef
from research_platform.registry import ArtifactRegistry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(root_dir=tmp_path)


@pytest.fixture
def agent(registry: ArtifactRegistry) -> LiteratureAgent:
    return LiteratureAgent(registry=registry, llm_backend=MagicMock())


def _make_article_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="article-match",
        assistant="literature_reviewer",
        kind="primary_article",
        title="Test Article",
        summary="A test article",
        content="Full text content",
        metadata={
            "abstract": "An abstract about physics",
            "title": "Test Article",
            "year": 2024,
            "authors": ["Smith", "Jones"],
            "arxiv_id": "2401.00001",
            "url": "https://arxiv.org/abs/2401.00001",
        },
    )


def _make_brief_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="article-brief",
        assistant="literature_reviewer",
        kind="article_brief",
        title="Article Brief",
        summary="A test brief",
        content="Brief content",
        metadata={
            "model_description": "Ising model",
            "key_parameters": {"T": "2.27", "L": "32"},
            "procedure": "Monte Carlo simulation",
        },
    )


def _make_review_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="literature-review",
        assistant="literature_reviewer",
        kind="literature_review",
        title="Literature Review",
        summary="5 accepted papers",
        content="Synthesis of related work...",
        metadata={"accepted_count": 5, "synthesis": "Related work synthesis"},
    )


# =========================================================================
# Agent identity & tool registration
# =========================================================================

class TestLiteratureAgentStructure:
    def test_agent_id(self, agent: LiteratureAgent):
        assert agent.agent_id() == AGENT_ID

    def test_exposes_literature_tools(self, agent: LiteratureAgent):
        tools = agent.tools()
        assert len(tools) == 7

    def test_tool_names(self, agent: LiteratureAgent):
        names = {t.name for t in agent.tools()}
        assert names == {
            "prepare_article_lookup",
            "find_primary_article",
            "parse_article_with_intention",
            "build_article_brief",
            "review_related_literature",
            "answer_from_article",
            "answer_from_review",
        }

    def test_all_tools_have_agent_id(self, agent: LiteratureAgent):
        for tool in agent.tools():
            assert tool.agent_id == AGENT_ID

    def test_all_tools_have_descriptions(self, agent: LiteratureAgent):
        for tool in agent.tools():
            assert len(tool.description) > 20, f"{tool.name} has a short description"

    def test_all_tools_have_display_names(self, agent: LiteratureAgent):
        for tool in agent.tools():
            assert tool.display_name, f"{tool.name} missing display_name"

    def test_tool_by_name(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("find_primary_article")
        assert tool is not None
        assert tool.name == "find_primary_article"

    def test_tool_by_name_missing(self, agent: LiteratureAgent):
        assert agent.tool_by_name("nonexistent") is None

    def test_unknown_tool_raises(self, agent: LiteratureAgent):
        ctx = ToolContext(
            tool_name="nonexistent",
            directive_id="d1",
            instruction="test",
        )
        with pytest.raises(KeyError):
            agent.invoke("nonexistent", ctx)

    def test_tool_catalog_json_valid(self, agent: LiteratureAgent):
        catalog = json.loads(agent.tool_catalog_json())
        assert len(catalog) == 7
        for entry in catalog:
            assert "name" in entry
            assert "requirements" in entry
            assert "produces" in entry


# =========================================================================
# Requirement validation tests
# =========================================================================

class TestPrepareArticleLookupValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("prepare_article_lookup")
        results = tool.validate_inputs(
            inputs={"instruction": "Find the Ising model paper by Onsager"},
            artifacts={},
        )
        assert all(r.ok for r in results)

    def test_empty_instruction_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("prepare_article_lookup")
        results = tool.validate_inputs(
            inputs={"instruction": ""},
            artifacts={},
        )
        assert not all(r.ok for r in results)

    def test_missing_instruction_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("prepare_article_lookup")
        results = tool.validate_inputs(inputs={}, artifacts={})
        assert not all(r.ok for r in results)


class TestFindPrimaryArticleValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("find_primary_article")
        results = tool.validate_inputs(
            inputs={
                "lookup_spec": {"query_string": "Ising model"},
            },
            artifacts={},
        )
        assert all(r.ok for r in results)

    def test_lookup_spec_without_query_string_accepted(self, agent: LiteratureAgent):
        """Phase 4: lookup_spec is fully optional — tool falls back to instruction as query."""
        tool = agent.tool_by_name("find_primary_article")
        results = tool.validate_inputs(
            inputs={
                "instruction": "Find the paper",
                "lookup_spec": {"authors": ["Smith"]},
            },
            artifacts={},
        )
        assert all(r.ok for r in results)

    def test_direct_query_or_lookup_spec_is_optional_for_planner_wiring(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("find_primary_article")
        results = tool.validate_inputs(inputs={}, artifacts={})
        assert all(r.ok for r in results)


class TestBuildArticleBriefValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("build_article_brief")
        results = tool.validate_inputs(
            inputs={},
            artifacts={"article": _make_article_artifact()},
        )
        assert all(r.ok for r in results)

    def test_article_without_abstract_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("build_article_brief")
        art = _make_article_artifact()
        art.metadata = {}  # no abstract
        results = tool.validate_inputs(inputs={}, artifacts={"article": art})
        assert not all(r.ok for r in results)


class TestParseArticleWithIntentionValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("parse_article_with_intention")
        results = tool.validate_inputs(
            inputs={"intention": "Summarize the simulation models and figures."},
            artifacts={"article": _make_article_artifact()},
        )
        assert all(r.ok for r in results)

    def test_missing_intention_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("parse_article_with_intention")
        results = tool.validate_inputs(
            inputs={},
            artifacts={"article": _make_article_artifact()},
        )
        assert not all(r.ok for r in results)

    def test_missing_article_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("build_article_brief")
        results = tool.validate_inputs(inputs={}, artifacts={})
        assert not all(r.ok for r in results)


class TestBuildArticleBriefExecution:
    def test_brief_exposes_generic_inquiries(self, agent: LiteratureAgent):
        article = _make_article_artifact()
        article.content = "Full article text. " * 30
        agent._llm_backend.complete.return_value = json.dumps({
            "model_description": "The article compares Model A and Model B.",
            "inquiries": [
                {
                    "id": "model_scope",
                    "kind": "selection",
                    "question": "Which model should be reproduced first?",
                    "reason": "The article contains two model variants.",
                    "applies_to": ["simulation"],
                    "blocking": True,
                    "options": [
                        {"label": "Model A", "description": "Local competition reproduction", "value": {"model": "A"}},
                        {"label": "Model B", "description": "Global competition reproduction", "value": {"model": "B"}},
                    ],
                },
            ],
            "key_parameters": {"N": "population size"},
            "procedure": "Iterate stochastic births and deaths.",
            "expected_figures": ["Model A abundance distribution"],
            "assumptions": [],
            "validation_criteria": [],
        })
        ctx = ToolContext(
            tool_name="build_article_brief",
            directive_id="d1",
            instruction="Reproduce both model variants.",
            artifacts={"article": article},
        )

        result = agent.invoke("build_article_brief", ctx)

        assert result.ok
        inquiries = result.artifacts[0].metadata["inquiries"]
        assert inquiries[0]["id"] == "model_scope"
        assert inquiries[0]["kind"] == "selection"
        assert inquiries[0]["options"][1]["label"] == "Model B"
        assert result.inquiries[0].request_kind == "inquiry"
        assert result.inquiries[0].metadata["inquiry"]["id"] == "model_scope"

    def test_parse_with_intention_passes_intention_to_extractor(self, agent: LiteratureAgent):
        article = _make_article_artifact()
        article.content = "Full article text. " * 30
        agent._llm_backend.complete.return_value = json.dumps({
            "model_description": "The article studies a stochastic neutral model with two variants.",
            "inquiries": [],
            "key_parameters": {"N": "population size"},
            "procedure": "1. Initialize. 2. Iterate births and deaths. 3. Measure abundance.",
            "expected_figures": ["Simulation figure"],
            "assumptions": [],
            "validation_criteria": [],
        })
        ctx = ToolContext(
            tool_name="parse_article_with_intention",
            directive_id="d1",
            instruction="Read the article.",
            inputs={"intention": "Summarize the models and simulation details."},
            artifacts={"article": article},
        )

        result = agent.invoke("parse_article_with_intention", ctx)

        assert result.ok
        first_call = agent._llm_backend.complete.call_args_list[0]
        assert "Downstream parsing intention" in first_call.kwargs["messages"][0]["content"]
        assert "Summarize the models and simulation details" in first_call.kwargs["messages"][0]["content"]

    def test_parse_falls_back_when_llm_json_fails(self, agent: LiteratureAgent):
        article = _make_article_artifact()
        article.metadata["abstract"] = (
            "Here we consider two generic time-averaged neutral models. "
            "The first (model A) describes local competition, while model B "
            "uses global competition. Analytic expressions fit extensive "
            "Monte-Carlo simulations."
        )
        article.content = article.metadata["abstract"]
        agent._llm_backend.complete.return_value = "not json"
        ctx = ToolContext(
            tool_name="parse_article_with_intention",
            directive_id="d1",
            instruction="Summarize simulations.",
            inputs={"intention": "Summarize the models and simulation details."},
            artifacts={"article": article},
        )

        result = agent.invoke("parse_article_with_intention", ctx)

        assert result.ok
        assert result.artifacts[0].metadata["extraction_fallback"] is True
        assert "Model A" in "\n".join(result.artifacts[0].metadata["expected_figures"])


class TestStructuredLookup:
    def test_lookup_splits_article_target_from_task_intent(self, agent: LiteratureAgent):
        agent._llm_backend.complete.return_value = json.dumps({
            "research_intent": "Find the Danino and Shnerb TNTB paper.",
            "article_target": "Danino Shnerb Time Averaged Neutral Model TNTB",
            "task_intent": "Summarize the models and simulation details.",
            "article_type": "primary_source",
            "required_authors": ["Danino", "Shnerb"],
            "year_constraints": {"preferred_year": None, "year_from": None, "year_to": None},
            "title_phrases": ["Time Averaged Neutral Model"],
            "topic_terms": ["TNTB", "time averaged neutral dynamics"],
            "excluded_search_terms": ["read", "summarize", "models and simulation details"],
            "scope_criteria": ["Paper is by Danino and Shnerb", "Paper concerns time averaged neutral dynamics"],
            "what_to_look_for": "Simulation details after finding the paper.",
            "queries": [{
                "label": "title-author",
                "rationale": "Specific title and author match.",
                "title_terms": ["Time Averaged Neutral Model"],
                "abstract_keywords": ["TNTB"],
                "authors": ["Danino", "Shnerb"],
                "categories": [],
                "year_from": None,
                "year_to": None,
                "sort_by": "relevance",
                "max_results": 5,
                "query_string": "bad LLM query with summarize simulation details",
            }],
            "known_ids_to_exclude": [],
            "fallback_query": "Danino Shnerb time averaged neutral",
        })
        ctx = ToolContext(
            tool_name="prepare_article_lookup",
            directive_id="d1",
            instruction="- Read Danino and Shnerb article about Time Averaged Neutral Model (TNTB) - Summarize the models and simulations details",
            inputs={"instruction": "- Read Danino and Shnerb article about Time Averaged Neutral Model (TNTB) - Summarize the models and simulations details"},
        )

        result = agent.invoke("prepare_article_lookup", ctx)

        assert result.ok
        spec = result.metadata["lookup_spec"]
        assert spec["required_authors"] == ["Danino", "Shnerb"]
        assert "Summarize" in spec["task_intent"]
        query_strings = " ".join(q["query_string"] for q in spec["queries"])
        assert "summarize" not in query_strings.lower()
        assert "simulation details" not in query_strings.lower()
        assert 'au:"Danino"' in query_strings
        assert 'au:"Shnerb"' in query_strings
        assert "max_tokens" not in agent._llm_backend.complete.call_args.kwargs

    def test_find_article_iterates_query_plan_after_zero_results(self, registry: ArtifactRegistry):
        from research_platform.agents.literature.find_article import execute

        llm = MagicMock()
        llm.complete.return_value = json.dumps({
            "research_intent": "Find the target paper.",
            "article_target": "time averaged neutral dynamics",
            "task_intent": "Summarize simulations.",
            "article_type": "primary_source",
            "required_authors": [],
            "year_constraints": {"preferred_year": None, "year_from": None, "year_to": None},
            "title_phrases": ["time averaged neutral dynamics"],
            "topic_terms": ["environmental stochasticity"],
            "excluded_search_terms": ["summarize", "simulations"],
            "scope_criteria": ["By Danino and Shnerb"],
            "what_to_look_for": "",
            "queries": [{
                "label": "revised",
                "rationale": "Broader title variant.",
                "title_terms": ["time averaged neutral dynamics"],
                "abstract_keywords": [],
                "authors": [],
                "categories": [],
                "year_from": None,
                "year_to": None,
                "sort_by": "relevance",
                "max_results": 5,
                "query_string": "",
            }],
            "known_ids_to_exclude": [],
            "fallback_query": "time averaged neutral dynamics",
        })
        target = {
            "title": "Theory of time-averaged neutral dynamics with environmental stochasticity",
            "authors": ["M. Danino", "N. M. Shnerb"],
            "abstract": "Neutral dynamics with environmental stochasticity.",
            "year": 2017,
            "arxiv_id": "1701.00001",
            "doi": None,
            "url": "https://arxiv.org/abs/1701.00001",
            "categories": ["q-bio.PE"],
            "source": "arxiv",
        }
        lookup_spec = {
            "article_target": "Time Averaged Neutral Model",
            "task_intent": "Summarize simulations.",
            "required_authors": ["Danino", "Shnerb"],
            "year_constraints": {"preferred_year": None, "year_from": None, "year_to": None},
            "queries": [{
                "label": "too narrow",
                "query_string": 'ti:"Time Averaged Neutral Model" AND au:"Danino" AND au:"Shnerb"',
                "max_results": 5,
            }],
            "fallback_query": "",
            "known_ids_to_exclude": [],
        }
        ctx = ToolContext(
            tool_name="find_primary_article",
            directive_id="d1",
            instruction="Find Danino and Shnerb paper.",
            inputs={"lookup_spec": lookup_spec, "max_query_iterations": 2},
        )

        with patch("research_platform.agents.literature.find_article._arxiv_search", side_effect=[[], [target], [target]]) as search:
            result = execute(ctx, registry=registry, llm_backend=llm)

        assert result.ok
        assert search.call_count >= 2
        assert result.artifacts[0].metadata["authors"] == ["M. Danino", "N. M. Shnerb"]


class TestReviewRelatedLiteratureValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("review_related_literature")
        results = tool.validate_inputs(
            inputs={},
            artifacts={
                "article": _make_article_artifact(),
                "brief": _make_brief_artifact(),
            },
        )
        assert all(r.ok for r in results)

    def test_wrong_article_kind_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("review_related_literature")
        bad_art = _make_article_artifact()
        bad_art = ArtifactRef(
            artifact_id="x", assistant="x", kind="wrong_kind",
            title="x", metadata={},
        )
        results = tool.validate_inputs(
            inputs={},
            artifacts={"article": bad_art, "brief": _make_brief_artifact()},
        )
        assert not all(r.ok for r in results)


class TestAnswerFromArticleValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("answer_from_article")
        results = tool.validate_inputs(
            inputs={"question": "What is the critical temperature?"},
            artifacts={"article": _make_article_artifact()},
        )
        assert all(r.ok for r in results)

    def test_empty_question_fails(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("answer_from_article")
        results = tool.validate_inputs(
            inputs={"question": ""},
            artifacts={"article": _make_article_artifact()},
        )
        assert not all(r.ok for r in results)


class TestAnswerFromReviewValidation:
    def test_valid_inputs(self, agent: LiteratureAgent):
        tool = agent.tool_by_name("answer_from_review")
        results = tool.validate_inputs(
            inputs={"question": "What methods are used in the field?"},
            artifacts={"review": _make_review_artifact()},
        )
        assert all(r.ok for r in results)
