"""
Tests for CodingAgent, WriterAgent, MathAgent, and RuntimeAgent
tool descriptors, validation, and dispatch routing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from research_platform.agents.base import ToolContext, ToolResult
from research_platform.agents.coding import CodingAgent
from research_platform.agents.writer import WriterAgent
from research_platform.agents.math import MathAgent
from research_platform.agents.runtime import RuntimeAgent
from research_platform.contracts import ArtifactRef
from research_platform.registry import ArtifactRegistry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(root_dir=tmp_path)


def _art(kind: str = "article_brief", **kw: Any) -> ArtifactRef:
    defaults = dict(artifact_id=f"test-{kind}", assistant="test", kind=kind,
                    title="Test", summary="test", metadata={"abstract": "abs"})
    defaults.update(kw)
    return ArtifactRef(**defaults)


# =========================================================================
# CodingAgent
# =========================================================================

class TestCodingAgentStructure:
    def test_agent_id(self, registry):
        a = CodingAgent(registry)
        assert a.agent_id() == "coding"

    def test_six_tools(self, registry):
        assert len(CodingAgent(registry).tools()) == 6

    def test_tool_names(self, registry):
        names = {t.name for t in CodingAgent(registry).tools()}
        assert names == {
            "plan_simulation_functions", "generate_function",
            "design_simulation_spec", "run_sample",
            "analyze_results", "run_full_sweep",
        }

    def test_unknown_tool(self, registry):
        with pytest.raises(KeyError):
            CodingAgent(registry).invoke("nope", ToolContext(
                tool_name="nope", directive_id="d", instruction="x"))

    def test_catalog_json(self, registry):
        cat = json.loads(CodingAgent(registry).tool_catalog_json())
        assert len(cat) == 6


class TestCodingValidation:
    def test_plan_functions_valid(self, registry):
        t = CodingAgent(registry).tool_by_name("plan_simulation_functions")
        r = t.validate_inputs(
            inputs={"instruction": "Simulate Ising"},
            artifacts={"brief": _art("article_brief")})
        assert all(x.ok for x in r)

    def test_generate_function_valid(self, registry):
        t = CodingAgent(registry).tool_by_name("generate_function")
        r = t.validate_inputs(
            inputs={"function_spec": {"name": "step", "signature": "def step(s):"}},
            artifacts={})
        assert all(x.ok for x in r)

    def test_generate_function_missing_name(self, registry):
        t = CodingAgent(registry).tool_by_name("generate_function")
        r = t.validate_inputs(inputs={"function_spec": {"signature": "def f():"}}, artifacts={})
        assert not all(x.ok for x in r)

    def test_run_sample_needs_spec(self, registry):
        t = CodingAgent(registry).tool_by_name("run_sample")
        r = t.validate_inputs(inputs={}, artifacts={})
        assert not all(x.ok for x in r)

    def test_run_sample_valid(self, registry):
        t = CodingAgent(registry).tool_by_name("run_sample")
        r = t.validate_inputs(inputs={}, artifacts={"spec": _art("simulation_spec")})
        assert all(x.ok for x in r)


# =========================================================================
# WriterAgent
# =========================================================================

class TestWriterAgentStructure:
    def test_agent_id(self, registry):
        assert WriterAgent(registry).agent_id() == "writer"

    def test_three_tools(self, registry):
        assert len(WriterAgent(registry).tools()) == 3

    def test_tool_names(self, registry):
        names = {t.name for t in WriterAgent(registry).tools()}
        assert names == {"plan_review_structure", "draft_section", "assemble_review"}

    def test_unknown_tool(self, registry):
        with pytest.raises(KeyError):
            WriterAgent(registry).invoke("nope", ToolContext(
                tool_name="nope", directive_id="d", instruction="x"))


class TestWriterValidation:
    def test_plan_valid(self, registry):
        t = WriterAgent(registry).tool_by_name("plan_review_structure")
        r = t.validate_inputs(
            inputs={"instruction": "Write review"},
            artifacts={"sources": [_art()]})
        # sources is list[ArtifactRef] — validated in artifacts dict
        assert all(x.ok for x in r)

    def test_draft_section_valid(self, registry):
        t = WriterAgent(registry).tool_by_name("draft_section")
        r = t.validate_inputs(
            inputs={"section_spec": {"title": "Introduction"}},
            artifacts={})
        assert all(x.ok for x in r)


# =========================================================================
# MathAgent
# =========================================================================

class TestMathAgentStructure:
    def test_agent_id(self):
        assert MathAgent().agent_id() == "math"

    def test_five_tools(self):
        assert len(MathAgent().tools()) == 5

    def test_tool_names(self):
        names = {t.name for t in MathAgent().tools()}
        assert names == {
            "evaluate_expression", "solve_equation", "solve_ode",
            "verify_numeric_claim", "verify_statistical_claim",
        }


class TestMathInvoke:
    def test_evaluate(self):
        ctx = ToolContext(tool_name="evaluate_expression", directive_id="d",
                          instruction="", inputs={"expression": "2+3"})
        r = MathAgent().invoke("evaluate_expression", ctx)
        assert r.ok
        assert r.metadata["value"] == 5.0

    def test_verify_numeric(self):
        ctx = ToolContext(tool_name="verify_numeric_claim", directive_id="d",
                          instruction="",
                          inputs={"expression": "2+3", "claimed_value": 5.0})
        r = MathAgent().invoke("verify_numeric_claim", ctx)
        assert r.ok
        assert r.metadata["status"] == "verified"


# =========================================================================
# RuntimeAgent
# =========================================================================

class TestRuntimeAgentStructure:
    def test_agent_id(self, registry):
        assert RuntimeAgent(registry).agent_id() == "runtime"

    def test_two_tools(self, registry):
        assert len(RuntimeAgent(registry).tools()) == 2

    def test_tool_names(self, registry):
        names = {t.name for t in RuntimeAgent(registry).tools()}
        assert names == {"materialize_runtime", "execute_run"}


class TestRuntimeValidation:
    def test_materialize_needs_spec(self, registry):
        t = RuntimeAgent(registry).tool_by_name("materialize_runtime")
        r = t.validate_inputs(inputs={}, artifacts={})
        assert not all(x.ok for x in r)

    def test_execute_run_needs_bundle(self, registry):
        t = RuntimeAgent(registry).tool_by_name("execute_run")
        r = t.validate_inputs(inputs={}, artifacts={})
        assert not all(x.ok for x in r)

    def test_execute_run_valid(self, registry):
        t = RuntimeAgent(registry).tool_by_name("execute_run")
        r = t.validate_inputs(
            inputs={"bundle": {"script_path": "/x"}, "spec_payload": {"name": "s"}},
            artifacts={})
        assert all(x.ok for x in r)
