"""
Tests for the tool-first agent foundation types.

Covers:
  • ToolRequirement — type checking, validation, serialisation
  • ToolDescriptor — input validation, serialisation, catalog generation
  • ToolContext / ToolResult — construction and properties
  • BaseAgent — contract enforcement with a mock agent
  • Validators — all factories from validators.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from research_platform.agents.base import (
    BaseAgent,
    RequirementValidationResult,
    ToolContext,
    ToolDescriptor,
    ToolRequirement,
    ToolResult,
)
from research_platform.agents.validators import (
    all_of,
    any_of,
    artifact_has_content,
    artifact_has_metadata_key,
    artifact_kind_is,
    dict_has_keys,
    dict_non_empty,
    is_non_empty_string,
    is_positive_number,
    list_max_length,
    list_min_length,
    list_non_empty,
    number_in_range,
    string_matches_any,
    string_max_length,
)
from research_platform.contracts import ArtifactRef


# =========================================================================
# Fixtures
# =========================================================================

def _make_artifact(**overrides: Any) -> ArtifactRef:
    defaults = {
        "artifact_id": "test-art-001",
        "assistant": "test_agent",
        "kind": "primary_article",
        "title": "Test Article",
        "summary": "A test article",
        "content": "Some content",
        "metadata": {"abstract": "An abstract", "year": 2024},
    }
    defaults.update(overrides)
    return ArtifactRef(**defaults)


def _make_requirement(**overrides: Any) -> ToolRequirement:
    defaults = {
        "name": "instruction",
        "description": "The PI directive text",
        "type": "str",
    }
    defaults.update(overrides)
    return ToolRequirement(**defaults)


def _make_descriptor(**overrides: Any) -> ToolDescriptor:
    defaults = {
        "name": "test_tool",
        "display_name": "Test Tool",
        "description": "A tool for testing",
        "agent_id": "test_agent",
        "requirements": [],
        "produces": ["test_output"],
    }
    defaults.update(overrides)
    return ToolDescriptor(**defaults)


class MockAgent(BaseAgent):
    """Minimal concrete agent for testing the BaseAgent interface."""

    def __init__(self, tools_list: Optional[list[ToolDescriptor]] = None):
        self._tools = tools_list or []

    def agent_id(self) -> str:
        return "mock_agent"

    def tools(self) -> list[ToolDescriptor]:
        return list(self._tools)

    def invoke(self, tool_name: str, context: ToolContext) -> ToolResult:
        descriptor = self.tool_by_name(tool_name)
        if descriptor is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        return ToolResult(status="completed", message=f"Executed {tool_name}")


# =========================================================================
# ToolRequirement tests
# =========================================================================

class TestToolRequirement:
    def test_required_missing_fails(self):
        req = _make_requirement(required=True)
        result = req.validate(None)
        assert not result.ok
        assert "missing" in result.message.lower()

    def test_optional_missing_passes(self):
        req = _make_requirement(required=False, default="fallback")
        result = req.validate(None)
        assert result.ok

    def test_correct_type_passes(self):
        req = _make_requirement(type="str")
        result = req.validate("hello")
        assert result.ok

    def test_wrong_type_fails(self):
        req = _make_requirement(type="str")
        result = req.validate(42)
        assert not result.ok
        assert "wrong type" in result.message.lower()

    def test_int_type(self):
        req = _make_requirement(type="int")
        assert req.validate(42).ok
        assert not req.validate(True).ok
        assert not req.validate("42").ok

    def test_float_accepts_int(self):
        req = _make_requirement(type="float")
        assert req.validate(3.14).ok
        assert req.validate(42).ok  # int is a valid float
        assert not req.validate(False).ok
        assert not req.validate("3.14").ok

    def test_bool_type(self):
        req = _make_requirement(type="bool")
        assert req.validate(True).ok
        assert not req.validate(1).ok  # int is not bool in strict check
        # Actually bool is a subclass of int in Python, so this might pass
        # Let's just check that str doesn't pass
        assert not req.validate("true").ok

    def test_dict_type(self):
        req = _make_requirement(type="dict")
        assert req.validate({"key": "value"}).ok
        assert not req.validate([1, 2]).ok

    def test_list_type(self):
        req = _make_requirement(type="list")
        assert req.validate([1, 2, 3]).ok
        assert not req.validate("not a list").ok

    def test_list_str_type_checks_elements(self):
        req = _make_requirement(type="list[str]")
        assert req.validate(["a", "b"]).ok
        result = req.validate(["a", 42])
        assert not result.ok
        assert "element" in result.message.lower()

    def test_list_str_empty_passes(self):
        req = _make_requirement(type="list[str]")
        assert req.validate([]).ok

    def test_list_float_rejects_bool_elements(self):
        req = _make_requirement(type="list[float]")
        result = req.validate([1.0, True])
        assert not result.ok
        assert "bool" in result.message.lower()

    def test_artifact_ref_type(self):
        req = _make_requirement(type="ArtifactRef")
        art = _make_artifact()
        assert req.validate(art).ok
        assert not req.validate({"not": "an artifact"}).ok

    def test_list_artifact_ref_type(self):
        req = _make_requirement(type="list[ArtifactRef]")
        arts = [_make_artifact(), _make_artifact(artifact_id="art-002")]
        assert req.validate(arts).ok

    def test_validator_passes(self):
        req = _make_requirement(
            validator=is_non_empty_string,
            validator_description="Must be non-empty",
        )
        assert req.validate("hello").ok

    def test_validator_fails(self):
        req = _make_requirement(
            validator=is_non_empty_string,
            validator_description="Must be non-empty",
        )
        result = req.validate("   ")
        assert not result.ok
        assert "must be non-empty" in result.message.lower()

    def test_validator_exception_fails_gracefully(self):
        def _bad_validator(value: Any) -> bool:
            raise RuntimeError("boom")

        req = _make_requirement(
            validator=_bad_validator,
            validator_description="A buggy validator",
        )
        result = req.validate("hello")
        assert not result.ok
        assert "boom" in result.message

    def test_to_dict_roundtrip(self):
        req = _make_requirement(
            validator=is_non_empty_string,
            validator_description="Non-empty check",
        )
        d = req.to_dict()
        assert d["name"] == "instruction"
        assert d["type"] == "str"
        assert d["required"] is True
        assert d["validator_description"] == "Non-empty check"
        assert d["source"] == "auto"
        # validator callable is NOT in the dict (not serialisable)
        assert "validator" not in d

    def test_unknown_type_fails(self):
        req = _make_requirement(type="SomeCustomType")
        result = req.validate("anything")
        assert not result.ok
        assert "unsupported" in result.message.lower()


# =========================================================================
# ToolDescriptor tests
# =========================================================================

class TestToolDescriptor:
    def test_to_dict(self):
        desc = _make_descriptor(
            tags=["search", "literature"],
            idempotent=True,
            estimated_seconds=30.0,
        )
        d = desc.to_dict()
        assert d["name"] == "test_tool"
        assert d["agent_id"] == "test_agent"
        assert d["tags"] == ["search", "literature"]
        assert d["idempotent"] is True
        assert d["estimated_seconds"] == 30.0
        assert isinstance(d["requirements"], list)

    def test_to_json_is_valid(self):
        desc = _make_descriptor()
        parsed = json.loads(desc.to_json())
        assert parsed["name"] == "test_tool"

    def test_validate_inputs_all_pass(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(name="query", type="str"),
                _make_requirement(name="max_results", type="int", required=False, default=10),
            ]
        )
        results = desc.validate_inputs(
            inputs={"query": "physics", "max_results": 5},
            artifacts={},
        )
        assert all(r.ok for r in results)

    def test_validate_inputs_missing_required(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(name="query", type="str", required=True),
            ]
        )
        results = desc.validate_inputs(inputs={}, artifacts={})
        assert not all(r.ok for r in results)

    def test_validate_inputs_optional_uses_default(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(name="limit", type="int", required=False, default=10),
            ]
        )
        results = desc.validate_inputs(inputs={}, artifacts={})
        assert all(r.ok for r in results)

    def test_validate_inputs_artifact_requirement(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(
                    name="article",
                    type="ArtifactRef",
                    description="The primary article",
                    validator=artifact_kind_is("primary_article"),
                    validator_description="Must be a primary_article artifact",
                ),
            ]
        )
        art = _make_artifact(kind="primary_article")
        results = desc.validate_inputs(inputs={}, artifacts={"article": art})
        assert all(r.ok for r in results)

    def test_validate_inputs_artifact_wrong_kind(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(
                    name="article",
                    type="ArtifactRef",
                    validator=artifact_kind_is("primary_article"),
                    validator_description="Must be primary_article",
                ),
            ]
        )
        art = _make_artifact(kind="literature_review")
        results = desc.validate_inputs(inputs={}, artifacts={"article": art})
        assert not all(r.ok for r in results)

    def test_validate_inputs_explicit_metadata_source(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(name="step_index", type="int", source="metadata"),
            ]
        )
        results = desc.validate_inputs(inputs={}, artifacts={}, metadata={"step_index": 3})
        assert all(r.ok for r in results)

    def test_contract_errors_reject_bad_requirement_source(self):
        desc = _make_descriptor(
            requirements=[
                _make_requirement(name="x", type="str", source="elsewhere"),
            ]
        )
        errors = desc.contract_errors(expected_agent_id="test_agent")
        assert any("unsupported source" in err for err in errors)


# =========================================================================
# ToolContext / ToolResult tests
# =========================================================================

class TestToolContext:
    def test_defaults(self):
        ctx = ToolContext(tool_name="t", directive_id="d1", instruction="do it")
        assert ctx.inputs == {}
        assert ctx.artifacts == {}
        assert ctx.metadata == {}

    def test_with_values(self):
        art = _make_artifact()
        ctx = ToolContext(
            tool_name="find",
            directive_id="d1",
            instruction="find the paper",
            inputs={"query": "physics"},
            artifacts={"article": art},
            output_dir=Path("/tmp/test"),
            metadata={"key": "value"},
        )
        assert ctx.inputs["query"] == "physics"
        assert ctx.artifacts["article"].artifact_id == "test-art-001"


class TestToolResult:
    def test_completed(self):
        r = ToolResult(status="completed", message="done")
        assert r.ok
        assert not r.needs_input
        assert not r.failed

    def test_needs_input(self):
        r = ToolResult(status="needs_input", message="clarify")
        assert not r.ok
        assert r.needs_input
        assert not r.failed

    def test_failed(self):
        r = ToolResult(status="failed", message="error")
        assert not r.ok
        assert not r.needs_input
        assert r.failed

    def test_follow_up_steps_default(self):
        r = ToolResult(status="completed")
        assert r.follow_up_steps == []
        assert r.inquiries == []


# =========================================================================
# BaseAgent tests
# =========================================================================

class TestBaseAgent:
    def test_mock_agent_tools(self):
        tools = [_make_descriptor(name="tool_a"), _make_descriptor(name="tool_b")]
        agent = MockAgent(tools_list=tools)
        assert agent.agent_id() == "mock_agent"
        assert len(agent.tools()) == 2

    def test_tool_by_name_found(self):
        tools = [_make_descriptor(name="find"), _make_descriptor(name="parse")]
        agent = MockAgent(tools_list=tools)
        found = agent.tool_by_name("find")
        assert found is not None
        assert found.name == "find"

    def test_tool_by_name_not_found(self):
        agent = MockAgent(tools_list=[_make_descriptor(name="find")])
        assert agent.tool_by_name("nonexistent") is None

    def test_invoke_known_tool(self):
        agent = MockAgent(tools_list=[_make_descriptor(name="test")])
        ctx = ToolContext(tool_name="test", directive_id="d1", instruction="go")
        result = agent.invoke("test", ctx)
        assert result.ok
        assert "test" in result.message

    def test_invoke_unknown_tool_raises(self):
        agent = MockAgent(tools_list=[])
        ctx = ToolContext(tool_name="missing", directive_id="d1", instruction="go")
        with pytest.raises(KeyError):
            agent.invoke("missing", ctx)

    def test_tool_catalog_json(self):
        tools = [_make_descriptor(name="a"), _make_descriptor(name="b")]
        agent = MockAgent(tools_list=tools)
        catalog = json.loads(agent.tool_catalog_json())
        assert len(catalog) == 2
        assert catalog[0]["name"] == "a"
        assert catalog[1]["name"] == "b"

    def test_catalog_validation_rejects_duplicate_tools(self):
        tools = [_make_descriptor(name="same"), _make_descriptor(name="same")]
        agent = MockAgent(tools_list=tools)
        errors = agent.catalog_errors()
        assert any("duplicate tool" in err for err in errors)


# =========================================================================
# Validator tests
# =========================================================================

class TestStringValidators:
    def test_is_non_empty_string(self):
        assert is_non_empty_string("hello")
        assert not is_non_empty_string("")
        assert not is_non_empty_string("   ")
        assert not is_non_empty_string(42)

    def test_string_max_length(self):
        check = string_max_length(5)
        assert check("abc")
        assert check("12345")
        assert not check("123456")

    def test_string_matches_any(self):
        check = string_matches_any("ok", "cancel", "retry")
        assert check("ok")
        assert check("OK")
        assert check(" Cancel ")
        assert not check("delete")


class TestNumericValidators:
    def test_is_positive_number(self):
        assert is_positive_number(1)
        assert is_positive_number(0.5)
        assert not is_positive_number(0)
        assert not is_positive_number(-1)
        assert not is_positive_number("1")

    def test_number_in_range(self):
        check = number_in_range(0.0, 1.0)
        assert check(0.0)
        assert check(0.5)
        assert check(1.0)
        assert not check(-0.1)
        assert not check(1.1)


class TestArtifactValidators:
    def test_artifact_kind_is(self):
        check = artifact_kind_is("primary_article")
        assert check(_make_artifact(kind="primary_article"))
        assert not check(_make_artifact(kind="other"))
        assert not check("not an artifact")

    def test_artifact_has_metadata_key(self):
        check = artifact_has_metadata_key("abstract")
        assert check(_make_artifact(metadata={"abstract": "text"}))
        assert not check(_make_artifact(metadata={}))

    def test_artifact_has_content_with_content(self):
        assert artifact_has_content(_make_artifact(content="text"))

    def test_artifact_has_content_empty(self):
        assert not artifact_has_content(_make_artifact(content="", path=None))

    def test_artifact_has_content_not_artifact(self):
        assert not artifact_has_content("string")


class TestCollectionValidators:
    def test_list_non_empty(self):
        assert list_non_empty([1])
        assert not list_non_empty([])
        assert not list_non_empty("not a list")

    def test_list_min_length(self):
        check = list_min_length(3)
        assert check([1, 2, 3])
        assert not check([1, 2])

    def test_list_max_length(self):
        check = list_max_length(2)
        assert check([1, 2])
        assert not check([1, 2, 3])


class TestDictValidators:
    def test_dict_has_keys(self):
        check = dict_has_keys("a", "b")
        assert check({"a": 1, "b": 2, "c": 3})
        assert not check({"a": 1})

    def test_dict_non_empty(self):
        assert dict_non_empty({"a": 1})
        assert not dict_non_empty({})
        assert not dict_non_empty("not a dict")


class TestCombinators:
    def test_all_of(self):
        check = all_of(is_non_empty_string, string_max_length(10))
        assert check("hello")
        assert not check("")
        assert not check("a" * 11)

    def test_any_of(self):
        check = any_of(
            artifact_kind_is("article"),
            artifact_kind_is("brief"),
        )
        assert check(_make_artifact(kind="article"))
        assert check(_make_artifact(kind="brief"))
        assert not check(_make_artifact(kind="review"))
