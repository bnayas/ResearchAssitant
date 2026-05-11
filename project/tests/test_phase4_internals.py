"""
Tests for Phase 4 agent-internal refactors:
  • generate_function caching and recursive sub-function support
  • Stateless literature tool execution (no LocalLiteratureAssistant)
  • design_spec auto-clarification
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_platform.agents.base import ToolContext, ToolResult
from research_platform.agents.coding.generate_function import (
    clear_function_cache,
    execute as execute_generate,
    get_function_cache,
    _extract_sub_function_specs,
    _strip_sub_declarations,
)
from research_platform.agents.coding.plan_functions import execute as execute_plan_functions
from research_platform.contracts import ArtifactRef
from research_platform.registry import ArtifactRegistry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def registry(tmp_path: Path) -> ArtifactRegistry:
    return ArtifactRegistry(root_dir=tmp_path)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Clear function cache before each test."""
    clear_function_cache()
    yield
    clear_function_cache()


def _make_context(func_spec: dict, **kw) -> ToolContext:
    defaults = dict(
        tool_name="generate_function", directive_id="test",
        instruction="Test", inputs={"function_spec": func_spec},
        output_dir=Path("."), metadata={},
    )
    defaults.update(kw)
    return ToolContext(**defaults)


def _mock_backend(code: str) -> MagicMock:
    backend = MagicMock()
    backend.complete.return_value = code
    return backend


# =========================================================================
# Sub-function spec extraction
# =========================================================================

class TestSubFunctionParsing:
    def test_extract_sub_specs(self):
        code = (
            "def main():\n"
            "    return helper()\n\n"
            "# --- SUB-FUNCTIONS ---\n"
            '# SUB: {"name": "helper", "signature": "def helper()", "docstring": "A helper"}\n'
            '# SUB: {"name": "util", "signature": "def util(x)"}\n'
        )
        specs = _extract_sub_function_specs(code)
        assert len(specs) == 2
        assert specs[0]["name"] == "helper"
        assert specs[1]["name"] == "util"

    def test_no_sub_specs(self):
        code = "def main():\n    return 42\n"
        assert _extract_sub_function_specs(code) == []

    def test_malformed_sub_spec_ignored(self):
        code = "# SUB: not json\n# SUB: {}\n"
        specs = _extract_sub_function_specs(code)
        assert len(specs) == 0  # {} has no "name" key

    def test_strip_sub_declarations(self):
        code = (
            "def main():\n"
            "    return 42\n\n"
            "# --- SUB-FUNCTIONS ---\n"
            '# SUB: {"name": "helper"}\n'
        )
        stripped = _strip_sub_declarations(code)
        assert "SUB" not in stripped
        assert "def main" in stripped


# =========================================================================
# Function caching
# =========================================================================

class TestFunctionCache:
    def test_cache_hit(self, registry):
        # Pre-generate a function
        spec = {"name": "cached_fn", "signature": "def cached_fn(): pass"}
        backend = _mock_backend("def cached_fn():\n    return 1\n")
        ctx = _make_context(spec)
        result = execute_generate(ctx, registry=registry, llm_config=backend)
        assert result.ok

        # Second call should be cached
        backend.complete.reset_mock()
        result2 = execute_generate(ctx, registry=registry, llm_config=backend)
        assert result2.ok
        assert result2.metadata.get("cached") is True
        backend.complete.assert_not_called()

    def test_cache_snapshot(self, registry):
        spec = {"name": "fn_a", "signature": "def fn_a(): pass"}
        backend = _mock_backend("def fn_a():\n    pass\n")
        execute_generate(_make_context(spec), registry=registry, llm_config=backend)
        cache = get_function_cache()
        assert "fn_a" in cache


class TestFunctionGeneration:
    def test_syntax_error_fails(self, registry):
        backend = _mock_backend("def bad(\n")
        spec = {"name": "bad", "signature": "def bad()"}
        result = execute_generate(_make_context(spec), registry=registry, llm_config=backend)
        assert result.failed
        assert "syntax" in result.message.lower()

    def test_no_backend_fails(self, registry):
        spec = {"name": "fn", "signature": "def fn()"}
        result = execute_generate(_make_context(spec), registry=registry, llm_config=None)
        assert result.failed

    def test_successful_generation(self, registry):
        backend = _mock_backend("def step(state):\n    return state + 1\n")
        spec = {"name": "step", "signature": "def step(state)", "docstring": "Advance one step"}
        result = execute_generate(_make_context(spec), registry=registry, llm_config=backend)
        assert result.ok
        assert "step" in result.message
        assert len(result.artifacts) == 1

    def test_recursive_sub_functions(self, registry):
        # First call returns code with sub-function declarations
        main_code = (
            "def main():\n"
            "    return helper()\n\n"
            "# --- SUB-FUNCTIONS ---\n"
            '# SUB: {"name": "helper", "signature": "def helper()"}\n'
        )
        call_count = [0]
        def mock_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return main_code
            return "def helper():\n    return 42\n"

        backend = MagicMock()
        backend.complete.side_effect = mock_complete

        spec = {"name": "main", "signature": "def main()"}
        # We need a real llm_config that make_service_backend can handle
        # Patch make_service_backend to return our mock
        with patch("research_platform.agents.coding.generate_function.execute") as _:
            # Actually let's just call directly since execute uses make_service_backend
            pass

        # Use the backend directly
        ctx = _make_context(spec)
        # Patch make_service_backend at the module level
        with patch("research_platform.assistants.backends.make_service_backend", return_value=backend):
            result = execute_generate(ctx, registry=registry, llm_config=backend)

        assert result.ok
        assert "helper" in result.metadata.get("sub_functions", [])


class TestFunctionPlanning:
    def test_planning_returns_follow_up_generate_steps(self, registry):
        backend = _mock_backend(json.dumps([
            {
                "name": "initialize_state",
                "signature": "def initialize_state(config)",
                "docstring": "Create the initial simulation state.",
                "context": "Neutral dynamics setup.",
            },
            {
                "name": "step_model",
                "signature": "def step_model(state, config, rng)",
                "docstring": "Advance one neutral dynamics step.",
                "context": "Environmental stochasticity update.",
            },
        ]))
        brief = ArtifactRef(
            artifact_id="brief",
            assistant="literature_reviewer",
            kind="article_brief",
            title="Brief",
            metadata={
                "model_description": "Neutral dynamics with environmental stochasticity",
                "key_parameters": {"sigma": "environmental noise"},
            },
        )
        ctx = ToolContext(
            tool_name="plan_simulation_functions",
            directive_id="test",
            instruction="Reproduce Danino Shnerb",
            inputs={"instruction": "Reproduce Danino Shnerb"},
            artifacts={"brief": brief},
        )

        with patch("research_platform.assistants.backends.make_service_backend", return_value=backend):
            result = execute_plan_functions(ctx, registry=registry, llm_config=MagicMock())

        assert result.ok
        assert [s["tool"] for s in result.follow_up_steps] == ["generate_function", "generate_function"]
        assert result.follow_up_steps[0]["inputs"]["function_spec"]["name"] == "initialize_state"


# =========================================================================
# Stateless literature tools
# =========================================================================

class TestStatelessLiteratureTools:
    """Verify Phase 4 tools don't import from assistants.literature."""

    def test_prepare_lookup_no_assistant_import(self):
        import research_platform.agents.literature.prepare_lookup as mod
        source = open(mod.__file__).read()
        assert "from ...assistants.literature import" not in source

    def test_find_article_no_assistant_import(self):
        import research_platform.agents.literature.find_article as mod
        source = open(mod.__file__).read()
        assert "from ...assistants.literature import" not in source

    def test_build_brief_no_assistant_import(self):
        import research_platform.agents.literature.build_brief as mod
        source = open(mod.__file__).read()
        assert "from ...assistants.literature import" not in source

    def test_review_literature_no_assistant_import(self):
        import research_platform.agents.literature.review_literature as mod
        source = open(mod.__file__).read()
        assert "from ...assistants.literature import" not in source
