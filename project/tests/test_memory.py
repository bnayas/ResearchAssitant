"""
test_memory.py
───────────────
Unit tests for memory.py — per-agent AGENTS.md read/write with LLM consolidation.

Covers:
  - AgentMemory init: file creation, header, zero entry count
  - append_entry: writes structured markdown entries
  - as_prompt_prefix: truncation, empty when no entries, correct header
  - consolidate_session: LLM path and no-backend fallback
  - clear: resets to header only
  - entry_count: correct after multiple appends
  - show / read_all: returns full file content
  - path property: correct location

Run:  python test_memory.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.memory import AgentMemory, _MAX_PROMPT_CHARS
except ImportError:
    from sim_tool.memory import AgentMemory, _MAX_PROMPT_CHARS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mem(tmp_path: Path, agent: str = "designer") -> AgentMemory:
    return AgentMemory(agent, memory_dir=tmp_path)


def _mock_backend(response: str = None) -> MagicMock:
    backend = MagicMock()
    if response is None:
        response = '{"what_failed": ["acceptance_rate missed lower bound"], "rules": ["Always assert both bounds for rate variables."]}'
    backend.complete.return_value = response
    backend.model_name = "mock-model"
    return backend


def _sample_facts(outcomes=None, flags=None, patch_changes=None) -> dict:
    return {
        "outcomes": outcomes or [{"status": "success", "reason": "converged"}],
        "flags": flags or [],
        "patch_changes": patch_changes or [],
        "assertion_errors": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:
    def test_creates_file_on_first_use(self, tmp_path):
        mem = _mem(tmp_path)
        assert mem.path.exists()

    def test_file_has_header(self, tmp_path):
        mem = _mem(tmp_path)
        content = mem.path.read_text()
        assert "AGENTS.md" in content

    def test_entry_count_zero_on_fresh_file(self, tmp_path):
        mem = _mem(tmp_path)
        assert mem.entry_count == 0

    def test_analyst_memory_separate_file(self, tmp_path):
        d = _mem(tmp_path, "designer")
        a = _mem(tmp_path, "analyst")
        assert d.path != a.path

    def test_designer_path_contains_designer(self, tmp_path):
        mem = _mem(tmp_path, "designer")
        assert "designer" in str(mem.path)

    def test_analyst_path_contains_analyst(self, tmp_path):
        mem = _mem(tmp_path, "analyst")
        assert "analyst" in str(mem.path)

    def test_loads_existing_file_without_overwriting(self, tmp_path):
        mem1 = _mem(tmp_path)
        mem1.append_entry("s1", "Ising", "Monte Carlo",
                          ["assertion missed lower bound"],
                          ["Always check both bounds."])
        mem2 = _mem(tmp_path)
        assert mem2.entry_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# append_entry
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendEntry:
    def test_entry_count_increments(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Ising", "Monte Carlo", ["bug"], ["rule"])
        assert mem.entry_count == 1

    def test_multiple_entries_accumulate(self, tmp_path):
        mem = _mem(tmp_path)
        for i in range(3):
            mem.append_entry(f"s{i}", f"Sim{i}", "MC", ["bug"], ["rule"])
        assert mem.entry_count == 3

    def test_session_id_in_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("sess-xyz", "Ising", "MC", [], ["some rule"])
        assert "sess-xyz" in mem.path.read_text()

    def test_sim_name_in_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "MySpecialSim", "MC", [], ["rule"])
        assert "MySpecialSim" in mem.path.read_text()

    def test_what_failed_in_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC",
                         ["acceptance_rate missing lower bound"], [])
        assert "acceptance_rate missing lower bound" in mem.path.read_text()

    def test_rules_in_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC",
                         [], ["Always assert both bounds for rate fields."])
        assert "Always assert both bounds" in mem.path.read_text()

    def test_empty_lists_not_written(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", [], [])
        # Nothing added — count stays 0
        assert mem.entry_count == 0

    def test_sim_type_in_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "ODE/PDE", ["bug"], ["rule"])
        assert "ODE/PDE" in mem.path.read_text()

    def test_multiple_rules_all_written(self, tmp_path):
        rules = ["Rule A.", "Rule B.", "Rule C."]
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", [], rules)
        content = mem.path.read_text()
        for r in rules:
            assert r in content

    def test_multiple_failures_all_written(self, tmp_path):
        failures = ["Bug one.", "Bug two."]
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", failures, ["rule"])
        content = mem.path.read_text()
        for f in failures:
            assert f in content

    def test_timestamp_in_entry(self, tmp_path):
        import re
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        content = mem.path.read_text()
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", content)


# ─────────────────────────────────────────────────────────────────────────────
# as_prompt_prefix
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptPrefix:
    def test_empty_when_no_entries(self, tmp_path):
        mem = _mem(tmp_path)
        assert mem.as_prompt_prefix() == ""

    def test_non_empty_after_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        prefix = mem.as_prompt_prefix()
        assert len(prefix) > 0

    def test_header_present_in_prefix(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        prefix = mem.as_prompt_prefix()
        assert "LESSONS LEARNED" in prefix

    def test_rule_text_in_prefix(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", [], ["Always assert both bounds."])
        prefix = mem.as_prompt_prefix()
        assert "Always assert both bounds." in prefix

    def test_prefix_truncated_at_max_chars(self, tmp_path):
        mem = _mem(tmp_path)
        # Add many entries to exceed limit
        for i in range(30):
            mem.append_entry(f"s{i}", f"Sim{i}", "MC",
                             [f"failure {i} " * 20],
                             [f"rule {i} " * 20])
        prefix = mem.as_prompt_prefix()
        # Prefix length should be bounded (with some slack for header/footer)
        assert len(prefix) < _MAX_PROMPT_CHARS + 500

    def test_truncation_message_present_when_long(self, tmp_path):
        mem = _mem(tmp_path)
        for i in range(30):
            mem.append_entry(f"s{i}", f"Sim{i}", "MC",
                             [f"failure {i} " * 20],
                             [f"rule {i} " * 20])
        prefix = mem.as_prompt_prefix()
        assert "truncated" in prefix

    def test_returns_string(self, tmp_path):
        mem = _mem(tmp_path)
        assert isinstance(mem.as_prompt_prefix(), str)


# ─────────────────────────────────────────────────────────────────────────────
# consolidate_session
# ─────────────────────────────────────────────────────────────────────────────

class TestConsolidateSession:
    def test_no_backend_returns_empty_list(self, tmp_path):
        mem = _mem(tmp_path)
        rules = mem.consolidate_session(
            "s1", "Sim", "MC", _sample_facts(), backend=None
        )
        assert rules == []

    def test_no_backend_does_not_write_entry(self, tmp_path):
        mem = _mem(tmp_path)
        mem.consolidate_session("s1", "Sim", "MC", _sample_facts(), backend=None)
        assert mem.entry_count == 0

    def test_with_backend_returns_rules(self, tmp_path):
        mem = _mem(tmp_path)
        backend = _mock_backend()
        rules = mem.consolidate_session("s1", "Ising", "Monte Carlo",
                                        _sample_facts(), backend=backend)
        assert len(rules) > 0
        assert isinstance(rules[0], str)

    def test_with_backend_writes_entry(self, tmp_path):
        mem = _mem(tmp_path)
        backend = _mock_backend()
        mem.consolidate_session("s1", "Ising", "Monte Carlo",
                                _sample_facts(), backend=backend)
        assert mem.entry_count == 1

    def test_rules_from_backend_written_to_file(self, tmp_path):
        mem = _mem(tmp_path)
        backend = _mock_backend(
            '{"what_failed": [], "rules": ["My specific rule ABC."]}'
        )
        mem.consolidate_session("s1", "Sim", "MC", _sample_facts(), backend=backend)
        assert "My specific rule ABC." in mem.path.read_text()

    def test_backend_failure_returns_empty_list(self, tmp_path):
        mem = _mem(tmp_path)
        backend = MagicMock()
        backend.complete.side_effect = RuntimeError("API down")
        backend.model_name = "mock"
        rules = mem.consolidate_session("s1", "Sim", "MC",
                                        _sample_facts(), backend=backend)
        assert rules == []

    def test_backend_bad_json_returns_empty(self, tmp_path):
        mem = _mem(tmp_path)
        backend = _mock_backend("not valid json at all")
        rules = mem.consolidate_session("s1", "Sim", "MC",
                                        _sample_facts(), backend=backend)
        assert rules == []

    def test_flags_included_in_backend_call(self, tmp_path):
        mem = _mem(tmp_path)
        backend = _mock_backend()
        facts = _sample_facts(flags=[
            {"kind": "max_steps_hit", "severity": "warning",
             "message": "ran to limit without outcome"}
        ])
        mem.consolidate_session("s1", "Sim", "MC", facts, backend=backend)
        call_content = str(backend.complete.call_args)
        assert "max_steps_hit" in call_content

    def test_patch_changes_included_in_backend_call(self, tmp_path):
        mem = _mem(tmp_path)
        backend = _mock_backend()
        facts = _sample_facts(patch_changes=[
            {"field": "max_steps", "why": "hit limit"}
        ])
        mem.consolidate_session("s1", "Sim", "MC", facts, backend=backend)
        call_content = str(backend.complete.call_args)
        assert "max_steps" in call_content


# ─────────────────────────────────────────────────────────────────────────────
# clear / show / properties
# ─────────────────────────────────────────────────────────────────────────────

class TestMiscOperations:
    def test_clear_resets_entry_count(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        mem.append_entry("s2", "Sim", "MC", ["bug2"], ["rule2"])
        assert mem.entry_count == 2
        mem.clear()
        assert mem.entry_count == 0

    def test_clear_preserves_file(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        mem.clear()
        assert mem.path.exists()

    def test_clear_file_has_header(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        mem.clear()
        assert "AGENTS.md" in mem.path.read_text()

    def test_show_returns_full_content(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "SpecialSim", "MC", [], ["rule text here"])
        content = mem.show()
        assert "SpecialSim" in content
        assert "rule text here" in content

    def test_read_all_same_as_show(self, tmp_path):
        mem = _mem(tmp_path)
        mem.append_entry("s1", "Sim", "MC", ["bug"], ["rule"])
        assert mem.show() == mem.read_all()

    def test_path_property_is_path(self, tmp_path):
        mem = _mem(tmp_path)
        assert isinstance(mem.path, Path)

    def test_path_inside_memory_dir(self, tmp_path):
        mem = _mem(tmp_path)
        assert mem.path.parent == tmp_path

    def test_entry_count_property(self, tmp_path):
        mem = _mem(tmp_path)
        assert mem.entry_count == 0
        mem.append_entry("s1", "Sim", "MC", ["b"], ["r"])
        assert mem.entry_count == 1
        mem.append_entry("s2", "Sim", "MC", ["b"], ["r"])
        assert mem.entry_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestInit,
        TestAppendEntry,
        TestPromptPrefix,
        TestConsolidateSession,
        TestMiscOperations,
    ]

    passed = failed = 0
    for cls in classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            with __import__("tempfile").TemporaryDirectory() as tmp:
                try:
                    getattr(instance, name)(Path(tmp))
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
