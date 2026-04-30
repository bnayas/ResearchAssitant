"""
test_launcher_log.py
─────────────────────
Unit tests for LauncherLog — the append-only markdown FAQ log.

Run:  python test_launcher_log.py
      or: python -m pytest test_launcher_log.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# ── Allow running from any directory ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

try:
    from sim_tool.launcher_log import LauncherLog, LogEntry, _FAQ_MIN_OCCURRENCES
except ImportError:
    from launcher_log import LauncherLog, LogEntry, _FAQ_MIN_OCCURRENCES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_log(tmp_path: Path) -> LauncherLog:
    return LauncherLog(log_dir=tmp_path)


def _make_ticket(
    kind_value="runtime_error",
    exit_code=1,
    pattern="RuntimeError",
    fix="Check stderr.",
    retryable=False,
    wall_time=1.5,
    script="sim.py",
):
    """Build a minimal mock BugTicket without importing contract.py."""
    t = MagicMock()
    t.kind.value = kind_value
    t.exit_code = exit_code
    t.failure_pattern = pattern
    t.suggested_fix = fix
    t.is_retryable = retryable
    t.wall_time_seconds = wall_time
    t.error_line = "RuntimeError: bad state at step 12"
    t.pipeline_script_path = None
    return t


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLauncherLogInit:
    def test_creates_file_on_first_use(self, tmp_path):
        log = _make_log(tmp_path)
        assert log.path.exists()

    def test_file_has_header(self, tmp_path):
        log = _make_log(tmp_path)
        content = log.path.read_text()
        assert "# Launcher Log" in content

    def test_entry_count_starts_at_zero(self, tmp_path):
        log = _make_log(tmp_path)
        assert log.entry_count == 0

    def test_loads_existing_log_without_overwriting(self, tmp_path):
        log1 = _make_log(tmp_path)
        log1.record_success("sid1", "sim.py", "uv 0.11.2", 1.2, configs_run=1)
        log2 = _make_log(tmp_path)
        assert log2.entry_count == 1


class TestRecordSuccess:
    def test_entry_count_increments(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("sid1", "sim.py", "python 3.11", 2.5, configs_run=3)
        assert log.entry_count == 1

    def test_success_appears_in_content(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("sid-abc", "ising.py", "uv 0.11", 5.0, configs_run=7)
        content = log.show()
        assert "ising.py" in content
        assert "sid-abc" in content
        assert "✓ success" in content

    def test_configs_count_in_entry(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("s1", "sim.py", "python", 1.0, configs_run=12)
        assert "12" in log.show()

    def test_multiple_successes(self, tmp_path):
        log = _make_log(tmp_path)
        for i in range(3):
            log.record_success(f"sid{i}", "sim.py", "python", 1.0)
        assert log.entry_count == 3


class TestRecordFailure:
    def test_entry_count_increments(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket()
        log.record_failure("sid1", "sim.py", ticket)
        assert log.entry_count == 1

    def test_failure_kind_in_content(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket(kind_value="env_setup")
        log.record_failure("sid1", "sim.py", ticket)
        content = log.show()
        assert "env_setup" in content

    def test_pattern_in_content(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket(pattern="ModuleNotFoundError")
        log.record_failure("sid1", "sim.py", ticket)
        assert "ModuleNotFoundError" in log.show()

    def test_fix_in_content(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket(fix="Run uv add numpy")
        log.record_failure("sid1", "sim.py", ticket)
        assert "Run uv add numpy" in log.show()

    def test_retryable_flag_recorded(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket(retryable=True)
        log.record_failure("sid1", "sim.py", ticket)
        assert "True" in log.show()

    def test_error_line_recorded_when_present(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket()
        ticket.error_line = "ZeroDivisionError: division by zero"
        log.record_failure("sid1", "sim.py", ticket)
        assert "ZeroDivisionError" in log.show()

    def test_pipeline_path_recorded_when_present(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket()
        ticket.pipeline_script_path = "/tmp/run_pipeline.sh"
        log.record_failure("sid1", "sim.py", ticket)
        assert "run_pipeline.sh" in log.show()


class TestFAQ:
    def test_no_faq_below_threshold(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket(pattern="UniquePattern999")
        log.record_failure("sid1", "sim.py", ticket)
        faq = log.get_faq()
        assert not any(e["pattern"] == "UniquePattern999" for e in faq)

    def test_faq_appears_at_threshold(self, tmp_path):
        log = _make_log(tmp_path)
        pattern = "ModuleNotFoundError: No module named 'numpy'"
        for i in range(_FAQ_MIN_OCCURRENCES):
            ticket = _make_ticket(pattern=pattern)
            log.record_failure(f"sid{i}", "sim.py", ticket)
        faq = log.get_faq()
        assert any(e["pattern"] == pattern for e in faq)

    def test_faq_count_correct(self, tmp_path):
        log = _make_log(tmp_path)
        pattern = "AssertionError: bad config"
        for i in range(4):
            ticket = _make_ticket(pattern=pattern)
            log.record_failure(f"sid{i}", "sim.py", ticket)
        faq = log.get_faq()
        entry = next(e for e in faq if e["pattern"] == pattern)
        assert entry["count"] == 4

    def test_faq_stores_fix(self, tmp_path):
        log = _make_log(tmp_path)
        pattern = "SyntaxError: invalid syntax"
        for i in range(_FAQ_MIN_OCCURRENCES):
            ticket = _make_ticket(pattern=pattern, fix="Fix the syntax error")
            log.record_failure(f"sid{i}", "sim.py", ticket)
        faq = log.get_faq()
        entry = next(e for e in faq if e["pattern"] == pattern)
        assert entry["fix"] == "Fix the syntax error"

    def test_faq_section_written_to_file(self, tmp_path):
        log = _make_log(tmp_path)
        pattern = "RecursionError"
        for i in range(_FAQ_MIN_OCCURRENCES):
            ticket = _make_ticket(pattern=pattern)
            log.record_failure(f"sid{i}", "sim.py", ticket)
        content = log.show()
        assert "FAQ" in content
        assert pattern in content

    def test_faq_does_not_duplicate_on_reload(self, tmp_path):
        log = _make_log(tmp_path)
        pattern = "TimeoutExpired"
        for i in range(_FAQ_MIN_OCCURRENCES):
            ticket = _make_ticket(pattern=pattern)
            log.record_failure(f"sid{i}", "sim.py", ticket)
        content_before = log.show().count("FAQ")
        # Add one more — should regenerate, not duplicate
        ticket = _make_ticket(pattern=pattern)
        log.record_failure("sid_extra", "sim.py", ticket)
        content_after = log.show().count("Frequently Seen")
        assert content_after == 1


class TestParseEntries:
    def test_parse_success_entry(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("parse-s1", "parse_sim.py", "uv 0.11", 3.7)
        entries = log._parse_entries()
        assert len(entries) == 1
        assert "parse_sim.py" in entries[0].script_name

    def test_parse_failure_entry(self, tmp_path):
        log = _make_log(tmp_path)
        ticket = _make_ticket(kind_value="runtime_error", pattern="RuntimeError")
        log.record_failure("parse-f1", "fail_sim.py", ticket)
        entries = log._parse_entries()
        assert len(entries) == 1
        assert entries[0].kind == "runtime_error"

    def test_parse_wall_time(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("wt-1", "sim.py", "python", 42.5)
        entries = log._parse_entries()
        assert abs(entries[0].wall_time - 42.5) < 0.01

    def test_mixed_entries_parse_correctly(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("s1", "sim.py", "python", 1.0)
        log.record_failure("f1", "sim.py", _make_ticket())
        log.record_success("s2", "sim.py", "python", 2.0)
        entries = log._parse_entries()
        assert len(entries) == 3


class TestShow:
    def test_show_returns_string(self, tmp_path):
        log = _make_log(tmp_path)
        assert isinstance(log.show(), str)

    def test_show_nonempty_after_record(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("s1", "sim.py", "python", 1.0)
        content = log.show()
        assert len(content) > 100

    def test_timestamps_present(self, tmp_path):
        log = _make_log(tmp_path)
        log.record_success("ts-1", "sim.py", "python", 1.0)
        content = log.show()
        import re
        timestamps = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)
        assert len(timestamps) >= 1


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    classes = [
        TestLauncherLogInit,
        TestRecordSuccess,
        TestRecordFailure,
        TestFAQ,
        TestParseEntries,
        TestShow,
    ]

    passed = failed = 0
    for cls in classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for name in methods:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                try:
                    getattr(instance, name)(tmp_path)
                    print(f"  ✓ {cls.__name__}.{name}")
                    passed += 1
                except Exception as exc:
                    print(f"  ✗ {cls.__name__}.{name}: {exc}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
