"""
conftest.py — pytest configuration for the research platform.

Adds the project root to sys.path so all sub-packages are importable
as top-level packages (sim_tool, literature_review, reviewer, writer).
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
