"""
ClickHouse analysis runner — internal module.

All CLI commands are handled by scripts/main_ck.py.
This module only exposes run_full_analysis() for programmatic use.
"""

from __future__ import annotations

from typing import Optional

from scripts.types import TimeRange
from scripts.analysis.orchestrator_ck import AnalysisOrchestratorCk


async def run_full_analysis(config, range_: Optional[TimeRange] = None) -> str:
    """Run full ClickHouse analysis and return the run_id."""
    orchestrator = AnalysisOrchestratorCk(config)
    return await orchestrator.run_full_analysis(range_)
