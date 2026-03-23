"""
Standalone analysis runner for ClickHouse (analyze_usage_ck.py).

Commands:
  (default)           Run full analysis with default time range from config
  --from YYYY-MM-DD --to YYYY-MM-DD   Run with custom date range
  --report <run_id>   Generate report for a previous analysis run
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from scripts.config_ck import load_config_ck
from scripts.db_ck import close_connection_pool
from scripts.types import TimeRange
from scripts.analysis.orchestrator_ck import AnalysisOrchestratorCk


async def run_full_analysis(config, range_: Optional[TimeRange] = None) -> str:
    """Run full ClickHouse analysis and return the run_id."""
    orchestrator = AnalysisOrchestratorCk(config)
    return await orchestrator.run_full_analysis(range_)


def _parse_command_line_args() -> dict[str, Optional[str]]:
    args = sys.argv[1:]
    result: dict[str, Optional[str]] = {"from": None, "to": None, "report": None}
    i = 0
    while i < len(args):
        if args[i] == "--from" and i + 1 < len(args):
            result["from"] = args[i + 1]
            i += 2
        elif args[i] == "--to" and i + 1 < len(args):
            result["to"] = args[i + 1]
            i += 2
        elif args[i] == "--report" and i + 1 < len(args):
            result["report"] = args[i + 1]
            i += 2
        else:
            i += 1
    return result


def _validate_date_format(date_string: str) -> bool:
    """Accept YYYY-MM-DD or YYYY-MM-DD HH:MM:SS (with optional fractional seconds)."""
    from datetime import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            datetime.strptime(date_string.split(".")[0], fmt)
            return True
        except ValueError:
            continue
    return False


async def main() -> None:
    config = load_config_ck()
    cli_args = _parse_command_line_args()

    # Report mode: query existing run results
    if cli_args["report"]:
        run_id = cli_args["report"]
        print(f"[analyze_usage_ck] Generating report for run_id: {run_id}")
        orchestrator = AnalysisOrchestratorCk(config)
        try:
            orchestrator.generate_report(run_id)
        finally:
            close_connection_pool()
        return

    # Analysis mode
    range_: Optional[TimeRange] = None

    if cli_args["from"] and cli_args["to"]:
        if not _validate_date_format(cli_args["from"]) or not _validate_date_format(cli_args["to"]):
            print("[analyze_usage_ck] Invalid date format. Please use ISO date format: YYYY-MM-DD")
            print("[analyze_usage_ck] Example: python3 -m scripts.analyze_usage_ck --from 2026-03-01 --to 2026-03-10")
            sys.exit(1)
        range_ = TimeRange(start_date=cli_args["from"], end_date=cli_args["to"])
        print(f"[analyze_usage_ck] Using custom time range: {range_.start_date} to {range_.end_date}")
    elif cli_args["from"] or cli_args["to"]:
        print("[analyze_usage_ck] Both --from and --to must be provided together")
        print("[analyze_usage_ck] Example: python3 -m scripts.analyze_usage_ck --from 2026-03-01 --to 2026-03-10")
        sys.exit(1)
    else:
        print("[analyze_usage_ck] Using default time range from configuration")

    try:
        run_id = await run_full_analysis(config, range_)
        print(f"\n[analyze_usage_ck] ✅ Analysis completed successfully. Run ID: {run_id}")
    except Exception as error:
        print(f"\n[analyze_usage_ck] ❌ Analysis failed: {error}")
        sys.exit(1)
    finally:
        close_connection_pool()


if __name__ == "__main__":
    import builtins
    _original_print = builtins.print
    def _flushed_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        _original_print(*args, **kwargs)
    builtins.print = _flushed_print

    asyncio.run(main())
