from __future__ import annotations

import asyncio
import os
import signal
import sys
from datetime import datetime

# Set up a dedicated log file that flushes every write, so we can monitor
# progress in real-time even when stdout is block-buffered (file redirect).
import builtins
_original_print = builtins.print
_log_file_path = os.environ.get("OPENCLAW_LOG_FILE", "")

if _log_file_path:
    _log_fh = open(_log_file_path, "w", buffering=1, encoding="utf-8")

    def _logged_print(*args, **kwargs):
        message = kwargs.get("sep", " ").join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        _log_fh.write(message + end)
        _log_fh.flush()
        kwargs.setdefault("flush", True)
        _original_print(*args, **kwargs)

    builtins.print = _logged_print
else:
    def _flushed_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        _original_print(*args, **kwargs)

    builtins.print = _flushed_print

def _parse_extra_args(args: list[str]) -> dict[str, str]:
    """Parse extra CLI arguments like --user, --from, --to, --run-id, --report."""
    result: dict[str, str] = {}
    index = 0
    while index < len(args):
        if args[index] == "--user" and index + 1 < len(args):
            result["user"] = args[index + 1]
            index += 2
        elif args[index] == "--from" and index + 1 < len(args):
            result["from"] = args[index + 1]
            index += 2
        elif args[index] == "--to" and index + 1 < len(args):
            result["to"] = args[index + 1]
            index += 2
        elif args[index] == "--run-id" and index + 1 < len(args):
            result["run-id"] = args[index + 1]
            index += 2
        else:
            index += 1
    return result


def _validate_date_format(date_string: str) -> bool:
    """Accept YYYY-MM-DD or YYYY-MM-DD HH:MM:SS (with optional fractional seconds)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            datetime.strptime(date_string.split(".")[0], fmt)
            return True
        except ValueError:
            continue
    return False


def _build_time_range(cli_args: dict[str, str], analysis_config) -> "TimeRange":
    """Build a TimeRange from CLI args, falling back to config defaults."""
    from scripts.types import TimeRange, last_n_days_range

    if cli_args.get("from") and cli_args.get("to"):
        if not _validate_date_format(cli_args["from"]) or not _validate_date_format(cli_args["to"]):
            print("❌ Invalid date format. Please use: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
            sys.exit(1)
        return TimeRange(start_date=cli_args["from"], end_date=cli_args["to"])
    elif cli_args.get("from") or cli_args.get("to"):
        print("❌ Both --from and --to must be provided together")
        sys.exit(1)

    window_days = analysis_config.analysis_window_days if analysis_config else 7
    return last_n_days_range(window_days)


async def main() -> None:
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=32))
    from scripts.config import load_config
    from scripts.collect_logs import collect_logs
    from scripts.analyze_usage import run_full_analysis
    from scripts.db import close_connection_pool

    print("=" * 60)
    print("🚀 OpenClaw Logger ADB Skill Starting")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 60)

    config = load_config()

    print("\n📋 Configuration:")
    print(f"  ADB address: {config.adb.host}:{config.adb.port}/{config.adb.database}")
    print(f"  Session table: {config.adb.session_table}")
    print(f"  Collection interval: {config.collection.interval_minutes} minutes")
    print(f"  Batch size: {config.collection.batch_size}")
    print(f"  Data retention: {config.collection.retention_days} days")
    print(f"  Log collection: {'enabled' if config.collection.enable_log_collection else 'disabled'}")
    print(f"  Token collection: {'enabled' if config.collection.enable_token_collection else 'disabled'}")

    command = sys.argv[1] if len(sys.argv) > 1 else "serve"

    # Parse extra arguments (--from, --to, --user, --run-id) from argv[2:]
    extra_args = _parse_extra_args(sys.argv[2:])

    if command == "collect":
        print("\n📥 Running one-time log collection...")
        await collect_logs(config)
        close_connection_pool()

    elif command == "analyze":
        print("\n📊 Running one-time usage analysis...")
        range_ = _build_time_range(extra_args, config.analysis)
        print(f"  Time range: {range_.start_date} → {range_.end_date}")
        await run_full_analysis(config, range_)
        close_connection_pool()

    elif command == "final-report":
        print("\n📄 Fetching latest final narrative report from database...")
        from scripts.analysis.orchestrator import AnalysisOrchestrator
        orchestrator = AnalysisOrchestrator(config)
        report_text = orchestrator.get_final_report()
        print(report_text)
        close_connection_pool()

    elif command == "report":
        run_id = extra_args.get("run-id") or (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None)
        if not run_id:
            print("❌ Run ID is required.")
            print("Usage: python -m scripts.main report <run_id>")
            print("       python -m scripts.main report --run-id <run_id>")
            sys.exit(1)
        print(f"\n📊 Generating report for run_id: {run_id}")
        from scripts.analysis.orchestrator import AnalysisOrchestrator
        orchestrator = AnalysisOrchestrator(config)
        orchestrator.generate_report(run_id)
        close_connection_pool()

    elif command == "describe-metrics":
        print("\n📘 Reading local insight metrics logic document...")
        from scripts.analysis.insight_logic_docs import generate_insight_logic_doc
        doc = await generate_insight_logic_doc(config)
        print(doc)
        close_connection_pool()

    elif command == "drilldown":
        from scripts.analysis.drilldown import (
            drilldown_user_tasks, drilldown_non_work_tasks,
            format_user_tasks_report, format_non_work_tasks_report,
        )
        from pathlib import Path

        sub_command = sys.argv[2] if len(sys.argv) > 2 else ""
        cli_args = _parse_extra_args(sys.argv[3:])

        if sub_command == "user-tasks":
            sender_id = cli_args.get("user")
            if not sender_id:
                print("❌ --user <sender_id> is required for user-tasks drilldown")
                print("Usage: python -m scripts.main drilldown user-tasks --user <sender_id> --from <date> --to <date>")
                sys.exit(1)
            range_ = _build_time_range(cli_args, config.analysis)
            result = drilldown_user_tasks(config.adb, config.adb.session_table, range_, sender_id)
            report = format_user_tasks_report(result)
            output_path = Path("output") / "drilldown_user_tasks.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(report)
            print(f"\n📄 Report saved to {output_path}")

        elif sub_command == "non-work-tasks":
            range_ = _build_time_range(cli_args, config.analysis)
            run_id = cli_args.get("run-id")
            result = drilldown_non_work_tasks(
                config.adb, config.adb.session_table, range_, run_id=run_id,
            )
            report = format_non_work_tasks_report(result)
            output_path = Path("output") / "drilldown_non_work_tasks.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(report)
            print(f"\n📄 Report saved to {output_path}")

        else:
            print("❌ Unknown drilldown sub-command. Available sub-commands:")
            print("  user-tasks       — Query a specific user's task count and complexity")
            print("  non-work-tasks   — Find non-work intent tasks with full details")
            print("")
            print("Examples:")
            print("  python -m scripts.main drilldown user-tasks --user 363779 --from 2026-03-01 --to 2026-03-10")
            print("  python -m scripts.main drilldown non-work-tasks --from 2026-03-01 --to 2026-03-10")
            print("  python -m scripts.main drilldown non-work-tasks --run-id <uuid>")
            sys.exit(1)

        close_connection_pool()

    else:
        # serve mode (default)
        if config.collection.enable_log_collection:
            print("\n📥 Running initial log collection...")
            try:
                inserted_count = await collect_logs(config)
                print(f"Initial collection completed, inserted {inserted_count} records")
            except Exception as error:
                print(f"Initial collection failed (will retry on next scheduled run): {error}")

        from scripts.scheduler import start_scheduler
        scheduler = start_scheduler(config)

        print("\n✅ Service started, press Ctrl+C to exit")

        def graceful_shutdown(signum, frame) -> None:
            print("\n🛑 Received shutdown signal, gracefully shutting down...")
            scheduler.shutdown(wait=False)
            close_connection_pool()
            print("👋 Exited")
            sys.exit(0)

        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

        import time
        while True:
            time.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        print(f"❌ Startup failed: {error}")
        sys.exit(1)
