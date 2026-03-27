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
    """Parse extra CLI arguments like --user, --from, --to, --run-id."""
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

    from scripts.config_ck import load_config_ck
    from scripts.collect_logs_ck import collect_logs
    from scripts.analyze_usage_ck import run_full_analysis
    from scripts.db_ck import close_connection_pool

    print("=" * 60)
    print("🚀 OpenClaw Logger ClickHouse Skill Starting")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 60)

    config = load_config_ck()

    print("\n📋 Configuration (ClickHouse):")
    print(f"  CK address: {config.ck.host}:{config.ck.port}/{config.ck.database}")
    print(f"  Session table: {config.ck.session_table}")
    print(f"  Collection interval: {config.collection.interval_minutes} minutes")
    print(f"  Batch size: {config.collection.batch_size}")
    print(f"  Data retention: {config.collection.retention_days} days")
    print(f"  Log collection: {'enabled' if config.collection.enable_log_collection else 'disabled'}")
    print(f"  Token collection: {'enabled' if config.collection.enable_token_collection else 'disabled'}")

    command = sys.argv[1] if len(sys.argv) > 1 else "serve"

    # Parse extra arguments (--from, --to, --user, --run-id) from argv[2:]
    extra_args = _parse_extra_args(sys.argv[2:])

    if command == "collect":
        print("\n📥 Running one-time log collection (ClickHouse)...")
        await collect_logs(config)
        close_connection_pool()

    elif command == "analyze":
        print("\n📊 Running one-time usage analysis (ClickHouse)...")
        range_ = _build_time_range(extra_args, config.analysis)
        print(f"  Time range: {range_.start_date} → {range_.end_date}")
        await run_full_analysis(config, range_)
        close_connection_pool()

    elif command == "final-report":
        print("\n📄 Fetching latest final narrative report from ClickHouse database...")
        from scripts.analysis.orchestrator_ck import AnalysisOrchestratorCk
        orchestrator = AnalysisOrchestratorCk(config)
        report_text = orchestrator.get_final_report()
        print(report_text)
        close_connection_pool()

    elif command == "report":
        run_id = extra_args.get("run-id") or (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None)
        if not run_id:
            print("❌ Run ID is required.")
            print("Usage: python -m scripts.main_ck report <run_id>")
            print("       python -m scripts.main_ck report --run-id <run_id>")
            sys.exit(1)
        print(f"\n📊 Generating report for run_id: {run_id}")
        from scripts.analysis.orchestrator_ck import AnalysisOrchestratorCk
        orchestrator = AnalysisOrchestratorCk(config)
        orchestrator.generate_report(run_id)
        close_connection_pool()

    elif command == "describe-metrics":
        print("\n📘 Reading local insight metrics logic document...")
        from scripts.analysis.insight_logic_docs import generate_insight_logic_doc
        doc = await generate_insight_logic_doc(config)
        print(doc)
        close_connection_pool()

    else:
        # serve mode (default)
        if config.collection.enable_log_collection:
            print("\n📥 Running initial log collection (ClickHouse)...")
            try:
                inserted_count = await collect_logs(config)
                print(f"Initial collection completed, inserted {inserted_count} records")
            except Exception as error:
                print(f"Initial collection failed (will retry on next scheduled run): {error}")

        from scripts.scheduler_ck import start_scheduler_ck
        # Scheduler expects an AppConfig-like object with collection/adb attributes.
        # For CK we wrap the config so the scheduler can read interval_minutes.
        scheduler = start_scheduler_ck(config)

        print("\n✅ Service started (ClickHouse), press Ctrl+C to exit")

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
