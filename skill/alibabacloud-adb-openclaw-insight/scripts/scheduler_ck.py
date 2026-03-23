"""
Scheduled task runner for ClickHouse (scheduler_ck.py).
- Log collection at the configured interval
- Usage analysis daily at 02:00
- Data cleanup daily at 03:00
"""

from __future__ import annotations

import asyncio

from apscheduler.schedulers.background import BackgroundScheduler

from scripts.config_ck import AppConfigCk


def _run_async(coro) -> None:
    """Run an async coroutine in a new event loop (for APScheduler callbacks)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


def start_scheduler_ck(config: AppConfigCk) -> BackgroundScheduler:
    interval_minutes = config.collection.interval_minutes

    print("[Scheduler-CK] Starting scheduler")
    print(f"[Scheduler-CK] Log collection interval: every {interval_minutes} minutes")
    print("[Scheduler-CK] Usage analysis: daily at 02:00")
    print("[Scheduler-CK] Data cleanup: daily at 03:00")

    scheduler = BackgroundScheduler()

    if config.collection.enable_log_collection:
        def collect_job() -> None:
            from datetime import datetime
            from scripts.collect_logs_ck import collect_logs
            print(f"\n[Scheduler-CK] {datetime.now().isoformat()} - Starting scheduled log collection")
            try:
                _run_async(collect_logs(config))
                print("[Scheduler-CK] Log collection completed")
            except Exception as error:
                print(f"[Scheduler-CK] Log collection failed: {error}")

        scheduler.add_job(collect_job, "interval", minutes=interval_minutes, id="collect_logs_ck")
        print("[Scheduler-CK] ✅ Log collection scheduled task started")
    else:
        print("[Scheduler-CK] ⚠️ Log collection is disabled")

    def analyze_job() -> None:
        from datetime import datetime
        from scripts.analyze_usage_ck import run_full_analysis
        print(f"\n[Scheduler-CK] {datetime.now().isoformat()} - Starting daily usage analysis")
        try:
            _run_async(run_full_analysis(config))
            print("[Scheduler-CK] Daily usage analysis completed")
        except Exception as error:
            print(f"[Scheduler-CK] Daily usage analysis failed: {error}")

    scheduler.add_job(analyze_job, "cron", hour=2, minute=0, id="daily_analysis_ck")
    print("[Scheduler-CK] ✅ Daily analysis scheduled task started")

    def cleanup_job() -> None:
        from datetime import datetime
        from scripts.collect_logs_ck import clean_expired_data
        print(f"\n[Scheduler-CK] {datetime.now().isoformat()} - Starting expired data cleanup")
        try:
            _run_async(clean_expired_data(config))
            print("[Scheduler-CK] Expired data cleanup completed")
        except Exception as error:
            print(f"[Scheduler-CK] Expired data cleanup failed: {error}")

    scheduler.add_job(cleanup_job, "cron", hour=3, minute=0, id="data_cleanup_ck")
    print("[Scheduler-CK] ✅ Data cleanup scheduled task started")

    scheduler.start()
    return scheduler
