"""
Initialize ClickHouse database tables for OpenClaw Insight.

Key ClickHouse DDL differences from the MySQL/ADB version (init_db.py):
  - MergeTree engine family (no AUTO_INCREMENT, no DISTRIBUTED BY)
  - PARTITION BY toYYYYMMDD(timestamp)
  - ORDER BY clause replaces PRIMARY KEY
  - DateTime64(3, 'Asia/Shanghai') instead of DATETIME(3)
  - Nullable(String) / Int32 / Int8 instead of TEXT / INT / TINYINT
  - No secondary INDEX definitions (MergeTree uses the sorting key)
  - execute_ddl() (client.command) is used instead of execute_query()
"""

from __future__ import annotations

import asyncio
import sys

from scripts.config_ck import load_config_ck
from scripts.db_ck import close_connection_pool, execute_ddl


async def init_database() -> None:
    config = load_config_ck()

    print("[Init-CK] Starting ClickHouse database initialization...")
    print(f"[Init-CK] Target: {config.ck.host}:{config.ck.port}/{config.ck.database}")
    print(f"[Init-CK] Session table: {config.ck.session_table}")

    # ── Session table ──
    create_session_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{config.ck.session_table}` (
            row_id              UInt64,
            session_id          String,
            type                String,
            id                  Nullable(String),
            parent_id           Nullable(String),
            timestamp           DateTime64(3, 'Asia/Shanghai'),
            hostname            Nullable(String),
            complete_session    String,
            role                Nullable(String)  COMMENT 'Message role: user/assistant/tool',
            model               Nullable(String)  COMMENT 'Model name',
            api                 Nullable(String)  COMMENT 'API identifier',
            provider            Nullable(String)  COMMENT 'Provider name',
            stop_reason         Nullable(String)  COMMENT 'Stop reason',
            input_tokens        Int32  DEFAULT 0  COMMENT 'Input token count',
            output_tokens       Int32  DEFAULT 0  COMMENT 'Output token count',
            cache_read_tokens   Int32  DEFAULT 0  COMMENT 'Cache read token count',
            cache_write_tokens  Int32  DEFAULT 0  COMMENT 'Cache write token count',
            total_tokens        Int32  DEFAULT 0  COMMENT 'Total token count',
            total_cost          Decimal(12, 6) DEFAULT 0 COMMENT 'Call cost',
            tool_name           Nullable(String)  COMMENT 'Tool name',
            tool_input          Nullable(String)  COMMENT 'Tool input parameters JSON',
            tool_use_id         Nullable(String)  COMMENT 'Tool call ID',
            is_error            Int8   DEFAULT 0  COMMENT 'Whether the tool call errored',
            content_text        Nullable(String)  COMMENT 'Plain text content',
            content_length      Int32  DEFAULT 0  COMMENT 'Character length of content_text',
            thinking_text       Nullable(String)  COMMENT 'Model thinking process text',
            sender_id           Nullable(String)  COMMENT 'Sender user ID',
            created_at          DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (session_id, timestamp, row_id)
        SETTINGS index_granularity = 8192
    """

    try:
        execute_ddl(config.ck, create_session_table_sql)
        print(f"[Init-CK] ✅ Session table `{config.ck.session_table}` ready")
    except Exception as error:
        print(f"[Init-CK] ❌ Failed to create session table: {error}")
        raise

    # ── Schema migrations: add columns that may be missing from older tables ──
    migrations = [
        (
            "sender_id",
            f"ALTER TABLE `{config.ck.session_table}` ADD COLUMN IF NOT EXISTS sender_id Nullable(String) COMMENT 'Sender user ID'",
        ),
    ]
    for col_name, migration_sql in migrations:
        try:
            execute_ddl(config.ck, migration_sql)
            print(f"[Init-CK] ✅ Column `{col_name}` ensured on `{config.ck.session_table}`")
        except Exception as error:
            print(f"[Init-CK] ⚠️ Migration for column `{col_name}` failed (may already exist): {error}")

    # ── Logs table ──
    logs_table = config.ck.logs_table or "openclaw_logs"
    print(f"[Init-CK] Logs table: {logs_table}")

    create_logs_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{logs_table}` (
            id                   UInt64,
            timestamp            DateTime64(3, 'Asia/Shanghai'),
            level                String,
            subsystem            Nullable(String),
            raw_field_0          Nullable(String),
            raw_field_1          Nullable(String),
            raw_field_2          Nullable(String),
            meta_runtime         Nullable(String),
            meta_runtime_version Nullable(String),
            hostname             Nullable(String),
            meta_name            Nullable(String),
            meta_parent_names    Nullable(String),
            meta_date            Nullable(DateTime64(3, 'Asia/Shanghai')),
            meta_log_level_id    Nullable(Int32),
            meta_log_level_name  Nullable(String),
            meta_path            Nullable(String),
            complete_log         String,
            created_at           DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, id)
        SETTINGS index_granularity = 8192
    """

    try:
        execute_ddl(config.ck, create_logs_table_sql)
        print(f"[Init-CK] ✅ Logs table `{logs_table}` ready")
    except Exception as error:
        print(f"[Init-CK] ❌ Failed to create logs table: {error}")
        raise

    # ── Analysis results table ──
    create_analysis_result_table_sql = """
        CREATE TABLE IF NOT EXISTS `openclaw_analysis_results` (
            row_id           UInt64,
            run_id           String          COMMENT 'Unique ID for each analysis run (UUID)',
            case_name        String          COMMENT 'Analysis case name, e.g. L1, L2-1, ...',
            analysis_type    String          COMMENT 'L1_OPERATIONAL / L2_BEHAVIOR / L3_ORGANIZATIONAL / FINAL_REPORT',
            status           String  DEFAULT 'success' COMMENT 'success / failure / skipped',
            elapsed_seconds  Nullable(Float64) COMMENT 'Execution time in seconds',
            time_range_start Nullable(String) COMMENT 'Analysis window start timestamp',
            time_range_end   Nullable(String) COMMENT 'Analysis window end timestamp',
            summary          Nullable(String) COMMENT 'Human-readable summary of the result',
            details          String          COMMENT 'Full analysis result JSON',
            error_message    Nullable(String) COMMENT 'Error message if status is failure',
            created_at       DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        ORDER BY (created_at, run_id, case_name)
        SETTINGS index_granularity = 8192
    """

    try:
        execute_ddl(config.ck, create_analysis_result_table_sql)
        print("[Init-CK] ✅ Analysis results table `openclaw_analysis_results` ready")
    except Exception as error:
        print(f"[Init-CK] ❌ Failed to create analysis results table: {error}")
        raise

    print("[Init-CK] ✅ ClickHouse database initialization completed")
    close_connection_pool()


if __name__ == "__main__":
    try:
        asyncio.run(init_database())
    except Exception as error:
        print(f"[Init-CK] Execution failed: {error}")
        sys.exit(1)
