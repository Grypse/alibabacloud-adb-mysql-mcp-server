"""
ClickHouse connection utilities using clickhouse-connect.

Each operation creates a fresh client and closes it immediately after use,
mirroring the per-call connection pattern of the MySQL version (db.py).

Key differences from db.py (MySQL):
  - Uses clickhouse_connect.get_client() instead of mysql.connector.connect()
  - Query results are extracted from QueryResult.result_rows / column_names
  - Batch insert uses client.insert() (no placeholders needed)
  - SQL parameters are substituted Python-side via _substitute_params()
    since clickhouse-connect's native binding uses {name:Type} syntax while
    all existing queries use the MySQL-style %s positional syntax.
  - execute_ddl() runs DDL statements (CREATE TABLE IF NOT EXISTS, ALTER TABLE)
    which do not return row data.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import clickhouse_connect

from scripts.config_ck import CkConfig

SqlValue = str | int | float | bool | None

# Dedicated thread pool for async DB wrappers
_db_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="db_ck")


# ─── Parameter Substitution ───

def _substitute_params(sql: str, params: tuple | list) -> str:
    """
    Substitute %s positional placeholders with safely-escaped values.

    Handles: str, int, float, bool, None.
    This is safe for our internal use where all params are datetime strings,
    integers, floats, or None — never raw user input.
    """
    if not params:
        return sql
    result: list[str] = []
    param_index = 0
    i = 0
    while i < len(sql):
        if sql[i] == "%" and i + 1 < len(sql) and sql[i + 1] == "s":
            val = params[param_index]
            if val is None:
                result.append("NULL")
            elif isinstance(val, bool):
                result.append("1" if val else "0")
            elif isinstance(val, (int, float)):
                result.append(str(val))
            elif isinstance(val, str):
                escaped = val.replace("\\", "\\\\").replace("'", "\\'")
                result.append(f"'{escaped}'")
            else:
                escaped = str(val).replace("\\", "\\\\").replace("'", "\\'")
                result.append(f"'{escaped}'")
            param_index += 1
            i += 2
        else:
            result.append(sql[i])
            i += 1
    return "".join(result)


# ─── Client Factory ───

def _create_client(ck_config: CkConfig):
    """Create a new clickhouse-connect client (HTTP)."""
    return clickhouse_connect.get_client(
        host=ck_config.host,
        port=ck_config.port,
        username=ck_config.username,
        password=ck_config.password,
        database=ck_config.database,
        secure=ck_config.secure,
        connect_timeout=30,
        send_receive_timeout=300,
    )


# ─── Public API ───

def execute_query(
    ck_config: CkConfig,
    sql: str,
    params: Optional[tuple | list] = None,
) -> list[dict]:
    """Execute a SELECT (or any result-returning) query and return rows as list of dicts."""
    start_time = time.time()
    client = _create_client(ck_config)
    try:
        formatted_sql = _substitute_params(sql, params) if params else sql
        result = client.query(formatted_sql)
        rows = [
            dict(zip(result.column_names, row))
            for row in result.result_rows
        ]
        elapsed = time.time() - start_time
        print(f"[DB-CK] Query returned {len(rows)} rows in {elapsed:.1f}s")
        return rows
    finally:
        client.close()


async def execute_query_async(
    ck_config: CkConfig,
    sql: str,
    params: Optional[tuple | list] = None,
) -> list[dict]:
    """Async wrapper for execute_query using a dedicated thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_db_executor, execute_query, ck_config, sql, params)


def execute_ddl(ck_config: CkConfig, sql: str) -> None:
    """Execute a DDL statement (CREATE TABLE, ALTER TABLE, etc.) that returns no rows."""
    client = _create_client(ck_config)
    try:
        client.command(sql)
    finally:
        client.close()


def execute_batch_insert(
    ck_config: CkConfig,
    table_name: str,
    columns: list[str],
    rows: list[list[SqlValue]],
) -> int:
    """
    Batch insert rows into a ClickHouse table.
    Returns the number of rows submitted (clickhouse-connect insert() has no row-count return).
    """
    if not rows:
        return 0

    client = _create_client(ck_config)
    try:
        client.insert(
            table_name,
            rows,
            column_names=columns,
            database=ck_config.database,
        )
        return len(rows)
    finally:
        client.close()


def close_connection_pool() -> None:
    """Shut down the shared DB thread-pool executor so the process can exit cleanly."""
    print("[DB-CK] Shutting down DB executor pool...")
    _db_executor.shutdown(wait=True)
    print("[DB-CK] DB executor pool shut down")
