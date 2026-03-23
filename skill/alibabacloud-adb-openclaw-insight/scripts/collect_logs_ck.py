"""
OpenClaw session and log file collector — ClickHouse edition.

Adapts collect_logs.py for ClickHouse storage:
  - Uses db_ck (clickhouse-connect) instead of db (mysql-connector)
  - row_id is generated in Python (time-based monotonic UInt64) since ClickHouse
    has no AUTO_INCREMENT concept
  - Timestamps are always converted to Asia/Shanghai (matching the DateTime64
    column definition) — no need to query the DB for its timezone
  - Data cleanup uses ALTER TABLE ... DELETE (ClickHouse mutation) instead of
    a standard DELETE statement
"""

from __future__ import annotations

import glob
import json
import os
import re
import socket
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from scripts.config_ck import AppConfigCk
from scripts.db_ck import SqlValue, execute_batch_insert, execute_query

# ─── Constants ───

STATE_FILE_NAME = ".collect_state_ck.json"
LOG_DIRECTORY = "/tmp/openclaw"
LOG_FILE_PATTERN = re.compile(r"^openclaw-\d{4}-\d{2}-\d{2}\.log$")

# ClickHouse sessions table columns (row_id prepended, generated in Python)
SESSION_COLUMNS = [
    "row_id",
    "session_id", "type", "id", "parent_id", "timestamp", "hostname", "sender_id",
    "complete_session", "role", "model", "api", "provider", "stop_reason",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "total_tokens", "total_cost", "tool_name", "tool_input", "tool_use_id",
    "is_error", "content_text", "content_length", "thinking_text",
]

# ClickHouse logs table columns (id prepended)
LOG_COLUMNS = [
    "id",
    "timestamp", "level", "subsystem", "raw_field_0", "raw_field_1", "raw_field_2",
    "meta_runtime", "meta_runtime_version", "hostname", "meta_name", "meta_parent_names",
    "meta_date", "meta_log_level_id", "meta_log_level_name", "meta_path", "complete_log",
]

LEVEL_MAP = {0: "silly", 1: "trace", 2: "debug", 3: "info", 4: "warn", 5: "error", 6: "fatal"}

# ─── Timezone (fixed to Asia/Shanghai matching DateTime64 column definition) ───

_CK_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _get_ck_timezone() -> ZoneInfo:
    """Return the timezone used for ClickHouse DateTime64 columns (always Asia/Shanghai)."""
    return _CK_TIMEZONE


# ─── Monotonic row_id Generator ───
#
# ClickHouse has no AUTO_INCREMENT.  We generate a time-based UInt64 that is:
#   - Globally unique across restarts (high bits = microsecond timestamp)
#   - Monotonically increasing within a process (low bits = counter)
#
# Formula:  base = int(time.time() * 1_000_000) * 10_000
#           id   = base + per_session_counter (max 9999 IDs per microsecond)

_row_id_lock = threading.Lock()
_row_id_base = int(time.time() * 1_000_000) * 10_000
_row_id_counter: list[int] = [0]


def _next_row_id() -> int:
    with _row_id_lock:
        _row_id_counter[0] += 1
        return _row_id_base + _row_id_counter[0]


# ─── Timestamp Conversion (reused from collect_logs.py) ───

def _convert_iso_timestamp(iso_timestamp: str, target_tz: ZoneInfo) -> str:
    """Convert an ISO 8601 timestamp string to a DATETIME string in the target timezone."""
    try:
        normalized = iso_timestamp.rstrip("Z") + ("+00:00" if iso_timestamp.endswith("Z") else "")
        dt_with_tz = datetime.fromisoformat(normalized)
        dt_local = dt_with_tz.astimezone(target_tz)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt_local.microsecond // 1000:03d}"
    except ValueError:
        return iso_timestamp.replace("T", " ").split("+")[0].rstrip("Z")


# ─── State Management ───

def _get_state_path() -> str:
    scripts_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(scripts_dir, "..", STATE_FILE_NAME))


def _load_collect_state() -> dict:
    state_path = _get_state_path()
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as state_file:
            return json.load(state_file)
    return {}


def _save_collect_state(state: dict) -> None:
    state_path = _get_state_path()
    with open(state_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)


# ─── Content Extraction (identical to collect_logs.py) ───

def _extract_content_parts(content: Any) -> dict:
    result = {"content_text": None, "thinking_text": None, "tool_calls": []}

    if isinstance(content, str):
        result["content_text"] = content
        return result

    if not isinstance(content, list):
        return result

    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
        elif item_type == "thinking" and isinstance(item.get("thinking"), str):
            thinking_parts.append(item["thinking"])
        elif item_type in ("toolCall", "tool_use"):
            arguments = item.get("arguments") or item.get("input") or {}
            result["tool_calls"].append({
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            })

    result["content_text"] = "\n".join(text_parts) if text_parts else None
    result["thinking_text"] = "\n".join(thinking_parts) if thinking_parts else None
    return result


def _parse_message_fields(message_obj: dict, record_type: str) -> dict:
    result = {
        "role": None, "model": None, "api": None, "provider": None,
        "stop_reason": None, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0, "total_tokens": 0,
        "total_cost": 0, "tool_name": None, "tool_input": None,
        "tool_use_id": None, "is_error": 0, "content_text": None,
        "content_length": 0, "thinking_text": None, "sender_id": None,
    }

    if not isinstance(message_obj, dict):
        return result

    if isinstance(message_obj.get("role"), str):
        result["role"] = message_obj["role"]
    if isinstance(message_obj.get("model"), str):
        result["model"] = message_obj["model"]
    if isinstance(message_obj.get("api"), str):
        result["api"] = message_obj["api"]
    if isinstance(message_obj.get("provider"), str):
        result["provider"] = message_obj["provider"]
    if isinstance(message_obj.get("stopReason"), str):
        result["stop_reason"] = message_obj["stopReason"]

    usage = message_obj.get("usage")
    if isinstance(usage, dict):
        result["input_tokens"] = usage.get("input", 0) or 0
        result["output_tokens"] = usage.get("output", 0) or 0
        result["cache_read_tokens"] = usage.get("cacheRead", 0) or 0
        result["cache_write_tokens"] = usage.get("cacheWrite", 0) or 0
        result["total_tokens"] = usage.get("totalTokens", 0) or 0
        cost = usage.get("cost")
        if isinstance(cost, dict) and isinstance(cost.get("total"), (int, float)):
            result["total_cost"] = cost["total"]

    content_parts = _extract_content_parts(message_obj.get("content"))
    result["thinking_text"] = content_parts["thinking_text"]
    if content_parts["content_text"]:
        result["content_text"] = content_parts["content_text"]
        result["content_length"] = len(content_parts["content_text"])

    if content_parts["tool_calls"]:
        first_tool = content_parts["tool_calls"][0]
        result["tool_name"] = first_tool["name"] or None
        result["tool_use_id"] = first_tool["id"] or None
        result["tool_input"] = first_tool["arguments"] or None

    if record_type == "tool_use" or message_obj.get("type") == "tool_use":
        if isinstance(message_obj.get("name"), str):
            result["tool_name"] = message_obj["name"]
        if message_obj.get("input") is not None:
            inp = message_obj["input"]
            result["tool_input"] = inp if isinstance(inp, str) else json.dumps(inp)
        if isinstance(message_obj.get("id"), str):
            result["tool_use_id"] = message_obj["id"]

    if record_type == "tool_result" or message_obj.get("type") == "tool_result":
        if isinstance(message_obj.get("toolCallId"), str):
            result["tool_use_id"] = message_obj["toolCallId"]
        elif isinstance(message_obj.get("tool_use_id"), str):
            result["tool_use_id"] = message_obj["tool_use_id"]
        if isinstance(message_obj.get("toolName"), str):
            result["tool_name"] = message_obj["toolName"]
        result["is_error"] = 1 if message_obj.get("isError") is True else 0
        raw_content = message_obj.get("content")
        if not result["content_text"]:
            if isinstance(raw_content, str):
                result["content_text"] = raw_content
                result["content_length"] = len(raw_content)
            elif isinstance(raw_content, list):
                text_parts = [
                    item["text"]
                    for item in raw_content
                    if isinstance(item, dict) and item.get("type") == "text"
                    and isinstance(item.get("text"), str) and item["text"]
                ]
                if text_parts:
                    result["content_text"] = "\n".join(text_parts)
                    result["content_length"] = len(result["content_text"])

    return result


def _extract_sender_id(content_text: Optional[str]) -> Optional[str]:
    if not content_text:
        return None
    match = re.search(r'"sender_id"\s*:\s*"([^"]+)"', content_text)
    return match.group(1) if match else None


# ─── JSONL Session Parsing ───

def _parse_jsonl_line(line: str, session_id: str, target_tz: ZoneInfo) -> Optional[dict]:
    trimmed = line.strip()
    if not trimmed:
        return None
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None

    record_type = parsed.get("type")
    timestamp = parsed.get("timestamp")
    if not record_type or not isinstance(timestamp, str) or not timestamp:
        return None

    timestamp = _convert_iso_timestamp(timestamp, target_tz)

    if record_type == "message":
        message_obj = parsed.get("message") or {}
        if not isinstance(message_obj, dict):
            message_obj = {}
    else:
        message_obj = {}

    wide_fields = _parse_message_fields(message_obj, record_type)
    sender_id = _extract_sender_id(wide_fields["content_text"]) if wide_fields["role"] == "user" else None

    return {
        "session_id": session_id,
        "type": record_type,
        "id": parsed.get("id"),
        "parent_id": parsed.get("parentId"),
        "timestamp": timestamp,
        "hostname": parsed.get("hostname") or socket.gethostname(),
        **wide_fields,
        "sender_id": sender_id,
    }


def _should_filter(record: dict, config: AppConfigCk) -> bool:
    filters = config.filters
    if filters.exclude_subsystems and record["type"] in filters.exclude_subsystems:
        return True
    if filters.include_subsystems and record["type"] not in filters.include_subsystems:
        return True
    return False


def _find_session_files() -> list[str]:
    home_dir = os.path.expanduser("~")
    pattern = os.path.join(home_dir, ".openclaw", "agents", "*", "sessions", "*.jsonl")
    files = glob.glob(pattern)
    return sorted(files)


async def _flush_session_batch(config: AppConfigCk, records: list[list[SqlValue]]) -> int:
    inserted = execute_batch_insert(config.ck, config.ck.session_table, SESSION_COLUMNS, records)
    print(f"[Collect-CK:Session] Batch inserted {inserted} records")
    return inserted


async def collect_sessions(config: AppConfigCk) -> int:
    """Collect session data from JSONL files under the OpenClaw agents directory."""
    print("[Collect-CK:Session] Scanning OpenClaw session files...")

    target_tz = _get_ck_timezone()
    session_files = _find_session_files()
    print(f"[Collect-CK:Session] Found {len(session_files)} session files")

    if not session_files:
        print("[Collect-CK:Session] No session files found, skipping")
        return 0

    state = _load_collect_state()
    total_inserted = 0

    for file_path in session_files:
        file_stat = os.stat(file_path)
        file_mtime_ms = file_stat.st_mtime * 1000
        file_state = state.get(file_path)

        if file_state and file_state.get("lastModified", 0) >= file_mtime_ms:
            continue

        print(f"[Collect-CK:Session] Processing: {os.path.basename(file_path)}")

        with open(file_path, "r", encoding="utf-8") as session_file:
            content = session_file.read()

        lines = content.split("\n")
        session_id = os.path.splitext(os.path.basename(file_path))[0]
        start_line = file_state["lastLineOffset"] if file_state else 0
        records: list[list[SqlValue]] = []

        for line_index in range(start_line, len(lines)):
            raw_line = lines[line_index]
            record = _parse_jsonl_line(raw_line, session_id, target_tz)
            if not record:
                continue
            if _should_filter(record, config):
                continue

            records.append([
                _next_row_id(),          # row_id (generated)
                record["session_id"],
                record["type"],
                record.get("id"),
                record.get("parent_id"),
                record["timestamp"],
                record["hostname"],
                record["sender_id"],
                raw_line.strip(),        # complete_session
                record["role"],
                record["model"],
                record["api"],
                record["provider"],
                record["stop_reason"],
                record["input_tokens"],
                record["output_tokens"],
                record["cache_read_tokens"],
                record["cache_write_tokens"],
                record["total_tokens"],
                record["total_cost"],
                record["tool_name"],
                record["tool_input"],
                record["tool_use_id"],
                record["is_error"],
                record["content_text"],
                record["content_length"],
                record["thinking_text"],
            ])

            if len(records) >= config.collection.batch_size:
                total_inserted += await _flush_session_batch(config, records)
                records.clear()

        if records:
            total_inserted += await _flush_session_batch(config, records)

        state[file_path] = {
            "lastLineOffset": len(lines),
            "lastModified": file_mtime_ms,
        }

    _save_collect_state(state)
    print(f"[Collect-CK:Session] Inserted {total_inserted} session records in this run")
    return total_inserted


# ─── Log File Parsing ───

def _find_log_files() -> list[str]:
    if not os.path.exists(LOG_DIRECTORY):
        return []
    entries = os.listdir(LOG_DIRECTORY)
    log_files = [
        os.path.join(LOG_DIRECTORY, entry)
        for entry in entries
        if LOG_FILE_PATTERN.match(entry)
    ]
    return sorted(log_files)


def _as_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _extract_timestamp(parsed: dict, meta: dict) -> Optional[str]:
    candidates = [
        parsed.get("time"), parsed.get("timestamp"),
        meta.get("date"), parsed.get("date"), meta.get("timestamp"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _extract_log_level(parsed: dict, meta: dict) -> str:
    candidates = [parsed.get("level"), meta.get("logLevelName"), meta.get("level")]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    level_id = meta.get("logLevelId")
    if isinstance(level_id, int):
        return LEVEL_MAP.get(level_id, "info")
    return "info"


def _extract_subsystem(parsed: dict, meta: dict) -> Optional[str]:
    field0 = parsed.get("0")
    if isinstance(field0, str) and field0.strip().startswith("{"):
        try:
            field0_obj = json.loads(field0)
            if isinstance(field0_obj.get("subsystem"), str):
                return field0_obj["subsystem"]
            if isinstance(field0_obj.get("module"), str):
                return field0_obj["module"]
        except json.JSONDecodeError:
            pass
    elif isinstance(field0, dict):
        if isinstance(field0.get("subsystem"), str):
            return field0["subsystem"]
        if isinstance(field0.get("module"), str):
            return field0["module"]

    if isinstance(parsed.get("subsystem"), str):
        return parsed["subsystem"]

    meta_name = meta.get("name")
    if isinstance(meta_name, str):
        if meta_name.startswith("{"):
            try:
                name_obj = json.loads(meta_name)
                if isinstance(name_obj.get("subsystem"), str):
                    return name_obj["subsystem"]
            except json.JSONDecodeError:
                pass
        return meta_name
    return None


def _parse_log_line(line: str, target_tz: ZoneInfo) -> Optional[dict]:
    trimmed = line.strip()
    if not trimmed:
        return None
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return None

    meta = parsed.get("_meta") or parsed.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    raw_timestamp = _extract_timestamp(parsed, meta)
    if not raw_timestamp:
        return None

    timestamp = _convert_iso_timestamp(raw_timestamp, target_tz)
    level = _extract_log_level(parsed, meta)
    meta_parent_names = meta.get("parentNames")

    return {
        "timestamp": timestamp,
        "level": level,
        "subsystem": _extract_subsystem(parsed, meta),
        "raw_field_0": _as_str_or_none(parsed.get("0")),
        "raw_field_1": _as_str_or_none(parsed.get("1")),
        "raw_field_2": _as_str_or_none(parsed.get("2")),
        "meta_runtime": _as_str_or_none(meta.get("runtime")),
        "meta_runtime_version": _as_str_or_none(meta.get("runtimeVersion")),
        "hostname": _as_str_or_none(parsed.get("hostname") or meta.get("hostname")) or socket.gethostname(),
        "meta_name": _as_str_or_none(meta.get("name")),
        "meta_parent_names": json.dumps(meta_parent_names) if meta_parent_names is not None else None,
        "meta_date": (
            _convert_iso_timestamp(meta["date"], target_tz)
            if isinstance(meta.get("date"), str) and meta["date"] else None
        ),
        "meta_log_level_id": meta.get("logLevelId") if isinstance(meta.get("logLevelId"), int) else None,
        "meta_log_level_name": _as_str_or_none(meta.get("logLevelName")),
        "meta_path": (
            meta["path"] if isinstance(meta.get("path"), str) else json.dumps(meta["path"])
        ) if meta.get("path") is not None else None,
    }


async def collect_log_files(config: AppConfigCk) -> int:
    """Collect log data from /tmp/openclaw/openclaw-YYYY-MM-DD.log files."""
    print(f"[Collect-CK:Log] Scanning log files in {LOG_DIRECTORY}...")

    target_tz = _get_ck_timezone()
    log_files = _find_log_files()
    print(f"[Collect-CK:Log] Found {len(log_files)} log files")

    if not log_files:
        print("[Collect-CK:Log] No log files found, skipping")
        return 0

    logs_table = config.ck.logs_table or "openclaw_logs"
    state = _load_collect_state()
    total_inserted = 0

    for file_path in log_files:
        file_stat = os.stat(file_path)
        file_mtime_ms = file_stat.st_mtime * 1000
        file_state = state.get(file_path)

        if file_state and file_state.get("lastModified", 0) >= file_mtime_ms:
            continue

        print(f"[Collect-CK:Log] Processing: {os.path.basename(file_path)}")

        with open(file_path, "r", encoding="utf-8") as log_file:
            content = log_file.read()

        lines = content.split("\n")
        start_line = file_state["lastLineOffset"] if file_state else 0
        records: list[list[SqlValue]] = []

        for line_index in range(start_line, len(lines)):
            raw_line = lines[line_index]
            log_record = _parse_log_line(raw_line, target_tz)
            if not log_record:
                continue

            records.append([
                _next_row_id(),          # row_id (generated)
                log_record["timestamp"],
                log_record["level"],
                log_record["subsystem"],
                log_record["raw_field_0"],
                log_record["raw_field_1"],
                log_record["raw_field_2"],
                log_record["meta_runtime"],
                log_record["meta_runtime_version"],
                log_record["hostname"],
                log_record["meta_name"],
                log_record["meta_parent_names"],
                log_record["meta_date"],
                log_record["meta_log_level_id"],
                log_record["meta_log_level_name"],
                log_record["meta_path"],
                raw_line.strip(),
            ])

            if len(records) >= config.collection.batch_size:
                inserted = execute_batch_insert(config.ck, logs_table, LOG_COLUMNS, records)
                total_inserted += inserted
                records.clear()
                print(f"[Collect-CK:Log] Batch inserted {inserted} records")

        if records:
            inserted = execute_batch_insert(config.ck, logs_table, LOG_COLUMNS, records)
            total_inserted += inserted
            print(f"[Collect-CK:Log] Batch inserted {inserted} records")

        state[file_path] = {
            "lastLineOffset": len(lines),
            "lastModified": file_mtime_ms,
        }

    _save_collect_state(state)
    print(f"[Collect-CK:Log] Inserted {total_inserted} log records in this run")
    return total_inserted


async def collect_logs(config: AppConfigCk) -> int:
    """Collect both session data and log data. Main entry point called by the scheduler."""
    total_inserted = 0
    total_inserted += await collect_sessions(config)
    total_inserted += await collect_log_files(config)
    return total_inserted


async def clean_expired_data(config: AppConfigCk) -> None:
    """
    Clean up expired data from both tables using ClickHouse mutations.

    ClickHouse does not support standard DELETE FROM.  ALTER TABLE ... DELETE
    is used instead (an asynchronous mutation — the actual deletion happens
    in the background; data becomes invisible to new queries quickly).
    """
    retention_days = config.collection.retention_days
    if retention_days <= 0:
        return

    cutoff_dt = datetime.now(_CK_TIMEZONE) - timedelta(days=retention_days)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

    session_sql = (
        f"ALTER TABLE `{config.ck.session_table}` "
        f"DELETE WHERE timestamp < '{cutoff_str}'"
    )
    print(f"[Cleanup-CK] Cleaning session data older than {retention_days} days (before {cutoff_str})...")
    execute_query(config.ck, session_sql)

    logs_table = config.ck.logs_table or "openclaw_logs"
    logs_sql = (
        f"ALTER TABLE `{logs_table}` "
        f"DELETE WHERE timestamp < '{cutoff_str}'"
    )
    print(f"[Cleanup-CK] Cleaning log data older than {retention_days} days...")
    execute_query(config.ck, logs_sql)

    print("[Cleanup-CK] Expired data cleanup mutation submitted")


# ─── Standalone entry point ───

if __name__ == "__main__":
    import asyncio
    from scripts.config_ck import load_config_ck
    from scripts.db_ck import close_connection_pool

    async def main() -> None:
        config = load_config_ck()
        await collect_logs(config)
        await clean_expired_data(config)
        close_connection_pool()

    asyncio.run(main())
