"""
Drilldown queries for deep-diving into analysis report metrics.

Provides two main capabilities:
1. User Task Drilldown: query a specific user's task count and complexity details
2. Non-Work Task Drilldown: find non-work intent tasks with full task chain,
   token consumption, and original user messages for manual review
"""

from __future__ import annotations

import json
from typing import Optional

from scripts.config import AdbConfig
from scripts.db import execute_query
from scripts.types import TimeRange, time_range_to_sql_params
from scripts.analysis.behavior_insight import _extract_user_message


# ─── Non-work intent categories (both Chinese and English labels) ───

NON_WORK_CATEGORIES = {
    "闲聊互动", "Casual Interaction",
    "安全测试", "Security Testing",
}


def drilldown_user_tasks(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    sender_id: str,
) -> dict:
    """Query a specific user's task chains with count and complexity breakdown.

    Returns:
        {
            "senderId": "363779",
            "timeRange": {"start": "...", "end": "..."},
            "totalTaskChains": 42,
            "complexityDistribution": {"low": 30, "medium": 8, "high": 3, "very_high": 1},
            "avgComplexityScore": 2.35,
            "totalTokens": 123456,
            "totalCost": 0.05,
            "taskChains": [ ... top 50 by complexity ... ]
        }
    """
    print(f"[Drilldown] Querying tasks for user '{sender_id}'...")

    start_time, end_time = time_range_to_sql_params(range_)

    sql = f"""
        WITH ordered_msgs AS (
            SELECT
                row_id, session_id, role, sender_id, stop_reason,
                tool_name, total_tokens, total_cost, thinking_text,
                content_text, is_error, input_tokens, output_tokens,
                timestamp,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                    OVER (PARTITION BY session_id ORDER BY timestamp, row_id) AS task_chain_id
            FROM `{table_name}`
            WHERE timestamp >= %s AND timestamp < %s
        ),
        chain_sender AS (
            SELECT session_id, task_chain_id,
                MIN(CASE WHEN role = 'user' THEN sender_id END) AS sender_id
            FROM ordered_msgs
            GROUP BY session_id, task_chain_id
        ),
        chain_metrics AS (
            SELECT
                o.session_id,
                o.task_chain_id,
                cs.sender_id,
                COUNT(DISTINCT CASE WHEN o.role = 'user' THEN o.row_id END) AS user_turns,
                COUNT(*) AS total_messages,
                SUM(CASE WHEN o.tool_name IS NOT NULL THEN 1 ELSE 0 END) AS tool_call_count,
                SUM(CASE WHEN o.is_error = 1 THEN 1 ELSE 0 END) AS error_count,
                SUM(CASE WHEN o.role = 'assistant'
                    THEN LENGTH(COALESCE(o.thinking_text, '')) ELSE 0 END) AS thinking_length,
                SUM(COALESCE(o.total_tokens, 0)) AS total_tokens,
                SUM(COALESCE(o.input_tokens, 0)) AS input_tokens,
                SUM(COALESCE(o.output_tokens, 0)) AS output_tokens,
                SUM(COALESCE(o.total_cost, 0)) AS total_cost,
                MIN(o.timestamp) AS start_time,
                MAX(o.timestamp) AS end_time,
                TIMESTAMPDIFF(SECOND, MIN(o.timestamp), MAX(o.timestamp)) AS duration_seconds,
                MAX(CASE WHEN o.stop_reason IN ('stop', 'end_turn') THEN 1 ELSE 0 END) AS has_normal_stop,
                MAX(CASE WHEN o.stop_reason IN ('error', 'aborted', 'cancelled', 'timeout', 'content_filter')
                    THEN 1 ELSE 0 END) AS has_abnormal_stop
            FROM ordered_msgs o
            JOIN chain_sender cs
                ON o.session_id = cs.session_id AND o.task_chain_id = cs.task_chain_id
            WHERE cs.sender_id = %s
            GROUP BY o.session_id, o.task_chain_id, cs.sender_id
        )
        SELECT *,
            ROUND(
                (user_turns * 2 + tool_call_count * 1.5 + thinking_length / 1000 + total_tokens / 10000) / 4,
                2
            ) AS complexity_score,
            CASE
                WHEN has_normal_stop = 1 AND error_count = 0 AND has_abnormal_stop = 0 THEN 'success'
                WHEN has_normal_stop = 1 AND error_count > 0 THEN 'partial'
                WHEN has_abnormal_stop = 1 THEN 'failure'
                ELSE 'failure'
            END AS outcome
        FROM chain_metrics
        ORDER BY complexity_score DESC
    """

    rows = execute_query(adb_config, sql, (start_time, end_time, sender_id))

    distribution = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
    total_tokens_sum = 0
    total_cost_sum = 0.0
    all_scores = []
    task_chains = []

    for row in rows:
        score = float(row.get("complexity_score") or 0)
        all_scores.append(score)
        total_tokens_sum += int(row.get("total_tokens") or 0)
        total_cost_sum += float(row.get("total_cost") or 0)

        if score < 2:
            distribution["low"] += 1
        elif score < 5:
            distribution["medium"] += 1
        elif score < 10:
            distribution["high"] += 1
        else:
            distribution["very_high"] += 1

        if len(task_chains) < 50:
            task_chains.append({
                "sessionId": row["session_id"],
                "taskChainId": row.get("task_chain_id") or 0,
                "complexityScore": score,
                "complexityLevel": _score_to_level(score),
                "outcome": row.get("outcome", "unknown"),
                "userTurns": row.get("user_turns") or 0,
                "totalMessages": row.get("total_messages") or 0,
                "toolCallCount": row.get("tool_call_count") or 0,
                "errorCount": row.get("error_count") or 0,
                "totalTokens": row.get("total_tokens") or 0,
                "inputTokens": row.get("input_tokens") or 0,
                "outputTokens": row.get("output_tokens") or 0,
                "totalCost": float(row.get("total_cost") or 0),
                "durationSeconds": row.get("duration_seconds") or 0,
                "startTime": str(row.get("start_time") or ""),
                "endTime": str(row.get("end_time") or ""),
            })

    average_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0

    result = {
        "senderId": sender_id,
        "timeRange": {"start": start_time, "end": end_time},
        "totalTaskChains": len(rows),
        "complexityDistribution": distribution,
        "avgComplexityScore": average_score,
        "totalTokens": total_tokens_sum,
        "totalCost": round(total_cost_sum, 6),
        "taskChains": task_chains,
    }

    print(f"[Drilldown] User '{sender_id}': {len(rows)} task chains, "
          f"avg complexity {average_score}, total tokens {total_tokens_sum:,}")
    return result


def drilldown_non_work_tasks(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    run_id: Optional[str] = None,
    categories: Optional[set[str]] = None,
) -> dict:
    """Find non-work intent tasks from L2-1 analysis results.

    Looks up the latest L2-1 (intent classification) result from the analysis
    results table, filters for non-work categories, then queries the full task
    chain details including original user messages and token consumption.

    Args:
        adb_config: Database configuration
        table_name: Session table name
        range_: Time range for the query
        run_id: Specific analysis run ID to use. If None, uses the latest.
        categories: Custom set of non-work categories. Defaults to NON_WORK_CATEGORIES.

    Returns:
        {
            "timeRange": {"start": "...", "end": "..."},
            "nonWorkCategories": ["闲聊互动", ...],
            "totalNonWorkTasks": 5,
            "tasks": [
                {
                    "category": "闲聊互动",
                    "senderId": "363779",
                    "sessionId": "abc-123",
                    "taskChainId": 2,
                    "userMessage": "今天天气怎么样？",
                    "userMessageFull": "今天天气怎么样？...",
                    "totalTokens": 1234,
                    "totalCost": 0.001,
                    "taskChainMessages": [ ... full chain ... ]
                }
            ]
        }
    """
    print("[Drilldown] Searching for non-work tasks...")

    target_categories = categories or NON_WORK_CATEGORIES
    start_time, end_time = time_range_to_sql_params(range_)

    # Step 1: Load L2-1 intent classification results from DB
    intent_items = _load_intent_items(adb_config, run_id)
    if not intent_items:
        print("[Drilldown] No L2-1 intent classification results found. "
              "Please run analysis first.")
        return {
            "timeRange": {"start": start_time, "end": end_time},
            "nonWorkCategories": sorted(target_categories),
            "totalNonWorkTasks": 0,
            "tasks": [],
        }

    # Step 2: Filter for non-work categories
    non_work_items = [
        item for item in intent_items
        if item.get("category") in target_categories
    ]

    if not non_work_items:
        print("[Drilldown] No non-work tasks found in the analysis results.")
        return {
            "timeRange": {"start": start_time, "end": end_time},
            "nonWorkCategories": sorted(target_categories),
            "totalNonWorkTasks": 0,
            "tasks": [],
        }

    print(f"[Drilldown] Found {len(non_work_items)} non-work messages, "
          f"fetching full task chain details...")

    # Step 3: For each non-work message, fetch the full task chain
    tasks = []
    for item in non_work_items:
        session_id = item.get("sessionId")
        row_id = item.get("rowId")
        if not session_id or not row_id:
            continue

        chain_detail = _fetch_task_chain_for_message(
            adb_config, table_name, session_id, row_id, start_time, end_time,
        )
        if chain_detail:
            chain_detail["category"] = item.get("category", "unknown")
            chain_detail["confidence"] = item.get("confidence", 0)
            chain_detail["senderId"] = item.get("senderId", "unknown")
            tasks.append(chain_detail)

    # Sort by total tokens descending (highest cost first)
    tasks.sort(key=lambda task: task.get("totalTokens", 0), reverse=True)

    result = {
        "timeRange": {"start": start_time, "end": end_time},
        "nonWorkCategories": sorted(target_categories),
        "totalNonWorkTasks": len(tasks),
        "tasks": tasks,
    }

    print(f"[Drilldown] Found {len(tasks)} non-work task chains with full details")
    return result


def _load_intent_items(
    adb_config: AdbConfig,
    run_id: Optional[str] = None,
) -> list[dict]:
    """Load L2-1 intent classification items from the analysis results table."""
    from scripts.analysis.orchestrator import RESULTS_TABLE

    if run_id:
        sql = f"""
            SELECT details
            FROM `{RESULTS_TABLE}`
            WHERE run_id = %s AND case_name = 'L2-1' AND status = 'success'
            LIMIT 1
        """
        rows = execute_query(adb_config, sql, (run_id,))
    else:
        sql = f"""
            SELECT details
            FROM `{RESULTS_TABLE}`
            WHERE case_name = 'L2-1' AND status = 'success'
            ORDER BY created_at DESC
            LIMIT 1
        """
        rows = execute_query(adb_config, sql)

    if not rows:
        return []

    details_raw = rows[0].get("details", "{}")
    details = json.loads(details_raw) if isinstance(details_raw, str) else details_raw
    return details.get("items", [])


def _fetch_task_chain_for_message(
    adb_config: AdbConfig,
    table_name: str,
    session_id: str,
    user_row_id: int,
    start_time: str,
    end_time: str,
) -> Optional[dict]:
    """Fetch the complete task chain that contains a specific user message.

    Uses the task chain segmentation pattern: find which chain the user_row_id
    belongs to, then return all messages in that chain.
    """
    sql = f"""
        WITH ordered_msgs AS (
            SELECT
                row_id, session_id, role, sender_id, stop_reason,
                tool_name, tool_input, content_text, thinking_text,
                input_tokens, output_tokens, total_tokens, total_cost,
                is_error, timestamp,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                    OVER (PARTITION BY session_id ORDER BY timestamp, row_id) AS task_chain_id
            FROM `{table_name}`
            WHERE session_id = %s
              AND timestamp >= %s AND timestamp < %s
        ),
        target_chain AS (
            SELECT task_chain_id
            FROM ordered_msgs
            WHERE row_id = %s
            LIMIT 1
        )
        SELECT o.*
        FROM ordered_msgs o
        JOIN target_chain t ON o.task_chain_id = t.task_chain_id
        ORDER BY o.timestamp, o.row_id
    """

    rows = execute_query(adb_config, sql, (session_id, start_time, end_time, user_row_id))

    if not rows:
        return None

    # Build the task chain detail
    chain_messages = []
    total_tokens = 0
    total_cost = 0.0
    user_message_full = ""
    user_message_preview = ""

    for row in rows:
        role = row.get("role", "")
        content_text = row.get("content_text") or ""

        # Extract the original user message (strip metadata)
        if role == "user" and row.get("row_id") == user_row_id:
            extracted = _extract_user_message(content_text)
            user_message_full = extracted
            user_message_preview = extracted[:200]

        total_tokens += int(row.get("total_tokens") or 0)
        total_cost += float(row.get("total_cost") or 0)

        message_entry = {
            "rowId": row.get("row_id"),
            "role": role,
            "timestamp": str(row.get("timestamp") or ""),
            "inputTokens": row.get("input_tokens") or 0,
            "outputTokens": row.get("output_tokens") or 0,
            "totalTokens": row.get("total_tokens") or 0,
            "stopReason": row.get("stop_reason"),
            "isError": bool(row.get("is_error")),
        }

        if role == "user":
            extracted = _extract_user_message(content_text)
            message_entry["contentPreview"] = extracted[:500]
            message_entry["contentFull"] = extracted
        elif role == "assistant":
            message_entry["contentPreview"] = (content_text or "")[:200]
            if row.get("thinking_text"):
                message_entry["thinkingPreview"] = row["thinking_text"][:200]
        elif role == "tool":
            message_entry["toolName"] = row.get("tool_name")
            message_entry["contentPreview"] = (content_text or "")[:200]

        chain_messages.append(message_entry)

    return {
        "sessionId": session_id,
        "taskChainId": rows[0].get("task_chain_id") or 0,
        "userMessage": user_message_preview,
        "userMessageFull": user_message_full,
        "totalTokens": total_tokens,
        "totalCost": round(total_cost, 6),
        "messageCount": len(chain_messages),
        "taskChainMessages": chain_messages,
    }


def _score_to_level(score: float) -> str:
    """Convert a complexity score to a human-readable level."""
    if score < 2:
        return "low"
    elif score < 5:
        return "medium"
    elif score < 10:
        return "high"
    return "very_high"


def format_user_tasks_report(result: dict) -> str:
    """Format user task drilldown result as a readable Markdown report."""
    lines = []
    lines.append(f"## 用户任务下钻报告")
    lines.append("")
    lines.append(f"- **用户 ID**: `{result['senderId']}`")
    lines.append(f"- **时间范围**: {result['timeRange']['start']} ~ {result['timeRange']['end']}")
    lines.append(f"- **任务链总数**: {result['totalTaskChains']}")
    lines.append(f"- **平均复杂度**: {result['avgComplexityScore']}")
    lines.append(f"- **总 Token 消耗**: {result['totalTokens']:,}")
    lines.append(f"- **总成本**: {result['totalCost']}")
    lines.append("")

    dist = result.get("complexityDistribution", {})
    lines.append("### 复杂度分布")
    lines.append("")
    lines.append("| 级别 | 数量 | 占比 |")
    lines.append("| --- | --- | --- |")
    total = sum(dist.values()) or 1
    for level in ("low", "medium", "high", "very_high"):
        count = dist.get(level, 0)
        pct = count / total * 100
        label = {"low": "低", "medium": "中", "high": "高", "very_high": "极高"}[level]
        lines.append(f"| {label} ({level}) | {count} | {pct:.1f}% |")
    lines.append("")

    chains = result.get("taskChains", [])
    if chains:
        lines.append(f"### 任务链详情（Top {len(chains)}，按复杂度排序）")
        lines.append("")
        lines.append("| # | 会话 ID | 链 ID | 复杂度 | 级别 | 结果 | 消息数 | 工具调用 | 错误 | Token | 时长(s) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for index, chain in enumerate(chains, 1):
            session_short = chain["sessionId"][:16] + "..."
            lines.append(
                f"| {index} | {session_short} | {chain['taskChainId']} "
                f"| {chain['complexityScore']} | {chain['complexityLevel']} "
                f"| {chain['outcome']} | {chain['totalMessages']} "
                f"| {chain['toolCallCount']} | {chain['errorCount']} "
                f"| {chain['totalTokens']:,} | {chain['durationSeconds']} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_non_work_tasks_report(result: dict) -> str:
    """Format non-work task drilldown result as a readable Markdown report."""
    lines = []
    lines.append("## 非工作任务下钻报告")
    lines.append("")
    lines.append(f"- **时间范围**: {result['timeRange']['start']} ~ {result['timeRange']['end']}")
    lines.append(f"- **检测类别**: {', '.join(result['nonWorkCategories'])}")
    lines.append(f"- **非工作任务总数**: {result['totalNonWorkTasks']}")
    lines.append("")

    tasks = result.get("tasks", [])
    if not tasks:
        lines.append("> ✅ 未发现非工作任务。")
        return "\n".join(lines)

    for index, task in enumerate(tasks, 1):
        lines.append(f"### 任务 {index}: {task.get('category', '未知')}")
        lines.append("")
        lines.append(f"- **用户**: `{task.get('senderId', '?')}`")
        lines.append(f"- **会话 ID**: `{task.get('sessionId', '?')}`")
        lines.append(f"- **任务链 ID**: {task.get('taskChainId', '?')}")
        lines.append(f"- **分类置信度**: {task.get('confidence', 0):.2f}")
        lines.append(f"- **总 Token**: {task.get('totalTokens', 0):,}")
        lines.append(f"- **总成本**: {task.get('totalCost', 0)}")
        lines.append(f"- **消息数**: {task.get('messageCount', 0)}")
        lines.append("")
        lines.append("**用户原始问题/指令（完整内容）：**")
        lines.append("")
        lines.append("```")
        lines.append(task.get("userMessageFull", "(无内容)"))
        lines.append("```")
        lines.append("")

        chain_messages = task.get("taskChainMessages", [])
        if chain_messages:
            lines.append(f"**任务链详情（共 {len(chain_messages)} 条消息）：**")
            lines.append("")
            lines.append("| # | 角色 | 时间 | Token | 停止原因 | 错误 | 内容预览 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for message_index, message in enumerate(chain_messages, 1):
                role = message.get("role", "?")
                timestamp = message.get("timestamp", "")[:19]
                tokens = message.get("totalTokens", 0)
                stop_reason = message.get("stopReason") or "-"
                is_error = "⚠️" if message.get("isError") else "-"
                preview = message.get("contentPreview", "")[:80].replace("\n", " ").replace("|", "\\|")

                if role == "tool":
                    tool_name = message.get("toolName") or ""
                    preview = f"[{tool_name}] {preview}"

                lines.append(
                    f"| {message_index} | {role} | {timestamp} "
                    f"| {tokens:,} | {stop_reason} | {is_error} | {preview} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
