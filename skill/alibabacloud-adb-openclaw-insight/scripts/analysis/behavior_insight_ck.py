"""
L2: User Behavior Analysis Layer (ClickHouse version).
Combines SQL queries with LLM-based semantic understanding.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from scripts.config_ck import CkConfig
from scripts.db_ck import execute_query_async
from scripts.llm_client import LlmClient
from scripts.types import TimeRange, time_range_to_sql_params


# ─── L2-1: Intent Classification ───

_INTENT_SYSTEM_PROMPT = """You are classifying user intents in AI-assisted conversations.
The user messages may be in Chinese or English — you must handle both languages.

Classify each message into one of the following categories. Each category has a Chinese label and an English label — use whichever language matches the dominant language of the user message:

| Chinese Label | English Label | Description |
|---|---|---|
| 代码开发 | Code Development | Writing new code, implementing features, creating functions/classes/modules |
| 问题修复 | Bug Fixing | Debugging, fixing bugs, resolving errors, troubleshooting |
| 代码优化 | Code Optimization | Refactoring, performance optimization, code review, improving code quality |
| 测试验证 | Testing & Validation | Writing tests, running tests, verifying results, validation |
| 架构设计 | Architecture Design | System design, architecture decisions, technical planning, schema design |
| 知识问答 | Knowledge Q&A | Asking questions about concepts, seeking explanations, learning |
| 配置部署 | Configuration & Deployment | DevOps, CI/CD, environment setup, configuration, installation |
| 数据分析 | Data Analysis | Data processing, analysis, visualization, statistics, mathematical computation |
| 信息检索 | Information Retrieval | Searching for information, looking up documentation, web search |
| 内容生成 | Content Generation | Writing documents, generating text content, summarization, translation |
| 任务管理 | Task Management | Task assignment, progress tracking, project coordination, workflow management |
| 工具使用 | Tool Usage | Using specific tools, plugins, skills, integrations |
| 闲聊互动 | Casual Interaction | Casual conversation, greetings, social interaction, role-play, relationship setting |
| 安全测试 | Security Testing | Security testing, penetration testing, vulnerability probing, prompt injection attempts |
| 多媒体处理 | Multimedia Processing | Image recognition, file operations, audio/video processing |

Language detection rules:
- If the user message is predominantly Chinese, use the Chinese label (e.g. "代码开发")
- If the user message is predominantly English, use the English label (e.g. "Code Development")
- If the message is mixed, use the label of the language that makes up the majority

If a message truly does not fit any category above, use "其他" (Chinese) or "Other" (English) — but this should be rare.
If multiple intents exist in one message, pick the primary/dominant one.

Return a JSON array where each element has:
- category: the label in the appropriate language as described above
- confidence: number 0-1"""


def _extract_user_message(raw_content: str) -> str:
    """Extract the actual user message from content_text that may contain metadata headers.

    The raw content may look like:
        Conversation info (untrusted metadata):
        ```json
        { ... }
        ```

        Sender (untrusted metadata):
        ```json
        { ... }
        ```

        <actual user message here>

    This function strips all metadata blocks and returns only the real user message.
    """
    if not raw_content:
        return ""

    # Strategy: find the last closing ``` of metadata blocks, take everything after it
    # Metadata blocks follow the pattern: <label>:\n```json\n{...}\n```
    last_metadata_end = -1
    search_pos = 0
    while True:
        # Find a metadata header pattern like "Conversation info" or "Sender"
        header_idx = raw_content.find("(untrusted metadata):", search_pos)
        if header_idx == -1:
            break
        # Find the closing ``` after this header
        code_block_start = raw_content.find("```", header_idx)
        if code_block_start == -1:
            break
        code_block_end = raw_content.find("```", code_block_start + 3)
        if code_block_end == -1:
            break
        last_metadata_end = code_block_end + 3
        search_pos = last_metadata_end

    if last_metadata_end > 0:
        extracted = raw_content[last_metadata_end:].strip()
        return extracted if extracted else raw_content.strip()

    return raw_content.strip()


async def classify_intents(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
    max_items: int = 500,
) -> dict:
    print("[L2-1] Starting intent classification...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            SELECT row_id, session_id, sender_id, content_text
            FROM `{table_name}`
            WHERE role = 'user'
              AND timestamp >= %s AND timestamp < %s
              AND content_text IS NOT NULL AND content_text != ''
              AND sender_id IS NOT NULL AND sender_id != ''
            ORDER BY session_id, timestamp
            LIMIT %s
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time, max_items))

        if not rows:
            return {"distribution": {}, "byUser": {}, "items": []}

        # Extract actual user messages by stripping metadata headers
        for row in rows:
            row["message_text"] = _extract_user_message(row["content_text"])

        # Filter out rows where extracted message is empty
        rows = [row for row in rows if row["message_text"]]

        if not rows:
            print("[L2-1] No valid user messages after metadata extraction")
            return {"distribution": {}, "byUser": {}, "items": []}

        messages = [row["message_text"] for row in rows]

        total_chars = sum(len(msg) for msg in messages)
        estimated_tokens = int(total_chars * 1.5)
        max_single_batch_tokens = 128_000

        if estimated_tokens < max_single_batch_tokens:
            batch_size = len(messages)
            print(f"[L2-1] Estimated {estimated_tokens} tokens (< 128K), sending all in one batch")
        else:
            batch_size = 15
            print(f"[L2-1] Estimated {estimated_tokens} tokens (>= 128K), splitting into batches of {batch_size}")

        def build_user_prompt(batch: list[str], start_index: int) -> str:
            numbered = "\n\n".join(f"{start_index + i + 1}. {msg}" for i, msg in enumerate(batch))
            return f"请对以下用户消息进行意图分类：\n\n{numbered}\n\n返回 JSON 数组，每个元素包含 category 和 confidence。"

        print(f"[L2-1] Queried {len(rows)} user messages ({total_chars} chars), sending to LLM for classification...")
        results = await llm_client.batch_classify(
            messages, _INTENT_SYSTEM_PROMPT, batch_size, build_user_prompt, label="L2-1:intent"
        )

        distribution: dict[str, int] = {}
        by_user: dict[str, dict[str, int]] = {}
        senders_by_category: dict[str, set] = {}
        items: list[dict] = []

        for i, row in enumerate(rows):
            result = results[i] if i < len(results) else {"category": "未分类", "confidence": 0}
            category = result.get("category", "未分类")
            confidence = float(result.get("confidence", 0))
            sender_id = row["sender_id"] or "unknown"

            distribution[category] = distribution.get(category, 0) + 1

            if sender_id not in by_user:
                by_user[sender_id] = {}
            by_user[sender_id][category] = by_user[sender_id].get(category, 0) + 1

            # Senders per category (deduplicated)
            if category not in senders_by_category:
                senders_by_category[category] = set()
            senders_by_category[category].add(sender_id)

            items.append({
                "rowId": row["row_id"],
                "sessionId": row["session_id"],
                "senderId": sender_id,
                "category": category,
                "confidence": confidence,
                "preview": row["message_text"][:100],
            })

        print(f"[L2-1] Classified {len(rows)} messages into {len(distribution)} intent categories")
        return {
            "distribution": distribution,
            "byUser": by_user,
            "sendersByCategory": {cat: sorted(sids) for cat, sids in senders_by_category.items()},
            "items": items,
        }

    except Exception as error:
        print(f"[L2-1] Intent classification failed: {error}")
        return {"distribution": {}, "byUser": {}, "items": []}


# ─── L2-2: Task Complexity Assessment ───

async def assess_task_complexity(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
) -> dict:
    """Assess task complexity at the task chain level (ClickHouse version).

    Key SQL adaptations from MySQL:
    - MIN(CASE WHEN role='user' THEN sender_id END)  →  anyIf(sender_id, role='user')
    - COUNT(DISTINCT CASE WHEN role='user' THEN row_id END)  →  countIf(role='user')
    - SUM(CASE WHEN tool_name IS NOT NULL THEN 1 ELSE 0 END)  →  countIf(isNotNull(tool_name))
    - sumIf replaces SUM(CASE WHEN ... THEN val ELSE 0 END)
    - TIMESTAMPDIFF(MINUTE,...)  →  dateDiff('minute',...)
    """
    print("[L2-2] Starting task complexity assessment...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            WITH ordered_msgs AS (
                SELECT
                    row_id, session_id, role, sender_id, stop_reason,
                    tool_name, total_tokens, thinking_text, timestamp,
                    SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                        OVER (PARTITION BY session_id ORDER BY timestamp, row_id) AS task_chain_id
                FROM `{table_name}`
                WHERE timestamp >= %s AND timestamp < %s
            ),
            chain_sender AS (
                SELECT session_id, task_chain_id,
                    anyIf(sender_id, role = 'user') AS sender_id
                FROM ordered_msgs
                GROUP BY session_id, task_chain_id
            ),
            chain_metrics AS (
                SELECT
                    o.session_id,
                    o.task_chain_id,
                    cs.sender_id,
                    countIf(o.role = 'user') AS user_turns,
                    countIf(isNotNull(o.tool_name)) AS tool_call_count,
                    sumIf(length(ifNull(o.thinking_text, '')), o.role = 'assistant') AS thinking_length,
                    sum(ifNull(o.total_tokens, 0)) AS total_tokens,
                    dateDiff('minute', min(o.timestamp), max(o.timestamp)) AS duration_minutes
                FROM ordered_msgs o
                JOIN chain_sender cs
                    ON o.session_id = cs.session_id AND o.task_chain_id = cs.task_chain_id
                GROUP BY o.session_id, o.task_chain_id, cs.sender_id
            )
            SELECT *,
                round(
                    (user_turns * 2 + tool_call_count * 1.5 + thinking_length / 1000 + total_tokens / 10000) / 4,
                    2
                ) AS complexity_score
            FROM chain_metrics
            ORDER BY complexity_score DESC
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time))

        distribution: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        by_user_scores: dict[str, list[float]] = {}
        top_complex = []

        for row in rows:
            score = float(row.get("complexity_score") or 0)
            sender_id = row.get("sender_id") or "unknown"

            if score < 2:
                distribution["low"] += 1
            elif score < 5:
                distribution["medium"] += 1
            elif score < 10:
                distribution["high"] += 1
            else:
                distribution["very_high"] += 1

            if sender_id not in by_user_scores:
                by_user_scores[sender_id] = []
            by_user_scores[sender_id].append(score)

            if len(top_complex) < 20:
                top_complex.append({
                    "sessionId": row["session_id"],
                    "taskChainId": row.get("task_chain_id") or 0,
                    "senderId": sender_id,
                    "complexityScore": score,
                    "userTurns": row.get("user_turns") or 0,
                    "toolCallCount": row.get("tool_call_count") or 0,
                    "thinkingLength": row.get("thinking_length") or 0,
                    "durationMinutes": row.get("duration_minutes") or 0,
                })

        by_user = [
            {"senderId": sender_id, "avgComplexity": round(sum(scores) / len(scores), 2)}
            for sender_id, scores in by_user_scores.items()
        ]
        by_user.sort(key=lambda x: x["avgComplexity"], reverse=True)

        print(f"[L2-2] Assessed complexity for {len(rows)} task chains")
        return {"distribution": distribution, "byUser": by_user, "topComplex": top_complex}

    except Exception as error:
        print(f"[L2-2] Task complexity assessment failed: {error}")
        return {"distribution": {}, "byUser": [], "topComplex": []}


# ─── L2-3: Task Success Rate Estimation ───

async def estimate_task_success_rate(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
) -> dict:
    """Estimate task success rate at the task chain level (ClickHouse version).

    Key SQL adaptations from MySQL:
    - anyIf replaces MIN(CASE WHEN role='user' THEN sender_id END)
    - countIf replaces SUM(CASE WHEN ... THEN 1 ELSE 0 END)
    - max(if(..., 1, 0)) replaces MAX(CASE WHEN ... THEN 1 ELSE 0 END)
    - CASE/WHEN for outcome/score is kept (CK supports it)
    """
    print("[L2-3] Starting task success rate estimation...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            WITH ordered_msgs AS (
                SELECT
                    row_id, session_id, role, sender_id, stop_reason,
                    tool_name, is_error, content_length, timestamp,
                    SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                        OVER (PARTITION BY session_id ORDER BY timestamp, row_id) AS task_chain_id
                FROM `{table_name}`
                WHERE timestamp >= %s AND timestamp < %s
            ),
            chain_sender AS (
                SELECT session_id, task_chain_id,
                    anyIf(sender_id, role = 'user') AS sender_id
                FROM ordered_msgs
                GROUP BY session_id, task_chain_id
            ),
            chain_metrics AS (
                SELECT
                    o.session_id,
                    o.task_chain_id,
                    cs.sender_id,
                    min(o.row_id) AS start_row_id,
                    max(o.row_id) AS end_row_id,
                    countIf(o.role = 'user') AS user_turns,
                    countIf(isNotNull(o.tool_name)) AS tool_count,
                    countIf(o.is_error = 1) AS error_count,
                    max(if(o.stop_reason IN ('stop', 'end_turn'), 1, 0)) AS has_normal_stop,
                    max(if(o.stop_reason = 'max_tokens', 1, 0)) AS has_truncation,
                    max(if(o.stop_reason IN ('error', 'aborted', 'cancelled', 'timeout', 'content_filter'), 1, 0)) AS has_abnormal_stop
                FROM ordered_msgs o
                JOIN chain_sender cs
                    ON o.session_id = cs.session_id AND o.task_chain_id = cs.task_chain_id
                GROUP BY o.session_id, o.task_chain_id, cs.sender_id
            )
            SELECT *,
                CASE
                    WHEN has_normal_stop = 1 AND error_count = 0 AND has_abnormal_stop = 0 THEN 'success'
                    WHEN has_normal_stop = 1 AND (error_count > 0 OR has_truncation = 1) THEN 'partial'
                    WHEN has_abnormal_stop = 1 THEN 'failure'
                    WHEN has_truncation = 1 THEN 'partial'
                    ELSE 'failure'
                END AS outcome,
                CASE
                    WHEN has_normal_stop = 1 AND error_count = 0 AND has_abnormal_stop = 0 THEN 1.0
                    WHEN has_normal_stop = 1 AND (error_count > 0 OR has_truncation = 1) THEN 0.6
                    WHEN has_truncation = 1 THEN 0.4
                    ELSE 0.0
                END AS score
            FROM chain_metrics
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time))

        overall: dict[str, int] = {"success": 0, "partial": 0, "failure": 0}
        by_user: dict[str, dict[str, int]] = {}
        items = []
        known_failures: list[dict] = []   # failures/partials with a real sender_id
        unknown_failure_count = 0          # failures/partials without sender_id

        for row in rows:
            outcome = row.get("outcome", "failure")
            raw_sender = (row.get("sender_id") or "").strip()
            sender_id = raw_sender if raw_sender else "unknown"
            score = float(row.get("score") or 0)

            overall[outcome] = overall.get(outcome, 0) + 1
            if sender_id not in by_user:
                by_user[sender_id] = {"success": 0, "partial": 0, "failure": 0}
            by_user[sender_id][outcome] = by_user[sender_id].get(outcome, 0) + 1

            item = {
                "sessionId": row["session_id"],
                "taskChainId": row.get("task_chain_id") or 1,
                "senderId": sender_id,
                "startRowId": row.get("start_row_id"),
                "endRowId": row.get("end_row_id"),
                "score": score,
                "outcome": outcome,
            }
            items.append(item)

            if outcome in ("failure", "partial"):
                if raw_sender:
                    known_failures.append(item)
                else:
                    unknown_failure_count += 1

        # Keep at most 5 representative known-user failures to reduce report payload
        failures: list[dict] = known_failures[:5]
        if unknown_failure_count > 0:
            failures.append({
                "type": "unknown",
                "count": unknown_failure_count,
                "note": f"任务类型未知（{unknown_failure_count} 条无 sender_id 的失败/部分失败任务）",
            })

        total_failures = len(known_failures) + unknown_failure_count
        print(f"[L2-3] Estimated success rate for {len(rows)} task chains "
              f"({total_failures} failures/partial: {len(known_failures)} known, {unknown_failure_count} unknown)")
        return {"overall": overall, "byUser": by_user, "items": items, "failures": failures}

    except Exception as error:
        print(f"[L2-3] Task success rate estimation failed: {error}")
        return {"overall": {"success": 0, "partial": 0, "failure": 0}, "byUser": {}, "items": [], "failures": []}


# ─── L2-4: Prompt Quality Scoring ───

_PROMPT_QUALITY_SYSTEM_PROMPT = """You are an expert prompt engineer evaluating the quality of user prompts sent to an AI coding assistant.

Rate each prompt on 6 dimensions using a 1-5 scale (1 = absent/poor, 5 = excellent).
These dimensions are derived from Google Cloud's "Strategies for writing better prompts":

1. **goal_clarity** — Does the prompt set a clear goal or objective?
   - 1: No discernible goal; vague or ambiguous request.
   - 3: Goal is implied but not explicitly stated.
   - 5: Goal is explicitly stated with a clear desired outcome or deliverable.

2. **context_provided** — Does the prompt provide context and background information?
   - 1: No context at all; the request is completely bare.
   - 3: Some context is given (e.g., mentions a file or technology) but lacks important details.
   - 5: Rich context including project background, relevant file paths, error messages, constraints, or environment details.

3. **chain_of_thought** — Does the prompt encourage or demonstrate chain-of-thought reasoning?
   - 1: No reasoning structure; single imperative sentence.
   - 3: Implicitly suggests a multi-step approach (e.g., "first do X, then Y").
   - 5: Explicitly breaks down the problem into logical steps, asks the AI to reason through the problem, or provides a structured thinking framework.

4. **few_shot_examples** — Does the prompt include examples (few-shot) to guide the expected output?
   - 1: No examples provided.
   - 3: One vague example or a reference to an existing pattern.
   - 5: Multiple clear input-output examples or references to concrete code samples that demonstrate the expected behavior.

5. **iteration_signals** — Does the prompt show signs of iterative refinement or experimentation?
   - 1: First attempt with no refinement signals.
   - 3: References a previous attempt or asks for an alternative approach.
   - 5: Explicitly iterates on a prior result, provides feedback on what worked/didn't, or requests comparison of multiple approaches.

6. **specificity** — Is the prompt specific and detailed in its requirements?
   - 1: Extremely vague (e.g., "help me with code").
   - 3: Moderately specific (e.g., mentions a function name or file but lacks detailed requirements).
   - 5: Highly specific with exact file paths, function signatures, expected behavior, edge cases, or acceptance criteria.

Return a JSON array where each element has:
- goal_clarity: number 1-5
- context_provided: number 1-5
- chain_of_thought: number 1-5
- few_shot_examples: number 1-5
- iteration_signals: number 1-5
- specificity: number 1-5"""

_SCORING_DIMENSIONS = [
    "goal_clarity", "context_provided", "chain_of_thought",
    "few_shot_examples", "iteration_signals", "specificity",
]

_DEFAULT_SCORES = {dim: 1 for dim in _SCORING_DIMENSIONS}


async def score_prompt_quality(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    print("[L2-4] Starting prompt quality scoring...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            SELECT row_id, session_id, sender_id, content_text
            FROM `{table_name}`
            WHERE role = 'user'
              AND timestamp >= %s AND timestamp < %s
              AND content_text IS NOT NULL AND content_text != ''
              AND sender_id IS NOT NULL AND sender_id != ''
            ORDER BY session_id, timestamp
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time))

        if not rows:
            return _empty_prompt_quality_result()

        for row in rows:
            row["user_prompt"] = _extract_user_message(row.get("content_text") or "")

        rows = [row for row in rows if row["user_prompt"]]

        if not rows:
            print("[L2-4] No valid user prompts after metadata extraction")
            return _empty_prompt_quality_result()

        messages = [row["user_prompt"] for row in rows]

        total_chars = sum(len(msg) for msg in messages)
        estimated_tokens = int(total_chars * 1.5)
        max_single_batch_tokens = 128_000

        if estimated_tokens < max_single_batch_tokens:
            batch_size = len(messages)
            print(f"[L2-4] Estimated {estimated_tokens} tokens (< 128K), sending all in one batch")
        else:
            batch_size = 10
            print(f"[L2-4] Estimated {estimated_tokens} tokens (>= 128K), splitting into batches of {batch_size}")

        def build_user_prompt(batch: list[str], start_index: int) -> str:
            numbered = "\n\n---\n\n".join(
                f"[{start_index + i + 1}]\n{msg}" for i, msg in enumerate(batch)
            )
            return (
                f"Evaluate these user prompts on the 6 dimensions described above.\n\n"
                f"{numbered}\n\n"
                f"Return a JSON array with goal_clarity, context_provided, chain_of_thought, "
                f"few_shot_examples, iteration_signals, specificity for each prompt."
            )

        print(f"[L2-4] Queried {len(rows)} sessions, sending to LLM for prompt quality scoring...")
        results = await llm_client.batch_classify(
            messages, _PROMPT_QUALITY_SYSTEM_PROMPT, batch_size,
            build_user_prompt, label="L2-4:prompt_quality",
        )

        user_prompt_scores: dict[str, list[dict]] = {}
        for i, row in enumerate(rows):
            sender_id = row.get("sender_id") or "unknown"
            raw_score = results[i] if i < len(results) else dict(_DEFAULT_SCORES)
            score_entry = {dim: raw_score.get(dim, 1) for dim in _SCORING_DIMENSIONS}
            score_entry["overall"] = sum(score_entry[d] for d in _SCORING_DIMENSIONS) / len(_SCORING_DIMENSIONS)
            score_entry["user_prompt"] = row["user_prompt"]
            score_entry["session_id"] = row["session_id"]
            score_entry["row_id"] = row["row_id"]

            if sender_id not in user_prompt_scores:
                user_prompt_scores[sender_id] = []
            user_prompt_scores[sender_id].append(score_entry)

        by_user: list[dict] = []
        all_dim_avgs: dict[str, list[float]] = {dim: [] for dim in _SCORING_DIMENSIONS}

        for sender_id, entries in user_prompt_scores.items():
            user_result: dict[str, Any] = {"senderId": sender_id, "promptCount": len(entries)}

            for dim in _SCORING_DIMENSIONS:
                avg_val = sum(e[dim] for e in entries) / len(entries)
                user_result[dim] = round(avg_val, 2)
                all_dim_avgs[dim].append(avg_val)

            user_result["overall"] = round(
                sum(user_result[d] for d in _SCORING_DIMENSIONS) / len(_SCORING_DIMENSIONS), 2
            )

            best_entry = max(entries, key=lambda e: e["overall"])
            user_result["bestPrompt"] = {
                "sessionId": best_entry["session_id"],
                "rowId": best_entry["row_id"],
                "overall": round(best_entry["overall"], 2),
                "content": best_entry["user_prompt"],
                "scores": {d: best_entry[d] for d in _SCORING_DIMENSIONS},
            }

            worst_entry = min(entries, key=lambda e: e["overall"])
            user_result["worstPrompt"] = {
                "sessionId": worst_entry["session_id"],
                "rowId": worst_entry["row_id"],
                "overall": round(worst_entry["overall"], 2),
                "content": worst_entry["user_prompt"],
                "scores": {d: worst_entry[d] for d in _SCORING_DIMENSIONS},
            }

            by_user.append(user_result)

        by_user.sort(key=lambda u: u["overall"], reverse=True)

        def safe_avg(values: list[float]) -> float:
            return round(sum(values) / len(values), 2) if values else 0

        team_average: dict[str, float] = {}
        for dim in _SCORING_DIMENSIONS:
            team_average[dim] = safe_avg(all_dim_avgs[dim])
        team_average["overall"] = safe_avg(
            [sum(all_dim_avgs[d][i] for d in _SCORING_DIMENSIONS) / len(_SCORING_DIMENSIONS)
             for i in range(len(all_dim_avgs[_SCORING_DIMENSIONS[0]]))]
        )

        top_users = [
            {
                "senderId": u["senderId"],
                "overall": u["overall"],
                "bestPrompt": {
                    "sessionId": u["bestPrompt"]["sessionId"],
                    "rowId": u["bestPrompt"]["rowId"],
                    "content": u["bestPrompt"]["content"],
                    "scores": u["bestPrompt"]["scores"],
                },
            }
            for u in by_user[:3]
        ]

        bottom_users = [
            {
                "senderId": u["senderId"],
                "overall": u["overall"],
                "worstPrompt": {
                    "sessionId": u["worstPrompt"]["sessionId"],
                    "rowId": u["worstPrompt"]["rowId"],
                    "content": u["worstPrompt"]["content"],
                    "scores": u["worstPrompt"]["scores"],
                },
            }
            for u in by_user[-3:]
        ]

        print(f"[L2-4] Scored prompt quality for {len(user_prompt_scores)} users, top/bottom 3 identified")
        return {
            "teamAverage": team_average,
            "byUser": by_user,
            "topUsers": top_users,
            "bottomUsers": bottom_users,
        }

    except Exception as error:
        print(f"[L2-4] Prompt quality scoring failed: {error}")
        return _empty_prompt_quality_result()


def _empty_prompt_quality_result() -> dict:
    zero_dims = {dim: 0 for dim in _SCORING_DIMENSIONS}
    zero_dims["overall"] = 0
    return {"teamAverage": zero_dims, "byUser": [], "topUsers": [], "bottomUsers": []}


# ─── L2-5: Topic Clustering ───

def _jaccard_similarity_sets(set1: set, set2: set) -> float:
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0


def _string_jaccard(str1: str, str2: str) -> float:
    words1 = set(re.split(r"[\s\W]+", str1.lower()))
    words2 = set(re.split(r"[\s\W]+", str2.lower()))
    words1.discard("")
    words2.discard("")
    return _jaccard_similarity_sets(words1, words2)


_TOPIC_SYSTEM_PROMPT = """You are analyzing conversation topics from an enterprise AI coding assistant.
Users are enterprise employees — they may discuss work-related technical topics, but also life, entertainment, casual chat, and other non-work subjects.

For each user message, classify it into ONE primary category and generate 1-2 specific topic tags.

### Primary Categories (pick exactly one):

Each category has a Chinese label and an English label. Use whichever language matches the dominant language of the user message.

**Work — Software Development**
| Chinese Label | English Label | Description |
|---|---|---|
| 代码开发 | Code Development | Writing, modifying, or generating code |
| 问题调试 | Debugging | Fixing bugs, troubleshooting errors |
| 架构设计 | Architecture & Design | System design, technical planning, schema design |
| 测试验证 | Testing | Writing tests, QA, validation |
| 代码审查 | Code Review | Reviewing, refactoring, optimizing code quality |
| 配置部署 | DevOps & Deployment | CI/CD, environment setup, K8s, Docker, configuration |
| 数据处理 | Data & Analytics | Data processing, SQL, visualization, statistics |

**Work — General**
| Chinese Label | English Label | Description |
|---|---|---|
| 文档写作 | Documentation | Writing docs, README, technical specs, API docs |
| 项目管理 | Project Management | Task tracking, planning, coordination |
| 学习研究 | Learning & Research | Asking about concepts, exploring new technologies |
| 工具使用 | Tooling | Using specific tools, plugins, integrations |

**Life & Personal**
| Chinese Label | English Label | Description |
|---|---|---|
| 生活日常 | Daily Life | Cooking, health, fitness, travel, shopping, personal advice |
| 情感社交 | Social & Emotional | Relationships, social interactions, emotional support |
| 教育学习 | Education | Non-work learning, language study, exam prep, homework help |

**Entertainment**
| Chinese Label | English Label | Description |
|---|---|---|
| 影视音乐 | Movies & Music | Movies, TV shows, music, celebrities |
| 游戏电竞 | Gaming | Video games, esports, game strategies |
| 体育运动 | Sports | Sports news, fitness activities, outdoor activities |
| 阅读创作 | Reading & Writing | Books, novels, creative writing, poetry |

**Other**
| Chinese Label | English Label | Description |
|---|---|---|
| 闲聊互动 | Casual Chat | Greetings, jokes, role-play, casual conversation |
| 安全测试 | Security Testing | Prompt injection, jailbreak attempts, security probing |
| 其他 | Miscellaneous | Anything that doesn't fit above categories |

Language detection rules:
- If the user message is predominantly Chinese, use the Chinese label (e.g., "代码开发", "生活日常")
- If the user message is predominantly English, use the English label (e.g., "Code Development", "Daily Life")
- If the message is mixed, use the label of the language that makes up the majority

Return a JSON array where each element has:
- category: the label in the appropriate language as described above
- tags: array of 1-2 specific topic tags in lowercase (e.g., ["react hooks", "state management"])"""


async def cluster_topics(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    print("[L2-5] Starting topic clustering...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            SELECT row_id, session_id, sender_id, content_text
            FROM `{table_name}`
            WHERE role = 'user'
              AND timestamp >= %s AND timestamp < %s
              AND content_text IS NOT NULL AND content_text != ''
              AND sender_id IS NOT NULL AND sender_id != ''
            ORDER BY session_id, timestamp
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time))

        if not rows:
            return {"categoryDistribution": {}, "topTags": [], "byUser": {}}

        for row in rows:
            row["user_message"] = _extract_user_message(row.get("content_text") or "")

        rows = [row for row in rows if row["user_message"]]

        if not rows:
            print("[L2-5] No valid user messages after metadata extraction")
            return {"categoryDistribution": {}, "topTags": [], "byUser": {}}

        messages = [row["user_message"] for row in rows]

        total_chars = sum(len(msg) for msg in messages)
        estimated_tokens = int(total_chars * 1.5)
        max_single_batch_tokens = 128_000

        if estimated_tokens < max_single_batch_tokens:
            batch_size = len(messages)
            print(f"[L2-5] Estimated {estimated_tokens} tokens (< 128K), sending all in one batch")
        else:
            batch_size = 15
            print(f"[L2-5] Estimated {estimated_tokens} tokens (>= 128K), splitting into batches of {batch_size}")

        def build_user_prompt(batch: list[str], start_index: int) -> str:
            numbered = "\n\n---\n\n".join(
                f"[{start_index + i + 1}]\n{msg}" for i, msg in enumerate(batch)
            )
            return (
                f"Classify these user messages into categories and generate topic tags.\n\n"
                f"{numbered}\n\n"
                f"Return a JSON array with category and tags for each message."
            )

        print(f"[L2-5] Queried {len(rows)} messages, sending to LLM for topic clustering...")
        results = await llm_client.batch_classify(
            messages, _TOPIC_SYSTEM_PROMPT, batch_size,
            build_user_prompt, label="L2-5:topics",
        )

        category_distribution: dict[str, int] = {}
        tag_counts: dict[str, dict] = {}
        by_user_categories: dict[str, dict[str, int]] = {}

        for i, row in enumerate(rows):
            result = results[i] if i < len(results) else {"category": "其他", "tags": []}
            category = result.get("category", "其他")
            tags = result.get("tags", [])
            sender_id = row.get("sender_id") or "unknown"

            category_distribution[category] = category_distribution.get(category, 0) + 1

            if sender_id not in by_user_categories:
                by_user_categories[sender_id] = {}
            by_user_categories[sender_id][category] = by_user_categories[sender_id].get(category, 0) + 1

            for tag in tags:
                normalized = tag.lower().strip()
                if not normalized:
                    continue
                if normalized not in tag_counts:
                    tag_counts[normalized] = {"count": 0, "users": set(), "category": category}
                tag_counts[normalized]["count"] += 1
                tag_counts[normalized]["users"].add(sender_id)

        merged_tags: dict[str, dict] = {}
        processed: set[str] = set()

        for tag, data in tag_counts.items():
            if tag in processed:
                continue
            merged = {"count": data["count"], "users": set(data["users"]), "category": data["category"]}
            processed.add(tag)

            for other_tag, other_data in tag_counts.items():
                if other_tag in processed:
                    continue
                if _string_jaccard(tag, other_tag) > 0.8:
                    merged["count"] += other_data["count"]
                    merged["users"].update(other_data["users"])
                    processed.add(other_tag)

            merged_tags[tag] = merged

        top_tags = sorted(
            [
                {
                    "tag": tag,
                    "category": data["category"],
                    "count": data["count"],
                    "uniqueUsers": len(data["users"]),
                }
                for tag, data in merged_tags.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )[:30]

        print(f"[L2-5] Clustered {len(rows)} messages into {len(category_distribution)} categories, {len(merged_tags)} tags")
        return {
            "categoryDistribution": category_distribution,
            "topTags": top_tags,
            "byUser": by_user_categories,
        }

    except Exception as error:
        print(f"[L2-5] Topic clustering failed: {error}")
        return {"categoryDistribution": {}, "topTags": [], "byUser": {}}


# ─── Combined: L2-1 + L2-5 + L3-1 (Per-User Aggregated) ───

_COMBINED_PER_MSG_SYSTEM_PROMPT = """You are analyzing enterprise AI assistant user messages. For each message, simultaneously:
1. Classify the user's primary intent
2. Classify the conversation topic with 1-2 tags
3. Identify all technologies mentioned or implied

### Intent Categories (use language matching the message — Chinese if predominantly Chinese, English if predominantly English):
代码开发/Code Development, 问题修复/Bug Fixing, 代码优化/Code Optimization, 测试验证/Testing & Validation, 架构设计/Architecture Design, 知识问答/Knowledge Q&A, 配置部署/Configuration & Deployment, 数据分析/Data Analysis, 信息检索/Information Retrieval, 内容生成/Content Generation, 任务管理/Task Management, 工具使用/Tool Usage, 闲聊互动/Casual Interaction, 安全测试/Security Testing, 多媒体处理/Multimedia Processing, 其他/Other

### Topic Categories (same language rule):
代码开发/Code Development, 问题调试/Debugging, 架构设计/Architecture & Design, 测试验证/Testing, 代码审查/Code Review, 配置部署/DevOps & Deployment, 数据处理/Data & Analytics, 文档写作/Documentation, 项目管理/Project Management, 学习研究/Learning & Research, 工具使用/Tooling, 生活日常/Daily Life, 情感社交/Social & Emotional, 教育学习/Education, 影视音乐/Movies & Music, 游戏电竞/Gaming, 体育运动/Sports, 阅读创作/Reading & Writing, 闲聊互动/Casual Chat, 安全测试/Security Testing, 其他/Miscellaneous

Return a JSON array with one object per message:
{"intent": "label", "confidence": 0.0-1.0, "topic": "label", "tags": ["tag1"], "technologies": ["React"]}"""

_COMBINED_PER_USER_SYSTEM_PROMPT = """You are analyzing enterprise AI assistant users based on their conversation histories.
For each user, analyze all their messages and return:
- intent_counts: estimated distribution of intent types ({"category_label": count})
- topic_counts: estimated distribution of topic types ({"category_label": count})
- tags: 2-3 representative lowercase topic tags summarizing the user's main interests
- technologies: all technology names mentioned or implied in their messages (canonical names)

### Intent Categories (Chinese if predominantly Chinese, English otherwise):
代码开发/Code Development, 问题修复/Bug Fixing, 代码优化/Code Optimization, 测试验证/Testing & Validation, 架构设计/Architecture Design, 知识问答/Knowledge Q&A, 配置部署/Configuration & Deployment, 数据分析/Data Analysis, 信息检索/Information Retrieval, 内容生成/Content Generation, 任务管理/Task Management, 工具使用/Tool Usage, 闲聊互动/Casual Interaction, 安全测试/Security Testing, 多媒体处理/Multimedia Processing, 其他/Other

### Topic Categories (same language rule):
代码开发/Code Development, 问题调试/Debugging, 架构设计/Architecture & Design, 测试验证/Testing, 代码审查/Code Review, 配置部署/DevOps & Deployment, 数据处理/Data & Analytics, 文档写作/Documentation, 项目管理/Project Management, 学习研究/Learning & Research, 工具使用/Tooling, 生活日常/Daily Life, 情感社交/Social & Emotional, 教育学习/Education, 影视音乐/Movies & Music, 游戏电竞/Gaming, 体育运动/Sports, 阅读创作/Reading & Writing, 闲聊互动/Casual Chat, 安全测试/Security Testing, 其他/Miscellaneous

Return a JSON array with one object per user (same order as input):
{"user_id": "string", "intent_counts": {"代码开发": 5}, "topic_counts": {"配置部署": 3}, "tags": ["docker"], "technologies": ["Docker"]}"""


def _build_top_tags(tag_counts: dict[str, dict]) -> list[dict]:
    """Build and deduplicate top tags using Jaccard similarity."""
    merged_tags: dict[str, dict] = {}
    processed: set[str] = set()

    for tag, data in tag_counts.items():
        if tag in processed:
            continue
        merged = {"count": data["count"], "users": set(data["users"]), "category": data["category"]}
        processed.add(tag)
        for other_tag, other_data in tag_counts.items():
            if other_tag in processed:
                continue
            if _string_jaccard(tag, other_tag) > 0.8:
                merged["count"] += other_data["count"]
                merged["users"].update(other_data["users"])
                processed.add(other_tag)
        merged_tags[tag] = merged

    return sorted(
        [
            {"tag": t, "category": d["category"], "count": d["count"], "uniqueUsers": len(d["users"])}
            for t, d in merged_tags.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:30]


async def _combined_per_message(rows: list[dict], llm_client: LlmClient) -> dict:
    """Small dataset path: one LLM batch returning intent+topic+tech per message."""
    messages = [r["msg_text"] for r in rows]
    print(f"[Combined:per-msg] {len(messages)} messages → single batch")

    def build_user_prompt(batch: list[str], start_index: int) -> str:
        numbered = "\n\n---\n\n".join(f"[{start_index + i + 1}]\n{msg}" for i, msg in enumerate(batch))
        return (
            f"Analyze these {len(batch)} user messages for intent, topic, and technologies.\n\n"
            f"{numbered}\n\n"
            f"Return a JSON array with one object per message: "
            f"{{intent, confidence, topic, tags[], technologies[]}}"
        )

    results = await llm_client.batch_classify(
        messages, _COMBINED_PER_MSG_SYSTEM_PROMPT, len(messages),
        build_user_prompt, label="combined:per-msg",
    )

    intent_dist: dict[str, int] = {}
    intent_by_user: dict[str, dict[str, int]] = {}
    intent_senders_by_cat: dict[str, set] = {}
    intent_items: list[dict] = []

    topic_cat_dist: dict[str, int] = {}
    topic_tag_counts: dict[str, dict] = {}
    topic_by_user: dict[str, dict[str, int]] = {}

    tech_users: dict[str, set[str]] = {}
    tech_sessions: dict[str, set[str]] = {}

    for i, row in enumerate(rows):
        res = results[i] if i < len(results) else {}
        sender_id = row["sender_id"]
        session_id = row["session_id"]

        intent_cat = res.get("intent", "其他")
        confidence = float(res.get("confidence", 0))
        intent_dist[intent_cat] = intent_dist.get(intent_cat, 0) + 1
        if sender_id not in intent_by_user:
            intent_by_user[sender_id] = {}
        intent_by_user[sender_id][intent_cat] = intent_by_user[sender_id].get(intent_cat, 0) + 1
        if intent_cat not in intent_senders_by_cat:
            intent_senders_by_cat[intent_cat] = set()
        intent_senders_by_cat[intent_cat].add(sender_id)
        intent_items.append({
            "rowId": row["row_id"], "sessionId": session_id, "senderId": sender_id,
            "category": intent_cat, "confidence": confidence, "preview": row["msg_text"][:100],
        })

        topic_cat = res.get("topic", "其他")
        tags = res.get("tags", [])
        topic_cat_dist[topic_cat] = topic_cat_dist.get(topic_cat, 0) + 1
        if sender_id not in topic_by_user:
            topic_by_user[sender_id] = {}
        topic_by_user[sender_id][topic_cat] = topic_by_user[sender_id].get(topic_cat, 0) + 1
        for tag in tags:
            norm = tag.lower().strip()
            if not norm:
                continue
            if norm not in topic_tag_counts:
                topic_tag_counts[norm] = {"count": 0, "users": set(), "category": topic_cat}
            topic_tag_counts[norm]["count"] += 1
            topic_tag_counts[norm]["users"].add(sender_id)

        for tech in res.get("technologies", []):
            if not isinstance(tech, str) or not tech.strip():
                continue
            t = tech.strip()
            if t not in tech_sessions:
                tech_sessions[t] = set()
                tech_users[t] = set()
            tech_sessions[t].add(session_id)
            tech_users[t].add(sender_id)

    print(f"[Combined:per-msg] intents={len(intent_dist)}, topics={len(topic_cat_dist)}, techs={len(tech_sessions)}")
    return {
        "intents": {
            "distribution": intent_dist,
            "byUser": intent_by_user,
            "sendersByCategory": {c: sorted(s) for c, s in intent_senders_by_cat.items()},
            "items": intent_items,
        },
        "topics": {
            "categoryDistribution": topic_cat_dist,
            "topTags": _build_top_tags(topic_tag_counts),
            "byUser": topic_by_user,
        },
        "tech_stack": {
            "technologies": sorted(
                [{"tech": t, "sessionCount": len(s), "uniqueUsers": len(tech_users[t])}
                 for t, s in tech_sessions.items()],
                key=lambda x: x["sessionCount"], reverse=True,
            )
        },
    }


async def _combined_per_user(rows: list[dict], llm_client: LlmClient) -> dict:
    """Large dataset path: group by user, one LLM request per user batch."""
    # Group rows by sender_id
    user_rows: dict[str, list[dict]] = {}
    for row in rows:
        uid = row["sender_id"]
        if uid not in user_rows:
            user_rows[uid] = []
        user_rows[uid].append(row)

    user_ids = list(user_rows.keys())
    user_sessions_map: dict[str, set[str]] = {
        uid: {r["session_id"] for r in u_rows} for uid, u_rows in user_rows.items()
    }

    # Build one context string per user (truncate each message to 500 chars)
    user_context_items: list[str] = []
    for uid in user_ids:
        snippets = [f"[{j+1}] {r['msg_text'][:500]}" for j, r in enumerate(user_rows[uid])]
        user_context_items.append(f"user_id: {uid}\n" + "\n".join(snippets))

    total_chars = sum(len(s) for s in user_context_items)
    estimated_tokens = int(total_chars * 2.5)
    batch_size = len(user_context_items) if estimated_tokens < 128_000 else 10
    print(f"[Combined:per-user] {len(user_ids)} users, ~{estimated_tokens} tokens, batch_size={batch_size}")

    def build_user_prompt(batch: list[str], start_index: int) -> str:
        combined = "\n\n===\n\n".join(batch)
        return (
            f"Analyze these {len(batch)} users' conversation histories.\n\n"
            f"{combined}\n\n"
            f"Return a JSON array with one object per user (same order as input): "
            f"{{user_id, intent_counts, topic_counts, tags[], technologies[]}}"
        )

    results = await llm_client.batch_classify(
        user_context_items, _COMBINED_PER_USER_SYSTEM_PROMPT, batch_size,
        build_user_prompt, label="combined:per-user",
    )

    intent_dist: dict[str, int] = {}
    intent_by_user: dict[str, dict[str, int]] = {}
    intent_senders_by_cat: dict[str, set] = {}

    topic_cat_dist: dict[str, int] = {}
    topic_tag_counts: dict[str, dict] = {}
    topic_by_user: dict[str, dict[str, int]] = {}

    tech_users: dict[str, set[str]] = {}
    tech_sessions: dict[str, set[str]] = {}

    for i, uid in enumerate(user_ids):
        res = results[i] if i < len(results) else {}
        u_sessions = user_sessions_map[uid]

        intent_counts = res.get("intent_counts") or {}
        if not isinstance(intent_counts, dict):
            intent_counts = {}
        intent_by_user[uid] = intent_counts
        for cat, cnt in intent_counts.items():
            intent_dist[cat] = intent_dist.get(cat, 0) + int(cnt)
            if cat not in intent_senders_by_cat:
                intent_senders_by_cat[cat] = set()
            intent_senders_by_cat[cat].add(uid)

        topic_counts = res.get("topic_counts") or {}
        if not isinstance(topic_counts, dict):
            topic_counts = {}
        topic_by_user[uid] = topic_counts
        for cat, cnt in topic_counts.items():
            topic_cat_dist[cat] = topic_cat_dist.get(cat, 0) + int(cnt)
        dominant_cat = max(topic_counts, key=lambda c: topic_counts[c]) if topic_counts else "其他"
        for tag in res.get("tags", []):
            norm = tag.lower().strip()
            if not norm:
                continue
            if norm not in topic_tag_counts:
                topic_tag_counts[norm] = {"count": 0, "users": set(), "category": dominant_cat}
            topic_tag_counts[norm]["count"] += 1
            topic_tag_counts[norm]["users"].add(uid)

        for tech in res.get("technologies", []):
            if not isinstance(tech, str) or not tech.strip():
                continue
            t = tech.strip()
            if t not in tech_sessions:
                tech_sessions[t] = set()
                tech_users[t] = set()
            tech_sessions[t].update(u_sessions)
            tech_users[t].add(uid)

    print(f"[Combined:per-user] intents={len(intent_dist)}, topics={len(topic_cat_dist)}, techs={len(tech_sessions)}")
    return {
        "intents": {
            "distribution": intent_dist,
            "byUser": intent_by_user,
            "sendersByCategory": {c: sorted(s) for c, s in intent_senders_by_cat.items()},
            "items": [],  # no per-row items in per-user mode
        },
        "topics": {
            "categoryDistribution": topic_cat_dist,
            "topTags": _build_top_tags(topic_tag_counts),
            "byUser": topic_by_user,
        },
        "tech_stack": {
            "technologies": sorted(
                [{"tech": t, "sessionCount": len(s), "uniqueUsers": len(tech_users[t])}
                 for t, s in tech_sessions.items()],
                key=lambda x: x["sessionCount"], reverse=True,
            )
        },
    }


async def classify_messages_combined(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
    max_items: int = 500,
) -> dict:
    """Combined analysis for L2-1 (intents), L2-5 (topics), and L3-1 (tech stack).

    Discards unknown users (null/empty sender_id).
    - Small datasets (estimated < 128K tokens at 2.5 tokens/char): one batch, per-message analysis.
    - Large datasets: group by user, truncate each message to 500 chars, one batch per user group.

    Returns dict with keys 'intents', 'topics', 'tech_stack'.
    """
    _empty: dict = {
        "intents": {"distribution": {}, "byUser": {}, "sendersByCategory": {}, "items": []},
        "topics": {"categoryDistribution": {}, "topTags": [], "byUser": {}},
        "tech_stack": {"technologies": []},
    }
    print("[Combined] Starting combined L2-1+L2-5+L3-1 analysis (per-user aggregation)...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)
        sql = f"""
            SELECT row_id, session_id, sender_id, content_text
            FROM `{table_name}`
            WHERE role = 'user'
              AND timestamp >= %s AND timestamp < %s
              AND content_text IS NOT NULL AND content_text != ''
              AND sender_id IS NOT NULL AND sender_id != ''
            ORDER BY sender_id, session_id, timestamp
            LIMIT %s
        """
        rows = await execute_query_async(ck_config, sql, (start_time, end_time, max_items))

        if not rows:
            print("[Combined] No valid user messages found")
            return _empty

        for row in rows:
            row["msg_text"] = _extract_user_message(row.get("content_text") or "")
        rows = [r for r in rows if r["msg_text"]]

        if not rows:
            print("[Combined] No valid messages after metadata extraction")
            return _empty

        total_chars = sum(len(r["msg_text"]) for r in rows)
        estimated_tokens = int(total_chars * 2.5)
        print(f"[Combined] {len(rows)} messages from {len({r['sender_id'] for r in rows})} users, "
              f"~{estimated_tokens} estimated tokens")

        if estimated_tokens < 128_000:
            return await _combined_per_message(rows, llm_client)
        else:
            return await _combined_per_user(rows, llm_client)

    except Exception as error:
        print(f"[Combined] Analysis failed: {error}")
        return _empty


# ─── L2-6: Retry Behavior Detection ───

async def detect_retry_behavior(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
) -> dict:
    """Detect retry behavior using window functions (ClickHouse version).

    lag() OVER (PARTITION BY ... ORDER BY ...) is supported in ClickHouse 21.1+,
    so the SQL logic is identical to the MySQL version.
    """
    print("[L2-6] Starting retry behavior detection...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            WITH consecutive_users AS (
                SELECT
                    session_id, sender_id, content_text, timestamp,
                    lag(content_text) OVER (PARTITION BY session_id ORDER BY timestamp) AS prev_content_text,
                    lag(role) OVER (PARTITION BY session_id ORDER BY timestamp) AS prev_role
                FROM `{table_name}`
                WHERE timestamp >= %s AND timestamp < %s AND role = 'user'
                  AND sender_id IS NOT NULL AND sender_id != ''
            )
            SELECT session_id, sender_id,
                content_text AS msg2, prev_content_text AS msg1
            FROM consecutive_users
            WHERE prev_content_text IS NOT NULL AND prev_role = 'user'
            LIMIT 500
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time))

        if not rows:
            return {"retryRate": 0, "items": []}

        retry_count = 0
        items = []

        for row in rows:
            msg1 = _extract_user_message(row.get("msg1") or "")
            msg2 = _extract_user_message(row.get("msg2") or "")
            similarity = _string_jaccard(msg1, msg2)

            if similarity > 0.5:
                classification = "retry"
                retry_count += 1
            elif similarity > 0.3:
                classification = "refinement"
            else:
                classification = "new_question"

            items.append({
                "sessionId": row["session_id"],
                "senderId": row.get("sender_id") or "unknown",
                "similarity": round(similarity, 4),
                "classification": classification,
                "msg1Preview": msg1[:100],
                "msg2Preview": msg2[:100],
            })

        retry_rate = retry_count / len(rows) if rows else 0
        print(f"[L2-6] Detected retry behavior in {len(rows)} consecutive pairs")
        return {"retryRate": retry_rate, "items": items}

    except Exception as error:
        print(f"[L2-6] Retry behavior detection failed: {error}")
        return {"retryRate": 0, "items": []}


# ─── L2-7: Thinking Depth Analysis ───

async def analyze_thinking_depth(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
) -> dict:
    """Analyze thinking depth distribution (ClickHouse version).

    Key SQL adaptations from MySQL:
    - AVG(COALESCE(..., 0))  →  avg(ifNull(..., 0))
    - COALESCE(model, 'unknown')  →  ifNull(model, 'unknown')
    - SUM(CASE WHEN thinking_text IS NOT NULL ... THEN 1 ELSE 0 END)  →  countIf(...)
    - AVG(CASE WHEN ... THEN LENGTH(thinking_text) ELSE NULL END)  →  avgIf(length(...), ...)
    - ORDER BY CASE WHEN ... END is supported in ClickHouse
    """
    print("[L2-7] Starting thinking depth analysis...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql1 = f"""
            SELECT
                CASE
                    WHEN thinking_text IS NULL OR thinking_text = '' THEN 'no_thinking'
                    WHEN length(thinking_text) < 500 THEN 'shallow'
                    WHEN length(thinking_text) < 2000 THEN 'moderate'
                    WHEN length(thinking_text) < 5000 THEN 'deep'
                    ELSE 'very_deep'
                END AS thinking_depth,
                count(*) AS message_count,
                avg(ifNull(output_tokens, 0)) AS avg_output_tokens,
                avg(ifNull(total_cost, 0)) AS avg_cost,
                avg(ifNull(content_length, 0)) AS avg_content_length
            FROM `{table_name}`
            WHERE timestamp >= %s AND timestamp < %s AND role = 'assistant'
            GROUP BY thinking_depth
            ORDER BY
                CASE thinking_depth
                    WHEN 'no_thinking' THEN 1
                    WHEN 'shallow' THEN 2
                    WHEN 'moderate' THEN 3
                    WHEN 'deep' THEN 4
                    WHEN 'very_deep' THEN 5
                END
        """

        sql2 = f"""
            SELECT
                ifNull(model, 'unknown') AS model,
                count(*) AS total_messages,
                countIf(isNotNull(thinking_text) AND thinking_text != '') AS thinking_messages,
                avgIf(length(thinking_text), isNotNull(thinking_text) AND thinking_text != '') AS avg_thinking_length
            FROM `{table_name}`
            WHERE timestamp >= %s AND timestamp < %s AND role = 'assistant'
            GROUP BY model
        """

        by_depth_rows, by_model_rows = await asyncio.gather(
            execute_query_async(ck_config, sql1, (start_time, end_time)),
            execute_query_async(ck_config, sql2, (start_time, end_time)),
        )

        by_depth = [
            {
                "thinkingDepth": row["thinking_depth"],
                "messageCount": row["message_count"],
                "avgOutputTokens": float(row.get("avg_output_tokens") or 0),
                "avgCost": float(row.get("avg_cost") or 0),
                "avgContentLength": float(row.get("avg_content_length") or 0),
            }
            for row in by_depth_rows
        ]

        by_model = [
            {
                "model": row["model"],
                "totalMessages": row["total_messages"],
                "thinkingCount": row.get("thinking_messages") or 0,
                "thinkingPct": (
                    (row.get("thinking_messages") or 0) / row["total_messages"]
                    if row["total_messages"] > 0 else 0
                ),
                "avgThinkingLength": float(row.get("avg_thinking_length") or 0),
            }
            for row in by_model_rows
        ]

        print(f"[L2-7] Analyzed thinking depth for {len(by_depth_rows)} buckets")
        return {"byDepth": by_depth, "byModel": by_model}

    except Exception as error:
        print(f"[L2-7] Thinking depth analysis failed: {error}")
        return {"byDepth": [], "byModel": []}


# ─── L2-8: User Maturity Tracking ───

_CONSISTENTLY_HIGH_THRESHOLD = 4.0

async def track_user_maturity(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    """Track user prompt quality growth over time (ClickHouse version).

    Key SQL adaptation:
    - DATE(timestamp)  →  toDate(timestamp)
    """
    print("[L2-8] Starting user maturity tracking...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        # toDate() replaces DATE() for ClickHouse
        sql = f"""
            SELECT row_id, session_id, sender_id, content_text,
                   toDate(timestamp) AS day_bucket, timestamp
            FROM `{table_name}`
            WHERE role = 'user'
              AND timestamp >= %s AND timestamp < %s
              AND content_text IS NOT NULL AND content_text != ''
              AND sender_id IS NOT NULL AND sender_id != ''
            ORDER BY sender_id, timestamp
        """

        rows = await execute_query_async(ck_config, sql, (start_time, end_time))

        if not rows:
            return {"users": []}

        for row in rows:
            row["user_prompt"] = _extract_user_message(row.get("content_text") or "")

        rows = [row for row in rows if row["user_prompt"]]

        if not rows:
            print("[L2-8] No valid user prompts after metadata extraction")
            return {"users": []}

        messages = [row["user_prompt"] for row in rows]

        total_chars = sum(len(msg) for msg in messages)
        estimated_tokens = int(total_chars * 1.5)
        max_single_batch_tokens = 128_000

        if estimated_tokens < max_single_batch_tokens:
            batch_size = len(messages)
            print(f"[L2-8] Estimated {estimated_tokens} tokens (< 128K), sending all in one batch")
        else:
            batch_size = 10
            print(f"[L2-8] Estimated {estimated_tokens} tokens (>= 128K), splitting into batches of {batch_size}")

        def build_user_prompt(batch: list[str], start_index: int) -> str:
            numbered = "\n\n---\n\n".join(
                f"[{start_index + i + 1}]\n{msg}" for i, msg in enumerate(batch)
            )
            return (
                f"Evaluate these user prompts on the 6 dimensions described above.\n\n"
                f"{numbered}\n\n"
                f"Return a JSON array with goal_clarity, context_provided, chain_of_thought, "
                f"few_shot_examples, iteration_signals, specificity for each prompt."
            )

        print(f"[L2-8] Scoring {len(messages)} user messages for maturity tracking...")
        results = await llm_client.batch_classify(
            messages, _PROMPT_QUALITY_SYSTEM_PROMPT, batch_size,
            build_user_prompt, label="L2-8:maturity",
        )

        user_daily_scores: dict[str, dict[str, list[float]]] = {}

        for i, row in enumerate(rows):
            sender_id = row["sender_id"]
            day_bucket = str(row["day_bucket"])
            raw_score = results[i] if i < len(results) else dict(_DEFAULT_SCORES)
            overall = sum(raw_score.get(dim, 1) for dim in _SCORING_DIMENSIONS) / len(_SCORING_DIMENSIONS)

            if sender_id not in user_daily_scores:
                user_daily_scores[sender_id] = {}
            if day_bucket not in user_daily_scores[sender_id]:
                user_daily_scores[sender_id][day_bucket] = []
            user_daily_scores[sender_id][day_bucket].append(overall)

        users = []
        for sender_id, day_data in user_daily_scores.items():
            sorted_days = sorted(day_data.keys())
            daily_scores: list[dict] = []
            all_scores_flat: list[float] = []

            for day in sorted_days:
                scores_list = day_data[day]
                avg_score = sum(scores_list) / len(scores_list)
                daily_scores.append({
                    "date": day,
                    "overall": round(avg_score, 2),
                    "promptCount": len(scores_list),
                })
                all_scores_flat.extend(scores_list)

            global_avg = sum(all_scores_flat) / len(all_scores_flat) if all_scores_flat else 0
            prompt_count = len(all_scores_flat)

            all_high = all(
                (sum(day_data[d]) / len(day_data[d])) >= _CONSISTENTLY_HIGH_THRESHOLD
                for d in sorted_days
            )

            slope = 0.0
            if len(sorted_days) >= 2:
                points_x = list(range(len(sorted_days)))
                points_y = [
                    sum(day_data[d]) / len(day_data[d]) for d in sorted_days
                ]
                n = len(points_x)
                sum_x = sum(points_x)
                sum_y = sum(points_y)
                sum_xy = sum(x * y for x, y in zip(points_x, points_y))
                sum_xx = sum(x * x for x in points_x)
                denominator = n * sum_xx - sum_x ** 2
                if denominator != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / denominator

            if all_high:
                trend = "consistently_high"
            elif slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"

            users.append({
                "senderId": sender_id,
                "promptCount": prompt_count,
                "overallAvg": round(global_avg, 2),
                "dailyScores": daily_scores,
                "trend": trend,
                "slope": round(slope, 4),
            })

        trend_priority = {"improving": 0, "declining": 1, "stable": 2, "consistently_high": 3}
        users.sort(key=lambda u: (trend_priority.get(u["trend"], 9), -u["slope"]))

        print(f"[L2-8] Tracked maturity for {len(users)} users across {len(set(d for dd in user_daily_scores.values() for d in dd))} days")
        return {"users": users}

    except Exception as error:
        print(f"[L2-8] User maturity tracking failed: {error}")
        return {"users": []}


# ─── Main L2 Analysis Runner ───

async def run_l2_analysis(
    ck_config: CkConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
    max_items: int = 500,
) -> dict:
    print("=== Starting L2 Behavior Analysis (ClickHouse) ===")

    complexity = await assess_task_complexity(ck_config, table_name, range_)
    success_rate = await estimate_task_success_rate(ck_config, table_name, range_)
    retry_behavior = await detect_retry_behavior(ck_config, table_name, range_)
    thinking_depth = await analyze_thinking_depth(ck_config, table_name, range_)
    intents = await classify_intents(ck_config, table_name, range_, llm_client, max_items)
    prompt_quality = await score_prompt_quality(ck_config, table_name, range_, llm_client)
    topics = await cluster_topics(ck_config, table_name, range_, llm_client)
    user_maturity = await track_user_maturity(ck_config, table_name, range_, llm_client)

    print("=== L2 Behavior Analysis Complete (ClickHouse) ===")

    return {
        "intents": intents,
        "complexity": complexity,
        "successRate": success_rate,
        "promptQuality": prompt_quality,
        "topics": topics,
        "retryBehavior": retry_behavior,
        "thinkingDepth": thinking_depth,
        "userMaturity": user_maturity,
    }
