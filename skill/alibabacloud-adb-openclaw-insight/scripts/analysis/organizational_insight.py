"""
L3: Organizational Cognition Layer.
Focuses on organizational-level insights, knowledge gaps, best practices, and skill candidates.
"""

from __future__ import annotations

import asyncio
import json

from scripts.config import AdbConfig
from scripts.db import execute_query_async
from scripts.llm_client import LlmClient, count_tokens, _MAX_BATCH_TOKENS as MAX_BATCH_TOKENS
from scripts.types import TimeRange, time_range_to_sql_params
from scripts.analysis.behavior_insight import _extract_user_message


# ─── Technology Dictionary ───

TECHNOLOGY_DICTIONARY: dict[str, dict[str, list[str]]] = {
    "Languages": {
        "Java": ["java", "spring", "maven", "gradle", "jvm"],
        "Python": ["python", "pip", "django", "flask", "fastapi", "pytorch"],
        "TypeScript": ["typescript", "ts-node", "tsx", "tsc"],
        "Go": ["golang", "go mod", "goroutine"],
        "Rust": ["rust", "cargo", "tokio"],
    },
    "Frameworks": {
        "React": ["react", "jsx", "next.js", "nextjs", "usestate", "useeffect"],
        "Vue": ["vue", "vuex", "nuxt", "pinia"],
    },
    "Databases": {
        "MySQL": ["mysql", "innodb", "mysqldump"],
        "PostgreSQL": ["postgresql", "postgres", "psql"],
        "Redis": ["redis", "jedis", "lettuce"],
        "MongoDB": ["mongodb", "mongoose", "mongosh"],
    },
    "Infrastructure": {
        "Kubernetes": ["kubernetes", "k8s", "kubectl", "helm", "pod", "deployment"],
        "Docker": ["docker", "dockerfile", "docker-compose", "container"],
    },
}


# ─── L3-1: Build Tech Stack Heatmap ───

_TECH_STACK_SYSTEM_PROMPT = """You are a technology stack analyst. For each user message from an enterprise AI agent assistant session, identify ALL technologies, frameworks, libraries, databases, infrastructure tools, and programming languages mentioned or implied.

Return a JSON array where each element has:
- technologies: array of strings — normalized technology names (e.g., "React", "Python", "Kubernetes", "MySQL", "Docker")

Rules:
- Use canonical names: "React" not "reactjs", "Kubernetes" not "k8s", "TypeScript" not "ts"
- Include both explicitly mentioned and strongly implied technologies (e.g., if user mentions "useEffect", include "React")
- If no technology is identifiable, return an empty array for that message
- Each technology name should be a single well-known technology (not a description)"""

async def build_tech_stack_heatmap(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    """Build a tech stack heatmap using LLM to identify technologies from user messages."""
    print("[L3-1] Building tech stack heatmap...")

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

        rows = await execute_query_async(adb_config, sql, (start_time, end_time))

        if not rows:
            return {"technologies": []}

        # Extract actual user messages
        for row in rows:
            row["user_prompt"] = _extract_user_message(row.get("content_text") or "")

        rows = [row for row in rows if row["user_prompt"]]

        if not rows:
            print("[L3-1] No valid user prompts after metadata extraction")
            return {"technologies": []}

        messages = [row["user_prompt"][:300] for row in rows]

        # Apply 128K single-batch strategy
        estimated_tokens = count_tokens(messages)

        if estimated_tokens < MAX_BATCH_TOKENS:
            batch_size = len(messages)
            print(f"[L3-1] Estimated {estimated_tokens} tokens (< 128K), sending all in one batch")
        else:
            batch_size = 15
            print(f"[L3-1] Estimated {estimated_tokens} tokens (>= 128K), splitting into batches of {batch_size}")

        def build_user_prompt(batch: list[str], start_index: int) -> str:
            numbered = "\n\n---\n\n".join(
                f"[{start_index + i + 1}]\n{msg}" for i, msg in enumerate(batch)
            )
            return (
                f"Identify technologies mentioned in these user messages.\n\n"
                f"{numbered}\n\n"
                f"Return a JSON array with a 'technologies' field (array of strings) for each message."
            )

        print(f"[L3-1] Queried {len(rows)} user messages, sending to LLM for tech stack identification...")
        results = await llm_client.batch_classify(
            messages, _TECH_STACK_SYSTEM_PROMPT, batch_size,
            build_user_prompt, label="L3-1:tech_stack",
        )

        # Aggregate: count sessions and unique users per technology
        tech_sessions: dict[str, set[str]] = {}
        tech_users: dict[str, set[str]] = {}

        for i, row in enumerate(rows):
            raw_result = results[i] if i < len(results) else {"technologies": []}
            techs = raw_result.get("technologies", [])
            if not isinstance(techs, list):
                continue

            session_id = row["session_id"]
            sender_id = row.get("sender_id") or "unknown"

            for tech_name in techs:
                if not isinstance(tech_name, str) or not tech_name.strip():
                    continue
                normalized = tech_name.strip()
                if normalized not in tech_sessions:
                    tech_sessions[normalized] = set()
                    tech_users[normalized] = set()
                tech_sessions[normalized].add(session_id)
                tech_users[normalized].add(sender_id)

        technologies = sorted(
            [
                {
                    "tech": tech_name,
                    "sessionCount": len(sessions),
                    "uniqueUsers": len(tech_users[tech_name]),
                }
                for tech_name, sessions in tech_sessions.items()
            ],
            key=lambda x: x["sessionCount"],
            reverse=True,
        )

        print(f"[L3-1] Found {len(technologies)} technologies")
        return {"technologies": technologies}

    except Exception as error:
        print(f"[L3-1] Error building tech stack heatmap: {error}")
        return {"technologies": []}


# ─── L3-2: Discover High-Frequency Repeated Questions ───

_REPEATED_QUESTION_SYSTEM_PROMPT = """You are an analyst identifying repeated questions across users in an enterprise AI agent assistant.

You will receive a list of user messages from different users. Your task is to:
1. Group messages that are essentially asking the SAME question (even if worded differently)
2. Only report groups where 2+ DIFFERENT users asked the same question
3. For each group, provide a canonical question summary

A "repeated question" means different people independently asked the AI the same thing — this wastes tokens and should be solved once (via documentation, a Skill, or a shared tool).

Return a JSON object:
{
  "repeatedQuestions": [
    {
      "canonicalQuestion": "string — a clear summary of what they all asked",
      "messageIndices": [1, 5, 12],
      "category": "one of: knowledge_query | routine_task | config_lookup | code_generation | debugging | other"
    }
  ]
}

Rules:
- Only group messages that are semantically the SAME question, not just the same topic
- Ignore messages that are unique one-off questions
- If no repeated questions are found, return an empty array"""


async def discover_repeated_questions(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    """Discover high-frequency repeated questions asked by multiple users.

    Identifies questions that multiple different users independently asked the AI,
    which wastes tokens and should be addressed via documentation, Skills, or shared tools.
    """
    print("[L3-2] Discovering high-frequency repeated questions...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            SELECT row_id, session_id, sender_id, content_text
            FROM `{table_name}`
            WHERE role = 'user'
              AND timestamp >= %s AND timestamp < %s
              AND content_text IS NOT NULL AND content_text != ''
              AND sender_id IS NOT NULL AND sender_id != ''
            ORDER BY timestamp
        """

        rows = await execute_query_async(adb_config, sql, (start_time, end_time))

        if not rows:
            print("[L3-2] No user messages found")
            return {"repeatedQuestions": [], "totalMessagesAnalyzed": 0}

        for row in rows:
            row["user_prompt"] = _extract_user_message(row.get("content_text") or "")

        rows = [row for row in rows if row["user_prompt"]]

        if not rows:
            print("[L3-2] No valid user prompts after extraction")
            return {"repeatedQuestions": [], "totalMessagesAnalyzed": 0}

        # Build numbered message list with sender info for LLM
        messages_for_llm = [
            f"[{i + 1}] (user: {row.get('sender_id', 'unknown')})\n{row['user_prompt'][:300]}"
            for i, row in enumerate(rows)
        ]

        estimated_tokens = count_tokens(messages_for_llm)

        if estimated_tokens < MAX_BATCH_TOKENS:
            print(f"[L3-2] Estimated {estimated_tokens} tokens (< 128K), sending all in one batch")
            combined_messages = "\n\n---\n\n".join(messages_for_llm)
            user_prompt = (
                f"Analyze these {len(messages_for_llm)} user messages from an enterprise AI agent assistant. "
                f"Find questions that DIFFERENT users asked independently but are essentially the same question.\n\n"
                f"{combined_messages}"
            )
            raw_result = await llm_client.chat_json(_REPEATED_QUESTION_SYSTEM_PROMPT, user_prompt)
        else:
            print(f"[L3-2] Estimated {estimated_tokens} tokens (>= 128K), splitting into batches")
            batch_size = 15
            all_repeated: list[dict] = []
            for batch_start in range(0, len(messages_for_llm), batch_size):
                batch = messages_for_llm[batch_start:batch_start + batch_size]
                combined = "\n\n---\n\n".join(batch)
                user_prompt = (
                    f"Analyze these {len(batch)} user messages. "
                    f"Find questions that DIFFERENT users asked independently but are essentially the same.\n\n"
                    f"{combined}"
                )
                batch_result = await llm_client.chat_json(_REPEATED_QUESTION_SYSTEM_PROMPT, user_prompt)
                all_repeated.extend(batch_result.get("repeatedQuestions", []))
            raw_result = {"repeatedQuestions": all_repeated}

        # Enrich results with actual user counts and sender details
        repeated_questions = []
        for group in raw_result.get("repeatedQuestions", []):
            indices = group.get("messageIndices", [])
            senders = set()
            sample_messages = []
            for idx in indices:
                actual_idx = idx - 1
                if 0 <= actual_idx < len(rows):
                    sender = rows[actual_idx].get("sender_id", "unknown")
                    senders.add(sender)
                    if len(sample_messages) < 3:
                        sample_messages.append({
                            "senderId": sender,
                            "sessionId": rows[actual_idx]["session_id"],
                            "preview": rows[actual_idx]["user_prompt"][:150],
                        })

            if len(senders) < 2:
                continue

            repeated_questions.append({
                "canonicalQuestion": group.get("canonicalQuestion", ""),
                "category": group.get("category", "other"),
                "uniqueUsers": len(senders),
                "totalOccurrences": len(indices),
                "senders": sorted(senders),
                "sampleMessages": sample_messages,
            })

        repeated_questions.sort(key=lambda x: x["uniqueUsers"], reverse=True)

        print(f"[L3-2] Found {len(repeated_questions)} repeated questions across multiple users")
        return {
            "repeatedQuestions": repeated_questions,
            "totalMessagesAnalyzed": len(rows),
        }

    except Exception as error:
        print(f"[L3-2] Error discovering repeated questions: {error}")
        return {"repeatedQuestions": [], "totalMessagesAnalyzed": 0}


# ─── L3-3: Extract Best Practices ───

async def extract_best_practices(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    print("[L3-3] Extracting best practices from successful sessions...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        sql = f"""
            WITH session_stats AS (
                SELECT session_id, sender_id,
                    SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_turns,
                    SUM(CASE WHEN is_error = 1 THEN 1 ELSE 0 END) AS error_count
                FROM `{table_name}`
                WHERE timestamp >= %s AND timestamp < %s
                GROUP BY session_id, sender_id
            ),
            last_stop AS (
                SELECT session_id, stop_reason,
                    ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp DESC) AS rn
                FROM `{table_name}`
                WHERE timestamp >= %s AND timestamp < %s AND stop_reason IS NOT NULL
            )
            SELECT ss.session_id, ss.sender_id
            FROM session_stats ss
            LEFT JOIN last_stop ls ON ss.session_id = ls.session_id AND ls.rn = 1
            WHERE ss.user_turns <= 3
              AND ss.error_count = 0
              AND ls.stop_reason IN ('stop', 'end_turn')
            LIMIT 100
        """

        session_rows = await execute_query_async(
            adb_config, sql,
            (start_time, end_time, start_time, end_time)
        )

        if not session_rows:
            print("[L3-3] No successful sessions found")
            return {"bestPractices": [], "commonPatterns": []}

        session_ids = [row["session_id"] for row in session_rows]
        placeholders = ", ".join(["%s"] * len(session_ids))

        message_sql = f"""
            WITH first_msgs AS (
                SELECT session_id, content_text,
                    ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp ASC) AS rn
                FROM `{table_name}`
                WHERE role = 'user'
                  AND session_id IN ({placeholders})
                  AND content_text IS NOT NULL AND content_text != ''
            )
            SELECT session_id, content_text FROM first_msgs WHERE rn = 1
        """

        message_rows = await execute_query_async(
            adb_config, message_sql, tuple(session_ids)
        )

        user_prompts = [row["content_text"][:500] for row in message_rows[:50]]

        system_prompt = """You are an expert in AI prompt engineering and enterprise AI agent assistant best practices.
Analyze successful user prompts to identify patterns and extract actionable best practices."""

        user_prompt = f"""Analyze the following successful user prompts from enterprise AI agent assistant sessions.
These prompts resulted in successful task completion (<=3 user turns, 0 errors, normal completion).

User Prompts:
{chr(10).join(f"{i + 1}. {prompt}..." for i, prompt in enumerate(user_prompts))}

Extract 5-10 best practices and identify common patterns.

Return a JSON object with this structure:
{{
  "bestPractices": [
    {{"title": "string", "description": "string", "example": "string"}}
  ],
  "commonPatterns": ["string"]
}}"""

        result = await llm_client.chat_json(system_prompt, user_prompt)
        print(f"[L3-3] Extracted {len(result.get('bestPractices', []))} best practices")
        return result

    except Exception as error:
        print(f"[L3-3] Error extracting best practices: {error}")
        return {"bestPractices": [], "commonPatterns": []}


# ─── L3-4: Discover Skill Candidates ───

_SKILL_SYSTEM_PROMPT = """You are an AI workflow automation specialist who understands the Skill specification for enterprise AI agent assistants.

## What is a Skill?

A Skill is a self-contained, reusable automation package that an AI agent can invoke to complete a specific, well-defined task. It is NOT a generic assistant or a vague capability.

A valid Skill MUST satisfy ALL of the following criteria:
1. **Deterministic workflow**: The Skill follows a clear, repeatable sequence of steps (e.g., read config → modify → validate → deploy). It is not an open-ended conversation.
2. **Clear input/output contract**: The Skill has well-defined inputs (e.g., a file path, a service name, a config template) and produces a concrete output (e.g., a deployed service, a generated report, a validated config).
3. **Tool-chain backed**: The Skill leverages specific tool calls (exec, read, write, web_fetch, etc.) in a predictable pattern. The tool chain pattern should be observable in the usage data.
4. **Domain-specific**: The Skill solves a specific domain problem (e.g., "K8s ingress configuration", "database migration"), NOT a generic capability (e.g., "answer questions", "run commands").
5. **Automatable end-to-end**: The entire workflow can be automated without human intervention once triggered. If it requires subjective judgment at every step, it is NOT a Skill.
6. **High frequency + multi-user**: The workflow is performed frequently (multiple times per week) by multiple different users, indicating organizational-level demand.

## What is NOT a Skill:
- Generic Q&A or knowledge retrieval (that's just the base agent capability)
- Running arbitrary shell commands (too generic, no domain specificity)
- Vague categories like "security testing" or "code review" without a specific workflow
- One-off tasks that don't recur

Only recommend candidates that can realistically be developed into a Skill package with a YAML/Markdown spec, system prompt, and tool definitions."""


async def discover_skill_candidates(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    """Discover skill candidates by querying task chain data directly from DB.

    Extracts the first user message and tool sequence for each task chain,
    then asks the LLM to identify repeatable, automatable workflow patterns.
    """
    print("[L3-4] Discovering skill candidates...")

    try:
        start_time, end_time = time_range_to_sql_params(range_)

        # Single SQL: for each task chain, get the first user message and the
        # ordered tool sequence.  Uses the standard task_chain_id window
        # (cumulative count of role='user' rows partitioned by session_id).
        sql = f"""
            WITH base AS (
                SELECT session_id, sender_id, role, content_text, tool_name,
                    SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                        OVER (PARTITION BY session_id ORDER BY timestamp, row_id) AS task_chain_id,
                    ROW_NUMBER()
                        OVER (PARTITION BY session_id ORDER BY timestamp, row_id) AS msg_seq
                FROM `{table_name}`
                WHERE timestamp >= %s AND timestamp < %s
            ),
            first_user AS (
                SELECT session_id, task_chain_id, sender_id, content_text,
                    ROW_NUMBER() OVER (
                        PARTITION BY session_id, task_chain_id ORDER BY msg_seq
                    ) AS rn
                FROM base
                WHERE role = 'user'
            ),
            tool_seq AS (
                SELECT session_id, task_chain_id,
                    GROUP_CONCAT(tool_name ORDER BY msg_seq SEPARATOR ' -> ') AS tool_chain
                FROM base
                WHERE tool_name IS NOT NULL AND tool_name != ''
                GROUP BY session_id, task_chain_id
            )
            SELECT
                fu.session_id,
                fu.task_chain_id,
                fu.sender_id,
                fu.content_text AS user_message,
                ts.tool_chain
            FROM first_user fu
            LEFT JOIN tool_seq ts
                ON fu.session_id = ts.session_id
                AND fu.task_chain_id = ts.task_chain_id
            WHERE fu.rn = 1
              AND fu.content_text IS NOT NULL AND fu.content_text != ''
            ORDER BY fu.session_id, fu.task_chain_id
        """

        rows = await execute_query_async(adb_config, sql, (start_time, end_time))

        if not rows:
            print("[L3-4] No task chains found")
            return {"skillCandidates": []}

        # Extract actual user messages and truncate
        for row in rows:
            row["user_prompt"] = _extract_user_message(row.get("user_message") or "")[:300]

        rows = [row for row in rows if row["user_prompt"]]

        if not rows:
            print("[L3-4] No valid task chains after extraction")
            return {"skillCandidates": []}

        # Build compact summaries for LLM: "[idx] (user: X) message | tools: A->B->C"
        task_chain_summaries = []
        for i, row in enumerate(rows):
            tool_chain = row.get("tool_chain") or "(no tools)"
            summary = (
                f"[{i + 1}] (user: {row.get('sender_id', 'unknown')})\n"
                f"  Request: {row['user_prompt']}\n"
                f"  Tools: {tool_chain}"
            )
            task_chain_summaries.append(summary)

        estimated_tokens = count_tokens(task_chain_summaries)

        if estimated_tokens < MAX_BATCH_TOKENS:
            print(f"[L3-4] {estimated_tokens} tokens (< 128K), sending all in one batch")
        else:
            # Trim to fit within token budget — keep the most recent chains
            while estimated_tokens >= MAX_BATCH_TOKENS and len(task_chain_summaries) > 50:
                task_chain_summaries = task_chain_summaries[-len(task_chain_summaries) // 2:]
                estimated_tokens = count_tokens(task_chain_summaries)
            print(f"[L3-4] Trimmed to {len(task_chain_summaries)} chains ({estimated_tokens} tokens)")

        combined_data = "\n\n---\n\n".join(task_chain_summaries)

        user_prompt = f"""Analyze the following {len(task_chain_summaries)} task chains from an enterprise AI agent assistant.
Each task chain shows a user's request and the tool sequence the Agent executed.

{combined_data}

Based on the Skill specification above, identify exactly 3 Skill candidates that:
1. Have a clear, deterministic workflow (observable tool chain pattern that repeats across multiple task chains)
2. Are domain-specific (not generic)
3. Have the highest combination of frequency x unique users x automation potential
4. Can be realistically packaged as a self-contained Skill with input/output contract

For each candidate, explain WHY it qualifies as a Skill (which recurring tool chain pattern supports it, what the concrete input/output would be).

Return a JSON object with this structure:
{{
  "skillCandidates": [
    {{
      "name": "string",
      "description": "string",
      "trigger": "string",
      "workflow": "string — the concrete step-by-step workflow (e.g., read config -> modify -> validate -> apply)",
      "inputContract": "string — what inputs the Skill needs",
      "outputContract": "string — what the Skill produces",
      "supportingEvidence": "string — which recurring tool chain pattern and user requests support this",
      "estimatedWeeklyUsage": number,
      "uniqueUsers": number,
      "automationPotential": "high|medium|low"
    }}
  ]
}}"""

        result = await llm_client.chat_json(_SKILL_SYSTEM_PROMPT, user_prompt)
        print(f"[L3-4] Discovered {len(result.get('skillCandidates', []))} skill candidates")
        return result

    except Exception as error:
        print(f"[L3-4] Error discovering skill candidates: {error}")
        return {"skillCandidates": []}


# ─── L3-5: Generate Narrative Report ───

def generate_narrative_report(
    all_results: dict,
    range_: TimeRange,
) -> dict:
    """Generate a narrative-style report using template-based rendering.

    No LLM call — pure Python template filling.  Produces a section-by-section
    data report covering L1 (Operational), L2 (Behavior), L3 (Organizational).
    """
    print("[L3-5] Generating narrative report (template mode)...")

    l1 = all_results.get("l1", {})
    l2 = all_results.get("l2", {})
    l3 = all_results.get("l3", {})
    period = f"{range_.start_date} ~ {range_.end_date}"

    sections: list[str] = []

    # ── Helper ──
    def _round2(val) -> str:
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return str(val)

    # ════════════════════════════════════════════════
    # 1. Executive Summary
    # ════════════════════════════════════════════════
    te = l1.get("tokenEfficiency", {}).get("overall", {})
    sr = l2.get("successRate", {}).get("overall", {})
    total_sessions = te.get("totalSessions", 0)
    total_cost = te.get("totalCost", 0)
    success = sr.get("success", 0)
    partial = sr.get("partial", 0)
    failure = sr.get("failure", 0)
    total_tasks = success + partial + failure
    anomaly_count = len(l1.get("anomalies", {}).get("anomalies", []))

    sections.append(
        f"## 1. Executive Summary\n\n"
        f"> 分析周期：{period}\n\n"
        f"本分析周期内共 **{total_sessions}** 个会话、**{total_tasks}** 条任务链，"
        f"总成本 **{_round2(total_cost)}**。"
        f"任务成功率 **{success}/{total_tasks}**"
        f"（失败 {failure}），检测到 **{anomaly_count}** 个异常。"
    )

    # ════════════════════════════════════════════════
    # 2. L1 Operational Efficiency
    # ════════════════════════════════════════════════
    l1_lines: list[str] = ["## 2. L1 运营效率\n"]

    # 2.1 Token Consumption
    te_full = l1.get("tokenEfficiency", {})
    by_model = te_full.get("byModel", [])
    by_user = te_full.get("byUser", [])
    if by_model or te:
        l1_lines.append("### 2.1 Token 消耗与成本效率\n")
        if te:
            l1_lines.append(_kv_lines(te) + "\n")
        if by_model:
            l1_lines.append(_md_table(
                ["模型", "会话数", "输入", "输出", "输出/输入", "缓存%", "成本", "均成本"],
                [[m.get("model", ""), m.get("sessionCount", ""), m.get("totalInput", ""),
                  m.get("totalOutput", ""), m.get("outputInputRatio", ""),
                  m.get("cacheHitRatePct", ""), m.get("totalCost", ""),
                  m.get("avgCostPerSession", "")] for m in by_model],
            ) + "\n")
        if by_user:
            # Sort by total tokens (input + output) descending, show Top 5
            top5_users = sorted(
                by_user,
                key=lambda u: (u.get("totalInput", 0) or 0) + (u.get("totalOutput", 0) or 0),
                reverse=True,
            )[:5]
            l1_lines.append("**Token 消耗 Top 5 用户：**\n")
            l1_lines.append(_md_table(
                ["用户", "会话数", "输入 token", "输出 token", "总 token", "成本"],
                [[u.get("senderId", ""),
                  u.get("sessionCount", ""),
                  u.get("totalInput", ""),
                  u.get("totalOutput", ""),
                  (u.get("totalInput", 0) or 0) + (u.get("totalOutput", 0) or 0),
                  u.get("totalCost", "")]
                 for u in top5_users],
            ) + "\n")

    # 2.2 Task Chain Depth
    sd = l1.get("sessionDepth", {})
    buckets = sd.get("bucketDistribution", [])
    if buckets:
        l1_lines.append("### 2.2 任务链深度分布\n")
        l1_lines.append(f"总任务链数：{sd.get('totalChains', '?')}\n")
        l1_lines.append(_md_table(
            ["深度", "链数", "均消息", "均时长(s)", "均工具", "均 token", "均成本"],
            [[b.get("depthBucket", ""), b.get("chainCount", ""), b.get("avgMessages", ""),
              b.get("avgDurationSeconds", ""), b.get("avgToolCalls", ""),
              b.get("avgTokens", ""), b.get("avgCost", "")] for b in buckets],
        ) + "\n")

    # 2.3 Tool Chain Patterns
    tc = l1.get("toolChains", {})
    bigrams = tc.get("topBigrams", [])[:10]
    trigrams = tc.get("topTrigrams", [])[:10]
    tool_success = tc.get("toolSuccessRates", [])[:10]
    if bigrams or trigrams:
        l1_lines.append("### 2.3 工具链模式\n")
        if bigrams:
            l1_lines.append("**Top Bigrams：**\n")
            l1_lines.append(_md_table(["模式", "次数"], [[b.get("pattern", ""), b.get("count", "")] for b in bigrams]) + "\n")
        if trigrams:
            l1_lines.append("**Top Trigrams：**\n")
            l1_lines.append(_md_table(["模式", "次数"], [[t.get("pattern", ""), t.get("count", "")] for t in trigrams]) + "\n")
        if tool_success:
            l1_lines.append("**工具成功率：**\n")
            l1_lines.append(_md_table(
                ["工具", "调用数", "成功率"],
                [[s.get("toolName", ""), s.get("totalCalls", ""), s.get("successRate", "")] for s in tool_success],
            ) + "\n")

    # 2.4 High-Cost Sessions
    hc = l1.get("highCostSessions", {})
    chains = hc.get("taskChains", [])[:10]
    if chains:
        l1_lines.append("### 2.4 高成本会话\n")
        l1_lines.append(_md_table(
            ["用户", "会话", "token", "成本", "消息", "工具", "错误", "时长(s)", "成本驱动"],
            [[c.get("senderId", ""), c.get("sessionId", ""), c.get("totalTokens", ""),
              c.get("totalCost", ""), c.get("messageCount", ""), c.get("toolCallCount", ""),
              c.get("toolErrorCount", ""), c.get("durationSeconds", ""),
              ", ".join(c.get("costDrivers", []))] for c in chains],
        ) + "\n")

    # 2.5 Anomaly Detection
    anomalies = l1.get("anomalies", {}).get("anomalies", [])
    if anomalies:
        l1_lines.append("### 2.5 异常检测\n")
        l1_lines.append(_md_table(
            ["用户", "类型", "实际值", "均值", "标准差", "Z 分", "严重度"],
            [[a.get("senderId", ""), a.get("anomalyType", ""), a.get("actualValue", ""),
              a.get("mean", ""), a.get("stddev", ""), a.get("zScore", ""),
              a.get("severity", "")] for a in anomalies],
        ) + "\n")

    sections.append("\n".join(l1_lines))

    # ════════════════════════════════════════════════
    # 3. L2 User Behavior
    # ════════════════════════════════════════════════
    l2_lines: list[str] = ["## 3. L2 用户行为\n"]

    # 3.1 Intent Classification
    intents = l2.get("intents", {})
    intent_dist = intents.get("distribution", {})
    if intent_dist:
        l2_lines.append("### 3.1 意图分类\n")
        total_intents = sum(intent_dist.values())
        l2_lines.append(_md_table(
            ["意图", "次数", "占比"],
            [[k, v, f"{v / total_intents * 100:.1f}%" if total_intents else "N/A"]
             for k, v in sorted(intent_dist.items(), key=lambda x: -x[1])],
        ) + "\n")

    # 3.2 Task Complexity
    complexity = l2.get("complexity", {})
    comp_dist = complexity.get("distribution", {})
    top_complex = complexity.get("topComplex", [])[:5]
    if comp_dist:
        l2_lines.append("### 3.2 任务复杂度\n")
        l2_lines.append(_kv_lines(comp_dist) + "\n")
    if top_complex:
        l2_lines.append(_md_table(
            ["用户", "会话", "复杂度", "轮次", "工具", "思考长度", "时长(min)"],
            [[c.get("senderId", ""), c.get("sessionId", ""), c.get("complexityScore", ""),
              c.get("userTurns", ""), c.get("toolCallCount", ""),
              c.get("thinkingLength", ""), c.get("durationMinutes", "")] for c in top_complex],
        ) + "\n")

    # 3.3 Task Success Rate
    sr_data = l2.get("successRate", {})
    sr_overall = sr_data.get("overall", {})
    if sr_overall:
        l2_lines.append("### 3.3 任务成功率\n")
        l2_lines.append(_kv_lines(sr_overall) + "\n")
    failures = sr_data.get("failures", [])[:10]
    if failures:
        l2_lines.append("**失败任务链：**\n")
        l2_lines.append(_md_table(
            ["用户", "会话", "链 ID", "结果"],
            [[f.get("senderId", ""), f.get("sessionId", ""), f.get("taskChainId", ""),
              f.get("outcome", "")] for f in failures],
        ) + "\n")

    # 3.4 Prompt Quality
    pq = l2.get("promptQuality", {})
    team_avg = pq.get("teamAverage", {})
    if team_avg:
        l2_lines.append("### 3.4 Prompt 质量\n")
        l2_lines.append(f"团队平均：{_kv_lines(team_avg)}\n")
    top_u = pq.get("topUsers", [])
    bot_u = pq.get("bottomUsers", [])
    if top_u:
        l2_lines.append("**Top 用户：**\n")
        l2_lines.append(_md_table(
            ["用户", "综合分", "最佳 Prompt 预览"],
            [[u.get("senderId", ""), _round2(u.get("overall", "")),
              (u.get("bestPrompt", {}).get("content", "") or "")[:100]] for u in top_u],
        ) + "\n")
    if bot_u:
        l2_lines.append("**Bottom 用户：**\n")
        l2_lines.append(_md_table(
            ["用户", "综合分", "最差 Prompt 预览"],
            [[u.get("senderId", ""), _round2(u.get("overall", "")),
              (u.get("worstPrompt", {}).get("content", "") or "")[:100]] for u in bot_u],
        ) + "\n")

    # 3.5 Topic Clustering
    topics = l2.get("topics", {})
    cat_dist = topics.get("categoryDistribution", {})
    top_tags = topics.get("topTags", [])
    if cat_dist:
        l2_lines.append("### 3.5 话题聚类\n")
        l2_lines.append(_kv_lines(cat_dist) + "\n")
    if top_tags:
        l2_lines.append(_md_table(
            ["标签", "类别", "次数", "用户数"],
            [[t.get("tag", ""), t.get("category", ""), t.get("count", ""),
              t.get("uniqueUsers", "")] for t in top_tags],
        ) + "\n")

    # 3.6 Retry Behavior
    retry = l2.get("retryBehavior", {})
    if retry:
        l2_lines.append("### 3.6 重试行为\n")
        l2_lines.append(
            f"- 重试率：{retry.get('retryRate', '?')}\n"
            f"- 总会话：{retry.get('totalSessions', '?')}\n"
            f"- 重试会话：{retry.get('retrySessionCount', '?')}\n"
        )

    # 3.7 Thinking Depth
    td = l2.get("thinkingDepth", {})
    by_depth = td.get("byDepth", [])
    by_model_td = td.get("byModel", [])
    if by_depth:
        l2_lines.append("### 3.7 思考深度\n")
        l2_lines.append(_md_table(
            ["深度", "消息数", "均输出 token", "均成本", "均内容长度"],
            [[d.get("thinkingDepth", ""), d.get("messageCount", ""),
              d.get("avgOutputTokens", ""), d.get("avgCost", ""),
              d.get("avgContentLength", "")] for d in by_depth],
        ) + "\n")
    if by_model_td:
        l2_lines.append(_md_table(
            ["模型", "总消息", "思考数", "思考%", "均思考长度"],
            [[m.get("model", ""), m.get("totalMessages", ""), m.get("thinkingCount", ""),
              m.get("thinkingPct", ""), m.get("avgThinkingLength", "")] for m in by_model_td],
        ) + "\n")

    # 3.8 User Maturity
    maturity = l2.get("userMaturity", {})
    maturity_users = maturity.get("users", [])[:20]
    if maturity_users:
        l2_lines.append("### 3.8 用户成熟度\n")
        l2_lines.append(_md_table(
            ["用户", "Prompt 数", "平均分", "趋势", "斜率"],
            [[u.get("senderId", ""), u.get("promptCount", ""), _round2(u.get("overallAvg", "")),
              u.get("trend", ""), _round2(u.get("slope", ""))] for u in maturity_users],
        ) + "\n")

    sections.append("\n".join(l2_lines))

    # ════════════════════════════════════════════════
    # 4. L3 Organizational Cognition
    # ════════════════════════════════════════════════
    l3_lines: list[str] = ["## 4. L3 组织认知\n"]

    # 4.1 Tech Stack
    tech_data = l3.get("techStack", {})
    tech_list = tech_data.get("technologies", [])
    if tech_list:
        l3_lines.append("### 4.1 技术栈热力图\n")
        l3_lines.append(_md_table(
            ["技术", "类别", "会话数", "用户数"],
            [[t.get("technology", ""), t.get("category", ""), t.get("sessionCount", ""),
              t.get("uniqueUsers", "")] for t in tech_list],
        ) + "\n")

    # 4.2 Repeated Questions
    rq = l3.get("repeatedQuestions", {})
    rq_list = rq.get("repeatedQuestions", [])
    if rq_list:
        l3_lines.append("### 4.2 高频重复问题\n")
        l3_lines.append(_md_table(
            ["问题", "类别", "用户数", "总次数", "涉及用户"],
            [[q.get("canonicalQuestion", ""), q.get("category", ""),
              q.get("uniqueUsers", ""), q.get("totalOccurrences", ""),
              ", ".join(q.get("senders", [])) if isinstance(q.get("senders"), list) else q.get("senders", "")]
             for q in rq_list],
        ) + "\n")

    # 4.3 Best Practices
    bp = l3.get("bestPractices", {})
    bp_list = bp.get("bestPractices", [])
    if bp_list:
        l3_lines.append("### 4.3 最佳实践\n")
        for practice in bp_list:
            title = practice.get("title", practice.get("name", ""))
            desc = practice.get("description", "")
            l3_lines.append(f"- **{title}**：{desc}\n")
        l3_lines.append("")

    # 4.4 Skill Candidates
    sc = l3.get("skillCandidates", {})
    sc_list = sc.get("skillCandidates", [])
    if sc_list:
        l3_lines.append("### 4.4 Skill 候选\n")
        for skill in sc_list:
            l3_lines.append(
                f"- **{skill.get('name', '')}**：{skill.get('description', '')}\n"
                f"  - 触发条件：{skill.get('trigger', '')}\n"
                f"  - 工作流：{skill.get('workflow', '')}\n"
            )
        l3_lines.append("")

    sections.append("\n".join(l3_lines))

    report_text = "\n\n---\n\n".join(sections)
    print("[L3-5] Narrative report generated successfully (template mode)")
    return {"report": report_text}

# ─── Data Formatting Helper (Markdown tables) ───

def _md_table(headers: list[str], rows: list[list]) -> str:
    """Build a compact Markdown table from headers and row data."""
    if not rows:
        return "(no data)"
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _kv_lines(data: dict, key_labels: dict[str, str] | None = None) -> str:
    """Format a flat dict as ``label: value`` lines."""
    if not data:
        return "(no data)"
    parts: list[str] = []
    for key, value in data.items():
        label = key_labels.get(key, key) if key_labels else key
        parts.append(f"{label}: {value}")
    return "\n".join(parts)


def _format_for_report(all_results: dict) -> str:
    """Convert analysis results into compact Markdown tables for LLM consumption.

    This replaces the old ``_summarize_for_report`` + ``json.dumps`` approach.
    Markdown tables eliminate JSON key-name bloat (quotes, braces, colons) and
    reduce token usage by ~40-60% while preserving all data without truncation.
    """
    l1 = all_results.get("l1", {})
    l2 = all_results.get("l2", {})
    l3 = all_results.get("l3", {})
    sections: list[str] = []

    def add(label: str, content: str) -> None:
        if content and content != "(no data)":
            sections.append(f"【{label}】\n{content}")

    # ── L1-1 Token Efficiency ──
    te = l1.get("tokenEfficiency", {})
    if te:
        overall = te.get("overall", {})
        overall_text = _kv_lines(overall) if overall else ""
        by_model = te.get("byModel", [])
        model_table = _md_table(
            ["model", "sessions", "input", "output", "out/in", "cache%", "cost", "avg_cost"],
            [[m.get("model", ""), m.get("sessionCount", ""), m.get("totalInput", ""),
              m.get("totalOutput", ""), m.get("outputInputRatio", ""),
              m.get("cacheHitRatePct", ""), m.get("totalCost", ""),
              m.get("avgCostPerSession", "")] for m in by_model],
        ) if by_model else ""
        by_user = te.get("byUser", [])
        # Sort by total tokens descending, cap at Top 5
        top5_users = sorted(
            by_user,
            key=lambda u: (u.get("totalInput", 0) or 0) + (u.get("totalOutput", 0) or 0),
            reverse=True,
        )[:5] if by_user else []
        user_table = _md_table(
            ["sender", "sessions", "input", "output", "total_tokens", "cost"],
            [[u.get("senderId", ""), u.get("sessionCount", ""), u.get("totalInput", ""),
              u.get("totalOutput", ""),
              (u.get("totalInput", 0) or 0) + (u.get("totalOutput", 0) or 0),
              u.get("totalCost", "")]
             for u in top5_users],
        ) if top5_users else ""
        parts = [p for p in [overall_text, model_table] if p]
        if user_table:
            parts.append(f"Token Top 5 Users:\n{user_table}")
        add("L1-1 Token 消耗与成本效率", "\n\n".join(parts))

    # ── L1-2 Session Depth ──
    sd = l1.get("sessionDepth", {})
    if sd:
        total = f"totalChains: {sd.get('totalChains', '?')}"
        buckets = sd.get("bucketDistribution", [])
        bucket_table = _md_table(
            ["depth", "chains", "avg_msgs", "avg_dur_s", "avg_tools", "avg_tokens", "avg_cost", "sum_tokens"],
            [[b.get("depthBucket", ""), b.get("chainCount", ""), b.get("avgMessages", ""),
              b.get("avgDurationSeconds", ""), b.get("avgToolCalls", ""),
              b.get("avgTokens", ""), b.get("avgCost", ""), b.get("sumTokens", "")]
             for b in buckets],
        ) if buckets else ""
        add("L1-2 任务链深度分布", f"{total}\n{bucket_table}" if bucket_table else total)

    # ── L1-3 Tool Chains (cap 10 each) ──
    tc = l1.get("toolChains", {})
    if tc:
        bigrams = _md_table(
            ["pattern", "count"],
            [[b.get("pattern", ""), b.get("count", "")] for b in tc.get("topBigrams", [])[:10]],
        )
        trigrams = _md_table(
            ["pattern", "count"],
            [[t.get("pattern", ""), t.get("count", "")] for t in tc.get("topTrigrams", [])[:10]],
        )
        success = _md_table(
            ["tool", "calls", "success%"],
            [[s.get("toolName", ""), s.get("totalCalls", ""), s.get("successRate", "")]
             for s in tc.get("toolSuccessRates", [])[:10]],
        )
        parts = []
        if bigrams != "(no data)":
            parts.append(f"Top Bigrams:\n{bigrams}")
        if trigrams != "(no data)":
            parts.append(f"Top Trigrams:\n{trigrams}")
        if success != "(no data)":
            parts.append(f"Tool Success Rates:\n{success}")
        if parts:
            add("L1-3 工具链模式", "\n\n".join(parts))

    # ── L1-4 High Cost Sessions (cap 10, drop userFirstMessage) ──
    hc = l1.get("highCostSessions", {})
    chains = hc.get("taskChains", [])[:10]
    if chains:
        add("L1-4 高成本会话", _md_table(
            ["sender", "session", "tokens", "cost", "msgs", "tools", "errors", "duration_s", "drivers"],
            [[c.get("senderId", ""), c.get("sessionId", ""), c.get("totalTokens", ""),
              c.get("totalCost", ""), c.get("messageCount", ""), c.get("toolCallCount", ""),
              c.get("toolErrorCount", ""), c.get("durationSeconds", ""),
              ", ".join(c.get("costDrivers", []))] for c in chains],
        ))

    # ── L1-5 Anomalies ──
    anomalies = l1.get("anomalies", {}).get("anomalies", [])
    if anomalies:
        add("L1-5 异常检测", _md_table(
            ["sender", "type", "actual", "mean", "stddev", "z", "severity"],
            [[a.get("senderId", ""), a.get("anomalyType", ""), a.get("actualValue", ""),
              a.get("mean", ""), a.get("stddev", ""), a.get("zScore", ""),
              a.get("severity", "")] for a in anomalies],
        ))

    # ── L2-1 Intents (drop items array) ──
    intents = l2.get("intents", {})
    if intents:
        dist = intents.get("distribution", {})
        dist_table = _md_table(
            ["intent", "count"],
            [[k, v] for k, v in sorted(dist.items(), key=lambda x: -x[1])],
        ) if dist else ""

        senders_by_cat = intents.get("sendersByCategory", {})
        sbc_lines = "\n".join(
            f"{cat}: {', '.join(senders)}"
            for cat, senders in senders_by_cat.items() if senders
        ) if senders_by_cat else ""

        # byUser intent table — keep for cross-referencing user behavior
        by_user = intents.get("byUser", {})
        if by_user:
            all_cats = sorted({cat for user_cats in by_user.values() for cat in user_cats})
            user_rows = []
            for sender, cats in by_user.items():
                user_rows.append([sender] + [cats.get(c, 0) for c in all_cats])
            user_table = _md_table(["sender"] + all_cats, user_rows)
        else:
            user_table = ""

        # NOTE: items array is intentionally skipped — too large, not needed for report
        parts = [p for p in [dist_table, sbc_lines, user_table] if p]
        if parts:
            add("L2-1 意图分类分布", "\n\n".join(parts))

    # ── L2-2 Complexity (drop byUser, cap topComplex at 5) ──
    complexity = l2.get("complexity", {})
    if complexity:
        dist = complexity.get("distribution", {})
        dist_text = _kv_lines(dist) if dist else ""
        top_complex = complexity.get("topComplex", [])[:5]
        tc_table = _md_table(
            ["sender", "session", "score", "turns", "tools", "thinking", "dur_min"],
            [[c.get("senderId", ""), c.get("sessionId", ""), c.get("complexityScore", ""),
              c.get("userTurns", ""), c.get("toolCallCount", ""),
              c.get("thinkingLength", ""), c.get("durationMinutes", "")]
             for c in top_complex],
        ) if top_complex else ""
        parts = [p for p in [dist_text, tc_table] if p]
        if parts:
            add("L2-2 任务复杂度", "\n\n".join(parts))

    # ── L2-3 Success Rate ──
    success_rate = l2.get("successRate", {})
    if success_rate:
        overall = success_rate.get("overall", {})
        overall_text = _kv_lines(overall) if overall else ""

        by_user = success_rate.get("byUser", {})
        if by_user:
            user_rows = [[sender, d.get("success", 0), d.get("partial", 0), d.get("failure", 0)]
                         for sender, d in by_user.items()]
            user_table = _md_table(["sender", "success", "partial", "failure"], user_rows)
        else:
            user_table = ""

        failures = success_rate.get("failures", [])[:10]
        fail_table = _md_table(
            ["sender", "session", "chain", "outcome"],
            [[f.get("senderId", ""), f.get("sessionId", ""), f.get("taskChainId", ""),
              f.get("outcome", "")] for f in failures],
        ) if failures else ""

        parts = [p for p in [overall_text, user_table, fail_table] if p]
        if parts:
            add("L2-3 任务成功率", "\n\n".join(parts))

    # ── L2-4 Prompt Quality ──
    pq = l2.get("promptQuality", {})
    if pq:
        team_avg = pq.get("teamAverage", {})
        avg_text = _kv_lines(team_avg) if team_avg else ""

        top_users = pq.get("topUsers", [])
        top_table = _md_table(
            ["sender", "overall", "best_prompt_preview"],
            [[u.get("senderId", ""), u.get("overall", ""),
              (u.get("bestPrompt", {}).get("content", "") or "")[:150]]
             for u in top_users],
        ) if top_users else ""

        bottom_users = pq.get("bottomUsers", [])
        bot_table = _md_table(
            ["sender", "overall", "worst_prompt_preview"],
            [[u.get("senderId", ""), u.get("overall", ""),
              (u.get("worstPrompt", {}).get("content", "") or "")[:150]]
             for u in bottom_users],
        ) if bottom_users else ""

        parts = [p for p in [avg_text] if p]
        if top_table:
            parts.append(f"Top Users:\n{top_table}")
        if bot_table:
            parts.append(f"Bottom Users:\n{bot_table}")
        if parts:
            add("L2-4 Prompt 质量评分", "\n\n".join(parts))

    # ── L2-5 Topics ──
    topics = l2.get("topics", {})
    if topics:
        cat_dist = topics.get("categoryDistribution", {})
        cat_text = _kv_lines(cat_dist) if cat_dist else ""
        top_tags = topics.get("topTags", [])
        tag_table = _md_table(
            ["tag", "category", "count", "users"],
            [[t.get("tag", ""), t.get("category", ""), t.get("count", ""),
              t.get("uniqueUsers", "")] for t in top_tags],
        ) if top_tags else ""
        parts = [p for p in [cat_text, tag_table] if p]
        if parts:
            add("L2-5 话题聚类", "\n\n".join(parts))

    # ── L2-6 Retry Behavior ──
    retry = l2.get("retryBehavior", {})
    if retry:
        summary = (f"retryRate: {retry.get('retryRate', '?')}\n"
                   f"totalSessions: {retry.get('totalSessions', '?')}\n"
                   f"retrySessionCount: {retry.get('retrySessionCount', '?')}")
        add("L2-6 重试行为检测", summary)

    # ── L2-7 Thinking Depth ──
    td = l2.get("thinkingDepth", {})
    if td:
        by_depth = td.get("byDepth", [])
        depth_table = _md_table(
            ["depth", "msgs", "avg_output", "avg_cost", "avg_content_len"],
            [[d.get("thinkingDepth", ""), d.get("messageCount", ""),
              d.get("avgOutputTokens", ""), d.get("avgCost", ""),
              d.get("avgContentLength", "")] for d in by_depth],
        ) if by_depth else ""
        by_model = td.get("byModel", [])
        model_table = _md_table(
            ["model", "total_msgs", "thinking_count", "thinking%", "avg_thinking_len"],
            [[m.get("model", ""), m.get("totalMessages", ""), m.get("thinkingCount", ""),
              m.get("thinkingPct", ""), m.get("avgThinkingLength", "")]
             for m in by_model],
        ) if by_model else ""
        parts = [p for p in [depth_table, model_table] if p]
        if parts:
            add("L2-7 思考深度分布", "\n\n".join(parts))

    # ── L2-8 User Maturity (cap 20, drop dailyScores) ──
    maturity = l2.get("userMaturity", {})
    users = maturity.get("users", [])[:20]
    if users:
        add("L2-8 用户成熟度趋势", _md_table(
            ["sender", "prompts", "avg_score", "trend", "slope"],
            [[u.get("senderId", ""), u.get("promptCount", ""), u.get("overallAvg", ""),
              u.get("trend", ""), u.get("slope", "")] for u in users],
        ))

    # ── L3-1 Tech Stack ──
    tech = l3.get("techStack", {})
    techs = tech.get("technologies", [])
    if techs:
        add("L3-1 技术栈热力图", _md_table(
            ["tech", "sessions", "users"],
            [[t.get("tech", ""), t.get("sessionCount", ""), t.get("uniqueUsers", "")]
             for t in techs],
        ))

    # ── L3-2 Repeated Questions ──
    rq = l3.get("repeatedQuestions", {})
    questions = rq.get("repeatedQuestions", [])
    if questions:
        add("L3-2 高频重复问题", _md_table(
            ["question", "category", "users", "occurrences", "senders"],
            [[q.get("canonicalQuestion", ""), q.get("category", ""),
              q.get("uniqueUsers", ""), q.get("totalOccurrences", ""),
              ", ".join(q.get("senders", []))] for q in questions],
        ))

    # ── L3-3 Best Practices ──
    bp = l3.get("bestPractices", {})
    practices = bp.get("bestPractices", [])
    if practices:
        practice_text = "\n\n".join(
            f"**{p.get('title', '')}**: {p.get('description', '')}\nExample: {p.get('example', '')}"
            for p in practices
        )
        patterns = bp.get("commonPatterns", [])
        pattern_text = "\n".join(f"- {p}" for p in patterns) if patterns else ""
        parts = [p for p in [practice_text, pattern_text] if p]
        if parts:
            add("L3-3 最佳实践", "\n\n".join(parts))

    # ── L3-4 Skill Candidates ──
    sc = l3.get("skillCandidates", {})
    candidates = sc.get("skillCandidates", [])
    if candidates:
        skill_text = "\n\n".join(
            f"**{c.get('name', '')}** (potential: {c.get('automationPotential', '?')}, "
            f"weekly: ~{c.get('estimatedWeeklyUsage', '?')}, users: {c.get('uniqueUsers', '?')})\n"
            f"Description: {c.get('description', '')}\n"
            f"Trigger: {c.get('trigger', '')}\n"
            f"Workflow: {c.get('workflow', '')}"
            for c in candidates
        )
        add("L3-4 技能候选", skill_text)

    return "\n\n".join(sections) if sections else "暂无分析数据"

# ─── L3-5-NEW: Structured Report (总体结论 → 关键问题 → 优化建议) ───

def generate_structured_report(
    all_results: dict,
    range_: TimeRange,
) -> dict:
    """Generate a structured final report using template-based rendering.

    No LLM call — pure Python template filling.  Structure:
    总体结论 → 关键问题 → 优化建议.
    """
    print("[L3-5] Generating structured report (template mode)...")

    l1 = all_results.get("l1", {})
    l2 = all_results.get("l2", {})
    l3 = all_results.get("l3", {})

    sections: list[str] = []
    period = f"{range_.start_date} ~ {range_.end_date}"

    # ────────────────────────────────────────────────
    # Helper: safe getters
    # ────────────────────────────────────────────────
    def _pct(num: float | int, total: float | int) -> str:
        if not total:
            return "N/A"
        return f"{num / total * 100:.1f}%"

    def _round2(val) -> str:
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return str(val)

    # ════════════════════════════════════════════════
    # 一、总体结论
    # ════════════════════════════════════════════════
    part1_lines: list[str] = [f"## 一、总体结论\n\n> 分析周期：{period}\n"]

    # ── 1.1 整体运营健康度 ──
    part1_lines.append("### 1.1 整体运营健康度\n")

    te = l1.get("tokenEfficiency", {})
    overall_te = te.get("overall", {})
    total_sessions = overall_te.get("totalSessions", 0)
    total_cost = overall_te.get("totalCost", 0)

    sr = l2.get("successRate", {})
    sr_overall = sr.get("overall", {})
    success_count = sr_overall.get("success", 0)
    partial_count = sr_overall.get("partial", 0)
    failure_count = sr_overall.get("failure", 0)
    total_tasks = success_count + partial_count + failure_count

    anomalies_data = l1.get("anomalies", {})
    anomaly_list = anomalies_data.get("anomalies", [])
    anomaly_count = len(anomaly_list)

    # Health rating
    success_rate_val = (success_count / total_tasks * 100) if total_tasks else 0
    if success_rate_val >= 85 and anomaly_count <= 2:
        health_icon = "🟢 良好"
    elif success_rate_val >= 70 or anomaly_count <= 5:
        health_icon = "🟡 待改善"
    else:
        health_icon = "🔴 需关注"

    part1_lines.append(f"**整体健康度：{health_icon}**\n")
    part1_lines.append(
        f"- 分析周期内共 **{total_sessions}** 个会话，"
        f"**{total_tasks}** 条任务链，"
        f"总成本 **{_round2(total_cost)}**\n"
        f"- 任务成功率 **{_pct(success_count, total_tasks)}**"
        f"（成功 {success_count} / 部分成功 {partial_count} / 失败 {failure_count}）\n"
        f"- 检测到 **{anomaly_count}** 个异常\n"
    )

    if anomaly_list:
        anomaly_senders = sorted({str(a.get("senderId") or "?") for a in anomaly_list})
        part1_lines.append(f"- 异常涉及用户：{', '.join(anomaly_senders)}\n")

    # Token efficiency by model table
    by_model = te.get("byModel", [])
    if by_model:
        part1_lines.append(_md_table(
            ["模型", "会话数", "输入 token", "输出 token", "输出/输入比", "缓存命中率", "总成本", "均成本"],
            [[m.get("model", ""), m.get("sessionCount", ""), m.get("totalInput", ""),
              m.get("totalOutput", ""), m.get("outputInputRatio", ""),
              m.get("cacheHitRatePct", ""), m.get("totalCost", ""),
              m.get("avgCostPerSession", "")] for m in by_model],
        ))
        part1_lines.append("")

    # Token consumption Top 5 users
    by_user_te = te.get("byUser", [])
    if by_user_te:
        top5_token_users = sorted(
            by_user_te,
            key=lambda u: (u.get("totalInput", 0) or 0) + (u.get("totalOutput", 0) or 0),
            reverse=True,
        )[:5]
        part1_lines.append("**Token 消耗 Top 5 用户：**\n")
        part1_lines.append(_md_table(
            ["用户", "会话数", "输入 token", "输出 token", "总 token", "成本"],
            [[u.get("senderId", ""),
              u.get("sessionCount", ""),
              u.get("totalInput", ""),
              u.get("totalOutput", ""),
              (u.get("totalInput", 0) or 0) + (u.get("totalOutput", 0) or 0),
              u.get("totalCost", "")]
             for u in top5_token_users],
        ))
        part1_lines.append("")

    # ── 1.2 核心使用场景 ──
    part1_lines.append("### 1.2 核心使用场景\n")

    intents = l2.get("intents", {})
    intent_dist = intents.get("distribution", {})
    topics = l2.get("topics", {})
    cat_dist = topics.get("categoryDistribution", {})
    tech = l3.get("techStack", {})
    techs = tech.get("technologies", [])

    # Only show intent classification results here
    total_intent_count = sum(intent_dist.values()) if intent_dist else 0
    scenario_rows: list[list] = []
    if intent_dist:
        for intent_name, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
            pct = f"{count / total_intent_count * 100:.1f}%" if total_intent_count else "N/A"
            scenario_rows.append([intent_name, f"{count} 次（{pct}）"])

    if scenario_rows:
        part1_lines.append(_md_table(["使用场景", "频次（占比）"], scenario_rows))
        part1_lines.append("")

    # Non-work intents
    non_work_categories = {
        "闲聊互动", "生活日常", "情感社交", "教育学习", "影视音乐",
        "游戏电竞", "体育运动", "阅读创作",
        "Casual Chat", "Daily Life", "Emotional Social", "Education",
        "Entertainment", "Gaming", "Sports", "Reading & Writing",
    }
    senders_by_cat = intents.get("sendersByCategory", {})
    non_work_rows: list[list] = []
    for cat, senders in senders_by_cat.items():
        if cat in non_work_categories and senders:
            count = intent_dist.get(cat, len(senders))
            non_work_rows.append([cat, ", ".join(str(s) for s in senders) if isinstance(senders, list) else str(senders), count])
    if non_work_rows:
        part1_lines.append("**非工作类意图：**\n")
        part1_lines.append(_md_table(["非工作意图类别", "涉及用户", "消息数"], non_work_rows))
        part1_lines.append("")

    # ── 1.3 用户行为画像 ──
    part1_lines.append("### 1.3 用户行为画像\n")

    # Prompt quality
    pq = l2.get("promptQuality", {})
    team_avg = pq.get("teamAverage", {})
    if team_avg:
        part1_lines.append(f"**Prompt 质量**：团队平均分 **{_round2(team_avg.get('overall', 'N/A'))}**\n")

    # Build user → top intents lookup from L2-1 byUser data
    intent_by_user = intents.get("byUser", {})

    def _user_top_intents(sender_id: str, top_n: int = 3) -> str:
        """Return the top N intent categories for a given user as a comma-separated string."""
        user_intents = intent_by_user.get(sender_id, {})
        if not user_intents:
            return "—"
        sorted_intents = sorted(user_intents.items(), key=lambda x: -x[1])[:top_n]
        return ", ".join(f"{name}({count})" for name, count in sorted_intents)

    top_users = pq.get("topUsers", [])
    bottom_users = pq.get("bottomUsers", [])
    if top_users:
        part1_lines.append("**Prompt 质量 Top 用户：**\n")
        part1_lines.append(_md_table(
            ["用户", "综合分", "主要使用场景", "最佳 Prompt 预览"],
            [[str(u.get("senderId") or "?"),
              _round2(u.get("overall", "")),
              _user_top_intents(str(u.get("senderId") or "")),
              (u.get("bestPrompt", {}).get("content", "") or "")[:80]]
             for u in top_users[:3]],
        ))
        part1_lines.append("")
    if bottom_users:
        part1_lines.append("**Prompt 质量 Bottom 用户：**\n")
        part1_lines.append(_md_table(
            ["用户", "综合分", "主要使用场景", "最差 Prompt 预览"],
            [[str(u.get("senderId") or "?"),
              _round2(u.get("overall", "")),
              _user_top_intents(str(u.get("senderId") or "")),
              (u.get("worstPrompt", {}).get("content", "") or "")[:80]]
             for u in bottom_users[:3]],
        ))
        part1_lines.append("")

    # Complexity
    complexity = l2.get("complexity", {})
    comp_dist = complexity.get("distribution", {})
    if comp_dist:
        part1_lines.append(f"\n**任务复杂度分布**：{_kv_lines(comp_dist)}\n")

    # Top 3 most complex tasks from high-cost sessions
    hc_for_complexity = l1.get("highCostSessions", {}).get("taskChains", [])[:3]
    if hc_for_complexity:
        part1_lines.append("**最复杂任务 Top 3**（按成本排序）：\n")
        part1_lines.append(_md_table(
            ["用户", "会话", "总成本", "消息数", "工具调用", "错误数", "时长(s)", "成本驱动"],
            [[str(c.get("senderId") or "?"), str(c.get("sessionId") or ""),
              _round2(c.get("totalCost", "")), c.get("messageCount", ""),
              c.get("toolCallCount", ""), c.get("toolErrorCount", ""),
              c.get("durationSeconds", ""),
              ", ".join(str(d) for d in c.get("costDrivers", []))]
             for c in hc_for_complexity],
        ) + "\n")

    # Retry
    retry = l2.get("retryBehavior", {})
    retry_rate = retry.get("retryRate", None)
    if retry_rate is not None:
        part1_lines.append(
            f"\n**重试率**：{retry_rate}"
            f"（{retry.get('retrySessionCount', '?')}/{retry.get('totalSessions', '?')} 会话）\n"
        )

    # Maturity
    maturity = l2.get("userMaturity", {})
    maturity_users = maturity.get("users", [])[:10]
    if maturity_users:
        part1_lines.append("\n**用户成熟度趋势**：\n")
        part1_lines.append(_md_table(
            ["用户", "Prompt 数", "平均分", "趋势", "斜率"],
            [[u.get("senderId", ""), u.get("promptCount", ""), _round2(u.get("overallAvg", "")),
              u.get("trend", ""), _round2(u.get("slope", ""))] for u in maturity_users],
        ))
        part1_lines.append("")

    sections.append("\n".join(part1_lines))

    # ════════════════════════════════════════════════
    # 二、关键问题（rule-based extraction）
    # ════════════════════════════════════════════════
    problems: list[str] = []
    problem_idx = 0

    def _add_problem(title: str, severity: str, phenomenon: str,
                     evidence: list[str], users: list[str], impact: str) -> None:
        nonlocal problem_idx
        problem_idx += 1
        user_str = ", ".join(str(u) for u in users) if users else "数据中无 sender_id 信息"
        evidence_str = "\n".join(f"- {e}" for e in evidence)
        problems.append(
            f"### 问题{problem_idx}：{title}（严重程度：{severity}）\n\n"
            f"**现象**：{phenomenon}\n\n"
            f"**数据依据**：\n{evidence_str}\n\n"
            f"**涉及用户**：{user_str}\n\n"
            f"**业务影响**：{impact}\n"
        )

    # Problem: High failure rate
    if total_tasks and failure_count / total_tasks > 0.1:
        fail_senders = sorted({str(f.get("senderId") or "?") for f in sr.get("failures", [])[:10]})
        _add_problem(
            "任务失败率偏高", "高",
            f"任务失败率达 {_pct(failure_count, total_tasks)}，共 {failure_count} 条任务链失败。",
            [f"L2-3 成功率：成功 {success_count}，部分 {partial_count}，失败 {failure_count}"],
            fail_senders,
            "失败任务浪费 token 且用户体验差，需排查失败原因。",
        )

    # Problem: High cost anomalies
    hc = l1.get("highCostSessions", {})
    hc_chains = hc.get("taskChains", [])[:5]
    if hc_chains:
        top_cost = hc_chains[0].get("totalCost", 0)
        avg_cost_val = overall_te.get("avgCostPerSession", 0)
        if avg_cost_val and top_cost > avg_cost_val * 5:
            hc_senders = sorted({str(c.get("senderId") or "?") for c in hc_chains})
            _add_problem(
                "高成本会话异常", "高",
                f"Top 高成本会话成本达 {_round2(top_cost)}，是平均值 {_round2(avg_cost_val)} 的 {top_cost / avg_cost_val:.0f} 倍。",
                [f"L1-4 高成本会话 Top 1 成本：{_round2(top_cost)}",
                 f"L1-1 平均会话成本：{_round2(avg_cost_val)}"],
                hc_senders,
                "少数会话消耗大量 token 预算，可能存在 Agent 失控或任务设计不合理。",
            )

    # Problem: Anomalies detected
    if anomaly_count >= 3:
        anomaly_types = {}
        for a in anomaly_list:
            atype = a.get("anomalyType", "unknown")
            anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        anomaly_senders_list = sorted({str(a.get("senderId") or "?") for a in anomaly_list})
        _add_problem(
            "多项异常指标触发", "中",
            f"检测到 {anomaly_count} 个异常，类型分布：{anomaly_types}。",
            [f"L1-5 异常检测：共 {anomaly_count} 个异常"],
            anomaly_senders_list,
            "异常集中可能指向系统性问题，需逐一排查。",
        )

    # Problem: High retry rate
    if retry_rate is not None:
        try:
            retry_val = float(str(retry_rate).rstrip("%")) / 100 if "%" in str(retry_rate) else float(retry_rate)
        except (TypeError, ValueError):
            retry_val = 0
        if retry_val > 0.15:
            _add_problem(
                "重试率偏高", "中",
                f"重试率达 {retry_rate}，表明用户频繁重试或 Agent 响应不符合预期。",
                [f"L2-6 重试率：{retry_rate}"],
                [],
                "高重试率浪费 token 并降低用户满意度。",
            )

    # Problem: Prompt quality polarization
    if bottom_users:
        worst_score = bottom_users[0].get("overall", 0)
        if worst_score and float(worst_score) < 5.0:
            low_senders = [str(u.get("senderId") or "?") for u in bottom_users[:3]]
            bottom_desc_parts = [
                f"{s.get('senderId', '?')}={_round2(s.get('overall', ''))}"
                for s in bottom_users[:3]
            ]
            bottom_desc = ", ".join(bottom_desc_parts)
            _add_problem(
                "部分用户 Prompt 质量偏低", "中",
                f"最低 Prompt 质量评分仅 {_round2(worst_score)}，与团队平均 {_round2(team_avg.get('overall', 'N/A'))} 差距明显。",
                [f"L2-4 Bottom 用户评分：{bottom_desc}"],
                low_senders,
                "低质量 Prompt 导致 Agent 理解偏差，增加重试和失败率。",
            )

    # Problem: Repeated questions (knowledge gaps)
    repeated_q = l3.get("repeatedQuestions", {})
    rq_list = repeated_q.get("repeatedQuestions", [])
    multi_user_questions = [q for q in rq_list if q.get("uniqueUsers", 0) >= 3]
    if multi_user_questions:
        _add_problem(
            "知识缺口：多用户重复提问", "中",
            f"发现 {len(multi_user_questions)} 个问题被 3 人以上独立提出，表明团队存在共性知识缺口。",
            [f"L3-2 重复问题：{q.get('canonicalQuestion', '?')}（{q.get('uniqueUsers', '?')} 人提问）"
             for q in multi_user_questions[:3]],
            sorted({str(s) for q in multi_user_questions for s in (q.get("senders", []) if isinstance(q.get("senders"), list) else []) if s is not None}),
            "重复提问浪费团队时间，应沉淀为文档或 Skill。",
        )

    part2_lines = ["## 二、关键问题\n"]
    if problems:
        part2_lines.extend(problems)
    else:
        part2_lines.append("本分析周期内未检测到显著问题。\n")
    sections.append("\n".join(part2_lines))

    # ════════════════════════════════════════════════
    # 三、优化建议
    # ════════════════════════════════════════════════
    part3_lines = ["## 三、优化建议\n"]

    # ── 3.1 成本优化类 ──
    part3_lines.append("### 3.1 成本优化类\n")
    cost_recs: list[str] = []

    if hc_chains and problem_idx > 0:
        hc_senders_str = ", ".join(sorted({str(c.get("senderId") or "?") for c in hc_chains}))
        cost_recs.append(
            "#### 建议：排查高成本会话根因（优先级：高）\n\n"
            f"**建议内容**：对高成本会话涉及用户（{hc_senders_str}）进行 1:1 沟通，"
            "了解任务场景，优化 Prompt 或拆分复杂任务。\n\n"
            "**数据依据**：关键问题中的高成本会话异常\n\n"
            "**预期收益**：降低 Top 会话成本 50%+\n\n"
            "**参考指标**：L1-4 高成本会话 Top 5 成本趋势\n"
        )

    if non_work_rows:
        cost_recs.append(
            "#### 建议：关注非工作类使用（优先级：中）\n\n"
            "**建议内容**：对非工作类意图使用进行团队沟通，明确使用规范。\n\n"
            "**数据依据**：1.2 节非工作类意图统计\n\n"
            "**预期收益**：减少非必要 token 消耗\n\n"
            "**参考指标**：L2-1 非工作类意图占比趋势\n"
        )

    if retry_rate is not None and retry_val > 0.15:
        cost_recs.append(
            "#### 建议：降低重试率（优先级：中）\n\n"
            "**建议内容**：分析高重试用户的 Prompt 模式，提供 Prompt 编写指南。\n\n"
            f"**数据依据**：L2-6 重试率 {retry_rate}\n\n"
            "**预期收益**：重试率降至 10% 以下，节省 token\n\n"
            "**参考指标**：L2-6 重试率\n"
        )

    if cost_recs:
        part3_lines.extend(cost_recs)
    else:
        part3_lines.append("暂无明显成本问题。\n")

    # ── 3.2 技能沉淀 / 效果优化类 ──
    part3_lines.append("### 3.2 技能沉淀 / 效果优化类\n")
    skill_recs: list[str] = []

    # Skill candidates
    skill_candidates = l3.get("skillCandidates", {})
    sc_list = skill_candidates.get("skillCandidates", [])
    if sc_list:
        top_skills = sc_list[:3]
        skill_names = ", ".join(str(s.get("name") or "?") for s in top_skills)
        skill_recs.append(
            f"#### 建议：封装高频工作流为 Skill（优先级：高）\n\n"
            f"**建议内容**：将以下候选 Skill 封装为标准化 Skill：{skill_names}。\n\n"
            f"**数据依据**：L3-4 Skill 候选列表\n\n"
            f"**预期收益**：提升 Agent 工作流一致性和可复用性\n\n"
            f"**参考指标**：Skill 使用次数和成功率\n"
        )

    if multi_user_questions:
        skill_recs.append(
            "#### 建议：沉淀高频重复问题为知识库（优先级：高）\n\n"
            "**建议内容**：将多用户重复提问的问题整理为团队知识库或 FAQ。\n\n"
            f"**数据依据**：L3-2 发现 {len(multi_user_questions)} 个 3 人以上重复问题\n\n"
            "**预期收益**：减少重复提问，提升团队效率\n\n"
            "**参考指标**：L3-2 重复问题数量趋势\n"
        )

    if bottom_users and float(bottom_users[0].get("overall", 10)) < 5.0:
        skill_recs.append(
            "#### 建议：针对低分用户开展 Prompt 培训（优先级：中）\n\n"
            f"**建议内容**：为 Prompt 质量评分较低的用户提供培训和最佳实践示例。\n\n"
            f"**数据依据**：L2-4 Bottom 用户评分偏低\n\n"
            f"**预期收益**：提升整体 Prompt 质量，减少失败和重试\n\n"
            f"**参考指标**：L2-4 团队平均 Prompt 质量评分\n"
        )

    # Best practices
    best_practices = l3.get("bestPractices", {})
    bp_list = best_practices.get("bestPractices", [])
    if bp_list:
        skill_recs.append(
            "#### 建议：推广最佳实践（优先级：中）\n\n"
            "**建议内容**：将已识别的最佳实践在团队内推广分享。\n\n"
            f"**数据依据**：L3-3 识别出 {len(bp_list)} 条最佳实践\n\n"
            "**预期收益**：提升团队整体使用水平\n\n"
            "**参考指标**：L2-4 Prompt 质量评分、L2-8 用户成熟度趋势\n"
        )

    if skill_recs:
        part3_lines.extend(skill_recs)
    else:
        part3_lines.append("暂无明显优化机会。\n")

    sections.append("\n".join(part3_lines))

    # ════════════════════════════════════════════════
    # Appendix: Data tables (compact reference)
    # ════════════════════════════════════════════════
    appendix_lines = ["## 附录：详细数据\n"]
    appendix_data = _format_for_report(all_results)
    if appendix_data.strip():
        appendix_lines.append(appendix_data)
    sections.append("\n".join(appendix_lines))

    report_text = "\n\n---\n\n".join(sections)
    print("[L3-5] Structured report generated successfully (template mode)")
    return {"report": report_text}


# ─── L3 Analysis Orchestration ───

async def run_l3_independent_cases(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    """Run L3-1/2/3 in parallel — no L1/L2 dependency.

    Returns a partial L3 results dict with keys:
    ``techStack``, ``repeatedQuestions``, ``bestPractices``.
    """
    print("[L3] Starting L3 independent cases (L3-1/2/3) in parallel...")

    tech_stack, repeated_questions, best_practices = await asyncio.gather(
        build_tech_stack_heatmap(adb_config, table_name, range_, llm_client),
        discover_repeated_questions(adb_config, table_name, range_, llm_client),
        extract_best_practices(adb_config, table_name, range_, llm_client),
    )

    print("[L3] L3 independent cases (L3-1/2/3) completed")
    return {
        "techStack": tech_stack,
        "repeatedQuestions": repeated_questions,
        "bestPractices": best_practices,
    }


async def run_l3_independent_cases_no_tech_stack(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    """Run L3-2/3/4 in parallel, skipping L3-1 (tech stack handled by combined analysis).

    Returns a partial L3 results dict with keys:
    ``repeatedQuestions``, ``bestPractices``, ``skillCandidates``.
    """
    print("[L3] Starting L3-2/3/4 (tech stack skipped — handled by combined analysis)...")

    repeated_questions, best_practices, skill_candidates = await asyncio.gather(
        discover_repeated_questions(adb_config, table_name, range_, llm_client),
        extract_best_practices(adb_config, table_name, range_, llm_client),
        discover_skill_candidates(adb_config, table_name, range_, llm_client),
    )

    print("[L3] L3-2/3/4 completed")
    return {
        "repeatedQuestions": repeated_questions,
        "bestPractices": best_practices,
        "skillCandidates": skill_candidates,
    }

async def run_l3_analysis(
    adb_config: AdbConfig,
    table_name: str,
    range_: TimeRange,
    llm_client: LlmClient,
) -> dict:
    print("[L3] Starting L3 organizational cognition analysis (parallel)...")

    (
        tech_stack,
        repeated_questions,
        best_practices,
        skill_candidates,
    ) = await asyncio.gather(
        build_tech_stack_heatmap(adb_config, table_name, range_, llm_client),
        discover_repeated_questions(adb_config, table_name, range_, llm_client),
        extract_best_practices(adb_config, table_name, range_, llm_client),
        discover_skill_candidates(adb_config, table_name, range_, llm_client),
    )

    print("[L3] L3 analysis completed successfully")


# ─── HTML Report Generator ───

_HTML_DESIGN_SYSTEM_PROMPT = """You are an elite frontend engineer and visual designer. Your task is to convert a Markdown analysis report into a single, self-contained HTML file that is breathtakingly beautiful, production-grade, and unforgettable.

## Design Thinking

**Purpose**: An executive-level AI usage insight report for engineering leadership. Dense, information-rich, authoritative.

**Aesthetic Direction**: Editorial / Data-Magazine hybrid — think Bloomberg Intelligence meets a dark-themed developer dashboard. Dark background, sharp typographic hierarchy, data visualizations rendered as pure CSS/HTML, with deliberate use of accent color.

**Differentiation**: A dramatic full-viewport header with the report title rendered in a large, editorial typeface. Sections separated by bold horizontal rules with section numbers. Key metrics surfaced as "stat cards" with large numerals. Subtle noise texture overlay on the background for depth.

## Implementation Rules

1. **Single self-contained file** — all CSS and JS must be inline; no external CDN dependencies.
2. **Font**: Use `@import` from Google Fonts for a distinctive pair:
   - Display/heading: `Bebas Neue` or `Space Grotesk` — NO. Instead use `Playfair Display` (editorial, authoritative) for H1/H2.
   - Body: `IBM Plex Mono` for code-heavy data sections, `Source Serif 4` for prose paragraphs.
   - NEVER use Inter, Roboto, Arial, or system-ui as a primary font.
3. **Color palette** (CSS variables):
   - `--bg`: `#0d0f14` (near-black with blue undertone)
   - `--surface`: `#161922`
   - `--border`: `#2a2e3d`
   - `--accent`: `#e8c547` (warm gold — data highlight color)
   - `--accent-dim`: `#7a6823`
   - `--text-primary`: `#e8e9ed`
   - `--text-secondary`: `#8b8fa8`
   - `--text-muted`: `#4a4e60`
   - `--danger`: `#e05c5c`
   - `--success`: `#4ead7a`
4. **Layout**: Max-width 960px, centered. Generous padding. Each Markdown `##` section becomes a `<section>` card with a border-left accent stripe and a bold section counter.
5. **Animations**: Use CSS `@keyframes` for a staggered fade-in-up on page load for section cards (`animation-delay: calc(var(--i) * 80ms)`). Add a subtle pulse on stat card numbers.
6. **Markdown conversion rules**:
   - `# Title` → Full-bleed dark hero header with title in Playfair Display, large (clamp(2.5rem, 8vw, 5rem))
   - `## Section` → Section card with bold left-border stripe, section number badge, heading in Playfair Display
   - `### Subsection` → Bold sub-heading with a thin separator line
   - `**bold**` → `<strong>` with gold color
   - `` `code` `` → inline styled `<code>` with monospace, dark background
   - Code blocks (``` ```) → styled `<pre><code>` with dark surface, subtle border
   - Tables → Styled with alternating row shading, header row in accent color
   - Lists → Custom bullet style using a gold dash `—`
   - Horizontal rules `---` → Dramatic `<hr>` with gradient fade
   - Numbers that look like metrics (e.g., "1,234 sessions", "87%") → auto-highlight with accent color span
7. **Stat cards**: If the Markdown contains lines like "总会话数: 1234" or metric summaries in the opening section, extract and render them as horizontal stat card strips with large numerals.
8. **Noise texture**: Add an SVG noise filter via a `<div class="noise-overlay">` positioned fixed, pointer-events none, opacity 0.03.
9. **Footer**: A minimal dark footer with "Generated by OpenClaw Insight" and the timestamp.
10. **Responsive**: The layout must be readable on mobile (max-width 768px breakpoint).

## Output

Return ONLY the complete HTML document — starting with `<!DOCTYPE html>` and ending with `</html>`. No markdown fences, no explanation, no preamble. The HTML must be fully self-contained and renderable by opening the file directly in a browser.
"""


async def generate_html_report(markdown_content: str, llm_client: LlmClient) -> str:
    """Convert a Markdown report to a beautiful, self-contained HTML page.

    Uses the LLM with a detailed design system prompt to produce editorial,
    dark-themed HTML. Returns the HTML string, or empty string on failure.
    """
    print("[HTML-Report] Starting HTML report generation from Markdown...")

    if not markdown_content.strip():
        print("[HTML-Report] ⚠️ Empty Markdown content, skipping HTML generation")
        return ""

    user_message = (
        "Convert the following Markdown analysis report into a beautiful, "
        "self-contained HTML page following all design rules in your system prompt.\n\n"
        "--- BEGIN MARKDOWN ---\n"
        f"{markdown_content}\n"
        "--- END MARKDOWN ---"
    )

    try:
        import time as _time
        start = _time.time()
        html_content = await llm_client.chat(
            system_prompt=_HTML_DESIGN_SYSTEM_PROMPT,
            user_prompt=user_message,
        )
        elapsed = _time.time() - start

        # Strip any accidental markdown code fences the LLM might prepend
        html_content = html_content.strip()
        if html_content.startswith("```"):
            lines = html_content.split("\n")
            # Remove first line (``` or ```html) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            html_content = "\n".join(lines)

        print(f"[HTML-Report] ✅ HTML generated in {elapsed:.1f}s ({len(html_content)} chars)")
        return html_content

    except Exception as exc:
        print(f"[HTML-Report] ❌ HTML generation failed: {exc}")
        return ""
