from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent
from typing import Any

from scripts.llm_client import LlmClient
from scripts.types import TimeRange

_INSIGHT_LOGIC_FILENAME = "insight_logic_explanation.md"

_GLOBAL_RULES = [
    "Task chain is the core unit: each role='user' message starts a new chain, and the chain ends only when an assistant message stops with stop_reason != 'toolUse'.",
    "The analysis window usually follows the configured range; anomaly detection also looks back 30 days to build a baseline.",
    "ADB and ClickHouse must stay logically aligned: SQL syntax may differ, but thresholds, labels, and business meanings stay the same.",
]

_CASE_GROUPS = {
    "L1 - Operational Insights": [
        "tokenEfficiency",
        "sessionDepth",
        "toolChains",
        "highCostSessions",
        "anomalies",
    ],
    "L2 - Behavior Insights": [
        "intents",
        "complexity",
        "successRate",
        "promptQuality",
        "topics",
        "retryBehavior",
        "thinkingDepth",
        "userMaturity",
    ],
    "L3 - Organizational Insights": [
        "techStack",
        "repeatedQuestions",
        "bestPractices",
        "skillCandidates",
    ],
}

_CASE_TITLES = {
    "tokenEfficiency": "L1-1 Token Efficiency",
    "sessionDepth": "L1-2 Task Chain Depth",
    "toolChains": "L1-3 Tool Chains",
    "highCostSessions": "L1-4 High Token Task Chains",
    "anomalies": "L1-5 Anomaly Detection",
    "intents": "L2-1 Intent Classification",
    "complexity": "L2-2 Task Complexity",
    "successRate": "L2-3 Task Success Rate",
    "promptQuality": "L2-4 Prompt Quality",
    "topics": "L2-5 Topic Clustering",
    "retryBehavior": "L2-6 Retry Behavior",
    "thinkingDepth": "L2-7 Thinking Depth",
    "userMaturity": "L2-8 User Maturity",
    "techStack": "L3-1 Tech Stack Heatmap",
    "repeatedQuestions": "L3-2 Repeated Questions",
    "bestPractices": "L3-3 Best Practices",
    "skillCandidates": "L3-4 Skill Candidates",
}

_CASE_LOGIC = {
    "tokenEfficiency": [
        "Counts distinct sessions in the window and sums input/output/cache tokens.",
        "Computes outputInputRatio = total_output / total_input and cacheHitRatePct = cache_read_tokens / (input_tokens + cache_read_tokens) * 100.",
        "Also aggregates by model and senderId so the overall platform picture can be traced back to specific users or models.",
    ],
    "sessionDepth": [
        "Works at task-chain granularity, not whole-session granularity.",
        "Each chain aggregates message count, duration, tool calls, tokens, and cost before being bucketed by depth.",
        "Buckets are: <=2 single, <=5 short, <=10 medium, <=20 deep, and >20 marathon.",
    ],
    "toolChains": [
        "Extracts tool-call sequences per task chain.",
        "Builds top bigrams and trigrams to reveal repeated execution workflows.",
        "Also measures tool-level success/failure rates to separate frequent workflows from fragile workflows.",
    ],
    "highCostSessions": [
        "Aggregates task-chain level totals such as tokens, cost, tool calls, tool errors, thinking length, and duration.",
        "Sorts by total_tokens descending and keeps the top 20 highest-consumption task chains.",
        "Adds rule-based cost drivers such as high tool error rate (>30%), deep reasoning (>10000 thinking chars), long dialog (>20 messages), heavy tool usage (>10 calls), and huge input (>500000 tokens).",
    ],
    "anomalies": [
        "Builds a 30-day baseline per user for daily cost, sessions, errors, abnormal stops, and message count.",
        "Runs z-score detection on the recent window; |z| > 3 is anomalous and |z| > 5 is critical.",
        "Adds OFF_HOURS anomalies when weekend or 22:00-06:00 activity exceeds 5 sessions, even if the main z-score severity is lower.",
    ],
    "intents": [
        "Uses the LLM to assign one primary intent and a confidence score to each user message.",
        "Aggregates intent distribution overall and by user.",
        "The distribution is persuasive because every counted message has already been normalized into exactly one dominant intent category.",
    ],
    "complexity": [
        "Computes complexity at task-chain level with complexityScore = (user_turns * 2 + tool_call_count * 1.5 + thinking_length / 1000 + total_tokens / 10000) / 4.",
        "Buckets the result into low, medium, high, and very_high.",
        "This combines interaction depth, tool usage, reasoning length, and token consumption into one comparable score.",
    ],
    "successRate": [
        "Aggregates error_count, normal completion, truncation, and abnormal stop signals at task-chain level.",
        "Classifies chains as success, partial, or failure depending on whether they ended normally and whether errors or truncation occurred.",
        "Also produces a numeric success score, but the main business reading comes from the categorical outcome mix.",
    ],
    "promptQuality": [
        "Uses the LLM to score each user prompt on six dimensions: goal clarity, context, chain-of-thought guidance, few-shot examples, iteration signals, and specificity.",
        "Computes overall as the average of the six dimension scores on a 1-5 scale.",
        "Aggregates to team averages, user averages, and best/worst prompt examples so the score is explainable rather than purely abstract.",
    ],
    "topics": [
        "Uses the LLM to assign one primary category and one or two topic tags per user message.",
        "Aggregates categoryDistribution, topTags, and per-user topic mix.",
        "This lets the current topic mix be explained by message-level semantic classification rather than simple keyword counting.",
    ],
    "retryBehavior": [
        "Compares adjacent user messages inside the same session using Jaccard similarity.",
        "Labels each pair as retry (>0.5), refinement (0.3-0.5], or new_question (<=0.3).",
        "Computes retryRate as retry_count / consecutive_user_pairs, so the rate directly reflects how often users repeated substantially similar asks.",
    ],
    "thinkingDepth": [
        "Looks only at assistant messages and buckets them by thinking_text length.",
        "Depth buckets are no_thinking, shallow, moderate, deep, and very_deep.",
        "Also tracks per-model thinking coverage and average thinking length, which explains whether deeper reasoning is concentrated in specific models.",
    ],
    "userMaturity": [
        "Reuses the same six-dimensional prompt-quality scoring model as L2-4.",
        "Aggregates daily overall scores per user, then fits a linear trend to detect improving, declining, stable, or consistently_high behavior.",
        "This makes maturity a trend metric, not a one-off prompt score.",
    ],
    "techStack": [
        "Uses the LLM to identify technologies explicitly mentioned or strongly implied in user prompts.",
        "Aggregates each technology by sessionCount and uniqueUsers.",
        "The current heatmap is persuasive because it reflects repeated conversation evidence across both session volume and user breadth.",
    ],
    "repeatedQuestions": [
        "Uses the LLM to cluster semantically equivalent questions across different users.",
        "Keeps only groups asked by at least two different users.",
        "Reports canonicalQuestion, category, uniqueUsers, totalOccurrences, and examples so repeated demand can be acted on.",
    ],
    "bestPractices": [
        "Samples successful sessions only: <=3 user turns, zero errors, and normal completion.",
        "Sends those successful prompts to the LLM to extract reusable best practices and shared patterns.",
        "Because the source set is success-filtered, the extracted practices are tied to prompts that already worked well in production data.",
    ],
    "skillCandidates": [
        "Combines L1 tool-chain evidence with L2 topic and intent patterns.",
        "The LLM proposes candidates only when the workflow looks deterministic, domain-specific, repeatable, automatable, and shared across users.",
        "Each candidate is grounded by workflow description, input/output contract, supporting evidence, usage estimate, and automation potential.",
    ],
}


def _format_range(range_: TimeRange | None) -> str:
    if not range_:
        return "not provided"
    return f"{range_.start_date} → {range_.end_date}"


def _truncate(text: Any, limit: int = 120) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _format_named_counts(mapping: dict[str, Any], limit: int = 4) -> str:
    if not mapping:
        return "none"
    items = sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{name}={count}" for name, count in items)


def _trim_for_prompt(value: Any, *, max_items: int = 4, max_depth: int = 4) -> Any:
    if max_depth <= 0:
        return "..."
    if isinstance(value, dict):
        return {
            key: _trim_for_prompt(sub_value, max_items=max_items, max_depth=max_depth - 1)
            for key, sub_value in value.items()
        }
    if isinstance(value, list):
        trimmed = value[:max_items]
        output = [_trim_for_prompt(item, max_items=max_items, max_depth=max_depth - 1) for item in trimmed]
        if len(value) > max_items:
            output.append(f"... ({len(value) - max_items} more items)")
        return output
    if isinstance(value, str):
        return _truncate(value, 220)
    return value


def _findings_for_case(case_key: str, result: dict | None) -> list[str]:
    if not result:
        return ["This case has no result in the current run, so only the calculation logic is available."]

    if case_key == "tokenEfficiency":
        overall = result.get("overall", {})
        return [
            (
                f"Current window contains {overall.get('totalSessions', 0)} sessions, "
                f"{overall.get('totalInput', 0):,} input tokens, and {overall.get('totalOutput', 0):,} output tokens."
            ),
            (
                f"The observed output/input ratio is {overall.get('outputInputRatio', 0):.2f} and cache hit rate is "
                f"{overall.get('cacheHitRatePct', 0):.1f}%, which means the efficiency conclusion is based on whole-window token totals rather than anecdotal sessions."
            ),
            f"User-level slices captured: {len(result.get('byUser', []))}; model-level slices captured: {len(result.get('byModel', []))}.",
        ]
    if case_key == "sessionDepth":
        buckets = result.get("bucketDistribution", [])
        bucket_text = ", ".join(
            f"{item.get('depthBucket', 'unknown')}={item.get('chainCount', 0)}" for item in buckets[:5]
        ) or "none"
        return [
            f"The run produced {result.get('totalChains', 0)} task chains in total.",
            f"Current depth mix: {bucket_text}.",
            "Because depth is assigned after each task chain is segmented and aggregated, this result reflects actual execution depth instead of raw session length.",
        ]
    if case_key == "toolChains":
        bigrams = result.get("topBigrams", [])
        trigrams = result.get("topTrigrams", [])
        top_pattern = (trigrams or bigrams or [{}])[0]
        return [
            f"The run found {len(result.get('toolSuccessRates', []))} tools with tracked reliability data.",
            f"Top reusable pattern in the snapshot: {_truncate(top_pattern.get('pattern', 'none'))} (count={top_pattern.get('count', 0)}).",
            "This is persuasive because the workflow signal comes from repeated tool sequences across task chains, not from one isolated session.",
        ]
    if case_key == "highCostSessions":
        chains = result.get("taskChains", [])
        if not chains:
            return ["No high-token task chains were returned in the current run."]
        top_chain = chains[0]
        return [
            f"{len(chains)} high-cost task chains were retained, and the top chain consumed {top_chain.get('totalTokens', 0):,} total tokens.",
            (
                f"Its top-level evidence includes cost={top_chain.get('totalCost', 0)}, messages={top_chain.get('messageCount', 0)}, "
                f"toolCalls={top_chain.get('toolCallCount', 0)}, thinkingLength={top_chain.get('thinkingLength', 0)}."
            ),
            f"Rule-based cost drivers attached to the top chain: {', '.join(top_chain.get('costDrivers', [])) or 'normal'}.",
        ]
    if case_key == "anomalies":
        anomalies = result.get("anomalies", [])
        sev = {}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "unknown")
            sev[severity] = sev.get(severity, 0) + 1
        top = anomalies[0] if anomalies else {}
        findings = [
            f"The current run flagged {len(anomalies)} anomalies in total; severity mix: {_format_named_counts(sev)}.",
        ]
        if top:
            findings.append(
                f"Top anomaly snapshot: senderId={top.get('senderId', 'unknown')}, metric={top.get('metric', 'unknown')}, zScore={top.get('zScore', 0)}."
            )
        findings.append("Each anomaly is grounded in deviation from that user's own historical baseline, which makes spikes easier to defend analytically.")
        return findings
    if case_key == "intents":
        dist = result.get("distribution", {})
        total = sum(dist.values())
        return [
            f"The intent classifier labeled {total} messages across {len(dist)} intent categories.",
            f"Current dominant intents: {_format_named_counts(dist)}.",
            "Because each message is mapped to one primary intent before aggregation, the resulting mix is interpretable as a portfolio of user demand.",
        ]
    if case_key == "complexity":
        dist = result.get("distribution", {})
        top_complex = result.get("topComplex", [])
        findings = [
            f"Complexity distribution in this run: {_format_named_counts(dist)}.",
        ]
        if top_complex:
            findings.append(
                f"The highest-complexity chain in the retained sample has complexityScore={top_complex[0].get('complexityScore', 0):.2f}."
            )
        findings.append("The conclusion is based on a blended formula that combines turns, tool calls, thinking length, and tokens instead of any single proxy.")
        return findings
    if case_key == "successRate":
        overall = result.get("overall", {})
        total = sum(overall.values())
        success = overall.get("success", 0)
        rate = (success / total * 100) if total else 0
        return [
            f"The run evaluated {total} task chains with success mix: {_format_named_counts(overall)}.",
            f"Observed success share is {rate:.1f}% and retained failure/partial details count is {len(result.get('failures', []))}.",
            "This mix is persuasive because the label is assigned from explicit stop-reason and error signals at task-chain level.",
        ]
    if case_key == "promptQuality":
        team_avg = result.get("teamAverage", {})
        return [
            f"Team prompt-quality average is {team_avg.get('overall', 0):.2f} / 5 across {len(result.get('byUser', []))} users.",
            f"The run retained {len(result.get('topUsers', []))} top users and {len(result.get('bottomUsers', []))} bottom users for attribution.",
            "Because the overall score is the average of six explainable dimensions, this result can be defended as structured prompt quality rather than vibe-based judgment.",
        ]
    if case_key == "topics":
        cat_dist = result.get("categoryDistribution", {})
        top_tags = result.get("topTags", [])
        top_tag = top_tags[0] if top_tags else {}
        findings = [
            f"The topic pass found {len(cat_dist)} primary categories; leading categories are {_format_named_counts(cat_dist)}.",
        ]
        if top_tag:
            findings.append(
                f"Top tag in the retained sample: {top_tag.get('tag', 'unknown')} (count={top_tag.get('count', 0)}, uniqueUsers={top_tag.get('uniqueUsers', 0)})."
            )
        findings.append("This turns qualitative demand into a measurable distribution because every message contributes a category and one or two tags.")
        return findings
    if case_key == "retryBehavior":
        items = result.get("items", [])
        retry_count = sum(1 for item in items if item.get("classification") == "retry")
        refinement_count = sum(1 for item in items if item.get("classification") == "refinement")
        new_count = sum(1 for item in items if item.get("classification") == "new_question")
        return [
            f"The run compared {len(items)} consecutive user-message pairs; retryRate={result.get('retryRate', 0):.4f}.",
            f"Classification mix: retry={retry_count}, refinement={refinement_count}, new_question={new_count}.",
            "This rate is defendable because it is derived from explicit text similarity thresholds, not manual labeling.",
        ]
    if case_key == "thinkingDepth":
        by_depth = result.get("byDepth", [])
        depth_text = ", ".join(
            f"{item.get('thinkingDepth', 'unknown')}={item.get('messageCount', 0)}" for item in by_depth[:5]
        ) or "none"
        by_model = result.get("byModel", [])
        top_model = by_model[0] if by_model else {}
        findings = [
            f"Assistant reasoning depth buckets in this run: {depth_text}.",
        ]
        if top_model:
            findings.append(
                f"One tracked model snapshot: {top_model.get('model', 'unknown')} with thinkingPct={top_model.get('thinkingPct', 0):.2f} and avgThinkingLength={top_model.get('avgThinkingLength', 0):.1f}."
            )
        findings.append("Because the bucket comes directly from thinking_text length, the result explains how much hidden reasoning is actually being used.")
        return findings
    if case_key == "userMaturity":
        users = result.get("users", [])
        trend_counts: dict[str, int] = {}
        for user in users:
            trend = user.get("trend", "unknown")
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        return [
            f"The maturity tracker evaluated {len(users)} users; trend mix: {_format_named_counts(trend_counts)}.",
            "This is a trend view, not a single-score view, because the label depends on day-level prompt quality over time and the slope of that trajectory.",
        ]
    if case_key == "techStack":
        techs = result.get("technologies", [])
        top = techs[0] if techs else {}
        findings = [f"The heatmap retained {len(techs)} technologies in the current run."]
        if top:
            findings.append(
                f"Top technology snapshot: {top.get('tech', 'unknown')} with sessionCount={top.get('sessionCount', 0)} and uniqueUsers={top.get('uniqueUsers', 0)}."
            )
        findings.append("Both depth of usage (sessions) and breadth of usage (unique users) are tracked, so popular stacks are not judged by a single dimension.")
        return findings
    if case_key == "repeatedQuestions":
        questions = result.get("repeatedQuestions", [])
        top = questions[0] if questions else {}
        findings = [
            f"The run found {len(questions)} repeated-question clusters from {result.get('totalMessagesAnalyzed', 0)} analyzed messages.",
        ]
        if top:
            findings.append(
                f"Largest repeated question: {_truncate(top.get('canonicalQuestion', 'unknown'))} (uniqueUsers={top.get('uniqueUsers', 0)}, totalOccurrences={top.get('totalOccurrences', 0)})."
            )
        findings.append("This is persuasive because a cluster survives only when at least two different users independently asked the same thing.")
        return findings
    if case_key == "bestPractices":
        return [
            f"The extractor returned {len(result.get('bestPractices', []))} best practices and {len(result.get('commonPatterns', []))} common patterns.",
            "These practices are derived only from sessions that were short, error-free, and normally completed, so they are grounded in successful behavior.",
        ]
    if case_key == "skillCandidates":
        candidates = result.get("skillCandidates", [])
        top = candidates[0] if candidates else {}
        findings = [f"The run proposed {len(candidates)} skill candidates."]
        if top:
            findings.append(
                f"Top candidate snapshot: {top.get('name', 'unknown')} (automationPotential={top.get('automationPotential', 'unknown')}, uniqueUsers={top.get('uniqueUsers', 0)})."
            )
        findings.append("Candidates are not free-form ideas: they must be backed by workflow repeatability, topic/intent demand, and automation feasibility.")
        return findings

    return [
        f"This case returned data with top-level keys: {', '.join(sorted(result.keys())[:8]) or 'none'}."
    ]


def _collect_case_payloads(
    l1_results: dict | None,
    l2_results: dict | None,
    l3_results: dict | None,
) -> list[dict[str, Any]]:
    all_results = {}
    all_results.update(l1_results or {})
    all_results.update(l2_results or {})
    all_results.update(l3_results or {})

    payloads: list[dict[str, Any]] = []
    for group_name, case_keys in _CASE_GROUPS.items():
        for case_key in case_keys:
            result = all_results.get(case_key)
            payloads.append({
                "group": group_name,
                "caseKey": case_key,
                "title": _CASE_TITLES[case_key],
                "logic": _CASE_LOGIC[case_key],
                "currentFindings": _findings_for_case(case_key, result),
                "resultSnapshot": _trim_for_prompt(result or {}),
            })
    return payloads


def _strip_outer_code_fence(markdown: str) -> str:
    text = markdown.strip()
    fenced_match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return fenced_match.group(1).strip() if fenced_match else text


def _build_fallback_markdown(
    run_id: str | None,
    range_: TimeRange | None,
    stack: str,
    l1_results: dict | None,
    l2_results: dict | None,
    l3_results: dict | None,
    generation_mode: str,
    output_dir: Path | None = None,
) -> str:
    case_payloads = _collect_case_payloads(l1_results, l2_results, l3_results)
    payload_by_key = {payload["caseKey"]: payload for payload in case_payloads}

    lines = [
        "# Insight Metrics Logic Documentation",
        "",
        "This local document explains both **how each insight is calculated** and **what the latest run currently shows**.",
        "",
        f"- Stack: {stack}",
        f"- Run ID: {run_id or 'not provided'}",
        f"- Time range: {_format_range(range_)}",
        f"- Generation mode: {generation_mode}",
        "",
        "## Cross-cutting rules",
        "",
    ]
    lines.extend(f"- {rule}" for rule in _GLOBAL_RULES)

    for group_name, case_keys in _CASE_GROUPS.items():
        lines.extend(["", f"## {group_name}", ""])
        for case_key in case_keys:
            payload = payload_by_key[case_key]
            lines.extend([f"### {payload['title']}", "", "**Current result**"])
            lines.extend(f"- {line}" for line in payload["currentFindings"])
            lines.extend(["", "**Calculation logic**"])
            lines.extend(f"- {line}" for line in payload["logic"])
            lines.append("")

    doc_path_hint = str(output_dir / _INSIGHT_LOGIC_FILENAME) if output_dir else f"output/{{run_id}}/{_INSIGHT_LOGIC_FILENAME}"
    lines.extend([
        "## Document generation behavior",
        "",
        f"- This document is generated automatically after every analysis run and saved to `{doc_path_hint}`.",
        "- It is co-located with the JSON result files for the same run inside the `output/{{run_id}}/` directory.",
        "- It is not included in the normal analysis response payload.",
    ])
    return "\n".join(lines).strip() + "\n"


async def _build_llm_markdown(
    run_id: str | None,
    range_: TimeRange | None,
    stack: str,
    l1_results: dict | None,
    l2_results: dict | None,
    l3_results: dict | None,
    llm_client: LlmClient,
    language: str = "zh",
) -> str:
    case_payloads = _collect_case_payloads(l1_results, l2_results, l3_results)
    source_payload = {
        "metadata": {
            "stack": stack,
            "runId": run_id,
            "timeRange": _format_range(range_),
        },
        "globalRules": _GLOBAL_RULES,
        "cases": case_payloads,
    }

    lang_instruction = (
        "Write the entire document in Chinese (中文). Do NOT use English."
        if language == "zh"
        else "Write the entire document in English. Do NOT use Chinese."
    )

    system_prompt = dedent(
        f"""
        You are writing an internal analytics explanation document for OpenClaw.

        {lang_instruction}

        Your job is not to merely list formulas. For each use case, connect:
        1. what the current result shows,
        2. how the metric is calculated,
        3. why that logic supports the interpretation.

        Requirements:
        - Return raw Markdown only, with no code fences.
        - Be precise, factual, and persuasive.
        - Do not invent data that is not present in the source payload.
        - If a case has no data, explicitly say it was skipped or unavailable.
        - Use headings for the three layers and subheadings for each use case.
        - Keep each use case compact but meaningful.
        - Include a short section explaining that this document is auto-generated per analysis run and saved alongside the result files.
        """
    ).strip()

    user_prompt = (
        "Write the full insight_logic_explanation.md document based on the following structured source.\n\n"
        + json.dumps(source_payload, ensure_ascii=False, indent=2)
    )
    response = await llm_client.chat(system_prompt, user_prompt)
    return _strip_outer_code_fence(response)


async def generate_insight_logic_doc(
    run_id: str | None = None,
    range_: TimeRange | None = None,
    stack: str = "ADB",
    l1_results: dict | None = None,
    l2_results: dict | None = None,
    l3_results: dict | None = None,
    llm_client: LlmClient | None = None,
    output_dir: Path | None = None,
    language: str = "zh",
) -> Path:
    """Generate a per-run insight logic explanation document.

    The document is saved to ``output_dir / insight_logic_explanation.md``.
    If ``output_dir`` is not provided, a default of ``output/{run_id}`` is used.
    """
    if output_dir is None:
        output_dir = Path("output") / (run_id or "unknown_run")

    markdown: str
    if llm_client is not None:
        try:
            markdown = await _build_llm_markdown(
                run_id,
                range_,
                stack,
                l1_results,
                l2_results,
                l3_results,
                llm_client,
                language=language,
            )
            if not markdown.lstrip().startswith("#"):
                markdown = "# Insight Metrics Logic Documentation\n\n" + markdown.lstrip()
            generation_mode = "LLM-enhanced"
        except Exception as error:
            print(f"[InsightLogicDoc] LLM generation failed, falling back to deterministic markdown: {error}")
            generation_mode = "deterministic fallback after LLM failure"
            markdown = _build_fallback_markdown(
                run_id,
                range_,
                stack,
                l1_results,
                l2_results,
                l3_results,
                generation_mode,
                output_dir,
            )
    else:
        generation_mode = "deterministic fallback"
        markdown = _build_fallback_markdown(
            run_id,
            range_,
            stack,
            l1_results,
            l2_results,
            l3_results,
            generation_mode,
            output_dir,
        )

    if generation_mode == "LLM-enhanced" and "Generation mode:" not in markdown:
        meta_lines = [
            "# Insight Metrics Logic Documentation",
            "",
            f"- Stack: {stack}",
            f"- Run ID: {run_id or 'not provided'}",
            f"- Time range: {_format_range(range_)}",
            "- Generation mode: LLM-enhanced",
            "",
        ]
        if markdown.startswith("# Insight Metrics Logic Documentation"):
            markdown = re.sub(
                r"^# Insight Metrics Logic Documentation\s*",
                "\n".join(meta_lines),
                markdown,
                count=1,
            )
        else:
            markdown = "\n".join(meta_lines) + markdown.lstrip()

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path = output_dir / _INSIGHT_LOGIC_FILENAME
    doc_path.write_text(markdown.strip() + "\n", encoding="utf-8")
    return doc_path