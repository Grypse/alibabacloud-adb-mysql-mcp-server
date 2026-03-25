"""
ClickHouse configuration for the OpenClaw Insight Analysis system.

Reads the "clickhouse" section from config.json.
Shared configuration classes (CollectionConfig, FiltersConfig, LlmConfig,
AnalysisConfig) are reused from scripts.config.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

from scripts.config import (
    AnalysisConfig,
    CollectionConfig,
    FiltersConfig,
    LlmConfig,
)


@dataclass
class CkConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    secure: bool
    session_table: str
    logs_table: str


@dataclass
class AppConfigCk:
    ck: CkConfig
    collection: CollectionConfig
    filters: FiltersConfig
    llm: Optional[LlmConfig] = None
    analysis: Optional[AnalysisConfig] = None


CONFIG_FILE_NAME = "config.json"


def load_config_ck() -> AppConfigCk:
    config_path = os.path.join(os.path.dirname(__file__), "..", CONFIG_FILE_NAME)
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        print(
            f"Configuration file not found: {config_path}\n"
            "Please copy config.example.json to config.json and fill in the actual configuration."
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = json.load(config_file)

    config = _parse_config_ck(raw)
    _validate_config_ck(config)
    return config


def _parse_config_ck(raw: dict) -> AppConfigCk:
    ck_raw = raw.get("clickhouse", {})
    ck = CkConfig(
        host=ck_raw.get("host", ""),
        port=int(ck_raw.get("port", 8123)),
        database=ck_raw.get("database", ""),
        username=ck_raw.get("username", "default"),
        password=ck_raw.get("password", ""),
        secure=bool(ck_raw.get("secure", False)),
        session_table=ck_raw.get("sessionTable", "openclaw_sessions"),
        logs_table=ck_raw.get("logsTable", "openclaw_logs"),
    )

    col_raw = raw.get("collection", {})
    collection = CollectionConfig(
        interval_minutes=int(col_raw.get("intervalMinutes", 5)),
        batch_size=int(col_raw.get("batchSize", 100)),
        retention_days=int(col_raw.get("retentionDays", 7)),
        enable_log_collection=bool(col_raw.get("enableLogCollection", True)),
        enable_token_collection=bool(col_raw.get("enableTokenCollection", True)),
    )

    fil_raw = raw.get("filters", {})
    filters = FiltersConfig(
        min_level=fil_raw.get("minLevel", "info"),
        include_subsystems=fil_raw.get("includeSubsystems", []),
        exclude_subsystems=fil_raw.get("excludeSubsystems", []),
    )

    llm: Optional[LlmConfig] = None
    if "llm" in raw and raw["llm"]:
        llm_raw = raw["llm"]
        llm = LlmConfig(
            endpoint=llm_raw.get("endpoint", ""),
            api_key=llm_raw.get("apiKey", ""),
            model=llm_raw.get("model", ""),
            api_type=llm_raw.get("apiType", "openai"),
            max_concurrency=int(llm_raw.get("maxConcurrency", 5)),
            temperature=float(llm_raw.get("temperature", 0.1)),
            max_tokens=llm_raw.get("maxTokens"),
        )

    analysis: Optional[AnalysisConfig] = None
    if "analysis" in raw and raw["analysis"]:
        ana_raw = raw["analysis"]
        analysis = AnalysisConfig(
            enable_l1=bool(ana_raw.get("enableL1", True)),
            enable_l2=bool(ana_raw.get("enableL2", True)),
            enable_l3=bool(ana_raw.get("enableL3", True)),
            analysis_window_days=int(ana_raw.get("analysisWindowDays", 7)),
            max_sessions_for_llm=int(ana_raw.get("maxSessionsForLlm", 500)),
            generate_html_report=bool(ana_raw.get("generateHtmlReport", False)),
        )

    return AppConfigCk(ck=ck, collection=collection, filters=filters, llm=llm, analysis=analysis)


def _validate_config_ck(config: AppConfigCk) -> None:
    if not config.ck.host or not config.ck.database:
        raise ValueError("ClickHouse configuration missing host or database")
    if not config.ck.username:
        raise ValueError("ClickHouse configuration missing username")
    if not config.ck.session_table:
        raise ValueError("ClickHouse configuration missing sessionTable")
    if config.collection.interval_minutes <= 0:
        raise ValueError("collection.intervalMinutes must be greater than 0")
    if config.collection.batch_size <= 0:
        raise ValueError("collection.batchSize must be greater than 0")

    if config.llm:
        if not config.llm.endpoint:
            raise ValueError("llm.endpoint is required when llm config is provided")
        if not config.llm.api_key:
            raise ValueError("llm.apiKey is required when llm config is provided")
        if not config.llm.model:
            raise ValueError("llm.model is required when llm config is provided")

    if config.analysis is None:
        config.analysis = AnalysisConfig()

    if (config.analysis.enable_l2 or config.analysis.enable_l3) and not config.llm:
        print("[Config] ⚠️ L2/L3 analysis requires LLM config. Only L1 analysis will run.")
        config.analysis.enable_l2 = False
        config.analysis.enable_l3 = False
