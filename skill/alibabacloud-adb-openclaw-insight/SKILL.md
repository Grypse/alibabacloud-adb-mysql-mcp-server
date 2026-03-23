---
name: alibabacloud-adb-openclaw-insight
description: >
  OpenClaw conversation log collection and deep insight analysis Skill. Collects OpenClaw
  session logs (JSONL format) in real time and pushes them to Alibaba Cloud AnalyticDB
  MySQL (ADB) or ClickHouse for storage. Provides a three-layer insight analysis architecture:
  L1 Operational Efficiency (Token efficiency, session depth, tool chain analysis,
  high-cost attribution, anomaly detection), L2 User Behavior (intent classification,
  task complexity, success rate, prompt quality, topic clustering,
  retry detection, thinking depth, user maturity), and L3 Organizational Cognition
  (tech stack heatmap, knowledge gap discovery, best practice extraction, skill
  candidate discovery, narrative report generation). Powered by SQL + Python + LLM.
  Supports two storage backends: ADB MySQL (OLTP) and ClickHouse (OLAP columnar engine,
  recommended for large-scale log analysis). Use this Skill when you need to monitor
  OpenClaw usage, analyze costs, understand user behavior patterns, or generate
  organizational intelligence reports.
---

# OpenClaw Logger Insight Skill

Collect OpenClaw session logs in real time and push them to **AnalyticDB MySQL** or **ClickHouse**. Analyze usage patterns with a three-layer insight architecture powered by **SQL + Python + LLM**.

> **两套后端并行支持（Dual Backend）**
> - **ADB MySQL 版**（`scripts/main.py`）：适合已有 AnalyticDB MySQL 实例的场景，兼容 OLTP 读写。
> - **ClickHouse 版**（`scripts/main_ck.py`）：适合大规模日志分析，列式存储、聚合查询更快，推荐用于生产环境高并发写入与分析。
> 两套版本**独立运行，互不干扰**，可以只用其中一套，也可以同时运行。

## Prerequisites

- Python >= 3.10 (use `python` or `python3` depending on your system)
- An accessible Alibaba Cloud AnalyticDB MySQL instance **and/or** Alibaba Cloud ClickHouse instance
- OpenClaw deployed and generating session files (`~/.openclaw/agents/*/sessions/*.jsonl`) and logs (`/tmp/openclaw/openclaw-YYYY-MM-DD.log`)
- (Optional) An OpenAI-compatible or Anthropic LLM API endpoint for L2/L3 analysis

---

## ADB MySQL 版使用说明

### Quick Start

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Copy the configuration template
cp config.example.json config.json
# Edit config.json: fill in the "adb" section and (optionally) LLM API config

# 4. Initialize the database tables
uv run python -m scripts.init_db

# 5. (Optional) Start the all-in-one service (collection + scheduled analysis)
uv run python -m scripts.main
```

### CLI Commands

#### Collect — One-shot data collection

```bash
uv run python -m scripts.main collect
```

#### Analyze — Run full insight analysis

```bash
uv run python -m scripts.main analyze
```

Custom time range:

```bash
uv run python -m scripts.analyze_usage --from "2026-03-01 00:00:00" --to "2026-03-10 23:59:59"
```

#### Final Report — Print the latest report

```bash
uv run python -m scripts.main final-report
```

---

## ClickHouse 版使用说明

### 前置条件

- 阿里云云原生 ClickHouse 实例（[购买地址](https://www.aliyun.com/product/clickhouse)）
  - 建议版本：22.8+（支持窗口函数 LAG/LEAD/ROW_NUMBER）
  - 默认 HTTP 端口：8123
- `config.json` 中已填写 `clickhouse` 配置节（见下文）

### Quick Start

```bash
# 1. 安装依赖（clickhouse-connect 已包含在 requirements.txt 中）
uv pip install -r requirements.txt

# 2. 复制配置模板（如未复制）
cp config.example.json config.json

# 3. 编辑 config.json，填写 clickhouse 节：
#    {
#      "clickhouse": {
#        "host": "cc-xxxxxxxx.clickhouse.ads.aliyuncs.com",
#        "port": 8123,
#        "database": "openclaw_ck",
#        "username": "admin",
#        "password": "xxxxxxxxxx",
#        "secure": false,
#        "sessionTable": "openclaw_sessions",
#        "logsTable": "openclaw_logs"
#      },
#      ...
#    }

# 4. 初始化 ClickHouse 数据表（三张表：sessions / logs / analysis_results）
uv run python -m scripts.init_db_ck

# 5. 启动一体化服务（采集 + 定时分析）
uv run python -m scripts.main_ck
```

### CLI Commands

#### Collect — 单次数据采集

扫描新增 JSONL 会话文件和当日日志，批量写入 ClickHouse，保存断点续传检查点后退出。

```bash
uv run python -m scripts.main_ck collect
```

#### Analyze — 运行完整洞察分析

按配置的时间窗口执行全量三层分析流水线（L1 运营效率 → L2 用户行为 → L3 组织认知 → 最终报告）。

```bash
uv run python -m scripts.main_ck analyze
```

自定义时间范围：

```bash
# 支持 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS 格式
uv run python -m scripts.analyze_usage_ck --from "2026-03-01 00:00:00" --to "2026-03-10 23:59:59"
```

查询历史运行结果（--report 模式）：

```bash
uv run python -m scripts.analyze_usage_ck --report <run_id>
```

#### Final Report — 打印最新分析报告

从 ClickHouse 读取最近一次成功生成的叙述性报告并打印到终端。

```bash
uv run python -m scripts.main_ck final-report
```

### 定时采集（OpenClaw Cron）

在 OpenClaw 中注册定时任务，每 30 秒执行一次采集（推荐方式）：

```json
{
  "cron": "*/30 * * * * *",
  "command": "python -m scripts.main_ck collect",
  "cwd": "/path/to/alibabacloud-adb-openclaw-insight"
}
```

每次调用会：
1. 扫描自上次运行以来的新增 JSONL 会话文件和日志文件
2. 按批次写入 ClickHouse（`execute_batch_insert` + `client.insert()`）
3. 保存断点检查点（`.collect_state.json`），下次从断点继续
4. 退出 — 无常驻进程

### 一体化服务模式（serve）

```bash
uv run python -m scripts.main_ck
# 或显式指定 serve
uv run python -m scripts.main_ck serve
```

启动后会：
- 执行一次初始采集
- 每隔 `collection.intervalMinutes` 分钟自动采集
- 每天 02:00 自动运行全量分析
- 每天 03:00 清理超过 `retentionDays` 天的过期数据

---

## Configuration

See `config.example.json` for all options.

### ClickHouse 配置节（`clickhouse`）

| 字段 | 说明 | 示例 |
|---|---|---|
| `host` | 阿里云 ClickHouse 实例域名 | `cc-xxxxxxxx.clickhouse.ads.aliyuncs.com` |
| `port` | HTTP 端口（默认 8123；HTTPS 用 8443） | `8123` |
| `database` | 数据库名 | `openclaw_ck` |
| `username` | 账号 | `admin` |
| `password` | 密码 | `xxxxxxxxxx` |
| `secure` | 是否启用 HTTPS/TLS | `false` |
| `sessionTable` | 会话数据表名 | `openclaw_sessions` |
| `logsTable` | 日志数据表名 | `openclaw_logs` |

### 通用配置节（ADB 和 CK 共用）

- **collection**: 采集参数（intervalMinutes、batchSize、retentionDays、enableLogCollection、enableTokenCollection）
- **filters**: 日志过滤（minLevel、includeSubsystems、excludeSubsystems）
- **llm**: LLM API 配置（endpoint、apiKey、model、apiType、maxConcurrency、temperature、maxTokens）
- **analysis**: 分析开关（enableL1/L2/L3、analysisWindowDays、maxSessionsForLlm）

> **注意**：L1 分析无需 LLM，L2 和 L3 需要配置 LLM endpoint。

---

## 数据库表结构

ClickHouse 版建表 DDL 位于 `sql/` 目录：

| 文件 | 表名 | 说明 |
|---|---|---|
| `sql/openclaw_sessions_ck.sql` | `openclaw_sessions` | 会话+消息主表（MergeTree，按天分区） |
| `sql/openclaw_logs_ck.sql` | `openclaw_logs` | 原始日志表（MergeTree，按天分区） |
| `sql/openclaw_analysis_results_ck.sql` | `openclaw_analysis_results` | 分析结果表（按 run_id 查询） |

运行 `python -m scripts.init_db_ck` 会自动创建以上三张表（`CREATE TABLE IF NOT EXISTS`）。

---

## ADB MySQL 版 vs ClickHouse 版对比

| 维度 | ADB MySQL 版 | ClickHouse 版 |
|---|---|---|
| 存储引擎 | AnalyticDB MySQL（OLTP/OLAP 混合） | 云原生 ClickHouse（纯 OLAP 列式） |
| 写入方式 | mysql-connector 批量 INSERT | clickhouse-connect `client.insert()` |
| 分析性能 | 中等（行存+列存混合） | 极高（纯列存，聚合查询快 5-10x） |
| 主键方式 | AUTO_INCREMENT | Python 生成单调 UInt64 |
| 时间戳类型 | DATETIME(3) | DateTime64(3, 'Asia/Shanghai') |
| 删除方式 | DELETE FROM | ALTER TABLE … DELETE WHERE（异步 mutation） |
| 入口脚本 | `scripts/main.py` | `scripts/main_ck.py` |
| 分析入口 | `scripts/analyze_usage.py` | `scripts/analyze_usage_ck.py` |
| 初始化 | `scripts/init_db.py` | `scripts/init_db_ck.py` |
| 调度器 | `scripts/scheduler.py` | `scripts/scheduler_ck.py` |

