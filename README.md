# MetricPilot —— 指标异常自主诊断 Agent Harness

面向 DAU、转化率、营收、付费用户数等业务指标异常，设计并实现的一套**自主诊断智能体**：
从指标事件出发，自动完成「调查规划 → 受控取数 → 根因假设 → 证据验证 → 修复建议」的
完整诊断链路，并通过工具护栏、证据校验、失败重试与确定性 Fallback 保障 Agent 输出可靠、可审计。

> 定位：AI Agent / 数据智能方向的求职型项目，覆盖 Planner、ReAct、Tool Calling、
> Structured Outputs、RAG、Tool Guardrail、Evidence Validation、Benchmark 对比实验与
> FastAPI 服务化等企业面试高频考察点。

---

## 1. 系统架构

```
                        ┌──────────────────────────────────────────────┐
 指标事件(metric/date/描述)│                                              │
        ───────────────▶ │  MetricDiagnosisHarness（主编排）             │
                         │                                              │
                         │  1. Planner        调查规划 / 初步假设        │
                         │  2. ReActAgent     推理-行动循环，工具调用决策  │
                         │  3. ToolRegistry   受控工具白名单 + 结果包装   │
                         │  4. EvidenceValidator 独立证据复核 / 置信校准 │
                         │  5. ReportBuilder  结构化报告 + 修复建议      │
                         └───────┬───────────────┬───────────────┬─────┘
                                 │               │               │
                    ┌────────────▼───┐   ┌──────▼──────┐   ┌───▼─────────┐
                    │  Guardrail     │   │ KnowledgeBase│   │  LLM 抽象层  │
                    │ SQL AST 只读护栏│   │ TF-IDF RAG  │   │ OpenAI 兼容 /│
                    │ 表白名单/禁写操作│   │ 历史案例检索 │   │ 规则回退引擎 │
                    └────────────┬───┘   └─────────────┘   └─────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  QueryExecutor（内存数据集）│
                    │  metric_daily / dimension │
                    └─────────────────────────┘
```

## 2. 核心模块与工程亮点

| 模块 | 位置 | 说明 |
|---|---|---|
| LLM 抽象层 | `metric_pilot/llm/` | 统一 `chat` / `chat_structured` 接口；支持 OpenAI 兼容协议，未配置密钥时自动回退内置规则引擎，保证离线可复现 |
| 结构化输出 | `metric_pilot/schemas.py` | Pydantic 强类型契约：工具调用、调查计划、根因假设、证据结论、诊断报告均有 Schema 校验 |
| SQL 工具护栏 | `metric_pilot/tools/guardrail.py` | AST 静态校验：仅允许 SELECT、表白名单、禁多语句/写操作/DDL，自动补 LIMIT，越权即拦截 |
| 工具注册表 | `metric_pilot/tools/registry.py` | 4 个受控工具：指标查询、维度下钻、基线对比、RAG 案例检索；统一超时、结果包装与 trace 记录 |
| RAG 知识库 | `metric_pilot/rag/knowledge_base.py` | TF-IDF + 余弦相似度检索历史故障案例，结构预留 Embedding / 向量库替换接口 |
| 证据校验 | `metric_pilot/agent/evidence.py` | 独立复核样本量、基线显著性、切片差异与案例匹配；证据不足自动降级置信度 |
| ReAct 循环 | `metric_pilot/agent/react.py` | 推理-行动闭环，支持步数上限、失败重试与轨迹回放 |
| Benchmark | `metric_pilot/benchmark/` | 6 类故障用例，对比规则基线 / ReAct / 全链路 Harness 三种架构的准确率与关键词命中 |
| API 服务 | `metric_pilot/api/server.py` | FastAPI 暴露健康检查、诊断、Benchmark 接口，便于集成到业务系统 |

## 3. 技术栈

- **语言 / 框架**：Python 3.10+、Pydantic v2、FastAPI、Uvicorn、pandas、pytest
- **Agent 能力**：ReAct、Planner、Tool Calling、Structured Outputs、RAG、Tool Guardrail、Evidence Validation
- **数据 / 实验**：内存模拟数据集（可替换 DuckDB/MySQL）、TF-IDF 检索、Benchmark 评分与多架构对比
- **模型接入**：OpenAI 兼容协议（DeepSeek / 通义千问 / Moonshot / 本地 Ollama），内置规则回退

## 4. 快速开始

### 4.1 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 4.2 （可选）配置真实模型

复制 `.env.example` 为 `.env`，填入 `OPENAI_API_KEY` 与模型信息；**不配置也能运行**，
系统会自动使用内置规则引擎完成全链路演示与测试。

```bash
# 环境变量方式（任选其一）
export OPENAI_API_KEY=sk-xxxx
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export METRIC_PILOT_MODEL=qwen-plus
```

### 4.3 运行诊断（命令行）

```bash
python main.py --metric dau --date 2026-11-17 --desc "DAU 较昨日明显下跌"
```

### 4.4 运行 Benchmark 多架构对比

```bash
python main.py --benchmark
# 结果落盘 data/benchmark/results.json
```

### 4.5 启动 API 服务

```bash
python main.py --serve --port 8000
# 接口
# GET  /health            健康检查
# POST /api/diagnose      提交指标事件，返回结构化诊断报告
# GET  /api/benchmark     返回 Benchmark 汇总对比
```

### 4.6 运行测试

```bash
pytest
```

## 5. 目录结构

```
metric-pilot-agent/
├── main.py                        # 命令行入口（诊断 / Benchmark / 服务）
├── requirements.txt
├── .env.example
├── metric_pilot/
│   ├── config.py                  # 全局配置
│   ├── schemas.py                 # Pydantic 结构化模型
│   ├── llm/                       # LLM 抽象：OpenAI 客户端 / 规则回退 / 工厂
│   ├── tools/                      # 护栏、执行器、数据集、工具注册表
│   ├── agent/                     # Planner / ReAct / 证据校验 / Harness 编排
│   ├── rag/                       # TF-IDF 知识库检索
│   ├── benchmark/                 # 用例集 / 评分器 / 运行器
│   └── api/                       # FastAPI 服务
├── data/benchmark/knowledge_base.json
├── tests/                         # 护栏 / 工具 / Harness / Benchmark 测试
└── docs/
```

## 6. 诊断报告示例（结构）

```json
{
  "metric": "dau",
  "anomaly_date": "2026-11-17",
  "severity": "medium",
  "summary": "调查目标：定位指标异常根因并给出修复建议；基线偏离 -32.4%，Z=-3.1",
  "root_causes": [
    {
      "category": "traffic_drop",
      "title": "外部流量下跌导致指标回落",
      "description": "付费渠道流量明显下滑，拉低整体 DAU",
      "evidence": ["基线对比：当前 ...", "维度下钻：paid 渠道显著偏低", "历史案例：..."],
      "confidence": 0.82
    }
  ],
  "recommendations": ["核查投放渠道与外部引流链路...", "排查版本发布对访问路径的影响..."],
  "evidence_summary": ["已收集 3 条有效工具证据", "基线复核：确属异常", "..."],
  "trace": [{"step": 1, "action": "tool_call", "tool": "compare_baseline", "...": "..."}]
}
```

## 7. 简历可用表述（参考）

> **MetricPilot —— 指标异常自主诊断 Agent Harness**（Python / FastAPI / Pydantic / RAG）
>
> - 面向 DAU、转化率、营收等指标异常，设计 Planner + ReAct + Tool Guardrail + Evidence
>   Validation 的自主诊断链路，实现调查规划、受控取数、根因假设、证据验证与修复建议自动化。
> - 基于 SQL AST 实现只读工具护栏：白名单表、禁写操作 / DDL / 多语句堆叠，并通过独立证据源
>   复核样本量与基线显著性，配合失败重试与确定性 Fallback 提升 Agent 可靠性。
> - 构建 5 类故障、6 个 Benchmark 用例，完成规则基线 / ReAct / 全链路 Harness 三种架构对比，
>   沉淀结构化报告、工具调用轨迹与 FastAPI 诊断接口，配套 pytest 测试体系。

## 8. 后续可扩展方向

- 接入 Embedding 模型与向量数据库（如 Chroma / Milvus）替换 TF-IDF 检索；
- 将内存数据集替换为 DuckDB / MySQL，接入真实业务指标与埋点数据；
- 增加多轮对话解释、报告图表可视化与告警工单联动；
- 补充更多故障类别与压力测试，完善 Prompt / Schema 评测集。
