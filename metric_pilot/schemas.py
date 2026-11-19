"""结构化数据模型（Pydantic）。

这里定义 Agent 全链路的输入 / 中间产物 / 输出契约：
- 工具调用、调查计划、根因假设、证据结论、最终诊断报告均为强类型结构，
  便于做 Schema 校验、序列化落盘与 Benchmark 对比。
- 对应简历中可写的 "Structured Outputs / Tool Calling" 能力点。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
class ToolName(str, Enum):
    """Agent 可调用的受控工具白名单。"""

    QUERY_METRIC = "query_metric"
    BREAKDOWN_DIMENSION = "breakdown_dimension"
    COMPARE_BASELINE = "compare_baseline"
    SEARCH_KNOWLEDGE = "search_knowledge"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RootCauseCategory(str, Enum):
    """常见指标故障类别（与 Benchmark 标签对齐）。"""

    TRAFFIC_DROP = "traffic_drop"            # 流量下跌
    CONVERSION_DROP = "conversion_drop"      # 转化率下跌
    PAYMENT_DROP = "payment_drop"            # 付费下跌
    DATA_PIPELINE = "data_pipeline"          # 数据管道 / 埋点异常
    SEASONAL = "seasonal"                    # 季节性波动
    NORMAL = "normal"                        # 正常波动


# ---------------------------------------------------------------------------
# 工具层
# ---------------------------------------------------------------------------
class ToolCall(BaseModel):
    """LLM 决策出的一次工具调用。"""

    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", description="调用该工具的理由")


class ToolResult(BaseModel):
    """工具执行结果（含护栏与执行元信息）。"""

    tool: ToolName
    ok: bool
    data: Any = None
    error: str | None = None
    guardrail_blocked: bool = False
    rows: int = 0
    elapsed_ms: float = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent 链路中间产物
# ---------------------------------------------------------------------------
class InvestigationPlan(BaseModel):
    """Planner 产出的调查计划。"""

    objective: str
    steps: list[str] = Field(default_factory=list)
    hypothesis: str = ""


class RootCause(BaseModel):
    """一条根因假设及其证据。"""

    category: RootCauseCategory
    title: str
    description: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceCheck(BaseModel):
    """证据校验结果：独立复核工具返回的数据是否支撑假设。"""

    supported: bool
    checks: list[str] = Field(default_factory=list)
    note: str = ""


class DiagnosisReport(BaseModel):
    """最终诊断报告（结构化输出，可直接序列化 JSON）。"""

    metric: str
    anomaly_date: str
    severity: Severity
    summary: str
    root_causes: list[RootCause] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
class BenchmarkCase(BaseModel):
    """一条 Benchmark 用例：输入指标事件 + 标准答案。"""

    case_id: str
    metric: str
    anomaly_date: str
    description: str
    expected_category: RootCauseCategory
    expected_keywords: list[str] = Field(default_factory=list)
    injected_fault: str = ""


class BenchmarkScore(BaseModel):
    """单条用例评分结果。"""

    case_id: str
    predicted_category: RootCauseCategory | str | None
    category_hit: bool
    keyword_hit: float
    confidence: float
    report: DiagnosisReport
