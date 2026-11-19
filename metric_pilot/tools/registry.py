"""工具注册表：把受控工具封装为 Agent 可调用的统一接口。

每个工具都经过：参数校验 -> 护栏 -> 执行 -> 结构化结果包装，
返回 :class:`ToolResult`，供 Agent 记录 trace 与做证据校验。
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from metric_pilot import config
from metric_pilot.rag.knowledge_base import KnowledgeBase
from metric_pilot.schemas import ToolName, ToolResult
from metric_pilot.tools.executor import QueryExecutor
from metric_pilot.tools.guardrail import GuardrailReject


class ToolRegistry:
    """注册并调度 Agent 工具。"""

    def __init__(self, executor: QueryExecutor, kb: KnowledgeBase) -> None:
        self.executor = executor
        self.kb = kb

    # ------------------------------------------------------------------
    # 工具描述（供 LLM 做 Tool Calling 选择）
    # ------------------------------------------------------------------
    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "name": ToolName.QUERY_METRIC.value,
                "description": "查询某指标在指定日期的整体数值，用于核对是否偏离基线。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "指标名，如 dau / conversion_rate / revenue"},
                        "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
                    },
                    "required": ["metric", "date"],
                },
            },
            {
                "name": ToolName.BREAKDOWN_DIMENSION.value,
                "description": "按渠道 / 平台 / 新老客维度下钻，定位异常集中在哪个切片。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "date": {"type": "string"},
                        "dimension": {"type": "string", "enum": ["channel", "platform", "user_type"]},
                    },
                    "required": ["metric", "date", "dimension"],
                },
            },
            {
                "name": ToolName.COMPARE_BASELINE.value,
                "description": "对比异常日与前 7 日基线，计算偏离幅度与 Z 分数。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["metric", "date"],
                },
            },
            {
                "name": ToolName.SEARCH_KNOWLEDGE.value,
                "description": "检索历史同类故障案例与处置经验（RAG）。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # 调度入口
    # ------------------------------------------------------------------
    def dispatch(self, tool: ToolName, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            handler = {
                ToolName.QUERY_METRIC: self._query_metric,
                ToolName.BREAKDOWN_DIMENSION: self._breakdown,
                ToolName.COMPARE_BASELINE: self._compare_baseline,
                ToolName.SEARCH_KNOWLEDGE: self._search_kb,
            }[tool]
            data = handler(arguments)
            rows = len(data) if isinstance(data, pd.DataFrame) else len(data) if isinstance(data, list) else 1
            return ToolResult(
                tool=tool,
                ok=True,
                data=data,
                rows=rows,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except GuardrailReject as exc:
            return ToolResult(
                tool=tool,
                ok=False,
                error=str(exc),
                guardrail_blocked=True,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool=tool,
                ok=False,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

    # ------------------------------------------------------------------
    # 具体工具实现
    # ------------------------------------------------------------------
    def _query_metric(self, args: dict[str, Any]) -> list[dict]:
        metric, date = args["metric"], args["date"]
        sql = (
            f"SELECT date, metric, value FROM {config.METRIC_TABLE} "
            f"WHERE metric = '{metric}' AND date = '{date}'"
        )
        df = self.executor.run_sql(sql)
        return df.to_dict("records")

    def _breakdown(self, args: dict[str, Any]) -> list[dict]:
        metric, date, dim = args["metric"], args["date"], args["dimension"]
        sql = (
            f"SELECT {dim}, AVG(value) AS value FROM {config.DIMENSION_TABLE} "
            f"WHERE metric = '{metric}' AND date = '{date}' GROUP BY {dim}"
        )
        df = self.executor.run_sql(sql)
        value_col = "value" if "value" in df.columns else df.columns[-1]
        return df.sort_values(value_col).to_dict("records")

    def _compare_baseline(self, args: dict[str, Any]) -> dict[str, Any]:
        metric, date = args["metric"], args["date"]
        daily = self.executor._tables[config.METRIC_TABLE]  # noqa: SLF001
        series = daily[daily["metric"] == metric].sort_values("date")
        current_row = series[series["date"] == date]
        if current_row.empty:
            return {"error": "异常日无数据"}
        current = float(current_row["value"].iloc[0])
        history = series[series["date"] < date].tail(7)["value"].astype(float)
        if history.empty:
            return {"current": current, "baseline": None, "deviation": None, "z_score": None}
        baseline = float(history.mean())
        std = float(history.std(ddof=0)) or 1.0
        z = (current - baseline) / std
        return {
            "current": round(current, 4),
            "baseline": round(baseline, 4),
            "deviation_pct": round((current - baseline) / baseline * 100, 2),
            "z_score": round(z, 3),
            "anomaly": abs(z) >= config.ANOMALY_Z_THRESHOLD,
        }

    def _search_kb(self, args: dict[str, Any]) -> list[dict]:
        return self.kb.search(args["query"], top_k=3)
