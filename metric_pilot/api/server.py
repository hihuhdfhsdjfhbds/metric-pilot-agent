"""FastAPI 服务层。

对外暴露诊断接口，便于把 Agent 作为后端服务集成到业务系统：

- ``GET  /health``            健康检查；
- ``POST /api/diagnose``      提交指标事件，返回结构化诊断报告；
- ``GET  /api/benchmark``     触发并返回 Benchmark 汇总结果。

服务内部复用同一套 Harness、护栏与证据校验逻辑。
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from metric_pilot.agent.harness import MetricDiagnosisHarness
from metric_pilot.benchmark.runner import run_benchmark
from metric_pilot.llm.factory import build_llm
from metric_pilot.rag.knowledge_base import KnowledgeBase
from metric_pilot.tools.dataset import generate_dataset
from metric_pilot.tools.executor import QueryExecutor
from metric_pilot.tools.registry import ToolRegistry


class DiagnoseRequest(BaseModel):
    metric: str = Field(..., description="指标名，如 dau / conversion_rate / revenue")
    anomaly_date: str = Field(..., description="异常日期 YYYY-MM-DD")
    description: str = Field("", description="异常现象描述")


def create_app() -> FastAPI:
    app = FastAPI(title="MetricPilot Agent", version="1.0.0")

    # 复用同一套数据与工具（演示用内存数据集）
    tables = generate_dataset()
    executor = QueryExecutor(tables)
    kb = KnowledgeBase.from_file()
    tools = ToolRegistry(executor, kb)
    llm = build_llm()
    harness = MetricDiagnosisHarness(llm, tools)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": harness.llm.name}

    @app.post("/api/diagnose")
    def diagnose(req: DiagnoseRequest) -> dict:
        report = harness.diagnose(req.metric, req.anomaly_date, req.description)
        return report.to_dict()

    @app.get("/api/benchmark")
    def benchmark() -> dict:
        return run_benchmark()

    return app


app = create_app()
