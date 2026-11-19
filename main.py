"""命令行入口：直接对单条指标事件跑诊断，输出结构化报告。

用法示例::

    python main.py --metric dau --date 2026-11-17 --desc "DAU 较昨日明显下跌"
    # 注：date 需落在数据集生成窗口内（最近 60 天），Benchmark 用例日期由程序自动计算。
    python main.py --benchmark          # 运行 Benchmark 多架构对比
    python main.py --serve --port 8000  # 启动 FastAPI 服务
"""

from __future__ import annotations

import argparse
import json

from metric_pilot.agent.harness import MetricDiagnosisHarness
from metric_pilot.benchmark.cases import get_injections
from metric_pilot.benchmark.runner import run_benchmark
from metric_pilot.llm.factory import build_llm
from metric_pilot.rag.knowledge_base import KnowledgeBase
from metric_pilot.tools.dataset import generate_dataset
from metric_pilot.tools.executor import QueryExecutor
from metric_pilot.tools.registry import ToolRegistry


def _build_harness() -> MetricDiagnosisHarness:
    tables = generate_dataset(injections=get_injections())
    executor = QueryExecutor(tables)
    kb = KnowledgeBase.from_file()
    tools = ToolRegistry(executor, kb)
    return MetricDiagnosisHarness(build_llm(), tools)


def diagnose(metric: str, date: str, desc: str) -> None:
    harness = _build_harness()
    report = harness.diagnose(metric, date, desc)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def benchmark() -> None:
    result = run_benchmark()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def serve(port: int) -> None:
    import uvicorn

    from metric_pilot.api.server import create_app

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="MetricPilot 指标异常诊断 Agent")
    parser.add_argument("--metric", help="指标名")
    parser.add_argument("--date", help="异常日期 YYYY-MM-DD")
    parser.add_argument("--desc", default="", help="现象描述")
    parser.add_argument("--benchmark", action="store_true", help="运行 Benchmark 对比实验")
    parser.add_argument("--serve", action="store_true", help="启动 API 服务")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.benchmark:
        benchmark()
    elif args.serve:
        serve(args.port)
    elif args.metric and args.date:
        diagnose(args.metric, args.date, args.desc)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
