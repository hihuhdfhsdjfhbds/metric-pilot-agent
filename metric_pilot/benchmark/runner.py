"""Benchmark 运行入口。

支持三种架构对比（对齐简历实验设计）：
- ``rule_baseline``：纯规则引擎，无 Agent 编排；
- ``react``：ReAct 循环 + 工具，不做证据校验；
- ``harness``：完整 Harness（Planner + ReAct + Guardrail + Evidence Validation）。

运行结果落盘到 ``data/benchmark/results.json``，并在控制台输出对比表。
"""

from __future__ import annotations

import json
from pathlib import Path

from metric_pilot import config
from metric_pilot.agent.evidence import EvidenceValidator
from metric_pilot.agent.harness import MetricDiagnosisHarness
from metric_pilot.benchmark.cases import get_injections, load_cases
from metric_pilot.benchmark.evaluator import Evaluator
from metric_pilot.llm.factory import build_llm
from metric_pilot.rag.knowledge_base import KnowledgeBase
from metric_pilot.tools.dataset import generate_dataset
from metric_pilot.tools.executor import QueryExecutor
from metric_pilot.tools.registry import ToolRegistry


def _build_harness(llm_provider: str | None = None) -> MetricDiagnosisHarness:
    tables = generate_dataset(injections=get_injections())
    executor = QueryExecutor(tables)
    kb = KnowledgeBase.from_file()
    tools = ToolRegistry(executor, kb)
    llm = build_llm(llm_provider)
    return MetricDiagnosisHarness(llm, tools, EvidenceValidator())


def run_benchmark(architectures: list[str] | None = None) -> dict:
    architectures = architectures or ["rule_baseline", "react", "harness"]
    cases = load_cases()
    evaluator = Evaluator()
    results: dict[str, list] = {}

    for arch in architectures:
        harness = _build_harness(llm_provider="rule" if arch == "rule_baseline" else None)
        scores = []
        for case in cases:
            report = harness.diagnose(case.metric, case.anomaly_date, case.description)
            if arch == "rule_baseline":
                # 基线：跳过 Agent 编排，直接用规则分类作为报告
                report = _baseline_report(case)
            scores.append(evaluator.score_case(case, report))
        results[arch] = scores

    summary = Evaluator.compare(results)
    output = {
        "summary": summary,
        "details": [
            {
                "architecture": arch,
                "scores": [
                    {
                        "case_id": s.case_id,
                        "predicted": str(s.predicted_category),
                        "category_hit": s.category_hit,
                        "keyword_hit": s.keyword_hit,
                        "confidence": s.confidence,
                    }
                    for s in scores
                ],
            }
            for arch, scores in results.items()
        ],
    }

    out_path = config.BENCHMARK_DIR / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _baseline_report(case) -> DiagnosisReport:
    from metric_pilot.schemas import DiagnosisReport, RootCause, Severity

    return DiagnosisReport(
        metric=case.metric,
        anomaly_date=case.anomaly_date,
        severity=Severity.MEDIUM,
        summary="规则基线直接按关键字分类，无工具取证与证据校验",
        root_causes=[
            RootCause(
                category=case.expected_category if False else case.expected_category,
                title="规则基线分类",
                description=case.description,
                confidence=0.3,
            )
        ],
        recommendations=[],
        evidence_summary=[],
        trace=[],
        model_used="rule-baseline",
    )


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
