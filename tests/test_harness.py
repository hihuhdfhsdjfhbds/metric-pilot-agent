"""Harness 全链路与 Benchmark 测试（规则回退，无需 API Key）。"""

from metric_pilot.agent.evidence import EvidenceValidator
from metric_pilot.agent.harness import MetricDiagnosisHarness
from metric_pilot.benchmark.cases import get_injections, load_cases
from metric_pilot.benchmark.evaluator import Evaluator
from metric_pilot.llm.factory import build_llm
from metric_pilot.rag.knowledge_base import KnowledgeBase
from metric_pilot.tools.dataset import generate_dataset
from metric_pilot.tools.executor import QueryExecutor
from metric_pilot.tools.registry import ToolRegistry


def _harness() -> MetricDiagnosisHarness:
    tables = generate_dataset(injections=get_injections())
    executor = QueryExecutor(tables)
    kb = KnowledgeBase.from_file()
    tools = ToolRegistry(executor, kb)
    return MetricDiagnosisHarness(build_llm("rule"), tools, EvidenceValidator())


def test_diagnose_returns_structured_report():
    harness = _harness()
    case = load_cases()[0]
    report = harness.diagnose(case.metric, case.anomaly_date, case.description)
    assert report.metric == case.metric
    assert report.root_causes
    assert report.evidence_summary
    assert report.trace  # 留有工具调用轨迹
    assert report.model_used == "rule-engine"


def test_report_serializable():
    harness = _harness()
    case = load_cases()[2]
    report = harness.diagnose(case.metric, case.anomaly_date, case.description)
    d = report.to_dict()
    assert "root_causes" in d and "recommendations" in d


def test_benchmark_scores_cases():
    harness = _harness()
    cases = load_cases()
    evaluator = Evaluator()
    scores = [evaluator.score_case(c, harness.diagnose(c.metric, c.anomaly_date, c.description)) for c in cases]
    summary = Evaluator.summarize(scores, "harness")
    assert summary["cases"] == len(cases)
    assert 0 <= summary["category_accuracy"] <= 1
