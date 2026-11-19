"""工具层测试：数据集生成、查询执行、维度下钻与 RAG 检索。"""

from metric_pilot.benchmark.cases import get_injections, load_cases
from metric_pilot.rag.knowledge_base import KnowledgeBase
from metric_pilot.tools.dataset import generate_dataset
from metric_pilot.tools.executor import QueryExecutor
from metric_pilot.tools.registry import ToolRegistry
from metric_pilot.schemas import ToolName


def _registry() -> ToolRegistry:
    tables = generate_dataset(injections=get_injections())
    executor = QueryExecutor(tables)
    kb = KnowledgeBase.from_file()
    return ToolRegistry(executor, kb)


def test_dataset_has_tables():
    tables = generate_dataset()
    assert "metric_daily" in tables and "metric_dimension" in tables
    assert not tables["metric_daily"].empty


def test_query_metric():
    reg = _registry()
    case = load_cases()[0]
    res = reg.dispatch(ToolName.QUERY_METRIC, {"metric": case.metric, "date": case.anomaly_date})
    assert res.ok
    assert res.rows >= 1


def test_breakdown_dimension():
    reg = _registry()
    case = load_cases()[0]
    res = reg.dispatch(
        ToolName.BREAKDOWN_DIMENSION,
        {"metric": case.metric, "date": case.anomaly_date, "dimension": "channel"},
    )
    assert res.ok
    assert len(res.data) == 4


def test_compare_baseline():
    reg = _registry()
    case = load_cases()[0]
    res = reg.dispatch(ToolName.COMPARE_BASELINE, {"metric": case.metric, "date": case.anomaly_date})
    assert res.ok
    assert "z_score" in res.data


def test_guardrail_blocks_bad_sql():
    tables = generate_dataset()
    executor = QueryExecutor(tables)
    from metric_pilot.tools.guardrail import GuardrailReject

    import pytest

    with pytest.raises(GuardrailReject):
        executor.run_sql("DELETE FROM metric_daily")


def test_kb_search():
    kb = KnowledgeBase.from_file()
    hits = kb.search("付费 支付 营收 下滑", top_k=2)
    assert hits and hits[0]["score"] > 0
