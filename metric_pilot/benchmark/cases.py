"""Benchmark 用例集。

构造覆盖多类故障的诊断用例：每条用例包含注入的故障、标准答案类别与命中关键词，
用于量化评估 Agent 的根因定位准确率。

注意：异常日期由 ``relative_days`` 相对运行当天计算，保证在数据生成窗口内、
任何时间 clone 下来都能复现，避免写死历史日期导致过滤为空。
"""

from __future__ import annotations

import datetime as dt

from metric_pilot.schemas import BenchmarkCase, RootCauseCategory


def _date(offset: int) -> str:
    return (dt.date.today() - dt.timedelta(days=offset)).isoformat()


# 注入参数：与 dataset 生成逻辑对齐（offset 为距今天数）
INJECTION_PLAN: list[dict] = [
    {
        "case_id": "C01",
        "metric": "dau",
        "offset": 2,
        "description": "DAU 较昨日明显下跌，付费与转化暂未同步波动",
        "expected_category": RootCauseCategory.TRAFFIC_DROP,
        "expected_keywords": ["流量", "渠道", "访问"],
        "injection": {"factor": 0.45, "dimension": "channel", "value": "paid", "fault": "traffic_drop"},
    },
    {
        "case_id": "C02",
        "metric": "conversion_rate",
        "offset": 3,
        "description": "注册到下单转化率下滑，访问量基本持平",
        "expected_category": RootCauseCategory.CONVERSION_DROP,
        "expected_keywords": ["转化", "下单", "漏斗"],
        "injection": {"factor": 0.5, "dimension": "user_type", "value": "new", "fault": "conversion_drop"},
    },
    {
        "case_id": "C03",
        "metric": "revenue",
        "offset": 4,
        "description": "营收与付费用户数同步走低，疑似支付环节异常",
        "expected_category": RootCauseCategory.PAYMENT_DROP,
        "expected_keywords": ["付费", "支付", "收入"],
        "injection": {"factor": 0.55, "dimension": "platform", "value": "ios", "fault": "payment_drop"},
    },
    {
        "case_id": "C04",
        "metric": "payment_users",
        "offset": 5,
        "description": "付费用户数突降，疑似埋点上报中断导致数据缺失",
        "expected_category": RootCauseCategory.DATA_PIPELINE,
        "expected_keywords": ["埋点", "管道", "缺失"],
        "injection": {"factor": 0.35, "fault": "data_pipeline"},
    },
    {
        "case_id": "C05",
        "metric": "dau",
        "offset": 6,
        "description": "周末后指标小幅回落，处于历史同期正常区间",
        "expected_category": RootCauseCategory.SEASONAL,
        "expected_keywords": ["季节", "周末", "正常"],
        "injection": None,
    },
    {
        "case_id": "C06",
        "metric": "conversion_rate",
        "offset": 7,
        "description": "加购到下单转化率下跌，老客群体受影响明显",
        "expected_category": RootCauseCategory.CONVERSION_DROP,
        "expected_keywords": ["转化", "下单", "老客"],
        "injection": {"factor": 0.5, "dimension": "user_type", "value": "old", "fault": "conversion_drop"},
    },
]


def load_cases() -> list[BenchmarkCase]:
    cases = []
    for c in INJECTION_PLAN:
        date = _date(c["offset"])
        cases.append(
            BenchmarkCase(
                case_id=c["case_id"],
                metric=c["metric"],
                anomaly_date=date,
                description=c["description"],
                expected_category=c["expected_category"],
                expected_keywords=c["expected_keywords"],
                injected_fault=c["injection"]["fault"] if c["injection"] else "",
            )
        )
    return cases


def get_injections() -> list[dict]:
    """把用例的故障注入转换为 dataset 可消费的参数（补齐日期与指标）。"""
    injections = []
    for c in INJECTION_PLAN:
        if not c["injection"]:
            continue
        injections.append(
            {
                "date": _date(c["offset"]),
                "metric": c["metric"],
                **c["injection"],
            }
        )
    return injections
