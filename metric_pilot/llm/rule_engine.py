"""内置规则引擎（确定性回退 LLM）。

在未配置 API Key、离线环境或做 Benchmark 基线对比时使用。它基于关键字与已知
故障特征产出结构化结果，保证：
1. 项目 clone 下来无需密钥即可跑通完整 Agent 链路与测试；
2. Benchmark 可对比 "规则基线 / ReAct / 全链路 Harness" 等不同架构的效果。

生产环境下应替换为真实 LLM；本类仅承担可复现的占位与基线职责。
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

from metric_pilot import config
from metric_pilot.llm.base import BaseLLM
from metric_pilot.schemas import (
    InvestigationPlan,
    RootCause,
    RootCauseCategory,
)

T = TypeVar("T", bound=BaseModel)

# 关键字 -> 故障类别映射（与 Benchmark 标签对齐）
KEYWORD_RULES: list[tuple[RootCauseCategory, tuple[str, ...]]] = [
    (RootCauseCategory.TRAFFIC_DROP, ("流量", "访问", "pv", "uv", "新客", "曝光", "traffic")),
    (RootCauseCategory.CONVERSION_DROP, ("转化率", "下单", "注册转化", "加购", "conversion")),
    (RootCauseCategory.PAYMENT_DROP, ("付费", "充值", "支付", "回款", "arpu", "payment")),
    (RootCauseCategory.DATA_PIPELINE, ("埋点", "管道", "上报", "延迟", "缺失", "断流", "pipeline", "etl")),
    (RootCauseCategory.SEASONAL, ("周末", "节假日", "大促", "季节性", "回落")),
]


def classify_by_keywords(text: str) -> RootCauseCategory:
    """按命中关键字最多的类别判定故障类型。"""
    lowered = text.lower()
    scores: dict[RootCauseCategory, int] = {c: 0 for c, _ in KEYWORD_RULES}
    for category, words in KEYWORD_RULES:
        for w in words:
            if w in lowered:
                scores[category] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else RootCauseCategory.NORMAL


class RuleLLM(BaseLLM):
    """确定性规则回退模型。"""

    name = "rule-engine"

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        text = "\n".join(m["content"] for m in messages)
        category = classify_by_keywords(text)
        return f"[规则引擎] 初步判断指标异常类别为 {category.value}，建议进一步下钻维度并核对数据管道。"

    def chat_structured(self, messages: list[dict[str, str]], schema: type[T], **kwargs) -> T:
        text = "\n".join(m["content"] for m in messages)

        if schema is InvestigationPlan:
            category = classify_by_keywords(text)
            return InvestigationPlan(
                objective="定位指标异常根因并给出修复建议",
                steps=[
                    "核对整体指标相对基线的偏离幅度",
                    "按核心维度下钻，定位异常集中群体",
                    "检索历史同类故障与处理经验",
                    "汇总证据并形成根因结论",
                ],
                hypothesis=f"初步怀疑类别：{category.value}",
            )  # type: ignore[return-value]

        if schema is RootCause:
            category = classify_by_keywords(text)
            title = self._title(category)
            return RootCause(
                category=category,
                title=title,
                description=f"规则引擎基于关键字将异常归类为 {category.value}，需结合工具证据进一步确认。",
                evidence=[],
                confidence=0.5,
            )  # type: ignore[return-value]

        raise ValueError(f"RuleLLM 不支持结构化 schema: {schema.__name__}")

    @staticmethod
    def _title(category: RootCauseCategory) -> str:
        mapping = {
            RootCauseCategory.TRAFFIC_DROP: "外部流量下跌导致指标回落",
            RootCauseCategory.CONVERSION_DROP: "转化链路效率下降导致指标走低",
            RootCauseCategory.PAYMENT_DROP: "付费 / 支付环节异常导致收入下滑",
            RootCauseCategory.DATA_PIPELINE: "数据管道或埋点异常造成指标失真",
            RootCauseCategory.SEASONAL: "季节性因素引起的正常波动",
            RootCauseCategory.NORMAL: "指标处于正常波动范围",
        }
        return mapping[category]
