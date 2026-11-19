"""调查规划器（Planner）。

根据指标事件描述，产出结构化调查计划：明确目标、分步动作与初步假设。
Planner 是 Agent Harness 的第一环，对应简历中可写的 "调查规划 / 任务分解" 能力。
"""

from __future__ import annotations

from metric_pilot.llm.base import BaseLLM
from metric_pilot.schemas import InvestigationPlan


class Planner:
    """基于 LLM（或规则回退）生成调查计划。"""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def plan(self, metric: str, anomaly_date: str, description: str) -> InvestigationPlan:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是数据指标异常诊断 Agent 的规划器。请围绕给定指标事件，"
                    "拆解出可执行的调查步骤，并给出初步根因假设。"
                ),
                "role": "user",
                "content": (
                    f"指标：{metric}\n异常日期：{anomaly_date}\n现象描述：{description}\n"
                    "请输出调查计划。"
                ),
            }
        ]
        return self.llm.chat_structured(messages, InvestigationPlan)
