"""ReAct 决策循环。

每一轮：
1. 把当前观察到的证据（工具结果摘要）交给 LLM；
2. LLM 决策下一步动作：调用工具或给出最终根因假设；
3. 工具经护栏执行后回写观察，进入下一轮；
4. 达到步数上限或 LLM 判定信息充分时退出。

这里的动作决策以结构化 :class:`ToolCall` 返回，便于做 Tool Calling 审计与回放。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from metric_pilot import config
from metric_pilot.llm.base import BaseLLM
from metric_pilot.schemas import RootCause, ToolCall, ToolResult
from metric_pilot.tools.registry import ToolRegistry


@dataclass
class ReactState:
    """ReAct 循环的状态快照。"""

    observations: list[ToolResult] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    step: int = 0

    def evidence_text(self) -> str:
        if not self.observations:
            return "（暂无工具观察）"
        lines = []
        for obs in self.observations:
            lines.append(f"- 工具 {obs.tool.value}：{'成功' if obs.ok else '失败'}；数据={obs.data}")
        return "\n".join(lines)


class ReactAgent:
    """ReAct 风格的推理-行动循环。"""

    def __init__(self, llm: BaseLLM, tools: ToolRegistry) -> None:
        self.llm = llm
        self.tools = tools

    def run(self, metric: str, anomaly_date: str, description: str) -> ReactState:
        state = ReactState()

        for step in range(1, config.MAX_AGENT_STEPS + 1):
            state.step = step
            decision = self._decide(metric, anomaly_date, description, state)

            if decision is None:
                # LLM 认为信息已充分，结束循环
                state.trace.append({"step": step, "action": "finish"})
                break

            state.trace.append({
                "step": step,
                "action": "tool_call",
                "tool": decision.tool.value,
                "arguments": decision.arguments,
                "reason": decision.reason,
            })
            result = self.tools.dispatch(decision.tool, decision.arguments)
            state.observations.append(result)

            if not result.ok and result.guardrail_blocked:
                state.trace[-1]["guardrail_blocked"] = True
        else:
            state.trace.append({"step": "limit", "action": "max_steps_reached"})

        return state

    # ------------------------------------------------------------------
    def _decide(
        self, metric: str, anomaly_date: str, description: str, state: ReactState
    ) -> ToolCall | None:
        """由 LLM 决策下一步；规则回退时使用确定性脚本策略。"""
        # 规则引擎：按固定顺序补齐基线 -> 下钻 -> RAG
        if self.llm.name == "rule-engine":
            return self._rule_policy(metric, anomaly_date, state)
        return self._llm_decide(metric, anomaly_date, description, state)

    def _rule_policy(self, metric: str, anomaly_date: str, state: ReactState) -> ToolCall | None:
        used = {obs.tool for obs in state.observations}
        from metric_pilot.schemas import ToolName

        if ToolName.COMPARE_BASELINE not in used:
            return ToolCall(
                tool=ToolName.COMPARE_BASELINE,
                arguments={"metric": metric, "date": anomaly_date},
                reason="先对比基线确认异常幅度",
            )
        if ToolName.BREAKDOWN_DIMENSION not in used:
            return ToolCall(
                tool=ToolName.BREAKDOWN_DIMENSION,
                arguments={"metric": metric, "date": anomaly_date, "dimension": "channel"},
                reason="按渠道下钻定位异常切片",
            )
        if ToolName.SEARCH_KNOWLEDGE not in used:
            return ToolCall(
                tool=ToolName.SEARCH_KNOWLEDGE,
                arguments={"query": f"{metric} {anomaly_date} 指标下跌"},
                reason="检索历史同类故障经验",
            )
        return None

    def _llm_decide(
        self, metric: str, anomaly_date: str, description: str, state: ReactState
    ) -> ToolCall | None:
        available = [t["name"] for t in self.tools.tool_schemas()]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是指标诊断 Agent 的决策器。根据当前已掌握的观察，决定下一步："
                    "要么调用一个工具继续取证，要么当信息充分时结束。"
                    f"可用工具：{available}。若结束，请在 arguments 中给出 finish=true。"
                ),
                "role": "user",
                "content": (
                    f"指标：{metric}\n日期：{anomaly_date}\n现象：{description}\n"
                    f"当前证据：\n{state.evidence_text()}"
                ),
            }
        ]
        # 结构化决策：优先 ToolCall；若模型返回 finish 标记则返回 None
        try:
            call = self.llm.chat_structured(messages, ToolCall)
            if call.arguments.get("finish"):
                return None
            return call
        except Exception:  # noqa: BLE001
            return None
