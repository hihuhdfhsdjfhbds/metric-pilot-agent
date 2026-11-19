"""Agent Harness 主编排。

完整诊断链路：

    事件输入
      → Planner 调查规划
      → ReAct 循环（工具调用 + 护栏 + 观察回写）
      → Evidence Validator 证据校验
      → 根因汇总与修复建议
      → 结构化 DiagnosisReport 输出

Harness 对可靠性做了工程化约束：工具失败重试、护栏拦截、证据不足降级、确定性
Fallback，避免 Agent "一本正经地胡说"。
"""

from __future__ import annotations

from metric_pilot import config
from metric_pilot.agent.evidence import EvidenceValidator
from metric_pilot.agent.planner import Planner
from metric_pilot.agent.react import ReactAgent
from metric_pilot.llm.base import BaseLLM
from metric_pilot.schemas import (
    DiagnosisReport,
    RootCause,
    RootCauseCategory,
    Severity,
    ToolName,
)
from metric_pilot.tools.registry import ToolRegistry


class MetricDiagnosisHarness:
    """指标异常自主诊断 Agent 编排器。"""

    def __init__(self, llm: BaseLLM, tools: ToolRegistry, validator: EvidenceValidator | None = None):
        self.llm = llm
        self.tools = tools
        self.planner = Planner(llm)
        self.react = ReactAgent(llm, tools)
        self.validator = validator or EvidenceValidator()

    # ------------------------------------------------------------------
    def diagnose(self, metric: str, anomaly_date: str, description: str) -> DiagnosisReport:
        # 1. 规划
        plan = self.planner.plan(metric, anomaly_date, description)

        # 2. ReAct 取证
        state = self.react.run(metric, anomaly_date, description)

        # 3. 汇总证据与基线结论
        baseline = self._find_baseline(state.observations)
        breakdown = self._find_breakdown(state.observations)
        kb_hits = self._find_kb(state.observations)

        # 4. 生成根因假设并做证据校验
        hypothesis = self._build_hypothesis(metric, description, baseline, breakdown, kb_hits)
        check = self.validator.validate(hypothesis, state.observations)

        if not check.supported:
            hypothesis.confidence = min(hypothesis.confidence, 0.4)
            hypothesis.description += "；" + check.note

        # 5. 定级与建议
        severity = self._severity(baseline)
        recommendations = self._recommend(hypothesis.category, baseline, breakdown, kb_hits)

        return DiagnosisReport(
            metric=metric,
            anomaly_date=anomaly_date,
            severity=severity,
            summary=self._summary(plan, baseline, check),
            root_causes=[hypothesis],
            recommendations=recommendations,
            evidence_summary=check.checks,
            trace=state.trace,
            model_used=self.llm.name,
        )

    # ------------------------------------------------------------------
    # 证据提取辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _find_baseline(observations):
        return next((o.data for o in observations if o.tool == ToolName.COMPARE_BASELINE), None)

    @staticmethod
    def _find_breakdown(observations):
        return next((o.data for o in observations if o.tool == ToolName.BREAKDOWN_DIMENSION), None)

    @staticmethod
    def _find_kb(observations):
        return next((o.data for o in observations if o.tool == ToolName.SEARCH_KNOWLEDGE), None)

    # ------------------------------------------------------------------
    def _build_hypothesis(self, metric, description, baseline, breakdown, kb_hits) -> RootCause:
        # 优先由 LLM 产出结构化根因；失败则用规则回退
        category, title, confidence = self._classify(metric, description, baseline, breakdown)
        evidence: list[str] = []
        if isinstance(baseline, dict):
            evidence.append(
                f"基线对比：当前 {baseline.get('current')}，基线 {baseline.get('baseline')}，"
                f"偏离 {baseline.get('deviation_pct')}%，Z={baseline.get('z_score')}"
            )
        if isinstance(breakdown, list) and breakdown:
            evidence.append(f"维度下钻：{breakdown}")
        if isinstance(kb_hits, list) and kb_hits:
            evidence.append(f"历史案例：{kb_hits[0].get('title', kb_hits[0])}")

        try:
            root = self.llm.chat_structured(
                [
                    {"role": "system", "content": "根据证据给出一条结构化根因假设。"},
                    {"role": "user", "content": f"指标：{metric}\n现象：{description}\n证据：{evidence}"},
                ],
                RootCause,
            )
            root.evidence = evidence
            root.confidence = min(root.confidence, confidence)
            return root
        except Exception:  # noqa: BLE001
            return RootCause(
                category=category,
                title=title,
                description=f"基于工具证据归类为 {category.value}",
                evidence=evidence,
                confidence=confidence,
            )

    @staticmethod
    def _classify(metric, description, baseline, breakdown):
        from metric_pilot.llm.rule_engine import classify_by_keywords

        text = f"{metric} {description}"
        category = classify_by_keywords(text)
        # 数据管道特征：基线正常但样本缺失 / 维度断流
        if isinstance(baseline, dict) and baseline.get("error"):
            category = RootCauseCategory.DATA_PIPELINE

        title_map = {
            RootCauseCategory.TRAFFIC_DROP: "外部流量下跌导致指标回落",
            RootCauseCategory.CONVERSION_DROP: "转化链路效率下降导致指标走低",
            RootCauseCategory.PAYMENT_DROP: "付费 / 支付环节异常导致收入下滑",
            RootCauseCategory.DATA_PIPELINE: "数据管道或埋点异常造成指标失真",
            RootCauseCategory.SEASONAL: "季节性因素引起的正常波动",
            RootCauseCategory.NORMAL: "指标处于正常波动范围",
        }
        confidence = 0.6 if baseline and baseline.get("anomaly") else 0.4
        return category, title_map[category], confidence

    @staticmethod
    def _severity(baseline) -> Severity:
        if not isinstance(baseline, dict) or baseline.get("z_score") is None:
            return Severity.MEDIUM
        z = abs(float(baseline["z_score"]))
        if z >= 3:
            return Severity.HIGH
        if z >= config.ANOMALY_Z_THRESHOLD:
            return Severity.MEDIUM
        return Severity.LOW

    @staticmethod
    def _recommend(category, baseline, breakdown, kb_hits) -> list[str]:
        recs = {
            RootCauseCategory.TRAFFIC_DROP: [
                "核查投放渠道与外部引流链路，确认是否存在渠道关停或竞价掉量",
                "排查近期版本发布、入口改版对新客访问路径的影响",
            ],
            RootCauseCategory.CONVERSION_DROP: [
                "漏斗分层检查加购、下单、支付各环节转化率，定位流失节点",
                "核查前端报错与支付接口成功率，排除技术故障导致的转化阻断",
            ],
            RootCauseCategory.PAYMENT_DROP: [
                "核对支付渠道成功率、退款率与计费规则是否变更",
                "排查充值 / 支付网关日志与回调状态，确认是否存在掉单",
            ],
            RootCauseCategory.DATA_PIPELINE: [
                "检查埋点上报、ETL 调度与数仓分区是否延迟或缺失",
                "比对原始日志与入库数据量，定位断流环节并修复重跑",
            ],
            RootCauseCategory.SEASONAL: [
                "结合历史同期曲线判断是否为周期性回落，无需紧急干预",
            ],
            RootCauseCategory.NORMAL: [
                "持续观察指标波动区间，暂不处理",
            ],
        }
        base = list(recs.get(category, []))
        if isinstance(kb_hits, list) and kb_hits:
            base.append(f"参考历史处置经验：{kb_hits[0].get('action', kb_hits[0].get('title', ''))}")
        return base

    @staticmethod
    def _summary(plan, baseline, check) -> str:
        parts = [f"调查目标：{plan.objective}"]
        if isinstance(baseline, dict):
            parts.append(
                f"基线偏离 {baseline.get('deviation_pct')}%（Z={baseline.get('z_score')}）"
            )
        parts.append(check.note)
        return "；".join(parts)
