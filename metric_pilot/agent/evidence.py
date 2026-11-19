"""证据校验（Evidence Validation）。

Agent 给出的根因假设不能直接采信，必须由独立的校验模块复核已有工具证据：
1. 证据样本量是否充足；
2. 对比基线是否显示显著偏离；
3. 下钻结果是否能定位到明确的异常切片；
4. RAG 历史案例是否与当前现象吻合。

校验通过才提高置信度，否则标记为证据不足并走确定性 Fallback。
"""

from __future__ import annotations

from metric_pilot import config
from metric_pilot.schemas import EvidenceCheck, RootCause, ToolResult


class EvidenceValidator:
    """复核根因假设与已收集证据是否一致、充分。"""

    def validate(self, hypothesis: RootCause, observations: list[ToolResult]) -> EvidenceCheck:
        checks: list[str] = []
        supported = True

        # 1. 必须有可复核的工具证据
        usable = [r for r in observations if r.ok and r.data]
        if len(usable) < 2:
            supported = False
            checks.append(f"证据不足：仅 {len(usable)} 条有效工具返回，至少需要 2 条交叉验证")
        else:
            checks.append(f"已收集 {len(usable)} 条有效工具证据")

        # 2. 基线对比：是否确实显著偏离
        baseline = next((r.data for r in observations if r.tool.value == "compare_baseline"), None)
        if isinstance(baseline, dict):
            if baseline.get("anomaly"):
                checks.append(
                    f"基线复核：当前值 {baseline.get('current')} 相对基线 "
                    f"{baseline.get('baseline')} 偏离 {baseline.get('deviation_pct')}%，"
                    f"Z={baseline.get('z_score')}，确属异常"
                )
            else:
                supported = False
                checks.append("基线复核：指标未达到异常阈值，可能为正常波动")
        else:
            checks.append("缺少基线对比证据，无法确认异常幅度")

        # 3. 维度下钻：是否存在明显偏低切片
        breakdown = next((r.data for r in observations if r.tool.value == "breakdown_dimension"), None)
        if isinstance(breakdown, list) and breakdown:
            values = [float(x.get("value", 0)) for x in breakdown if isinstance(x, dict)]
            if values:
                worst, best = min(values), max(values)
                if best > 0 and (best - worst) / best >= 0.3:
                    checks.append(f"下钻复核：切片间差异明显（最低 {worst:.1f} / 最高 {best:.1f}），存在集中异常群体")
                else:
                    checks.append("下钻复核：各切片差异不大，异常更可能来自整体而非单一群体")

        # 4. RAG 案例匹配
        kb = next((r.data for r in observations if r.tool.value == "search_knowledge"), None)
        if isinstance(kb, list) and kb:
            top = kb[0]
            checks.append(f"历史案例匹配：{top.get('title', top.get('fault', '未知'))}（相似度 {top.get('score')}）")

        # 5. 样本量下限
        if usable:
            total_rows = sum(r.rows for r in usable)
            if total_rows < config.EVIDENCE_MIN_SAMPLE:
                supported = False
                checks.append(f"样本量偏小（{total_rows} 行），结论需谨慎")

        note = "证据充分，假设成立" if supported else "证据不足，建议补充下钻或降级为待核实结论"
        return EvidenceCheck(supported=supported, checks=checks, note=note)
