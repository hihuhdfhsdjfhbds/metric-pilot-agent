"""Benchmark 评分器。

对每条用例评估两项指标：
- 类别命中（category_hit）：预测根因类别是否与标准答案一致；
- 关键词命中（keyword_hit）：报告正文命中预期关键词的比例。

汇总输出准确率、平均关键词命中与按架构分组的对比表，用于支撑简历中的
"构建 N 类故障、M 个 Benchmark Case，完成多种架构对比实验"表述。
"""

from __future__ import annotations

from collections import defaultdict

from metric_pilot.schemas import BenchmarkCase, BenchmarkScore, DiagnosisReport


class Evaluator:
    """按用例评分并汇总。"""

    def score_case(self, case: BenchmarkCase, report: DiagnosisReport) -> BenchmarkScore:
        predicted = report.root_causes[0].category if report.root_causes else None
        category_hit = predicted == case.expected_category

        text = (report.summary + " " + " ".join(r.title + r.description for r in report.root_causes)).lower()
        if case.expected_keywords:
            hits = sum(1 for kw in case.expected_keywords if kw.lower() in text)
            keyword_hit = hits / len(case.expected_keywords)
        else:
            keyword_hit = 1.0

        confidence = report.root_causes[0].confidence if report.root_causes else 0.0
        return BenchmarkScore(
            case_id=case.case_id,
            predicted_category=predicted,
            category_hit=category_hit,
            keyword_hit=round(keyword_hit, 2),
            confidence=round(confidence, 2),
            report=report,
        )

    @staticmethod
    def summarize(scores: list[BenchmarkScore], architecture: str) -> dict:
        if not scores:
            return {"architecture": architecture, "cases": 0}
        n = len(scores)
        return {
            "architecture": architecture,
            "cases": n,
            "category_accuracy": round(sum(s.category_hit for s in scores) / n, 3),
            "avg_keyword_hit": round(sum(s.keyword_hit for s in scores) / n, 3),
            "avg_confidence": round(sum(s.confidence for s in scores) / n, 3),
        }

    @staticmethod
    def compare(results: dict[str, list[BenchmarkScore]]) -> list[dict]:
        """多架构对比：{架构名: [Score]} -> 汇总表。"""
        table = []
        for arch, scores in results.items():
            table.append(Evaluator.summarize(scores, arch))
        table.sort(key=lambda x: x.get("category_accuracy", 0), reverse=True)
        return table
