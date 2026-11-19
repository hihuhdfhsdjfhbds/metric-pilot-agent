"""只读 SQL 执行器。

在内存 pandas DataFrame 上执行经护栏校验的 SELECT 查询。为避免引入 DuckDB 等
重型原生依赖，这里用轻量的 DataFrame 注册表 + pandas 表达式解析；SQL 本身仍走
AST 护栏做权限与安全性校验。
"""

from __future__ import annotations

import re
import time

import pandas as pd

from metric_pilot import config
from metric_pilot.tools.guardrail import GuardrailReject, validate_sql


class QueryExecutor:
    """持有注册表并执行只读查询。"""

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables

    def register(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df

    def run_sql(self, sql: str) -> pd.DataFrame:
        """执行 SQL：先过护栏，再在内存表上求值。"""
        validated = validate_sql(sql)
        started = time.perf_counter()
        try:
            result = self._evaluate(validated)
        except Exception as exc:  # noqa: BLE001
            raise GuardrailReject(f"查询执行失败: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        result.attrs["elapsed_ms"] = elapsed_ms
        return result

    # ------------------------------------------------------------------
    # 轻量求值：覆盖 Agent 实际用到的聚合 / 过滤 / 分组语句
    # ------------------------------------------------------------------
    def _evaluate(self, sql: str) -> pd.DataFrame:
        # 解析目标表
        table_match = re.search(r"FROM\s+`?(\w+)`?", sql, re.IGNORECASE)
        if not table_match:
            raise GuardrailReject("无法解析 FROM 表名")
        table = table_match.group(1)
        if table not in self._tables:
            raise GuardrailReject(f"未注册表: {table}")
        df = self._tables[table]

        # WHERE 条件（date = '...' / metric = '...' / channel = '...'）
        where_clause = self._extract_where(sql)
        if where_clause:
            df = self._apply_where(df, where_clause)

        # GROUP BY + 聚合
        group_match = re.search(r"GROUP\s+BY\s+([\w,\s]+?)(?:ORDER\s+BY|LIMIT|$)", sql, re.IGNORECASE)
        agg_match = re.search(r"(SUM|AVG|MEAN|COUNT|MAX|MIN)\s*\(\s*value\s*\)", sql, re.IGNORECASE)

        if group_match:
            cols = [c.strip() for c in group_match.group(1).split(",")]
            cols = [c for c in cols if c in df.columns]
            agg = agg_match.group(1).lower() if agg_match else "mean"
            agg_func = {"sum": "sum", "avg": "mean", "mean": "mean", "count": "count",
                        "max": "max", "min": "min"}[agg]
            out = df.groupby(cols, as_index=False).agg(**{"value": ("value", agg_func)})
            out = out.rename(columns={"value": agg})
            return out

        if agg_match:
            agg = agg_match.group(1).lower()
            agg_func = {"sum": "sum", "avg": "mean", "mean": "mean", "count": "count",
                        "max": "max", "min": "min"}[agg]
            return pd.DataFrame({"value": [df["value"].agg(agg_func)]})

        return df.copy()

    @staticmethod
    def _extract_where(sql: str) -> str | None:
        m = re.search(r"WHERE\s+(.*?)(?:GROUP\s+BY|ORDER\s+BY|LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _apply_where(df: pd.DataFrame, where: str) -> pd.DataFrame:
        cond = pd.Series(True, index=df.index)
        for part in re.split(r"\s+AND\s+", where, flags=re.IGNORECASE):
            m = re.match(r"(\w+)\s*=\s*'([^']+)'", part.strip())
            if not m:
                continue
            col, val = m.group(1), m.group(2)
            if col in df.columns:
                cond &= df[col].astype(str) == val
        return df[cond]
