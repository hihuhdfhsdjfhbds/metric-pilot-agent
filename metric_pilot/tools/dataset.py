"""模拟业务数据集（DuckDB / SQLite 兼容内存库）。

为了让项目在无外部依赖、无真实业务库的情况下可复现，这里在内存中构造两张表：

- ``metric_daily``：日期 + 指标 + 整体数值；
- ``metric_dimension``：日期 + 指标 + 渠道 / 端 / 新老客等维度的拆分数值。

数据由固定随机种子生成，并按 Benchmark 用例注入已知故障（流量下跌、转化率下跌、
数据管道断流等），用于根因定位与评分。
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pandas as pd

from metric_pilot import config

# 维度取值
CHANNELS = ["organic", "paid", "social", "referral"]
PLATFORMS = ["ios", "android", "web"]
USER_TYPES = ["new", "old"]


def _baseline(day: dt.date, metric: str) -> float:
    """根据日期给出基线水平（含轻微周末效应）。"""
    base = {
        "dau": 12000.0,
        "conversion_rate": 0.085,
        "revenue": 56000.0,
        "payment_users": 1800.0,
    }.get(metric, 1000.0)
    weekend = 1.08 if day.weekday() >= 5 else 1.0
    return base * weekend


def generate_dataset(
    days: int = 60,
    seed: int = config.RANDOM_SEED,
    injections: list[dict] | None = None,
) -> dict[str, pd.DataFrame]:
    """生成整体指标表与维度明细表。

    Parameters
    ----------
    days:
        生成最近多少天的数据。
    injections:
        故障注入列表，每项形如::

            {"date": "YYYY-MM-DD", "metric": "dau", "factor": 0.7,
             "dimension": "channel", "value": "paid", "fault": "traffic_drop"}
    """
    rng = random.Random(seed)
    today = dt.date.today()
    injections = injections or []

    daily_rows: list[dict] = []
    dim_rows: list[dict] = []

    for offset in range(days, 0, -1):
        day = today - dt.timedelta(days=offset)
        date_str = day.isoformat()

        for metric in ["dau", "conversion_rate", "revenue", "payment_users"]:
            base = _baseline(day, metric)
            noise = 1 + rng.uniform(-0.03, 0.03)
            value = base * noise

            # 维度拆分初始值
            dim_values: dict[tuple[str, str, str], float] = {}
            for channel in CHANNELS:
                for platform in PLATFORMS:
                    for utype in USER_TYPES:
                        share = rng.uniform(0.02, 0.12)
                        dim_values[(channel, platform, utype)] = base * share * noise

            # 应用故障注入
            for inj in injections:
                if inj["date"] != date_str or inj["metric"] != metric:
                    continue
                factor = inj.get("factor", 1.0)
                dim = inj.get("dimension")
                dim_value = inj.get("value")
                if dim is None:
                    # 整体指标整体下跌
                    value *= factor
                    dim_values = {k: v * factor for k, v in dim_values.items()}
                else:
                    # 仅某一维度切片下跌
                    for (channel, platform, utype), v in dim_values.items():
                        match = (
                            (dim == "channel" and channel == dim_value)
                            or (dim == "platform" and platform == dim_value)
                            or (dim == "user_type" and utype == dim_value)
                        )
                        if match:
                            dim_values[(channel, platform, utype)] = v * factor

            daily_rows.append({"date": date_str, "metric": metric, "value": round(value, 4)})
            for (channel, platform, utype), v in dim_values.items():
                dim_rows.append(
                    {
                        "date": date_str,
                        "metric": metric,
                        "channel": channel,
                        "platform": platform,
                        "user_type": utype,
                        "value": round(v, 4),
                    }
                )

    return {
        config.METRIC_TABLE: pd.DataFrame(daily_rows),
        config.DIMENSION_TABLE: pd.DataFrame(dim_rows),
    }
