"""SQL 护栏测试：只读、表白名单、禁多语句与写操作。"""

import pytest

from metric_pilot.tools.guardrail import GuardrailReject, validate_sql


def test_allow_select():
    sql = "SELECT date, value FROM metric_daily WHERE metric = 'dau'"
    out = validate_sql(sql)
    assert "SELECT" in out.upper()


def test_reject_insert():
    with pytest.raises(GuardrailReject):
        validate_sql("INSERT INTO metric_daily VALUES ('2026-11-10', 'dau', 1)")


def test_reject_drop():
    with pytest.raises(GuardrailReject):
        validate_sql("DROP TABLE metric_daily")


def test_reject_unknown_table():
    with pytest.raises(GuardrailReject):
        validate_sql("SELECT * FROM secret_table")


def test_reject_multi_statement():
    with pytest.raises(GuardrailReject):
        validate_sql("SELECT * FROM metric_daily; DELETE FROM metric_daily")


def test_reject_update():
    with pytest.raises(GuardrailReject):
        validate_sql("UPDATE metric_daily SET value = 0 WHERE metric = 'dau'")
