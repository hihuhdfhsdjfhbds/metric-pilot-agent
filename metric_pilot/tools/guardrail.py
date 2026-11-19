"""SQL 只读护栏（Tool Guardrail）。

这是 Agent 工程化中最能体现"可靠性设计"的模块：Agent 生成的 SQL 不能直接交给
数据库执行，必须先经过 AST 级别的静态校验，确保：
1. 只允许 SELECT，禁止 INSERT / UPDATE / DELETE / DROP / DDL / 写库；
2. 只能访问白名单内的表，禁止跨库 / 未授权表；
3. 禁止多语句堆叠、禁止危险函数；
4. 限制返回行数，避免全表扫描拖垮服务。

校验失败抛出 ``GuardrailReject``，由 Agent 捕获后重试或走确定性 Fallback。
"""

from __future__ import annotations

import re
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import DML, DDL, Keyword, Punctuation

from metric_pilot import config


class GuardrailReject(RuntimeError):
    """SQL 未通过护栏校验。"""


# 禁止出现的关键字（写操作 / DDL / 事务 / 权限）
_FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "MERGE", "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK",
    "ATTACH", "DETACH", "PRAGMA", "COPY", "EXPORT", "CALL", "EXEC", "EXECUTE",
    "SET", "USE", "INTO OUTFILE", "LOAD_FILE", "SYSTEM",
}

# 允许的表名
_ALLOWED_TABLES = config.ALLOWED_TABLES


def _strip_comments(sql: str) -> str:
    return sqlparse.format(sql, strip_comments=True).strip()


def validate_sql(sql: str) -> str:
    """校验 SQL 并返回归一化后的语句；不合法时抛 ``GuardrailReject``。"""
    if not sql or not sql.strip():
        raise GuardrailReject("SQL 为空")

    clean = _strip_comments(sql)
    if not clean:
        raise GuardrailReject("SQL 仅含注释")

    # 1. 禁止多语句
    statements = [s for s in sqlparse.parse(clean) if s.tokens]
    if len(statements) != 1:
        raise GuardrailReject("禁止一次执行多条 SQL")

    stmt: Statement = statements[0]
    tokens = list(stmt.flatten())

    # 2. 必须以 SELECT / WITH 开头
    first_keyword = next(
        (t.normalized.upper() for t in tokens if t.ttype in (DML, Keyword) and t.normalized),
        "",
    )
    if first_keyword not in {"SELECT", "WITH"}:
        raise GuardrailReject(f"仅允许只读查询，首关键字为 {first_keyword or '空'}")

    # 3. 扫描禁用关键字
    for t in tokens:
        word = t.normalized.upper()
        if word in _FORBIDDEN_KEYWORDS:
            raise GuardrailReject(f"SQL 包含被禁止的操作关键字: {word}")

    # 4. 表名白名单校验
    referenced_tables = _extract_table_names(clean)
    if not referenced_tables:
        raise GuardrailReject("未识别到有效的查询表")
    for table in referenced_tables:
        if table not in _ALLOWED_TABLES:
            raise GuardrailReject(f"访问了未授权表: {table}")

    # 5. 行数上限：若未写 LIMIT，自动补一个安全上限
    if not re.search(r"\bLIMIT\b", clean, re.IGNORECASE):
        clean = clean.rstrip(";") + f" LIMIT {config.TOOL_TIMEOUT_SECONDS and 1000}"

    return clean


def _extract_table_names(sql: str) -> set[str]:
    """从 FROM / JOIN 子句中提取表名（取反引号 / 方括号外的裸标识符）。"""
    tables: set[str] = set()
    for m in re.finditer(r"(?:FROM|JOIN)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?", sql, re.IGNORECASE):
        tables.add(m.group(1))
    return tables
