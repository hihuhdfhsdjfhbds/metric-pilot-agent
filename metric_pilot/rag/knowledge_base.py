"""轻量 RAG 知识库（TF-IDF + 余弦相似度）。

- 文档来自 ``data/benchmark/knowledge_base.json``，沉淀历史故障现象、根因与处置方案；
- 查询时对问题与文档做分词 + TF-IDF 向量化，返回 Top-K 最相似案例作为上下文，
  供 Agent 在形成根因结论时参考（对应简历中的 RAG 能力点）。

不依赖外部向量数据库，便于在实习面试环境中一键复现；结构上预留了替换为
Embedding 模型 / 向量库的接口。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from metric_pilot import config


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：英文按词，中文按 2-gram 切分。"""
    text = text.lower()
    english = re.findall(r"[a-z0-9]+", text)
    chinese = re.findall(r"[\u4e00-\u9fa5]", text)
    bigrams = ["".join(chinese[i : i + 2]) for i in range(len(chinese) - 1)]
    return english + bigrams


class KnowledgeBase:
    """基于 TF-IDF 的历史案例检索库。"""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self._tfidf: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._index()

    @classmethod
    def from_file(cls, path: Path | str | None = None) -> "KnowledgeBase":
        path = Path(path) if path else config.BENCHMARK_DIR / "knowledge_base.json"
        if not path.exists():
            return cls([])
        docs = json.loads(path.read_text(encoding="utf-8"))
        return cls(docs)

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    def _index(self) -> None:
        doc_tokens: list[list[str]] = []
        df: dict[str, int] = {}
        for doc in self.docs:
            tokens = _tokenize(doc.get("symptom", "") + " " + doc.get("root_cause", ""))
            doc_tokens.append(tokens)
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1

        n = max(len(self.docs), 1)
        self._idf = {tok: math.log((n + 1) / (cnt + 1)) + 1.0 for tok, cnt in df.items()}

        for tokens in doc_tokens:
            tf: dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            total = len(tokens) or 1
            self._tfidf.append(
                {tok: (cnt / total) * self._idf.get(tok, 1.0) for tok, cnt in tf.items()}
            )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.docs:
            return []
        q_tokens = _tokenize(query)
        q_tf: dict[str, float] = {}
        for tok in q_tokens:
            q_tf[tok] = q_tf.get(tok, 0) + 1
        q_vec = {tok: (cnt / len(q_tokens) or 1) * self._idf.get(tok, 1.0) for tok, cnt in q_tf.items()}

        scored: list[tuple[float, dict]] = []
        for doc, vec in zip(self.docs, self._tfidf):
            score = self._cosine(q_vec, vec)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**doc, "score": round(score, 4)}
            for score, doc in scored[:top_k]
            if score > 0
        ]

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
