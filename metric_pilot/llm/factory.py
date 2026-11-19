"""LLM 工厂：按配置选择真实模型或规则回退引擎。"""

from __future__ import annotations

from metric_pilot import config
from metric_pilot.llm.base import BaseLLM


def build_llm(provider: str | None = None) -> BaseLLM:
    """根据配置构造 LLM 客户端。

    - ``provider=openai``：强制使用 OpenAI 兼容客户端（缺密钥则报错）；
    - ``provider=rule``：强制使用规则引擎；
    - ``provider=auto``（默认）：有 Key 用真实模型，否则回退规则引擎。
    """
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider in ("openai", "auto") and config.LLM_API_KEY:
        from metric_pilot.llm.openai_client import OpenAICompatLLM

        return OpenAICompatLLM()

    if provider == "openai":
        # 显式要求 openai 但没有 key：直接抛出，避免静默降级
        from metric_pilot.llm.openai_client import OpenAICompatLLM

        return OpenAICompatLLM()

    from metric_pilot.llm.rule_engine import RuleLLM

    return RuleLLM()
