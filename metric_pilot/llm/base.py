"""LLM 客户端抽象基类。

统一对外暴露 ``chat`` 与 ``chat_structured`` 两个接口：
- ``chat``：返回纯文本，用于自然语言推理；
- ``chat_structured``：按给定 Pydantic 模型返回结构化对象（Structured Outputs）。

上层 Agent 只依赖该抽象，不关心底层是 OpenAI 兼容接口还是内置规则引擎，
便于后续替换模型或做 A/B 对比实验。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    """LLM 抽象接口。"""

    name: str = "base"

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """根据消息列表返回一段文本。"""

    @abstractmethod
    def chat_structured(self, messages: list[dict[str, str]], schema: type[T], **kwargs) -> T:
        """按 schema 返回结构化对象。"""
