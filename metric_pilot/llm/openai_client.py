"""OpenAI 兼容协议客户端。

可接入 DeepSeek、通义千问（DashScope 兼容模式）、Moonshot、本地 Ollama 等任何
提供 ``/chat/completions`` 与 JSON Schema 能力的服务。

结构化输出优先使用 ``response_format=json_schema``；若服务端不支持，则回退到
提示词约束 + JSON 解析校验，解析失败抛出 ``LLMStructuredError`` 由上层重试。
"""

from __future__ import annotations

import json
from typing import TypeVar

import requests
from pydantic import BaseModel

from metric_pilot import config
from metric_pilot.llm.base import BaseLLM

T = TypeVar("T", bound=BaseModel)


class LLMStructuredError(RuntimeError):
    """结构化输出解析 / 校验失败。"""


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容 Chat Completions 客户端。"""

    name = config.LLM_MODEL

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_API_KEY
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.timeout = timeout or config.LLM_TIMEOUT
        if not self.api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY，无法使用 OpenAI 兼容客户端")

    # ------------------------------------------------------------------
    # 底层请求
    # ------------------------------------------------------------------
    def _post(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - 统一重试后抛出
                last_err = exc
                if attempt < config.LLM_MAX_RETRIES:
                    continue
        raise RuntimeError(f"LLM 请求失败: {last_err}")

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return self._post(messages, json_mode=False)

    def chat_structured(self, messages: list[dict[str, str]], schema: type[T], **kwargs) -> T:
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        guided_messages = messages + [
            {
                "role": "system",
                "content": (
                    "你必须只输出符合给定 JSON Schema 的 JSON 对象，不要输出任何解释、"
                    "markdown 代码块或多余文本。Schema：\n" + schema_hint
                ),
            }
        ]
        raw = self._post(guided_messages, json_mode=True)
        return self._parse_json(raw, schema)

    @staticmethod
    def _parse_json(raw: str, schema: type[T]) -> T:
        text = raw.strip()
        # 去掉可能出现的 ```json 代码块包裹
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMStructuredError(f"模型输出不是合法 JSON: {exc}") from exc
        try:
            return schema.model_validate(obj)
        except Exception as exc:  # noqa: BLE001
            raise LLMStructuredError(f"结构化校验失败: {exc}") from exc
