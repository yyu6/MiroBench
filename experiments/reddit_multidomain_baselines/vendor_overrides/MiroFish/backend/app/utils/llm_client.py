"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import sys
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from openai import OpenAI

from ..config import Config

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    from token_usage_tracker import record_openai_usage
except Exception:  # pragma: no cover - tracking must never block simulation.
    def record_openai_usage(*args: Any, **kwargs: Any) -> None:
        return


def _supports_custom_temperature(model: str) -> bool:
    """Return whether *model* accepts non-default temperature overrides."""

    return not model.strip().lower().startswith("gpt-5")


def _resolve_reasoning_effort(model: str, configured: str) -> str | None:
    """Return the configured reasoning effort for simulation-side LLM calls."""

    if not model.strip().lower().startswith("gpt-5"):
        return None
    effort = (configured or "").strip().lower()
    if effort == "none":
        return None
    return effort or None


class LLMClient:
    """LLM客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.reasoning_effort = Config.LLM_REASONING_EFFORT

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if _supports_custom_temperature(self.model):
            kwargs["temperature"] = temperature
        reasoning_effort = _resolve_reasoning_effort(self.model, self.reasoning_effort)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        record_openai_usage(response, model=self.model, component="oasis_llm_client")
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
