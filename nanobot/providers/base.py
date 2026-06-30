"""LLM Provider 抽象基类。

定义了所有 LLM 服务商必须实现的接口规范：
- LLMResponse：统一响应格式
- ToolCallRequest：工具调用请求
- LLMProvider：抽象基类，子类必须实现 chat() 和 get_default_model()

设计上与参考项目保持一致的抽象层次，
但简化了重试策略和流式处理等高级特性。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any


@dataclass
class ToolCallRequest:
    """来自 LLM 回复中的工具调用请求。

    当模型决定调用外部工具时，会返回一个或多个 ToolCallRequest。
    每个请求包含工具的唯一标识、名称和参数。

    Attributes:
        id: 工具调用的唯一标识，用于关联结果回传
        name: 工具名称，对应注册的工具名
        arguments: 参数字符串（JSON 格式），或已解析的 dict
    """

    id: str  # 工具调用 ID，用于 tool_call_id 回填
    name: str  # 工具名称
    arguments: Any  # JSON 参数，可能是字符串或 dict


@dataclass
class LLMResponse:
    """LLM provider 的统一响应格式。

    不论底层是 OpenAI、DeepSeek 还是其他 API，
    都统一封装为此格式。上层代码无需关心 provider 差异。

    Attributes:
        content: 模型回复的文本内容，工具调用时可能为 None
        tool_calls: 工具调用请求列表（如果有）
        finish_reason: 结束原因，如 stop、tool_calls、error
        usage: token 用量统计，内含 prompt_tokens/completion_tokens 等
    """

    content: str | None  # 回复文本，工具调用时可能为 None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)  # 工具调用列表
    finish_reason: str = "stop"  # 结束原因
    usage: dict[str, int] = field(default_factory=dict)  # token 用量

    @property
    def has_tool_calls(self) -> bool:
        """判断是否包含工具调用。"""
        return len(self.tool_calls) > 0


class StreamDelta:
    def __init__(self, content: str = "", finish_reason: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls or []

    """流式响应中的单个增量片段。
        tool_calls: 末帧携带完整的工具调用列表
    """
    content: str = ""
    finish_reason: str | None = None
    tool_calls: list = []  # 末帧携带完整的工具调用列表


class LLMProvider(ABC):
    """LLM provider 抽象基类。

    所有 LLM 服务商（OpenAI、DeepSeek、Anthropic 等）
    都需要继承此类并实现 chat() 和 get_default_model() 方法。
    遵循参考项目"依赖抽象，不依赖具体实现"的设计原则。
    """

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        # API 认证信息，子类在 chat() 中使用
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """发送聊天补全请求。

        这是所有 provider 必须实现的核心方法。
        参数与 OpenAI Chat Completions API 保持兼容。

        Args:
            messages: 消息列表，每个元素包含 role 和 content
            tools: 可选的工具定义列表
            model: 模型名称，不传则使用默认模型
            max_tokens: 最大生成 token 数
            temperature: 采样温度 (0-2)
            tool_choice: 工具选择策略

        Returns:
            LLMResponse 统一响应
        """
        ...

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamDelta]:
        """流式聊天补全，默认回退到非流式 chat()。

        子类可覆盖此方法以原生支持 SSE 流式输出。
        默认实现调用 chat() 后一次性 yield 完整结果。
        """
        response = await self.chat(
            messages=messages, model=model,
            max_tokens=max_tokens, temperature=temperature,
            tools=tools,
        )
        yield StreamDelta(content=response.content or "", finish_reason=response.finish_reason)

    @abstractmethod
    def get_default_model(self) -> str:
        """返回此 provider 的默认模型名称。

        当调用方没有指定模型时，使用此方法提供的默认值。
        """
        ...
