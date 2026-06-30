"""工具基类。"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """工具抽象基类。

    所有工具继承此类，实现 name/description/schema 属性
    和 execute() 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，用于 LLM 引用。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，LLM 决定是否调用时参考。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """参数 JSON Schema，描述工具需要哪些参数。"""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具调用，返回结果字符串。"""
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """转为 OpenAI function calling 格式的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
