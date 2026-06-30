"""工具注册中心。"""

from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """管理所有可用工具的注册与执行。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名称获取工具。"""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        """获取所有工具的 OpenAI function calling schema。"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any] | str) -> str:
        """执行一个工具调用。

        Args:
            name: 工具名称
            arguments: 参数字典或 JSON 字符串

        Returns:
            工具执行结果文本
        """
        tool = self.get(name)
        if tool is None:
            return f"[错误: 未找到工具 '{name}']"
        import json
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"[错误: 参数字符串解析失败: {arguments}]"
        try:
            result = await tool.execute(**arguments)
            return result
        except Exception as e:
            return f"[工具 '{name}' 执行异常: {e!s}]"
