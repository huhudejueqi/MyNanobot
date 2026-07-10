"""MCP 计算器服务器 — 数学运算，测试复杂 inputSchema。"""

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import math


async def main():
    server = Server("calc-server")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="calc_add",
                description="两数相加",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "第一个数"},
                        "b": {"type": "number", "description": "第二个数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="calc_sub",
                description="两数相减 (a - b)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "被减数"},
                        "b": {"type": "number", "description": "减数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="calc_mul",
                description="两数相乘",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "第一个数"},
                        "b": {"type": "number", "description": "第二个数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="calc_div",
                description="两数相除 (a / b)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "被除数"},
                        "b": {"type": "number", "description": "除数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="calc_power",
                description="计算 a 的 b 次方",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "底数"},
                        "b": {"type": "number", "description": "指数"},
                    },
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="calc_sqrt",
                description="计算平方根",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "非负数"},
                    },
                    "required": ["x"],
                },
            ),
            types.Tool(
                name="calc_avg",
                description="计算多个数的平均值",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "数字列表，如 [1, 2, 3]",
                        },
                    },
                    "required": ["numbers"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        try:
            args = arguments or {}

            if name == "calc_add":
                a, b = args.get("a", 0), args.get("b", 0)
                result = a + b
                return [types.TextContent(type="text", text=f"{a} + {b} = {result}")]

            elif name == "calc_sub":
                a, b = args.get("a", 0), args.get("b", 0)
                result = a - b
                return [types.TextContent(type="text", text=f"{a} - {b} = {result}")]

            elif name == "calc_mul":
                a, b = args.get("a", 0), args.get("b", 0)
                result = a * b
                return [types.TextContent(type="text", text=f"{a} × {b} = {result}")]

            elif name == "calc_div":
                a, b = args.get("a", 0), args.get("b", 1)
                if b == 0:
                    return [types.TextContent(type="text", text="错误: 除数不能为 0")]
                result = a / b
                return [types.TextContent(type="text", text=f"{a} ÷ {b} = {result}")]

            elif name == "calc_power":
                a, b = args.get("a", 0), args.get("b", 1)
                result = a ** b
                return [types.TextContent(type="text", text=f"{a} ^ {b} = {result}")]

            elif name == "calc_sqrt":
                x = args.get("x", 0)
                if x < 0:
                    return [types.TextContent(type="text", text="错误: 不能对负数开平方")]
                result = math.sqrt(x)
                return [types.TextContent(type="text", text=f"√{x} = {result}")]

            elif name == "calc_avg":
                nums = args.get("numbers", [])
                if not nums:
                    return [types.TextContent(type="text", text="错误: 数字列表为空")]
                result = sum(nums) / len(nums)
                text = f"avg({nums}) = {result}"
                return [types.TextContent(type="text", text=text)]

            return [types.TextContent(type="text", text=f"未知工具: {name}")]

        except Exception as e:
            return [types.TextContent(type="text", text=f"计算错误: {type(e).__name__}: {e}")]

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="calc-server",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
