"""MCP 计数器服务器 — 带内存状态，测试跨调用状态保持。"""

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from datetime import datetime


_COUNTERS: dict[str, int] = {"default": 0}


async def main():
    server = Server("counter-server")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="count_get",
                description="获取指定计数器的当前值",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "计数器名称，默认 default",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="count_inc",
                description="将指定计数器加 1，返回新值",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "计数器名称，默认 default",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="count_add",
                description="将指定计数器加 n，返回新值",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "计数器名称，默认 default",
                        },
                        "value": {
                            "type": "number",
                            "description": "要加的值（可正可负）",
                        },
                    },
                    "required": ["value"],
                },
            ),
            types.Tool(
                name="count_reset",
                description="重置计数器为 0",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "计数器名称，默认 default",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="count_list",
                description="列出所有计数器名称和当前值",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        args = arguments or {}
        cname = args.get("name", "default")

        if name == "count_get":
            val = _COUNTERS.get(cname, 0)
            return [types.TextContent(type="text", text=f"计数器 {cname!r} = {val}")]

        elif name == "count_inc":
            _COUNTERS[cname] = _COUNTERS.get(cname, 0) + 1
            return [types.TextContent(type="text", text=f"计数器 {cname!r} = {_COUNTERS[cname]}")]

        elif name == "count_add":
            delta = args.get("value", 0)
            _COUNTERS[cname] = _COUNTERS.get(cname, 0) + delta
            return [types.TextContent(type="text", text=f"计数器 {cname!r} = {_COUNTERS[cname]} (变化: {delta:+d})")]

        elif name == "count_reset":
            _COUNTERS[cname] = 0
            return [types.TextContent(type="text", text=f"计数器 {cname!r} 已重置为 0")]

        elif name == "count_list":
            if not _COUNTERS:
                return [types.TextContent(type="text", text="(没有计数器)")]
            lines = [f"  {k!r} = {v}" for k, v in sorted(_COUNTERS.items())]
            return [types.TextContent(type="text", text="计数器列表:\n" + "\n".join(lines))]

        return [types.TextContent(type="text", text=f"未知工具: {name}")]

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="counter-server",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
