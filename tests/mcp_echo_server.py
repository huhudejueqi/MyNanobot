"""MCP 回显/时间服务器 — 最简单的基础测试。"""

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from datetime import datetime


async def main():
    server = Server("echo-server")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="echo",
                description="回显输入的文字",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "要回显的文字"},
                    },
                    "required": ["message"],
                },
            ),
            types.Tool(
                name="echo_json",
                description="回显一个 JSON 对象（测试复杂参数）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "键"},
                        "value": {"type": "string", "description": "值"},
                    },
                    "required": ["key", "value"],
                },
            ),
            types.Tool(
                name="ping",
                description="返回 pong 和当前服务器时间",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        if name == "echo":
            msg = (arguments or {}).get("message", "")
            return [types.TextContent(type="text", text=f"ECHO: {msg}")]
        elif name == "echo_json":
            args = arguments or {}
            return [types.TextContent(
                type="text",
                text=f"收到: key={args.get('key')!r}, value={args.get('value')!r}"
            )]
        elif name == "ping":
            return [types.TextContent(
                type="text",
                text=f"pong ({datetime.now().isoformat()})"
            )]
        return [types.TextContent(type="text", text=f"未知工具: {name}")]

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="echo-server",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
