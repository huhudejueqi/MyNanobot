"""MCP 测试服务器 — 提供 echo 和 time 两个工具供验证 MCP 链路。"""

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from datetime import datetime
from pathlib import Path


async def main() -> None:
    # 启动时记录时间到文件
    log_path = Path(__file__).resolve().parent.parent / "mcp_test_startup.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"MCP test server started at: {datetime.now().isoformat()}\n")
    
    server = Server("test-mcp")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="echo",
                description="回显输入的消息",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "要回显的文字",
                        },
                    },
                    "required": ["message"],
                },
            ),
            types.Tool(
                name="get_time",
                description="返回当前服务器时间",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent]:
        if name == "echo":
            msg = (arguments or {}).get("message", "")
            return [types.TextContent(type="text", text=f"ECHO: {msg}")]
        elif name == "get_time":
            return [types.TextContent(
                type="text",
                text=f"当前服务器时间: {datetime.now().isoformat()}"
            )]
        return [types.TextContent(type="text", text=f"未知工具: {name}")]

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="test-mcp",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
