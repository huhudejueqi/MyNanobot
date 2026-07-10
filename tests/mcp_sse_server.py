"""MCP 测试服务器 — SSE 传输。

FastMCP 底层用 Starlette + SseServerTransport。
生成 SSE endpoint http://HOST:PORT/sse
并自动挂载 /messages/ 用于 POST 消息。

配置:
{
  "url": "http://127.0.0.1:9811/sse"
}
"""

from mcp.server.fastmcp import FastMCP
from datetime import datetime

HOST = "127.0.0.1"
PORT = 9811

mcp = FastMCP(
    "sse-test-server",
    host=HOST,
    port=PORT,
    log_level="WARNING",
)


@mcp.tool()
def sse_echo(text: str) -> str:
    """回显输入（通过 SSE 传输）

    Args:
        text: 要回显的文字
    """
    return f"SSE ECHO: {text}"


@mcp.tool()
def sse_time() -> str:
    """返回当前服务器时间"""
    return f"服务器时间 (SSE): {datetime.now().isoformat()}"


if __name__ == "__main__":
    print(f"SSE MCP server listening on http://{HOST}:{PORT}/sse")
    mcp.run(transport="sse")
