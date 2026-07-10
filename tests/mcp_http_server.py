"""MCP 测试服务器 — Streamable HTTP 传输。

FastMCP 底层用 StreamableHTTPSessionManager + Starlette。
生成 endpoint http://HOST:PORT/mcp

配置:
{
  "url": "http://127.0.0.1:9812/mcp",
  "type": "streamableHttp"
}
"""

from mcp.server.fastmcp import FastMCP
from datetime import datetime

HOST = "127.0.0.1"
PORT = 9812

mcp = FastMCP(
    "http-test-server",
    host=HOST,
    port=PORT,
    log_level="WARNING",
)


@mcp.tool()
def http_echo(text: str) -> str:
    """回显输入（通过 Streamable HTTP 传输）

    Args:
        text: 要回显的文字
    """
    return f"HTTP ECHO: {text}"


@mcp.tool()
def http_time() -> str:
    """返回当前服务器时间"""
    return f"服务器时间 (HTTP): {datetime.now().isoformat()}"


@mcp.tool()
def http_info() -> str:
    """返回服务器信息"""
    return f"传输协议: Streamable HTTP\n端口: {PORT}"


if __name__ == "__main__":
    print(f"Streamable HTTP MCP server listening on http://{HOST}:{PORT}/mcp")
    mcp.run(transport="streamable-http")
