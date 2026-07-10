#!/usr/bin/env bash
# 启动三个 MCP 测试服务器，然后启动 agent
# 用法: bash start_test_mcp.sh

set -e
PROJECT="/home/huhu/workspace/nanobot/MyNanobot"
PYTHON="/home/huhu/miniconda3/envs/nanobot/bin/python"

echo "=== 启动 SSE MCP server (端口 9811) ==="
setsid $PYTHON "$PROJECT/tests/mcp_sse_server.py" > /tmp/mcp_sse.log 2>&1 &
SSE_PID=$!
sleep 2

echo "=== 启动 Streamable HTTP MCP server (端口 9812) ==="
setsid $PYTHON "$PROJECT/tests/mcp_http_server.py" > /tmp/mcp_http.log 2>&1 &
HTTP_PID=$!
sleep 2

echo "=== 启动 Agent (stdio MCP 会自动 spawn) ==="
$PYTHON "$PROJECT/main.py"

echo "=== 清理 ==="
kill $SSE_PID $HTTP_PID 2>/dev/null || true
