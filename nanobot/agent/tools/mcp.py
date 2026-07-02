"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

import asyncio
import os
import re
import shutil
import urllib.parse
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from typing import Any, Mapping
from weakref import WeakKeyDictionary

import httpx
import logging
logger = logging.getLogger("nanobot.agent.tools.mcp")

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry


# Transient connection errors that warrant a single retry.
# These typically happen when an MCP server restarts or a network
# connection is interrupted between calls.
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset((
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "ConnectionAbortedError",
    "ConnectionError",
))

_WINDOWS_SHELL_LAUNCHERS: frozenset[str] = frozenset(("npx", "npm", "pnpm", "yarn", "bunx"))

# Characters allowed in tool names by model providers (Anthropic, OpenAI, etc.).
# Replace anything outside [a-zA-Z0-9_-] with underscore and collapse runs.
_SANITIZE_RE = re.compile(r"_+")
_RELOAD_LOCKS: WeakKeyDictionary[Any, asyncio.Lock] = WeakKeyDictionary()
_ReconnectCallback = Callable[[str, str, Tool], Awaitable[Tool | None]]


def _sanitize_name(name: str) -> str:
    """Sanitize an MCP-derived name for model API compatibility."""
    return _SANITIZE_RE.sub("_", re.sub(r"[^a-zA-Z0-9_-]", "_", name))


def _is_transient(exc: BaseException) -> bool:
    """Check if an exception looks like a transient connection error."""
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _is_session_terminated(exc: BaseException) -> bool:
    """Return True when the MCP SDK reports a dead client session."""
    messages = [str(exc)]
    error = getattr(exc, "error", None)
    if error is not None:
        messages.append(str(getattr(error, "message", "")))
    return any(
        marker in message.lower()
        for marker in ("session terminated", "connection closed")
        for message in messages
    )


async def _probe_http_url(url: str, timeout: float = 3.0) -> bool:
    """Quick TCP probe to check if an HTTP MCP server is reachable.

    Avoids entering ``streamable_http_client`` / ``sse_client`` when the port is
    closed — those transports use anyio task groups whose cleanup can raise
    ``RuntimeError`` / ``ExceptionGroup`` that escape the caller's try/except
    and crash the event loop.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if not port:
        port = 443 if parsed.scheme == "https" else 80
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        with suppress(OSError, asyncio.TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
        return True
    except (OSError, asyncio.TimeoutError):
        return False



def _windows_command_basename(command: str) -> str:
    """Return the lowercase basename for a Windows command or path."""
    return command.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()


def _normalize_windows_stdio_command(
    command: str,
    args: list[str] | None,
    env: dict[str, str] | None,
) -> tuple[str, list[str], dict[str, str] | None]:
    """Wrap Windows shell launchers so MCP stdio servers start reliably.
    文档注释：对Windows系统下启动MCP stdio服务的命令做包装处理，保证进程正常拉起运行
    """
    # 1. 处理参数列表：如果args是None，转为空列表，统一格式
    normalized_args = list(args or [])

    # 2. 判断操作系统：不是Windows（nt代表Windows），直接原样返回，无需处理
    if os.name != "nt":
        return command, normalized_args, env

    # 提取命令文件名（不带路径，只保留程序名，如 C:\xxx\node.exe → node）
    basename = _windows_command_basename(command)

    # 3. 如果本身就是 cmd/powershell/pwsh 等系统shell，不需要包装，直接返回
    if basename in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return command, normalized_args, env

    # 4. 如果命令本身是exe/com可执行程序，原生可直接启动，不用包装
    if basename.endswith((".exe", ".com")):
        return command, normalized_args, env

    # 5. 根据环境变量PATH查找命令真实完整路径；找不到就沿用原始command
    resolved = shutil.which(command, path=(env or {}).get("PATH")) or command
    # 取出真实路径的文件名
    resolved_basename = _windows_command_basename(resolved)

    # 6. 判断是否需要用shell包装启动：满足任意一条就要包装
    should_wrap = (
        # 命令属于Windows脚本启动器
        basename in _WINDOWS_SHELL_LAUNCHERS
        # 原始命令是 .bat/.cmd 批处理脚本
        or basename.endswith((".cmd", ".bat"))
        # 真实解析后的命令是 .bat/.cmd 批处理脚本
        or resolved_basename.endswith((".cmd", ".bat"))
    )

    # 7. 不需要包装 → 直接返回原始命令、参数、环境变量
    if not should_wrap:
        return command, normalized_args, env

    comspec = (env or {}).get("COMSPEC") or os.environ.get("COMSPEC") or "cmd.exe"
    return comspec, ["/d", "/c", command, *normalized_args], env


def _extract_nullable_branch(options: Any) -> tuple[dict[str, Any], bool] | None:
    """Return the single non-null branch for nullable unions."""
    if not isinstance(options, list):
        return None

    non_null: list[dict[str, Any]] = []
    saw_null = False
    for option in options:
        if not isinstance(option, dict):
            return None
        if option.get("type") == "null":
            saw_null = True
            continue
        non_null.append(option)

    if saw_null and len(non_null) == 1:
        return non_null[0], True
    return None


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    """Normalize only nullable JSON Schema patterns for tool definitions."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)

    raw_type = normalized.get("type")
    if isinstance(raw_type, list):
        non_null = [item for item in raw_type if item != "null"]
        if "null" in raw_type and len(non_null) == 1:
            normalized["type"] = non_null[0]
            normalized["nullable"] = True

    for key in ("oneOf", "anyOf"):
        nullable_branch = _extract_nullable_branch(normalized.get(key))
        if nullable_branch is not None:
            branch, _ = nullable_branch
            merged = {k: v for k, v in normalized.items() if k != key}
            merged.update(branch)
            normalized = merged
            normalized["nullable"] = True
            break

    if "properties" in normalized and isinstance(normalized["properties"], dict):
        normalized["properties"] = {
            name: _normalize_schema_for_openai(prop) if isinstance(prop, dict) else prop
            for name, prop in normalized["properties"].items()
        }

    if "items" in normalized and isinstance(normalized["items"], dict):
        normalized["items"] = _normalize_schema_for_openai(normalized["items"])

    if normalized.get("type") != "object":
        return normalized

    normalized.setdefault("properties", {})
    normalized.setdefault("required", [])
    return normalized


class _MCPWrapperBase(Tool):
    """Common reconnect handling for wrappers bound to one MCP server session."""

    _plugin_discoverable = False

    def _set_mcp_connection(self, session: Any, server_name: str) -> None:
        self._session = session
        self._server_name = server_name
        self._reconnect: _ReconnectCallback | None = None

    def set_reconnect_handler(self, reconnect: _ReconnectCallback) -> None:
        self._reconnect = reconnect

    async def _refresh_session_after_termination(
        self,
        exc: BaseException,
        already_refreshed: bool,
        capability_kind: str,
    ) -> bool:
        if already_refreshed or not _is_session_terminated(exc) or self._reconnect is None:
            return False
        logger.warning(
            "MCP {} '{}' session terminated; reconnecting server '{}' before retry",
            capability_kind,
            self._name,
            self._server_name,
        )
        refreshed_tool = await self._reconnect(self._server_name, self._name, self)
        refreshed_session = getattr(refreshed_tool, "_session", None)
        if refreshed_session is None:
            logger.warning(
                "MCP {} '{}' could not refresh session for server '{}'",
                capability_kind,
                self._name,
                self._server_name,
            )
            return False
        self._session = refreshed_session
        return True


class MCPToolWrapper(_MCPWrapperBase):
    """Wraps a single MCP server tool as a nanobot Tool."""

    _plugin_discoverable = False

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        self._set_mcp_connection(session, server_name)
        self._original_name = tool_def.name
        self._name = _sanitize_name(f"mcp_{server_name}_{tool_def.name}")
        self._description = tool_def.description or tool_def.name
        raw_schema = tool_def.inputSchema or {"type": "object", "properties": {}}
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(self._original_name, arguments=kwargs),
                    timeout=self._tool_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP tool '{}' timed out after {}s", self._name, self._tool_timeout
                )
                return f"(MCP tool call timed out after {self._tool_timeout}s)"
            except asyncio.CancelledError:
                # MCP SDK's anyio cancel scopes can leak CancelledError on timeout/failure.
                # Re-raise only if our task was externally cancelled (e.g. /stop).
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP tool '%s' was cancelled by server/SDK", self._name)
                return "(MCP tool call was cancelled)"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "tool",
                ):
                    refreshed_session = True
                    continue
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP tool '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)  # Brief backoff before retry
                        continue
                    # Second transient failure — give up with retry-specific message
                    logger.exception(
                        "MCP tool '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP tool call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP tool '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP tool call failed: {type(exc).__name__})"
            else:
                # Success — extract result
                parts = []
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP tool call failed)"  # Unreachable, but satisfies type checkers


class MCPResourceWrapper(_MCPWrapperBase):
    """Wraps an MCP resource URI as a read-only nanobot Tool."""

    _plugin_discoverable = False

    def __init__(self, session, server_name: str, resource_def, resource_timeout: int = 30):
        self._set_mcp_connection(session, server_name)
        self._uri = resource_def.uri
        self._name = _sanitize_name(f"mcp_{server_name}_resource_{resource_def.name}")
        desc = resource_def.description or resource_def.name
        self._description = f"[MCP Resource] {desc}\nURI: {self._uri}"
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._resource_timeout = resource_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.read_resource(self._uri),
                    timeout=self._resource_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP resource '{}' timed out after {}s", self._name, self._resource_timeout
                )
                return f"(MCP resource read timed out after {self._resource_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP resource '%s' was cancelled by server/SDK", self._name)
                return "(MCP resource read was cancelled)"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "resource",
                ):
                    refreshed_session = True
                    continue
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP resource '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP resource '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP resource read failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP resource '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP resource read failed: {type(exc).__name__})"
            else:
                parts: list[str] = []
                for block in result.contents:
                    if isinstance(block, types.TextResourceContents):
                        parts.append(block.text)
                    elif isinstance(block, types.BlobResourceContents):
                        parts.append(f"[Binary resource: {len(block.blob)} bytes]")
                    else:
                        parts.append(str(block))
                return "\n".join(parts) or "(no output)"

        return "(MCP resource read failed)"  # Unreachable


class MCPPromptWrapper(_MCPWrapperBase):
    """Wraps an MCP prompt as a read-only nanobot Tool."""

    _plugin_discoverable = False

    def __init__(self, session, server_name: str, prompt_def, prompt_timeout: int = 30):
        self._set_mcp_connection(session, server_name)
        self._prompt_name = prompt_def.name
        self._name = _sanitize_name(f"mcp_{server_name}_prompt_{prompt_def.name}")
        desc = prompt_def.description or prompt_def.name
        self._description = (
            f"[MCP Prompt] {desc}\n"
            "Returns a filled prompt template that can be used as a workflow guide."
        )
        self._prompt_timeout = prompt_timeout

        # Build parameters from prompt arguments
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg in prompt_def.arguments or []:
            prop: dict[str, Any] = {"type": "string"}
            if getattr(arg, "description", None):
                prop["description"] = arg.description
            properties[arg.name] = prop
            if arg.required:
                required.append(arg.name)
        self._parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        from mcp.shared.exceptions import McpError

        retried_transient = False
        refreshed_session = False
        while True:
            try:
                result = await asyncio.wait_for(
                    self._session.get_prompt(self._prompt_name, arguments=kwargs),
                    timeout=self._prompt_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP prompt '{}' timed out after {}s", self._name, self._prompt_timeout
                )
                return f"(MCP prompt call timed out after {self._prompt_timeout}s)"
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise
                logger.warning("MCP prompt '%s' was cancelled by server/SDK", self._name)
                return "(MCP prompt call was cancelled)"
            except McpError as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                logger.exception(
                    "MCP prompt '{}' failed: code={} message={}",
                    self._name,
                    exc.error.code,
                    exc.error.message,
                )
                return f"(MCP prompt call failed: {exc.error.message} [code {exc.error.code}])"
            except Exception as exc:
                if await self._refresh_session_after_termination(
                    exc,
                    refreshed_session,
                    "prompt",
                ):
                    refreshed_session = True
                    continue
                if _is_transient(exc):
                    if not retried_transient:
                        retried_transient = True
                        logger.warning(
                            "MCP prompt '{}' hit transient error ({}), retrying once...",
                            self._name,
                            type(exc).__name__,
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception(
                        "MCP prompt '{}' failed after retry: {}",
                        self._name,
                        type(exc).__name__,
                    )
                    return f"(MCP prompt call failed after retry: {type(exc).__name__})"
                logger.exception(
                    "MCP prompt '{}' failed: {}: {}",
                    self._name,
                    type(exc).__name__,
                    exc,
                )
                return f"(MCP prompt call failed: {type(exc).__name__})"
            else:
                parts: list[str] = []
                for message in result.messages:
                    content = message.content
                    if isinstance(content, types.TextContent):
                        parts.append(content.text)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, types.TextContent):
                                parts.append(block.text)
                            else:
                                parts.append(str(block))
                    else:
                        parts.append(str(content))
                return "\n".join(parts) or "(no output)"

        return "(MCP prompt call failed)"  # Unreachable


async def connect_mcp_servers(
    mcp_servers: dict, registry: ToolRegistry
) -> dict[str, AsyncExitStack]:
    """Connect to configured MCP servers and register their tools, resources, prompts.

    Returns a dict mapping server name -> its dedicated AsyncExitStack.
    Each server gets its own stack to prevent cancel scope conflicts
    when multiple MCP servers are configured.
    """
# connect_mcp_servers(批量连接所有MCP服务)
# ├─ 延迟导入MCP官方客户端模块
# ├─ 内部子函数：connect_single_server(单个MCP服务连接核心逻辑)
# │  ├─ 创建独立 AsyncExitStack 资源栈（单服务隔离资源）
# │  ├─ 手动进入资源栈上下文
# │  ├─ try 主连接流程
# │  │  ├─ 步骤1：自动识别传输类型 transport_type
# │  │  │  ├─ 配置有type → 直接使用
# │  │  │  └─ 无type自动推断：
# │  │  │     ├─ 有command → stdio
# │  │  │     ├─ 有url：路径尾 /sse → sse，否则 streamableHttp
# │  │  │     └─ 无command无url → 警告、关闭栈、返回None跳过
# │  │  ├─ 步骤2：网络传输前置URL安全校验（sse / streamableHttp）
# │  │  │  ├─ URL不安全 → 警告、关栈、返回None
# │  │  ├─ 步骤3：按传输类型建立读写流 read/write
# │  │  │  ├─ 分支1：transport_type = stdio
# │  │  │  │  ├─ 标准化Windows启动命令/参数/环境变量
# │  │  │  │  ├─ 组装 StdioServerParameters
# │  │  │  │  └─ stdio_client 入栈，得到 read, write
# │  │  │  ├─ 分支2：transport_type = sse
# │  │  │  │  ├─ 探测URL连通性，不可达则跳过
# │  │  │  │  ├─ 自定义httpx客户端工厂（合并请求头、URL校验钩子）
# │  │  │  │  └─ sse_client 入栈，得到 read, write
# │  │  │  ├─ 分支3：transport_type = streamableHttp
# │  │  │  │  ├─ 探测URL连通性，不可达则跳过
# │  │  │  │  ├─ 创建 httpx.AsyncClient 并入栈管理
# │  │  │  │  └─ streamable_http_client 入栈，得到 read, write
# │  │  │  └─ 分支4：未知传输类型 → 警告、关栈、返回None
# │  │  ├─ 步骤4：初始化MCP ClientSession会话
# │  │  │  ├─ ClientSession 加入资源栈托管
# │  │  │  └─ session.initialize() 完成MCP握手
# │  │  ├─ 步骤5：拉取并注册 Tools 工具
# │  │  │  ├─ 获取服务全部工具列表 tools
# │  │  │  ├─ 读取 enabled_tools 配置，判断是否允许全部工具(*)
# │  │  │  ├─ 遍历所有工具：
# │  │  │  │  ├─ 生成全局唯一包装名 mcp_服务名_工具名
# │  │  │  │  ├─ 不在启用列表则跳过
# │  │  │  │  ├─ 封装 MCPToolWrapper，注册到 ToolRegistry
# │  │  │  │  └─ 统计已注册数量、记录匹配的启用工具
# │  │  │  └─ 校验 enabled_tools 不存在的工具，输出警告日志
# │  │  ├─ 步骤6：拉取并注册 Resources 资源（try捕获兼容无资源服务）
# │  │  │  └─ 封装 MCPResourceWrapper 注册进注册表
# │  │  ├─ 步骤7：拉取并注册 Prompts 提示词（try捕获兼容无提示词服务）
# │  │  │  └─ 封装 MCPPromptWrapper 注册进注册表
# │  │  ├─ 打印连接成功日志，输出总注册能力数量
# │  │  └─ 返回 (服务名, 当前服务专属AsyncExitStack)
# │  ├─ except 全局异常捕获（连接失败）
# │  │  ├─ 识别JSON/JSONRPC协议污染错误，补充提示文案
# │  │  ├─ 打印异常堆栈日志
# │  │  ├─ 安全关闭资源栈（忽略关闭异常）
# │  │  └─ 返回 (服务名, None)
# ├─ 外层主逻辑：遍历所有配置 mcp_servers
# │  ├─ 逐个调用 connect_single_server 连接单个服务
# │  ├─ 捕获单服务连接外层异常，打印日志并跳过
# │  └─ 连接成功（返回栈不为空）存入 server_stacks 字典
# └─ 返回 server_stacks {服务名: AsyncExitStack}
    # 延迟导入MCP客户端相关模块，避免顶层导入拖慢启动速度
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    async def connect_single_server(name: str, cfg) -> tuple[str, AsyncExitStack | None]:
        """单个MCP服务连接逻辑，返回服务名与对应资源栈，失败返回None"""
        # 当前服务独立异步资源栈，隔离多服务上下文，避免取消域冲突
        server_stack = AsyncExitStack()
        # 手动进入资源栈上下文，方便函数内按需关闭
        await server_stack.__aenter__()

        try:
            # 读取配置指定的传输类型
            transport_type = cfg.get('type', '')
            # 未配置传输类型时自动推断
            if not transport_type:
                if cfg.get('command'):
                    # 存在启动命令则判定为stdio标准流模式
                    transport_type = "stdio"
                elif cfg.get('url'):
                    # 存在url，根据路径后缀区分sse / streamableHttp
                    transport_type = (
                        "sse" if cfg.get("url", "").rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                else:
                    # 无命令、无地址，配置无效，跳过当前服务
                    logger.warning("MCP server '%s': no command or url configured, skipping", name)
                    await server_stack.aclose()
                    return name, None

            # SSE/流式HTTP 网络传输，先校验URL安全性，拦截危险地址
            if transport_type in {"sse", "streamableHttp"}:
                ok, error = validate_url_target(cfg.get("url", ""))
                if not ok:
                    logger.warning(
                        "MCP server '{}': blocked unsafe URL {} ({})",
                        name,
                        cfg.get("url", ""),
                        error,
                    )
                    await server_stack.aclose()
                    return name, None

            # 分支1：stdio本地子进程传输
            if transport_type == "stdio":
                # 标准化windows下启动命令、参数、环境变量兼容处理
                command, args, env = _normalize_windows_stdio_command(
                    cfg.get("command"),
                    cfg.get("args") or [],
                    cfg.get("env"),
                )
                # 组装子进程启动参数
                params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env,
                    cwd=cfg.get("cwd"),
                )
                # 创建stdio读写流，交由资源栈托管生命周期
                read, write = await server_stack.enter_async_context(stdio_client(params))

            # 分支2：SSE长连接传输
            elif transport_type == "sse":
                # 预探测目标地址是否可访问，不可达直接跳过
                if not await _probe_http_url(cfg.get("url", "")):
                    logger.warning("MCP server '%s': %s unreachable, skipping", name, cfg.get("url", ""))
                    await server_stack.aclose()
                    return name, None

                # 自定义httpx客户端工厂，合并全局/服务自定义请求头、URL校验钩子
                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    merged_headers = {
                        "Accept": "application/json, text/event-stream",
                        **(cfg.get("headers") or {}),
                        **(headers or {}),
                    }
                    return httpx.AsyncClient(
                        headers=merged_headers or None,
                    
                        follow_redirects=True,
                        timeout=timeout,
                        auth=auth,
                    )

                # 创建sse读写流，资源栈自动释放
                read, write = await server_stack.enter_async_context(
                    sse_client(cfg.get("url", ""), httpx_client_factory=httpx_client_factory)
                )

            # 分支3：streamableHttp流式HTTP传输
            elif transport_type == "streamableHttp":
                # 探测地址连通性
                if not await _probe_http_url(cfg.get("url", "")):
                    logger.warning("MCP server '%s': %s unreachable, skipping", name, cfg.get("url", ""))
                    await server_stack.aclose()
                    return name, None

                # 创建独立http客户端，加入资源栈管理
                http_client = await server_stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.get("headers") or None,
                    
                        follow_redirects=True,
                        timeout=None,
                    )
                )
                # 创建流式http读写流
                read, write, _ = await server_stack.enter_async_context(
                    streamable_http_client(cfg.get("url", ""), http_client=http_client)
                )

            # 未知传输类型，直接关闭资源并跳过
            else:
                logger.warning("MCP server '%s': unknown transport type '%s'", name, transport_type)
                await server_stack.aclose()
                return name, None

            # 创建MCP客户端会话，托管至资源栈
            session = await server_stack.enter_async_context(ClientSession(read, write))
            # 执行MCP握手初始化
            await session.initialize()

            # 获取服务提供的全部工具列表
            tools = await session.list_tools()
            # 用户配置启用的工具集合
            enabled_tools = set(cfg.get("enabled_tools", ["*"]))
            # *代表启用全部工具标识；未配置时默认全部启用
            allow_all_tools = "*" in enabled_tools
            registered_count = 0  # 统计注册成功的工具/资源/提示词总数
            matched_enabled_tools: set[str] = set()  # 记录匹配到的启用工具名
            # 原始工具名、包装后唯一工具名预生成，用于后续日志告警
            available_raw_names = [tool_def.name for tool_def in tools.tools]
            available_wrapped_names = [_sanitize_name(f"mcp_{name}_{tool_def.name}") for tool_def in tools.tools]

            # 遍历所有工具，过滤并注册启用工具
            for tool_def in tools.tools:
                # 拼接全局唯一工具名，防止多服务重名冲突
                wrapped_name = _sanitize_name(f"mcp_{name}_{tool_def.name}")
                # 不在启用列表中则跳过注册
                if (
                    not allow_all_tools
                    and tool_def.name not in enabled_tools
                    and wrapped_name not in enabled_tools
                ):
                    logger.debug(
                        "MCP: skipping tool '{}' from server '{}' (not in enabledTools)",
                        wrapped_name,
                        name,
                    )
                    continue
                # 封装MCP工具为内部可调用工具对象
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.get("tool_timeout", 30))
                registry.register(wrapper)
                logger.debug("MCP: registered tool '%s' from server '%s'", wrapper.name, name)
                registered_count += 1
                # 记录匹配到的启用工具
                if enabled_tools:
                    if tool_def.name in enabled_tools:
                        matched_enabled_tools.add(tool_def.name)
                    if wrapped_name in enabled_tools:
                        matched_enabled_tools.add(wrapped_name)

            # 校验启用列表中不存在的工具，输出告警日志
            if enabled_tools and not allow_all_tools:
                unmatched_enabled_tools = sorted(enabled_tools - matched_enabled_tools)
                if unmatched_enabled_tools:
                    logger.warning(
                        "MCP server '{}': enabledTools entries not found: {}. Available raw names: {}. "
                        "Available wrapped names: {}",
                        name,
                        ", ".join(unmatched_enabled_tools),
                        ", ".join(available_raw_names) or "(none)",
                        ", ".join(available_wrapped_names) or "(none)",
                    )

            # 注册服务资源列表，捕获异常兼容不支持resource的服务
            try:
                resources_result = await session.list_resources()
                for resource in resources_result.resources:
                    wrapper = MCPResourceWrapper(
                        session, name, resource, resource_timeout=cfg.get("tool_timeout", 30)
                    )
                    registry.register(wrapper)
                    registered_count += 1
                    logger.debug(
                        "MCP: registered resource '{}' from server '{}'", wrapper.name, name
                    )
            except Exception as e:
                logger.debug("MCP server '%s': resources not supported or failed: %s", name, e)

            # 注册服务提示词列表，捕获异常兼容不支持prompt的服务
            try:
                prompts_result = await session.list_prompts()
                for prompt in prompts_result.prompts:
                    wrapper = MCPPromptWrapper(
                        session, name, prompt, prompt_timeout=cfg.get("tool_timeout", 30)
                    )
                    registry.register(wrapper)
                    registered_count += 1
                    logger.debug("MCP: registered prompt '%s' from server '%s'", wrapper.name, name)
            except Exception as e:
                logger.debug("MCP server '%s': prompts not supported or failed: %s", name, e)

            # 输出连接成功日志，打印注册能力总数
            logger.info(
                "MCP server '%s': connected, %s capabilities registered", name, registered_count
            )
            # 返回当前服务名与专属资源栈
            return name, server_stack

        except Exception as e:
            # 识别JSON解析/协议类错误，补充提示信息
            hint = ""
            text = str(e).lower()
            if any(
                marker in text
                for marker in (
                    "parse error",
                    "invalid json",
                    "unexpected token",
                    "jsonrpc",
                    "content-length",
                )
            ):
                hint = (
                    " Hint: this looks like stdio protocol pollution. Make sure the MCP server writes "
                    "only JSON-RPC to stdout and sends logs/debug output to stderr instead."
                )
            logger.exception("MCP server '%s': failed to connect: %s", name, hint)
            # 忽略关闭时异常，安全释放资源栈
            with suppress(Exception):
                await server_stack.aclose()
            return name, None

    # 存储所有连接成功服务对应的资源栈
    server_stacks: dict[str, AsyncExitStack] = {}

    # 循环遍历所有配置的MCP服务依次连接
    for name, cfg in mcp_servers.items():
        try:
            result = await connect_single_server(name, cfg)
        except Exception as e:
            logger.exception("MCP server '%s' connection failed: %s", name, e)
            continue
        # 连接成功则存入映射字典
        if result is not None and result[1] is not None:
            server_stacks[result[0]] = result[1]

    # 返回服务名-资源栈映射
    return server_stacks



