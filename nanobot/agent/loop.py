"""Agent 核心处理引擎：事件驱动的状态机。"""

from __future__ import annotations

import asyncio
import json
import time
import signal
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine

import base64

from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanobot.command.router import CommandRouter, CommandContext
from nanobot.command.builtin import register_builtin_commands
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.mcp import connect_mcp_servers
from nanobot.agent.tools.shell import ShellTool
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.utils.document import extract_documents, is_image_file
from nanobot.config.loader import Config
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import ProviderSnapshot, build_provider_snapshot

logger = logging.getLogger("nanobot.agent.loop")


class TurnState(Enum):
    RESTORE = auto()
    COMPACT = auto()
    COMMAND = auto()
    BUILD = auto()
    RUN = auto()
    SAVE = auto()
    RESPOND = auto()
    DONE = auto()


@dataclass
class StateTraceEntry:
    state: TurnState
    started_at: float
    duration_ms: float
    event: str
    error: str | None = None


@dataclass
class TurnContext:
    msg: InboundMessage
    session_key: str
    state: TurnState
    session: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    final_content: str | None = None
    tools_used: list[str] = field(default_factory=list)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    outbound: OutboundMessage | None = None
    on_stream: Callable[[str], Awaitable[None]] | None = None
    on_stream_end: Callable[..., Awaitable[None]] | None = None
    turn_wall_started_at: float = field(default_factory=time.time)
    trace: list[StateTraceEntry] = field(default_factory=list)
    error: str | None = None


class _DebugHook(AgentHook):
    """调试钩子：把 hook 事件全部打印到 logger.info。"""

    async def before_run(self, ctx: AgentRunHookContext) -> None:
        logger.info("[_DebugHook] before_run: messages=%d", len(ctx.messages))

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        logger.info("[_DebugHook] after_run: final_content=%.200r, tools_used=%s, stop_reason=%s",
                     ctx.final_content, ctx.tools_used, ctx.stop_reason)

    async def on_error(self, ctx: AgentRunHookContext) -> None:
        logger.info("[_DebugHook] on_error: %s", ctx.error)

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        tool_hint = ""
        if ctx.tool_calls:
            names = [f"{tc.name}(...)" for tc in ctx.tool_calls]
            tool_hint = f", tool_calls={names}"
        logger.info("[_DebugHook] before_iteration #%d: messages=%d%s",
                     ctx.iteration, len(ctx.messages), tool_hint)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        if ctx.response and ctx.response.content:
            logger.info("[_DebugHook] after_iteration #%d: LLM 回复=%.200r",
                         ctx.iteration, ctx.response.content)
        if ctx.tool_results:
            for i, r in enumerate(ctx.tool_results):
                logger.info("[_DebugHook] after_iteration #%d: 工具%d 结果=%.200r",
                             ctx.iteration, i, r)

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
            logger.info("[_DebugHook] before_execute_tools: %s(args=%s)",
                         tc.name, tc.arguments)

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        logger.info("[_DebugHook] on_stream: delta=%.200r", delta)


class AgentLoop:
    """AgentLoop 是 MyNanobot 的核心处理引擎。
    状态机执行树：
                     消息到达 _dispatch()
                           │
               ┌──────▼────────────────────────┐
               │  RESTORE                      │
               │  ├─ 查找/创建 session          │
               │  ├─ 解析附件：                 │
               │  │   文档 → 提取文字拼入 content│
               │  │   图片 → 留在 media 列表    │
               │  └─ 初始化 ctx.history         │
               │  "ok"                          │
               └──────┬─────────────────────────┘
                      │
               ┌──────▼────────────────────────┐
               │  COMPACT                      │
               │  ├─ 检查消息数 > 60            │
               │  └─ 是 → 截断保留后半          │
               │  "ok"                          │
               └──────┬─────────────────────────┘
                      │
               ┌──────▼────────────────────────┐
               │  COMMAND                      │
               │  ├─ 遍历 _handlers 匹配        │
               │  ├─ 匹配 /ping /time /version  │
               │  └─ 匹配 → shortcut / 否则 dispatch
               └──────┬──────┬──────────────────┘
                      │      │
                 shortcut dispatch
                      │      │
                   ┌──▼──┐ ┌──▼────────────────────┐
                   │DONE │ │  BUILD                │
                   │     │ │  ├─ 拷贝 history      │
                   └─────┘ │  ├─ 有图片 media？     │
                           │  │   是 → base64 编码  │
                           │  │    → 多模态 content │
                           │  │   否 → 纯文本追加   │
                           │  └─ 构建 all_messages │
                           │  "ok"                 │
                           └──┬────────────────────┘
                              │
               ┌──────────────▼──────────────────────────────┐
               │  RUN (ReAct 循环)                            │
               │                                              │
               │  委托 AgentRunner.run()                       │
               │  ┌──────────────────────────────────────┐    │
               │  │  LLM call                             │    │
               │  │  ├─ 首轮+流式 → chat_stream()         │    │
               │  │  └─ 后续/非流式 → chat()              │    │
               │  └──────────┬───────────────────────────┘    │
               │             │                                │
               │  ┌──────────▼───────────────────────────┐    │
               │  │  tool_calls?                          │    │
               │  │  ├─ 是 → 执行工具 → 拼接结果 → 继续   │    │
               │  │  └─ 否 → 返回最终回复                  │    │
               │  └──────────────────────────────────────┘    │
               │ "ok"                    "error"              │
               └──┬──────────────────────────┬────────────────┘
                  │                          │
               ┌──▼──────────────────┐  ┌───▼──────┐
               │  SAVE                │  │  DONE    │
               │  ├─ 追加 user msg    │  │ (error)  │
               │  └─ 追加 assistant   │  └──────────┘
               │  msg 到 session      │
               │  "ok"                │
               └──┬───────────────────┘
                  │
               ┌──▼──────────────────┐  ┌──────────┐
               │  RESPOND             │  │  DONE    │
               │  ├─ 标记 _streamed   │──│ (error)  │
               │  ├─ 组装 OutboundMsg │  └──────────┘
               │  └─ 写入 ctx.outbound│
               │  "ok"                │
               └──┬───────────────────┘
                  │
               ┌──▼────┐
               │  DONE  │
               └────────┘
    """

    _TRANSITIONS: dict[tuple[TurnState, str], TurnState] = {
        (TurnState.RESTORE, "ok"): TurnState.COMPACT,
        (TurnState.COMPACT, "ok"): TurnState.COMMAND,
        (TurnState.COMMAND, "dispatch"): TurnState.BUILD,
        (TurnState.COMMAND, "shortcut"): TurnState.DONE,
        (TurnState.BUILD, "ok"): TurnState.RUN,
        (TurnState.RUN, "ok"): TurnState.SAVE,
        (TurnState.SAVE, "ok"): TurnState.RESPOND,
        (TurnState.RESPOND, "ok"): TurnState.DONE,
        (TurnState.RESTORE, "error"): TurnState.DONE,
        (TurnState.COMPACT, "error"): TurnState.DONE,
        (TurnState.COMMAND, "error"): TurnState.DONE,
        (TurnState.BUILD, "error"): TurnState.DONE,
        (TurnState.RUN, "error"): TurnState.DONE,
        (TurnState.SAVE, "error"): TurnState.DONE,
        (TurnState.RESPOND, "error"): TurnState.DONE,
    }

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        context_window_tokens: int = 128_000,
        max_iterations: int = 10,
        on_llm_start: Callable[[], Coroutine] | None = None,
        on_llm_end: Callable[[], Coroutine] | None = None,
        mcp_servers: dict[str, Any] | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.context_window_tokens = context_window_tokens
        self.on_llm_start = on_llm_start
        self.on_llm_end = on_llm_end

        self._snapshot: ProviderSnapshot | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._current_turn: TurnContext | None = None
        self._handlers: list[Callable[[InboundMessage], str | None]] = []
        self._session_store: dict[str, list[dict[str, Any]]] = {}
        self.context = ContextBuilder(workspace)
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)
        self._mcp_servers = dict(mcp_servers or {})
        self._mcp_stacks: dict[str, Any] = {}
        self._pending_queues: dict[str, asyncio.Queue] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

        # 工具系统
        self.tools = ToolRegistry()
        self.runner = AgentRunner(
            provider=self.provider, max_iterations=max_iterations or 10,
            on_progress=self._on_tool_progress,
            hook=_DebugHook(),
        )
        self.max_iterations = max_iterations or 10
        self._register_default_tools()
        self._mcp_connected = False
        self._mcp_connecting = False

        logger.info("初始化: model=%s, provider=%s", model, type(provider).__name__)

    @property
    def snapshot(self) -> ProviderSnapshot:
        if self._snapshot is None:
            self._snapshot = ProviderSnapshot(
                provider=self.provider,
                model=self.model,
                context_window_tokens=self.context_window_tokens,
                signature=(self.model,),
            )
        return self._snapshot

    def _refresh_provider_snapshot(self) -> None:
        self._snapshot = ProviderSnapshot(
            provider=self.provider,
            model=self.model,
            context_window_tokens=self.context_window_tokens,
            signature=(self.model, self.provider.api_key, self.provider.api_base),
        )

    def apply_snapshot(self, snapshot: ProviderSnapshot) -> None:
        old_model = self.model
        self.provider = snapshot.provider
        self.model = snapshot.model
        self.context_window_tokens = snapshot.context_window_tokens
        self._snapshot = snapshot
        logger.info("切换模型: %s -> %s", old_model, self.model)

    @classmethod
    def from_config(
        cls,
        config: Config | None = None,
        bus: MessageBus | None = None,
    ) -> AgentLoop:
        if config is None:
            from nanobot.config.loader import load_config
            config = load_config()
        snapshot = build_provider_snapshot(config)
        if bus is None:
            bus = MessageBus()
        workspace = Path.home() / ".nanobot" / "workspace"
        tools_cfg = config.raw.get("tools", {})
        mcp_servers = tools_cfg.get("mcpServers") or tools_cfg.get("mcp_servers") or {}
        return cls(
            bus=bus, provider=snapshot.provider,
            workspace=workspace, model=snapshot.model,
            context_window_tokens=snapshot.context_window_tokens,
            mcp_servers=mcp_servers,
        )

    # ── 状态机驱动（异步处理，不阻塞主循环） ──

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """状态机驱动：RESTORE → COMPACT → ... → DONE，返回出站消息。"""
        key = session_key or msg.session_key
        ctx = TurnContext(
            msg=msg, session_key=key, state=TurnState.RESTORE,
            turn_wall_started_at=time.time(),
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

        while ctx.state is not TurnState.DONE:
            handler = getattr(self, f"_state_{ctx.state.name.lower()}", None)
            if handler is None:
                raise RuntimeError(f"缺少 {ctx.state} 状态对应的处理函数")

            t0 = time.time()
            event, error = "ok", None
            try:
                event = await handler(ctx)
                # 打印当前状态的上下文快照
                sname = ctx.state.name
                if sname == "RESTORE":
                    logger.info("[%s] RESTORE: session_key=%s, session=%s, media=%d, history=%d",
                                ctx.turn_wall_started_at, ctx.session_key,
                                ctx.session is not None, len(ctx.msg.media or []), len(ctx.history))
                elif sname == "COMPACT":
                    logger.info("[%s] COMPACT: history_len=%d", ctx.turn_wall_started_at, len(ctx.history))
                elif sname == "COMMAND":
                    logger.info("[%s] COMMAND: event=%s", ctx.turn_wall_started_at, event)
                elif sname == "BUILD":
                    logger.info("[%s] BUILD: history=%d, all_messages=%d", ctx.turn_wall_started_at,
                                len(ctx.history), len(ctx.all_messages))
                elif sname == "RUN":
                    logger.info("[%s] RUN: final_content=%.200r, tools_used=%s, stop_reason=%s",
                                ctx.turn_wall_started_at, ctx.final_content or "",
                                ctx.tools_used, ctx.stop_reason)
                elif sname == "SAVE":
                    logger.info("[%s] SAVE: session_len=%d, content=%.100r",
                                ctx.turn_wall_started_at,
                                len(ctx.session.get("messages", [])) if ctx.session else 0,
                                ctx.final_content or "")
                elif sname == "RESPOND":
                    logger.info("[%s] RESPOND: outbound.content=%.100r",
                                ctx.turn_wall_started_at,
                                ctx.outbound.content if ctx.outbound else None)
            except Exception as e:
                event, error = "error", str(e)
                logger.error("%s 异常: %s", ctx.state.name, error)
            elapsed_ms = (time.time() - t0) * 1000
            ctx.trace.append(StateTraceEntry(ctx.state, t0, elapsed_ms, event, error))

            next_state = self._TRANSITIONS.get((ctx.state, event))
            ctx.state = next_state if next_state else TurnState.DONE

        if ctx.outbound is None and ctx.final_content:
            ctx.outbound = OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=ctx.final_content,
            )
        return ctx.outbound

    async def _dispatch(self, msg: InboundMessage) -> None:
        """分发消息：获取 session 锁后调用 _process_message，再推送回复。"""
        session_key = msg.session_key
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        logger.info(
            "dispatch: session=%s, content=%.60s", session_key, msg.content,
        )

        # 流式回调：当客户端声明 _wants_stream 时，逐 chunk 通过 bus 推送
        on_stream = on_stream_end = None
        if msg.metadata.get("_wants_stream"):
            async def _on_stream(delta: str) -> None:
                meta = dict(msg.metadata or {})
                meta["_stream_delta"] = True
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=delta, metadata=meta,
                ))

            async def _on_stream_end(*, resuming: bool = False) -> None:
                meta = dict(msg.metadata or {})
                meta["_stream_end"] = True
                meta["_resuming"] = resuming
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="", metadata=meta,
                ))

            on_stream = _on_stream
            on_stream_end = _on_stream_end

        try:
            async with lock:
                outbound = await self._process_message(
                    msg, session_key=session_key,
                    on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if outbound is not None:
                    await self.bus.publish_outbound(outbound)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id, content="",
                    ))
        except asyncio.CancelledError:
            logger.info("会话 %s 的任务被取消", session_key)
            raise
        except Exception as exc:
            logger.exception("处理会话 %s 的消息时发生异常", session_key)
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="抱歉，处理请求时出现异常。",
            ))

    def _register_default_tools(self) -> None:
        """注册内置工具。"""
        self.tools.register(ShellTool())
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        logger.info("已注册 %d 个工具: %s", len(self.tools.tool_names), self.tools.tool_names)

    async def _on_tool_progress(self, msg: str) -> None:
        """工具执行进度回调。"""
        logger.info("工具进度: %s", msg)

    async def _state_restore(self, ctx: TurnContext) -> str:
        session = self._session_store.get(ctx.session_key)
        if session is None:
            self._session_store[ctx.session_key] = []
            session = self._session_store[ctx.session_key]
        msg = ctx.msg
        if msg.media:
            logger.info("处理 media: %d 个附件", len(msg.media))
            # 分离图片和文档：文档抽取文字拼入 content，图片留在 media 列表
            new_content, image_only = extract_documents(msg.content, msg.media)
            # 更新消息内容（含文档提取的文字）和 media（仅图片）
            ctx.msg = msg.__class__(
                channel=msg.channel, sender_id=msg.sender_id, chat_id=msg.chat_id,
                content=new_content, media=image_only,
                timestamp=msg.timestamp, metadata=msg.metadata,
            )
            msg = ctx.msg
            if image_only:
                logger.info("图片附件: %s", image_only)
        ctx.session = {"key": ctx.session_key, "messages": session}
        ctx.history = list(session)
        return "ok"

    async def _state_compact(self, ctx: TurnContext) -> str:
        max_messages = 60
        session = ctx.session
        if session and len(session.get("messages", [])) > max_messages:
            keep = session["messages"][-(max_messages // 2):]
            session["messages"] = keep
            ctx.history = list(keep)
        return "ok"

    async def _state_command(self, ctx: TurnContext) -> str:
        cmd_ctx = CommandContext(
            msg=ctx.msg, session=ctx.session, key=ctx.session_key,
            raw=ctx.msg.content.strip(), loop=self,
        )
        result = await self.commands.dispatch(cmd_ctx)
        if result is not None:
            ctx.outbound = result
            return "shortcut"
        return "dispatch"

    async def _state_build(self, ctx: TurnContext) -> str:
        messages = list(ctx.history)
        content_text = ctx.msg.content
        media = ctx.msg.media

        if media:
            # 多模态消息：content 为数组，包含 text + image_url
            content_blocks: list[dict] = []
            if content_text:
                content_blocks.append({"type": "text", "text": content_text})
            for img_path in media:
                try:
                    p = Path(img_path)
                    with open(p, "rb") as f:
                        img_data = f.read()
                    mime = self._detect_image_mime(img_data[:16])
                    if not mime:
                        mime = "image/png"
                    b64 = base64.b64encode(img_data).decode("ascii")
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                except Exception as e:
                    logger.warning("图片读取失败 %s: %s", img_path, e)
                    if content_text:
                        content_blocks.append({"type": "text", "text": f"[图片加载失败: {img_path}]"})
            messages.append({"role": "user", "content": content_blocks})
        else:
            messages.append({"role": "user", "content": content_text})
        ctx.all_messages = messages
        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        logger.info("RUN: model=%s, messages=%d, tools=%s",
                     self.model, len(ctx.all_messages),
                     self.tools.tool_names if self.tools.tool_names else "无")
        if self.on_llm_start:
            await self.on_llm_start()
        try:
            t0 = time.time()

            # 刷新 runner 的流式回调和 session_key
            self.runner.on_stream = ctx.on_stream
            self.runner.on_stream_end = ctx.on_stream_end
            self.runner._session_key = ctx.session_key

            # 使用 AgentRunner 执行 ReAct 循环
            result = await self.runner.run(
                messages=ctx.all_messages,
                tools=self.tools,
                model=self.model,
            )
            ctx.final_content = result.final_content
            ctx.stop_reason = result.stop_reason
            ctx.tools_used = result.tools_used
            ctx.all_messages = result.all_messages

            elapsed = time.time() - t0
            logger.info("LLM 返回: finish_reason=%s, %.1fs, %d chars, tools=%s",
                        ctx.stop_reason, elapsed, len(ctx.final_content or ""),
                        ctx.tools_used)
        except Exception as e:
            ctx.error = str(e)
            logger.error("LLM 异常: %s", e)
            return "error"
        finally:
            if self.on_llm_end:
                await self.on_llm_end()
            if ctx.on_stream_end:
                await ctx.on_stream_end(resuming=False)
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        session = self._session_store.get(ctx.session_key)
        if session is not None and ctx.all_messages:
            # 保存完整上下文（含 tool_calls、tool_results）
            session.clear()
            session.extend(ctx.all_messages)
        return "ok"

    @staticmethod
    def _detect_image_mime(data: bytes) -> str | None:
        """从文件头部字节检测图片 MIME 类型。"""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if data[:2] in (b"\xff\xd8",):
            return "image/jpeg"
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    async def _state_respond(self, ctx: TurnContext) -> str:
        metadata: dict[str, Any] = {}
        if ctx.on_stream is not None:
            metadata["_streamed"] = True
        ctx.outbound = OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=ctx.final_content or "",
            metadata=metadata,
        )
        return "ok"

    # ── 主循环：1 秒轮询 + dispatch 异步化 ──

    async def run_forever(self) -> None:
        """主循环，每秒轮询一次 inbound 队列。

        收到消息后直接用 create_task 异步 dispatch，
        不阻塞主循环，可以同时处理多条消息。
        符合原版模式：run 是 1s 监听，dispatch 异步处理。
        """
        logger.debug(f"self._mcp_servers {self._mcp_servers},self._mcp_connected {self._mcp_connected}")
        if self._mcp_servers and not self._mcp_connected:
            logger.info("连接 MCP 服务器: %s", list(self._mcp_servers))
            connected = await connect_mcp_servers(self._mcp_servers, self.tools)
            self._mcp_stacks.update(connected)
            self._mcp_connected = bool(self._mcp_stacks)

        logger.info("主循环启动 (1s 轮询)")
        self._running = True
        loop = asyncio.get_running_loop()

        # 注册信号处理器
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: setattr(self, '_running', False))
            except NotImplementedError:
                pass

        try:
            while self._running:
                try:
                    # 1 秒超时轮询 inbound 队列
                    msg = await asyncio.wait_for(
                        self.bus.consume_inbound(),
                        timeout=1.0,
                    )
                    # 异步处理，不阻塞主循环
                    asyncio.create_task(self._dispatch(msg))
                except asyncio.TimeoutError:
                    # 超时是正常的，继续轮询
                    continue
        finally:
            logger.info("主循环结束")
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass

    async def start(self) -> None:
        """在后台启动主循环。"""
        self._task = asyncio.create_task(self.run_forever())
        logger.info("AgentLoop 已启动")

    async def stop(self) -> None:
        """停止主循环。"""
        self._running = False
        # 关闭 MCP 连接
        for name, stack in list(self._mcp_stacks.items()):
            try:
                await stack.aclose()
            except Exception:
                logger.debug("MCP server '%s' 关闭异常", name)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AgentLoop 已停止")

    def add_handler(self, handler):
        self._handlers.append(handler)

    @property
    def is_running(self) -> bool:
        return self._running
