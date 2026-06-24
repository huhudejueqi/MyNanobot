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
from typing import Any, Callable, Coroutine

from nanobot.agent.context import ContextBuilder
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
    turn_wall_started_at: float = field(default_factory=time.time)
    trace: list[StateTraceEntry] = field(default_factory=list)
    error: str | None = None


class AgentLoop:
    """AgentLoop 是 MyNanobot 的核心处理引擎。

    状态机执行树（原版模式）：
                     消息到达 _dispatch()
                           │
                    ┌──────▼──────┐
                    │  RESTORE    │
                    │  "ok"       │
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │  COMPACT    │
                    │  "ok"       │
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │  COMMAND    │
                    │  /     \\    │
                shortcut    dispatch
                    │          │
               ┌────▼───┐ ┌───▼────┐
               │  DONE   │ │  BUILD │
               │         │ │ "ok"   │
               └─────────┘ └───┬────┘
                           ┌───▼────┐
                           │  RUN   │
                           │ "ok"   │
                           └───┬────┘
                        ┌──────▼──────┐
                        │    SAVE     │
                        │   "ok"     │
                        └──────┬──────┘
                        ┌──────▼──────┐
                        │   RESPOND   │
                        │   "ok"     │
                        └──────┬──────┘
                        ┌──────▼──────┐
                        │    DONE     │◄── 任意 "error" 也跳到这里
                        └─────────────┘
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
        on_llm_start: Callable[[], Coroutine] | None = None,
        on_llm_end: Callable[[], Coroutine] | None = None,
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
        self._pending_queues: dict[str, asyncio.Queue] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

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
        return cls(
            bus=bus, provider=snapshot.provider,
            workspace=workspace, model=snapshot.model,
            context_window_tokens=snapshot.context_window_tokens,
        )

    # ── 状态机驱动（异步处理，不阻塞主循环） ──

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
    ) -> OutboundMessage | None:
        """状态机驱动：RESTORE → COMPACT → ... → DONE，返回出站消息。"""
        key = session_key or msg.session_key
        ctx = TurnContext(
            msg=msg, session_key=key, state=TurnState.RESTORE,
            turn_wall_started_at=time.time(),
        )

        while ctx.state is not TurnState.DONE:
            handler = getattr(self, f"_state_{ctx.state.name.lower()}", None)
            if handler is None:
                raise RuntimeError(f"缺少 {ctx.state} 状态对应的处理函数")

            t0 = time.time()
            event, error = "ok", None
            try:
                event = await handler(ctx)
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

        try:
            async with lock:
                outbound = await self._process_message(msg, session_key=session_key)
                if outbound and outbound.content:
                    await self.bus.publish_outbound(outbound)
        except asyncio.CancelledError:
            logger.info("会话 %s 的任务被取消", session_key)
            raise
        except Exception as exc:
            logger.exception("处理会话 %s 的消息时发生异常", session_key)
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="抱歉，处理请求时出现异常。",
            ))

    async def _state_restore(self, ctx: TurnContext) -> str:
        session = self._session_store.get(ctx.session_key)
        if session is None:
            self._session_store[ctx.session_key] = []
            session = self._session_store[ctx.session_key]
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
        content = ctx.msg.content.strip()
        for handler in self._handlers:
            result = handler(ctx.msg)
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None:
                ctx.final_content = result
                return "shortcut"
        if content in ("/ping", "/status"):
            ctx.final_content = "pong"
            return "shortcut"
        if content == "/time":
            ctx.final_content = f"当前时间: {datetime.now()}"
            return "shortcut"
        if content == "/version":
            ctx.final_content = "MyNanobot v0.1.0"
            return "shortcut"
        return "dispatch"

    async def _state_build(self, ctx: TurnContext) -> str:
        messages = list(ctx.history)
        messages.append({"role": "user", "content": ctx.msg.content})
        ctx.all_messages = messages
        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        logger.info("RUN: model=%s, messages=%d", self.model, len(ctx.all_messages))
        if self.on_llm_start:
            await self.on_llm_start()
        try:
            t0 = time.time()
            response = await self.provider.chat(
                messages=ctx.all_messages, model=self.model,
            )
            elapsed = time.time() - t0
            ctx.final_content = response.content or ""
            ctx.stop_reason = response.finish_reason
            logger.info("LLM 返回: finish_reason=%s, %.1fs, %d chars",
                        response.finish_reason, elapsed, len(ctx.final_content))
            if response.finish_reason == "error":
                err = response.usage.get("error", "unknown")
                ctx.final_content = f"[LLM 调用失败: {err}]"
                return "ok"
            if response.tool_calls:
                for tc in response.tool_calls:
                    ctx.tools_used.append(tc.name)
                    ctx.all_messages.append({
                        "role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": tc.id, "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments if isinstance(tc.arguments, str)
                                else json.dumps(tc.arguments),
                            },
                        }],
                    })
                    ctx.all_messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": f"[工具 {tc.name} 已执行，结果待实现]",
                    })
        except Exception as e:
            ctx.error = str(e)
            logger.error("LLM 异常: %s", e)
            return "error"
        finally:
            if self.on_llm_end:
                await self.on_llm_end()
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        session = self._session_store.get(ctx.session_key)
        if session is not None and ctx.final_content:
            session.append({"role": "user", "content": ctx.msg.content})
            session.append({"role": "assistant", "content": ctx.final_content})
        return "ok"

    async def _state_respond(self, ctx: TurnContext) -> str:
        ctx.outbound = OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=ctx.final_content or "",
        )
        return "ok"

    # ── 主循环：1 秒轮询 + dispatch 异步化 ──

    async def run_forever(self) -> None:
        """主循环，每秒轮询一次 inbound 队列。

        收到消息后直接用 create_task 异步 dispatch，
        不阻塞主循环，可以同时处理多条消息。
        符合原版模式：run 是 1s 监听，dispatch 异步处理。
        """
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
