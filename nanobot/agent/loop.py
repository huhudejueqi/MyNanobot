"""Agent 核心处理引擎：事件驱动的状态机。"""

from __future__ import annotations
import os
import asyncio
import dataclasses
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

from nanobot.agent import context as agent_context
from nanobot.agent import model_presets as preset_helpers
from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.context import ContextBuilder
from nanobot.agent.cron_turns import CronTurnCoordinator
from nanobot.agent.hook import AgentHook, CompositeHook
from nanobot.agent.memory import Consolidator
from nanobot.agent.progress_hook import AgentProgressHook
from nanobot.command.router import CommandRouter, CommandContext
from nanobot.command.builtin import register_builtin_commands
from nanobot.config.schema import AgentDefaults, ModelPresetConfig
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.self import MyTool
from nanobot.agent.tools.mcp import connect_mcp_servers
from nanobot.agent.tools.shell import ShellTool
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool


from nanobot.utils.document import extract_documents, is_image_file,reference_non_image_attachments
from nanobot.config.loader import Config
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import build_bus_progress_callback
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import (
    RuntimeEventBus,
    RuntimeEventPublisher,
    ensure_runtime_event_publisher,
)
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import ProviderSnapshot, build_provider_snapshot
from nanobot.security.workspace_access import (
    WorkspaceScopeResolver,
    bind_workspace_scope,
    reset_workspace_scope,
)
from nanobot.session import turn_continuation
from nanobot.session.goal_state import (
    goal_state_runtime_lines,
    runner_wall_llm_timeout_s,
    sustained_goal_active,
)
from nanobot.session.keys import UNIFIED_SESSION_KEY, session_key_for_channel
from nanobot.session.manager import Session, SessionManager

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
    turn_id: str
    session: Session | None = None

    history: list[dict[str, Any]] = field(default_factory=list)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)

    final_content: str | None = None
    tools_used: list[str] = field(default_factory=list)
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    had_injections: bool = False

    user_persisted_early: bool = False
    save_skip: int = 0

    outbound: OutboundMessage | None = None
    suppress_response: bool = False

    on_progress: Callable[..., Awaitable[None]] | None = None
    on_stream: Callable[[str], Awaitable[None]] | None = None
    on_stream_end: Callable[..., Awaitable[None]] | None = None
    on_retry_wait: Callable[[str], Awaitable[None]] | None = None

    pending_queue: asyncio.Queue | None = None
    pending_summary: str | None = None

    ephemeral: bool = False #设为 True 时，这一轮 agent 运行是"用完即弃"的
    # ephemeral=False（默认）       ephemeral=True
    # 保存会话到磁盘	            不持久化任何东西
    # 注入历史记忆到上下文	         跳过历史记忆注入
    # 运行 extra hooks（通知等）	跳过 extra hooks
    # 正常更新记忆	                跳过记忆更新
    run_extra_hooks_for_ephemeral: bool = False
    hooks: list[AgentHook] = field(default_factory=list)
    tools: ToolRegistry | None = None

    turn_wall_started_at: float = field(default_factory=time.time)
    visible_run_started_at: float | None = None
    turn_latency_ms: int | None = None

    trace: list[StateTraceEntry] = field(default_factory=list)

    error: str | None = None

    # def __repr__(self) -> str:
    #     lines = [f"TurnContext(session_key={self.session_key}, state={self.state.name}, turn_id={self.turn_id})"]
    #     lines.append(f"  final_content: {(self.final_content[:120] + '...') if self.final_content and len(self.final_content) > 120 else self.final_content}")
    #     lines.append(f"  stop_reason={self.stop_reason}, tools_used={self.tools_used}, ephemeral={self.ephemeral}")
    #     if self.trace:
    #         lines.append("  trace:")
    #         for t in self.trace:
    #             lines.append(f"    {t.state.name:12s} {t.duration_ms:>8.2f}ms  {t.event}")
    #     return "\n".join(lines)


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
               │  SAVE                │  │  DONE   │
               │  ├─ 追加 user msg    │  │ (error)  │
               │  └─ 追加 assistant   │  └──────────┘
               │  msg 到 session      │
               │  "ok"                │
               └──┬───────────────────┘
                  │
               ┌──▼──────────────────┐  ┌──────────┐
               │  RESPOND            │  │  DONE    │
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
    @property
    def current_iteration(self) -> int:
        """返回当前轮次的迭代次数。"""
        return self._current_iteration
    
    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    # 事件驱动型状态转换表
    # 处理器会返回事件字符串；驱动程序通过本表查询下一状态
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
        max_iterations: int | None = None,
        max_concurrent_subagents: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        tool_hint_max_length: int | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        consolidation_ratio: float = 0.5,
        max_messages: int = 120,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
        image_generation_provider_config: ProviderConfig | None = None,
        image_generation_provider_configs: dict[str, ProviderConfig] | None = None,
        provider_snapshot_loader: Callable[..., ProviderSnapshot] | None = None,
        provider_signature: tuple[object, ...] | None = None,
        model_presets: dict[str, ModelPresetConfig] | None = None,
        model_preset: str | None = None,
        preset_snapshot_loader: preset_helpers.PresetSnapshotLoader | None = None,
        runtime_events: RuntimeEventBus | None = None,
        runtime_model_publisher: Callable[[str, str | None], None] | None = None,
    ):
        from nanobot.config.schema import ToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self.runtime_events = runtime_events or RuntimeEventBus()
        self.runtime_event_publisher = RuntimeEventPublisher(self.runtime_events)
        self.channels_config = channels_config
        self.provider = provider
        self._provider_snapshot_loader = provider_snapshot_loader
        self._preset_snapshot_loader = preset_snapshot_loader
        self._runtime_model_publisher = runtime_model_publisher
        self._provider_signature = provider_signature
        self._default_selection_signature = preset_helpers.default_selection_signature(provider_signature)
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.tool_hint_max_length = (
            tool_hint_max_length if tool_hint_max_length is not None
            else defaults.tool_hint_max_length
        )
        self.tools_config = _tc
        self.web_config = _tc.web
        self.exec_config = _tc.exec
        self._image_generation_provider_configs = dict(image_generation_provider_configs or {})
        if (
            image_generation_provider_config is not None
            and "openrouter" not in self._image_generation_provider_configs
        ):
            self._image_generation_provider_configs["openrouter"] = image_generation_provider_config
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.workspace_scopes = WorkspaceScopeResolver(
            default_workspace=workspace,
            default_restrict_to_workspace=restrict_to_workspace,
        )
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        # One file-read/write tracker per logical session. The tool registry is
        # shared by this loop, so tools resolve the active state via contextvars.
        self._file_state_store = FileStateStore()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            tools_config=_tc,
            max_tool_result_chars=self.max_tool_result_chars,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
            max_iterations=self.max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            llm_wall_timeout_for_session=lambda sk: runner_wall_llm_timeout_s(self.sessions, sk),
        )
        self._unified_session = unified_session
        self._max_messages = max_messages if max_messages > 0 else 120
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        self._cron_turns = CronTurnCoordinator(
            publish_inbound=self.bus.publish_inbound,
            dispatch=self._dispatch,
            is_running=lambda: self._running,
        )
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
            consolidation_ratio=consolidation_ratio,
            unified_session=unified_session,
        )
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.model_presets: dict[str, ModelPresetConfig] = model_presets or {}
        self._active_preset: str | None = None
        if model_preset:
            self.set_model_preset(model_preset, publish_update=False)
        self._register_default_tools()
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)
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
        **extra: Any,
    ) -> AgentLoop:
        """根据配置创建一个 AgentLoop 实例，并传入通用参数集。
        额外的关键字实参会透传给 AgentLoop 的构造函数 __init__，
        调用方可以借此覆盖或扩展从配置读取的标准参数（例如 cron_service、session_manager）。
        """
        from nanobot.providers.factory import make_provider
        if bus is None:
            bus = MessageBus()
        defaults = config.agents.defaults
        provider = extra.pop("provider", None) or make_provider(config)
        resolved = config.resolve_preset()
        model = extra.pop("model", None) or resolved.model
        context_window_tokens = extra.pop("context_window_tokens", None) or resolved.context_window_tokens
        provider_snapshot_loader = extra.pop("provider_snapshot_loader", None)
        preset_snapshot_loader = extra.pop("preset_snapshot_loader", None) or preset_helpers.make_preset_snapshot_loader(
            config,
            provider_snapshot_loader,
        )
        return cls(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=model,
            context_window_tokens=context_window_tokens,
            max_iterations=defaults.max_tool_iterations,
            timezone=defaults.timezone,
            disabled_skills=defaults.disabled_skills,
            mcp_servers=config.tools.mcp_servers,
            consolidation_ratio=defaults.consolidation_ratio,
            unified_session=defaults.unified_session,
            **extra,
        )

    # ── 状态机驱动（异步处理，不阻塞主循环） ──

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        tools: ToolRegistry | None = None,
    ) -> OutboundMessage | None:
        """状态机驱动：RESTORE → COMPACT → ... → DONE，返回出站消息。"""
        self._refresh_provider_snapshot()
        logger.info("_process_message")
        if msg.channel == "system":
            return await self._process_system_message(
                msg,
                session_key=session_key,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                pending_queue=pending_queue,
            )
        key = session_key or msg.session_key
        t0 = time.time()
        ctx = TurnContext(
            msg=msg,
            session=None,
            session_key=key,
            state=TurnState.RESTORE,
            turn_id=f"{key}:{time.time_ns()}",
            turn_wall_started_at=t0,
            visible_run_started_at=turn_continuation.internal_continuation_run_started_at(
                msg.metadata,
            ),
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            pending_queue=pending_queue,
            ephemeral=ephemeral,
            run_extra_hooks_for_ephemeral=run_extra_hooks_for_ephemeral,
            hooks=list(hooks or []),
            tools=tools,
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
                                len(ctx.session.messages) if ctx.session else 0,
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
        """分发消息：获取 session 锁后调用 _process_message，再推送回复。

        执行流程：
        1. 检查 msg.metadata["_wants_stream"]，设置 on_stream / on_stream_end 回调
        2. 获取 session 锁（_session_locks[session_key]），防止并发写入同一会话
        3. 调用 _process_message 获取 OutboundMessage
        4. 通过 bus.publish_outbound() 推给前端

        流式回调：
        - _on_stream(delta):    每个文本 delta 包装成 OutboundMessage（_stream_delta=True）推送
        - _on_stream_end(...):  结束标记包装成 OutboundMessage（_stream_end=True）推送
        """
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
        from nanobot.agent.tools.context import ToolContext
        from nanobot.agent.tools.loader import ToolLoader
        ctx = ToolContext(
            config=self.tools_config,
            workspace=str(self.workspace),
            bus=self.bus,
            subagent_manager=self.subagents,
            cron_service=self.cron_service,
            sessions=self.sessions,
            provider_snapshot_loader=self._provider_snapshot_loader,
            image_generation_provider_configs=self._image_generation_provider_configs,
            timezone=self.context.timezone or "UTC",
            workspace_sandbox=self.workspace_scopes.sandbox_status,
            runtime_events=self.runtime_events,
        )
        loader = ToolLoader()
        registered = loader.load(ctx, self.tools)
        """MyTool 需要运行时状态引用 —— 手动注册。"""
        if self.tools_config.my.enable:
            self.tools.register(
                MyTool(runtime_state=self, modify_allowed=self.tools_config.my.allow_set)
            )
            registered.append("my")

        # self.tools.register(ShellTool())
        # self.tools.register(ReadFileTool())
        # self.tools.register(WriteFileTool())
        # self.tools.register(WebSearchTool())
        logger.info("已注册 %d 个工具: %s", len(self.tools.tool_names), self.tools.tool_names)

    async def _on_tool_progress(self, msg: str) -> None:
        """工具执行进度回调。"""
        logger.info("工具进度: %s", msg)

    def _runtime_events(self) -> RuntimeEventPublisher:
        """获取运行时事件发布器，用于推送内部状态变更事件。

        返回 RuntimeEventPublisher 实例，外部模块可监听：
        - session_turn_started:  新轮次开始
        - run_status_changed:    运行状态变更（running/done/error）
        
        与 MessageBus 的分工：
        - MessageBus:     负责 agent ↔ 外部队列的消息传递（InboundMessage / OutboundMessage）
        - RuntimeEventBus: 负责内部模块间的事件通知（如 UI 更新、日志跟踪）
        """
        return ensure_runtime_event_publisher(self)

    async def _state_restore(self, ctx: TurnContext) -> str:
        msg = ctx.msg
        if msg.media:
            # 分离content和media文本
            new_content, image_only = self._prepare_message_media(msg.content, msg.media)
            ctx.msg = dataclasses.replace(msg, content=new_content, media=image_only)
            msg = ctx.msg
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from %s:%s: %s", msg.channel, msg.sender_id, preview)
        # session = self._session_store.get(ctx.session_key)
        # if session is None:
        #     self._session_store[ctx.session_key] = []
        #     session = self._session_store[ctx.session_key]
        # msg = ctx.msg
        # if msg.media:
        #     logger.info("处理 media: %d 个附件", len(msg.media))
        #     # 分离图片和文档：文档抽取文字拼入 content，图片留在 media 列表
        #     new_content, image_only = extract_documents(msg.content, msg.media)
        #     # 更新消息内容（含文档提取的文字）和 media（仅图片）
        #     ctx.msg = msg.__class__(
        #         channel=msg.channel, sender_id=msg.sender_id, chat_id=msg.chat_id,
        #         content=new_content, media=image_only,
        #         timestamp=msg.timestamp, metadata=msg.metadata,
        #     )
        #     msg = ctx.msg
        #     if image_only:
        #         logger.info("图片附件: %s", image_only)
        if ctx.session is None:
            ctx.session = self.sessions.get_or_create(ctx.session_key)
        await self._runtime_events().session_turn_started(msg, ctx.session_key)
        self.workspace_scopes.persist_message_scope(ctx.session,msg)
        if self._restore_runtime_checkpoint(ctx.session):
            self.sessions.save(ctx.session)
        if self._restore_pending_user_turn(ctx.session):
            self.sessions.save(ctx.session)
        # ctx.history = list(ctx.session.messages)
        return "ok"
    
    def _prepare_message_media(self, content: str, media: list[str]) -> tuple[str, list[str]]:
        """处理消息中的媒体附件。

        根据频道配置决定处理方式：
        extract_document_text=True  -> 提取文档文字，media 只保留图片
        extract_document_text=False -> 图片留在 media，非图片转为引用文字拼入 content

        Returns:
            (处理后的 content, 过滤后的图片列表)
        """
        if self._should_extract_document_text():
            return extract_documents(content, media)
        return reference_non_image_attachments(content, media)

    def _should_extract_document_text(self) -> bool:
        if self.channels_config is None:
            return True
        return self.channels_config.extract_document_text
    async def _state_compact(self, ctx: TurnContext) -> str:
        """COMPACT 阶段：压缩/整理会话上下文。

        委托 AutoCompact.prepare_session() 处理：
        1. 检查 session 是否需要自动压缩（超时/标记归档）
        2. 如果需要压缩：重新加载 session，编译摘要并入会话头部
        3. 从摘要缓存中读取之前生成的摘要文本
        4. 将摘要文本设为 pending_summary，供 BUILD 阶段注入上下文

        输出：ctx.session（可能更新）、ctx.pending_summary（摘要文本供注入）
        """
        ctx.session, pending = self.auto_compact.prepare_session(ctx.session, ctx.session_key)
        ctx.pending_summary = pending
        return "ok"
       
    def _set_tool_context(
        self, channel: str, chat_id: str,
        message_id: str | None = None, metadata: dict | None = None,
        session_key: str | None = None,
    ) -> None:
        """为所有需要路由信息的工具更新上下文。"""
        from nanobot.agent.tools.context import ContextAware

        effective_key = session_key or session_key_for_channel(
            channel,
            chat_id,
            unified_session=self._unified_session,
        )
        request_ctx = RequestContext(
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=effective_key,
            metadata=dict(metadata or {}),
        )

        for name in self.tools.tool_names:
            tool = self.tools.get(name)
            if tool and isinstance(tool, ContextAware):
                tool.set_context(request_ctx)

    async def _state_command(self, ctx: TurnContext) -> str:
        """COMMAND 阶段：检查是否是内置命令（路由短路的命令）。

        通过 CommandRouter.dispatch() 尝试匹配内置命令（如 /help、/clear 等）：
        - 匹配成功 → 填充 ctx.outbound（命令的回复消息），返回 "shortcut" 直接跳到 RESPOND
        - 匹配失败 → 返回 "dispatch"，进入 BUILD 阶段走正常 LLM 对话

        CommandContext 包含：消息、session、key、原始文本、AgentLoop 引用
        """
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
        
        if not ctx.ephemeral:
            await self.consolidator.maybe_consolidate_by_tokens(
                ctx.session,
                replay_max_messages=self._max_messages,
            )
        self._set_tool_context(
            ctx.msg.channel,
            ctx.msg.chat_id,
            ctx.msg.metadata.get("message_id"),
            ctx.msg.metadata,
            session_key=ctx.session_key,
        )
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()
                
        logger.info("_state_build")
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
        if ctx.visible_run_started_at is None:
            ctx.visible_run_started_at = time.time()
        await self._runtime_events().run_status_changed(
            ctx.msg,
            ctx.session_key,
            "running",
            started_at=ctx.visible_run_started_at,
        )
        try:
            result = await self._run_agent_loop(
                    ctx.all_messages,
                    on_progress=ctx.on_progress,
                    on_stream=ctx.on_stream,
                    on_stream_end=ctx.on_stream_end,
                    on_retry_wait=ctx.on_retry_wait,
                    session=ctx.session,
                    channel=ctx.msg.channel,
                    chat_id=ctx.msg.chat_id,
                    message_id=ctx.msg.metadata.get("message_id"),
                    metadata=ctx.msg.metadata,
                    session_key=ctx.session_key,
                    pending_queue=ctx.pending_queue,
                    ephemeral=ctx.ephemeral,
                    run_extra_hooks_for_ephemeral=ctx.run_extra_hooks_for_ephemeral,
                    hooks=ctx.hooks,
                    tools=ctx.tools,
                )
            final_content, tools_used, all_msgs, stop_reason, had_injections = result
            ctx.final_content = final_content
            ctx.tools_used = tools_used
            ctx.all_messages = all_msgs
            ctx.stop_reason = stop_reason
            ctx.had_injections = had_injections
            # ctx.final_content = result.final_content
            # ctx.stop_reason = result.stop_reason
            # ctx.tools_used = result.tools_used
            # ctx.all_messages = result.all_messages
            await turn_continuation.maybe_continue_turn(ctx)
            elapsed = time.time() - ctx.visible_run_started_at
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
        logger.info("_state_save")
        turn_continuation.prepare_save_boundary(ctx)
        if ctx.session is not None and ctx.all_messages:
            ctx.session.messages = list(ctx.all_messages)
            self.sessions.save(ctx.session)
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

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        pending_queue: asyncio.Queue | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        tools: ToolRegistry | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """
        执行智能体迭代主循环。

        *on_stream*：流式输出过程中，每产生一段增量文本都会触发该回调。
        *on_stream_end(resuming)*：单次流式会话结束时触发。
        参数 `resuming=True` 代表后续还有工具调用（加载动画需重新启动）；
        参数 `resuming=False` 代表本次为最终回复，流程结束。

        返回元组：(完整输出文本, 已调用工具列表, 完整消息记录, 停止原因, 是否存在注入内容)
        """
        # self._sync_subagent_runtime_limits()
        loop_hook = AgentProgressHook(
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            metadata=metadata,
            session_key=session_key,
            tool_hint_max_length=self.tool_hint_max_length,
            set_tool_context=self._set_tool_context,
            on_iteration=lambda iteration: setattr(self, "_current_iteration", iteration),
        )
        # run_hooks = []
        # for h in self._extra_hooks:
        #     run_hooks.append(h)
        # for h in (hooks if hooks is not None else []):
        #     run_hooks.append(h)
        run_hooks = [*self._extra_hooks, *(hooks or [])]
        hook: AgentHook = loop_hook
        if run_hooks and (not ephemeral or run_extra_hooks_for_ephemeral):
            # 正常对话（非 ephemeral）
            # 临时对话但调用方明确要求跑 hook（比如某些需要日志追踪的内部操作）
            hook = CompositeHook([loop_hook, *run_hooks])# CompositeHook 是一个广播器，把多个 hook 串在一起，每个事件同时触发所有 hook。loop_hook 是主进度钩子（负责推流式内容给前端），*run_hooks 是额外的钩子（比如 _DebugHook 打日志、自定义插件钩子等）。
        async def _checkpoint(payload: dict[str, Any]) -> None:
            """保存本轮运行进度到 session.metadata，用于崩溃恢复。"""
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """排干待处理队列，收集后续子 agent 或用户发来的消息。

            用于 ReAct 循环中每轮迭代开始前，将 pending_queue 中的积压消息
            取出并注入到会话上下文中。

            工作流程：
            1. 先尽量从 pending_queue 非阻塞取消息（get_nowait），最多 limit 条
            2. 如果取空了，但还有子 agent（subagent）在运行：
               - 阻塞等待最多 300 秒，等第一个子 agent 结果
               - 继续取剩余消息
               （这是为了保持 runner 循环存活，让后续子 agent 完成结果
                按顺序注入，而不是被分散到独立的 dispatch 中）
            3. 每条消息通过 _to_user_message 转为 OpenAI 格式的 user message

            子 agent 场景：主 agent 启动了一组子 agent 并行执行，
            子 agent 完成后往 pending_queue 发 InboundMessage，
            _drain_pending 在下一轮迭代前把这些结果注入上下文。

            Returns:
                list[dict]: 格式为 {"role": "user", "content": ...} 的注入消息列表
            """
            if pending_queue is None:
                return []  # 没有队列可排干

            def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                """将 InboundMessage 转为 LLM 使用的 user message 格式。"""
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    # 处理附件：文档提取文字，图片分离
                    content, media = self._prepare_message_media(content, media)
                    media = media or None
                # 构建最终 user content（可能含图片/多模态）
                user_content = self.context._build_user_content(content, media)
                return {"role": "user", "content": user_content}

            items: list[dict[str, Any]] = []
            # 第一步：非阻塞取出所有积压消息
            while len(items) < limit:
                try:
                    items.append(_to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break  # 队列取空了，跳出

            # 第二步：没有排到消息，但子 agent 还在运行
            # 阻塞等第一个完成结果，避免 runner 提前退出
            # 这样后续子 agent 完成的结果可以顺序注入，而不是各自走独立 dispatch
            if (not items
                    and session is not None
                    and self.subagents.get_running_count_by_session(session.key) > 0):
                try:
                    msg = await asyncio.wait_for(pending_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for sub-agent completion in session {}",
                        session.key,
                    )
                    return items  # 超时了，返回已有的 items（此时为空列表）
                items.append(_to_user_message(msg))
                # 拿到第一个后，再尝试拿剩余积压的
                while len(items) < limit:
                    try:
                        items.append(_to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items


        # ── 运行时绑定 ──
        # 在调用 runner 前，将当前请求的上下文信息绑定到全局 ContextVar，
        # 供工具执行时的路径检查、权限判断等场景使用。
        active_session_key = session.key if session else session_key  # 当前活跃 session key
        effective_scope = self.workspace_scopes.for_turn(             # 计算工作区范围（项目路径+访问权限）
            # 解析策略（workspace_access.py）：
            # 1. 非 websocket 频道（如 cli）→ 直接返回默认 scope（full 访问整个 workspace）
            # 2. websocket 频道（WebUI）→ 按以下优先级取 workpace_scope：
            #    a. 消息元数据 message_metadata["workspace_scope"]（UI 发来的当前项目路径）
            #    b. 无则回退到 session.metadata["workspace_scope"]（上一轮保留的）
            #    c. 两边都没有就返回默认 scope
            # 3. 每个 WorkspaceScope 包含：
            #    - project_path:      限制文件操作只能在此目录内
            #    - access_mode:       "restricted" 或 "full"
            #    - restrict_to_workspace:  是否强制限制
            #    - sandbox_status:    系统级沙箱状态（基于环境的 NANOBOT_WORKSPACE_* 变量）
            #    - source_channel:    来源频道标识
            channel=channel,
            message_metadata=metadata,
            session_metadata=session.metadata if session is not None else None,
        )
        request_ctx = RequestContext(                                  # 请求上下文（路由/权限用）
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=active_session_key,
            metadata=dict(metadata or {}),
        )
        # 将当前 session 的文件状态快照绑定到 ContextVar。
        # FileStateStore 追踪 session 内文件创建/修改的路径和哈希，
        # 工具（如 read_file）用 FileStates.has_changed() 检测文件变更，
        # 检测到变更时自动重新读取，避免 LLM 使用过期内容。
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        # 将请求上下文（频道、会话 key、元数据）绑定到 ContextVar。
        # 供 ContextAware 工具（如 MessageTool）在 execute() 中读取，
        # 知道当前应该往哪个频道/会话发消息。
        request_token = bind_request_context(request_ctx)
        # effective_scope 绑定到 ContextVar _CURRENT_WORKSPACE_SCOPE。
        # 工具执行时通过 current_tool_workspace() 读取：
        #   - project_path:               限制文件操作只能在此目录内（如 /projects/foo）
        #   - restrict_to_workspace:      是否强制限制（True → 越界操作会被拒绝）
        # 按以下链路生效：
        #   bind_workspace_scope()
        #     → ContextVar _CURRENT_WORKSPACE_SCOPE
        #       → current_tool_workspace()  [工具内部调用]
        #         → is_path_within() 检查路径合法性  [workspace_policy.py]
        workspace_token = bind_workspace_scope(effective_scope)

        # ── 长期目标续写消息 ──
        # Compute lazily because long_task may create goal metadata during this run.
        def _goal_continue() -> str | None:
            """构造"长期目标继续执行"提示，在 ReAct 超限后注入给 LLM。"""
            _goal_lines = goal_state_runtime_lines(session.metadata if session is not None else None)
            if not _goal_lines:
                return None
            return (
                "You have an active sustained goal:\n\n"
                + "\n".join(_goal_lines)
                + "\n\nPlease continue working toward the objective using your tools, "
                "or call complete_goal if the work is truly finished."
            )

        session_metadata = session.metadata if session is not None else None
        try:
            # ── 调用 AgentRunner.run() 启动 ReAct 循环 ──
            result = await self.runner.run(AgentRunSpec(
                initial_messages=initial_messages,
                tools=tools or self.tools,
                model=self.model,
                max_iterations=self.max_iterations,               # ReAct 最大轮次
                max_tool_result_chars=self.max_tool_result_chars,  # 工具结果截断长度
                hook=hook,                                         # 钩子链（进度推送 + 调试日志 + 自定义）
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True,                              # 允许工具并行执行
                workspace=effective_scope.project_path,             # 工具执行的 cwd（项目隔离）
                session_key=session.key if session else None,       # session 标识，用于 checkpoint
                context_window_tokens=self.context_window_tokens,   # 上下文窗口大小
                context_block_limit=self.context_block_limit,       # 上下文块数上限（更精细的裁剪）
                provider_retry_mode=self.provider_retry_mode,       # LLM 重试策略
                progress_callback=on_progress,                       # 进度通知回调
                stream_progress_deltas=on_stream is not None,        # 非流式时是否用进度增量代替
                retry_wait_callback=on_retry_wait,                   # 重试间隔回调
                checkpoint_callback=_checkpoint,                     # 每步保存进度到 session.metadata
                injection_callback=_drain_pending,                   # 排干待处理消息队列
                # 长期目标期间不设外层超时；流式 provider 由 STREAM_IDLE_TIMEOUT_S 兜底
                llm_timeout_s=runner_wall_llm_timeout_s(
                    self.sessions,
                    session.key if session is not None else session_key,
                    metadata=session_metadata,
                    message_metadata=metadata,
                ),
                goal_active_predicate=lambda: sustained_goal_active(session.metadata) if session is not None else False,
                goal_continue_message=_goal_continue,
                finalize_on_max_iterations=turn_continuation.should_finalize_on_max_iterations(
                    pending_queue_available=pending_queue is not None and session is not None,
                    session_metadata=session_metadata,
                    message_metadata=metadata,
                ),
            ))
        finally:
            # ── 清理运行时绑定 ──
            # ContextVar 的 .set() 返回 token，reset 恢复旧值，防止影响后续请求
            reset_workspace_scope(workspace_token)
            reset_request_context(request_token)
            reset_file_states(file_state_token)

        # ── 结果后处理 ──
        self._last_usage = result.usage  # 保存 token 用量供外面查
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            should_stream = turn_continuation.should_stream_budget_response(
                stop_reason=result.stop_reason,
                pending_queue_available=pending_queue is not None and session is not None,
                session_metadata=session_metadata,
                message_metadata=metadata,
            )
            # 流式频道（如飞书）超限后推送最终内容刷新卡片，避免前端一直显示"加载中"
            if on_stream and on_stream_end and should_stream:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections




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

    async def _process_system_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """Process a system inbound message (e.g. subagent announce)."""
        channel, chat_id = (
            msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
        )
        logger.info("Processing system message from {}", msg.sender_id)
        key = msg.session_key_override or f"{channel}:{chat_id}"
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)
        if pending:
            logger.info("Memory compact triggered for session {}", key)

        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            replay_max_messages=self._max_messages,
        )
        is_subagent = msg.sender_id == "subagent"
        if is_subagent and self._persist_subagent_followup(session, msg):
            logger.debug("Subagent result persisted for session {}", key)
            self.sessions.save(session)
        self._set_tool_context(
            channel, chat_id, msg.metadata.get("message_id"),
            msg.metadata, session_key=key,
        )
        current_role = "assistant" if is_subagent else "user"
        _hist_kwargs: dict[str, Any] = {
            "max_messages": self._max_messages,
            "max_tokens": self._replay_token_budget(),
            "include_timestamps": True,
            "extend_to_user": is_subagent,
        }
        history = session.get_history(**_hist_kwargs)
        workspace_scope = self.workspace_scopes.for_message(msg, session.metadata)

        messages = self.context.build_messages(
            history=history,
            current_message="" if is_subagent else msg.content,
            channel=channel,
            chat_id=chat_id,
            current_role=current_role,
            sender_id=msg.sender_id,
            session_summary=pending,
            session_metadata=session.metadata,
            workspace=workspace_scope.project_path,
            runtime_state=self,
            inbound_message=msg,
            skip_runtime_lines=is_subagent,
            session_key=key,
            unified_session=self._unified_session,
        )
        t_wall = time.time()
        final_content, _, all_msgs, stop_reason, _ = await self._run_agent_loop(
            messages, session=session, channel=channel, chat_id=chat_id,
            message_id=msg.metadata.get("message_id"),
            metadata=msg.metadata,
            session_key=key,
            pending_queue=pending_queue,
        )
        wall_done = time.time()
        latency_ms = max(0, int((wall_done - t_wall) * 1000))
        self._save_turn(session, all_msgs, 1 + len(history), turn_latency_ms=latency_ms)
        self._runtime_events().record_turn_latency(key, latency_ms)
        session.enforce_file_cap(
            on_archive=partial(self.context.memory.raw_archive, session_key=key)
        )
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(
            self.consolidator.maybe_consolidate_by_tokens(
                session,
                replay_max_messages=self._max_messages,
            )
        )
        content = final_content or "Background task completed."
        outbound_metadata: dict[str, Any] = {}
        if channel == "slack" and key.startswith("slack:") and key.count(":") >= 2:
            outbound_metadata["slack"] = {"thread_ts": key.split(":", 2)[2]}
        if origin_message_id := msg.metadata.get("origin_message_id"):
            outbound_metadata["origin_message_id"] = origin_message_id
        return OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            metadata=outbound_metadata,
        )
    
    def add_handler(self, handler):
        self._handlers.append(handler)

    @property
    def is_running(self) -> bool:
        return self._running



    # @staticmethod
    def _clear_pending_user_turn(self, session: Session) -> None:
        """清除 session 中"待处理的用户 turn"标记。"""
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        """清除 session 中的运行时 checkpoint。"""
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)
    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        """生成消息的唯一标识元组，用于 checkpoint 恢复时对比消息是否已存在。"""
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )
    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """从 session.metadata 恢复中断前的运行 checkpoint。

        如果上一轮对话因进程重启/崩溃中断，checkpoint 里存有：
        - assistant_message：已输出的 assistant 回复（含 tool_calls）
        - completed_tool_results：已执行完的 tool 结果
        - pending_tool_calls：未执行完的 tool 调用（标注中断错误）

        恢复方式：把这些消息重新写回 session.messages，
        让下一轮对话能接着中断时的上下文继续。

        Returns:
            True 表示恢复成功，False 表示没有 checkpoint 可恢复
        """
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")# 已输出的 assistant 回复（含 tool_calls）
        completed_tool_results = checkpoint.get("completed_tool_results") or []# 已执行完的工具结果
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []# 没执行完的工具（标记中断）

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """恢复只存了用户消息就崩溃的 turn。

        如果上一轮只来得及把用户消息写进 session 就崩溃了（还没等到 LLM 回复），
        给这条用户消息补一条 "Task interrupted" 的 assistant 回复，
        避免下一次启动时只有用户消息没有回复，破坏消息顺序。

        Returns:
            True 表示补了中断回复，False 表示没有需要恢复的 turn
        """
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)
