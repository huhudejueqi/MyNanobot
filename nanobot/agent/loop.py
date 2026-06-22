"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from contextlib import AsyncExitStack, nullcontext, suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent import context as agent_context
from nanobot.agent import model_presets as preset_helpers
from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.context import ContextBuilder
from nanobot.agent.cron_turns import CronTurnCoordinator
from nanobot.agent.hook import AgentHook, CompositeHook
from nanobot.agent.memory import Consolidator
from nanobot.agent.progress_hook import AgentProgressHook
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.self import MyTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.progress import build_bus_progress_callback
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import (
    RuntimeEventBus,
    RuntimeEventPublisher,
    ensure_runtime_event_publisher,
)
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import AgentDefaults, ModelPresetConfig
from nanobot.cron.session_turns import (
    cron_history_overrides,
)
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import ProviderSnapshot
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
from nanobot.utils.document import extract_documents, reference_non_image_attachments
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.image_generation_intent import image_generation_prompt
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
)

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        ProviderConfig,
        ToolsConfig,
    )
    from nanobot.cron.service import CronService


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

    ephemeral: bool = False
    tools: ToolRegistry | None = None

    turn_wall_started_at: float = field(default_factory=time.time)
    visible_run_started_at: float | None = None
    turn_latency_ms: int | None = None

    trace: list[StateTraceEntry] = field(default_factory=list)


class AgentLoop:
    """AgentLoop 是 nanobot 的核心处理引擎。

    职责：
    1. 从消息总线接收用户消息
    2. 构建上下文（含历史记录、记忆、技能）
    3. 调用 LLM 获取回复
    4. 执行工具调用
    5. 将响应发送回对应频道
    """

    @property
    def current_iteration(self) -> int:
        """返回当前轮次的迭代次数。"""
        return self._current_iteration

    @property
    def tool_names(self) -> list[str]:
        """返回已注册的工具名称列表。"""
        return self.tools.tool_names

    def llm_runtime(self) -> LLMRuntime:
        """返回当前 LLM provider/model 配对。"""
        self._refresh_provider_snapshot()
        return LLMRuntime(self.provider, self.model)

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    # 事件驱动的状态转换表。
    # handler 返回事件字符串，driver 根据该表查找下一状态。
    _TRANSITIONS: dict[tuple[TurnState, str], TurnState] = {
        # (当前状态, 事件) -> 下一状态
        # 恢复会话 -> 压缩历史
        (TurnState.RESTORE, "ok"): TurnState.COMPACT,
        # 压缩完毕 -> 处理命令
        (TurnState.COMPACT, "ok"): TurnState.COMMAND,
        # 需分派 -> 构建上下文
        (TurnState.COMMAND, "dispatch"): TurnState.BUILD,
        # 快捷命令 -> 直接结束
        (TurnState.COMMAND, "shortcut"): TurnState.DONE,
        # 上下文就绪 -> 调用 LLM
        (TurnState.BUILD, "ok"): TurnState.RUN,
        # LLM 返回 -> 保存消息
        (TurnState.RUN, "ok"): TurnState.SAVE,
        # 消息已保存 -> 发响应
        (TurnState.SAVE, "ok"): TurnState.RESPOND,
        # 响应已发送 -> 本轮结束
        (TurnState.RESPOND, "ok"): TurnState.DONE,
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
        # 消息总线：负责接收和分发所有消息
        self.bus = bus
        # 运行时事件总线（model 切换、session 变更等）
        self.runtime_events = runtime_events or RuntimeEventBus()
        self.runtime_event_publisher = RuntimeEventPublisher(self.runtime_events)
        # 频道配置（Telegram、Discord 等）
        self.channels_config = channels_config
        # LLM provider 和 model
        self.provider = provider
        self._provider_snapshot_loader = provider_snapshot_loader
        self._preset_snapshot_loader = preset_snapshot_loader
        self._runtime_model_publisher = runtime_model_publisher
        self._provider_signature = provider_signature
        self._default_selection_signature = preset_helpers.default_selection_signature(provider_signature)
        # 工作目录：存放 session、记忆、cron 等数据
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
        # 工具配置（web 搜索、文件操作、shell 执行等）
        self.tools_config = _tc
        self.web_config = _tc.web
        self.exec_config = _tc.exec
        self._image_generation_provider_configs = dict(image_generation_provider_configs or {})
        if (
            image_generation_provider_config is not None
            and "openrouter" not in self._image_generation_provider_configs
        ):
            self._image_generation_provider_configs["openrouter"] = image_generation_provider_config
        # 定时任务服务
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        # 工作区作用域解析：控制 agent 能访问的目录范围
        self.workspace_scopes = WorkspaceScopeResolver(
            default_workspace=workspace,
            default_restrict_to_workspace=restrict_to_workspace,
        )
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        # 上下文构建器：组装历史消息、记忆、技能提示
        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        # Session 管理器：按 session 保存和恢复对话历史
        self.sessions = session_manager or SessionManager(workspace)
        # 工具注册表：所有可用工具的注册中心
        self.tools = ToolRegistry()
        # One file-read/write tracker per logical session. The tool registry is
        # shared by this loop, so tools resolve the active state via contextvars.
        # 文件读写状态追踪（每个逻辑 session 独立追踪）
        self._file_state_store = FileStateStore()
        # Agent 运行器：负责 LLM 多轮对话循环
        self.runner = AgentRunner(provider)
        # 子 Agent 管理器：支持 spawn 子任务并行执行
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
        # 统一 session：同一频道共享同一个 session
        self._unified_session = unified_session
        self._max_messages = max_messages if max_messages > 0 else 120
        # 运行状态标志
        self._running = False
        # MCP 服务器管理：连接、断开、重连
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        # 活跃 session 任务追踪（用于中断和并发控制）
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        # 后台任务列表（如定时记忆合并）
        self._background_tasks: list[asyncio.Task] = []
        # 每个 session 的锁，防止同一会话并发
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        # 每个 session 的待处理消息队列
        # 当 session 正在处理中，新消息先入队等待
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # Cron 任务轮次协调器
        self._cron_turns = CronTurnCoordinator(
            publish_inbound=self.bus.publish_inbound,
            dispatch=self._dispatch,
            is_running=lambda: self._running,
        )
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        # 并发请求限制（NANOBOT_MAX_CONCURRENT_REQUESTS，默认 3）
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        # 记忆合并器（Dream）：定期将短期记忆压缩为长期记忆
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
        # 自动历史压缩：超阈值时触发上下文压缩
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        # Model Preset 管理：预定义的模型/供应商切换方案
        self.model_presets: dict[str, ModelPresetConfig] = model_presets or {}
        self._active_preset: str | None = None
        if model_preset:
            self.set_model_preset(model_preset, publish_update=False)
        self._register_default_tools()
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        # 命令路由器：处理 /slash 命令
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    @classmethod
    def from_config(
        cls,
        config: Any,
        bus: MessageBus | None = None,
        **extra: Any,
    ) -> AgentLoop:
        """Create an AgentLoop from config with the common parameter set.

        Extra keyword arguments are forwarded to ``AgentLoop.__init__``,
        allowing callers to override or extend the standard config-derived
        parameters (e.g. ``cron_service``, ``session_manager``).
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
            max_iterations=defaults.max_tool_iterations,
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            context_window_tokens=context_window_tokens,
            context_block_limit=defaults.context_block_limit,
            max_tool_result_chars=defaults.max_tool_result_chars,
            provider_retry_mode=defaults.provider_retry_mode,
            tool_hint_max_length=defaults.tool_hint_max_length,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            consolidation_ratio=defaults.consolidation_ratio,
            max_messages=defaults.max_messages,
            tools_config=config.tools,
            model_presets=preset_helpers.configured_model_presets(config),
            model_preset=defaults.model_preset,
            provider_snapshot_loader=provider_snapshot_loader,
            preset_snapshot_loader=preset_snapshot_loader,
            **extra,
        )

    def _sync_subagent_runtime_limits(self) -> None:
        """Keep subagent runtime limits aligned with mutable loop settings."""
        self.subagents.max_iterations = self.max_iterations

    def _apply_provider_snapshot(
        self,
        snapshot: ProviderSnapshot,
        *,
        publish_update: bool = True,
        model_preset: str | None = None,
    ) -> None:
        """Swap model/provider for future turns without disturbing an active one."""
        provider = snapshot.provider
        model = snapshot.model
        context_window_tokens = snapshot.context_window_tokens
        old_model = self.model
        # LLM provider 和 model
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.runner.provider = provider
        self.subagents.set_provider(provider, model)
        self.consolidator.set_provider(provider, model, context_window_tokens)
        self._provider_signature = snapshot.signature
        if publish_update and self._runtime_model_publisher is not None:
            self._runtime_model_publisher(
                self.model,
                model_preset if model_preset is not None else self.model_preset,
            )
        if publish_update:
            self._runtime_events().runtime_model_changed(
                self.model,
                model_preset if model_preset is not None else self.model_preset,
            )
        logger.info("Runtime model switched for next turn: {} -> {}", old_model, model)

    def _refresh_provider_snapshot(self) -> None:
        if self._provider_snapshot_loader is None:
            return
        try:
            snapshot = self._provider_snapshot_loader()
        except Exception:
            logger.exception("Failed to refresh provider config")
            return
        default_selection = preset_helpers.default_selection_signature(snapshot.signature)
        if self._active_preset and self._default_selection_signature in (None, default_selection):
            self._default_selection_signature = default_selection
            try:
                snapshot = self._build_model_preset_snapshot(self._active_preset)
            except Exception:
                logger.exception("Failed to refresh active model preset")
                return
        else:
            self._active_preset = None
            self._default_selection_signature = default_selection
        if snapshot.signature == self._provider_signature:
            return
        self._default_selection_signature = preset_helpers.default_selection_signature(snapshot.signature)
        self._apply_provider_snapshot(snapshot)

    @property
    def model_preset(self) -> str | None:
        return self._active_preset

    @model_preset.setter
    def model_preset(self, name: str | None) -> None:
        self.set_model_preset(name)

    def _build_model_preset_snapshot(self, name: str) -> ProviderSnapshot:
        return preset_helpers.build_runtime_preset_snapshot(
            name=name,
            presets=self.model_presets,
            provider=self.provider,
            loader=self._preset_snapshot_loader,
        )

    def set_model_preset(self, name: str | None, *, publish_update: bool = True) -> None:
        """Resolve a preset by name and apply all runtime model dependents."""
        name = preset_helpers.normalize_preset_name(name, self.model_presets)
        snapshot = self._build_model_preset_snapshot(name)
        self._apply_provider_snapshot(snapshot, publish_update=publish_update, model_preset=name)
        self._active_preset = name

    def _register_default_tools(self) -> None:
        """Register the default set of tools via plugin loader."""
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

        # MyTool needs runtime state reference — manual registration
        if self.tools_config.my.enable:
            self.tools.register(
                MyTool(runtime_state=self, modify_allowed=self.tools_config.my.allow_set)
            )
            registered.append("my")

        logger.info("Registered {} tools: {}", len(registered), registered)

    async def _connect_mcp(self) -> None:
        """Connect configured MCP servers."""
        await agent_context.connect_mcp(self, self.tools)

    def _set_tool_context(
        self, channel: str, chat_id: str,
        message_id: str | None = None, metadata: dict | None = None,
        session_key: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
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

    @staticmethod
    def _runtime_chat_id(msg: InboundMessage) -> str:
        """Return the chat id shown in runtime metadata for the model."""
        return str(msg.metadata.get("context_chat_id") or msg.chat_id)

    async def _build_bus_progress_callback(
        self, msg: InboundMessage
    ) -> Callable[..., Awaitable[None]]:
        """Build a progress callback that publishes to the message bus."""
        return build_bus_progress_callback(self.bus, msg)

    async def _build_retry_wait_callback(
        self, msg: InboundMessage
    ) -> Callable[[str], Awaitable[None]]:
        """Build a retry-wait callback that publishes to the message bus."""

        async def _on_retry_wait(content: str) -> None:
            meta = dict(msg.metadata or {})
            meta["_retry_wait"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        return _on_retry_wait

    def _runtime_events(self) -> RuntimeEventPublisher:
        return ensure_runtime_event_publisher(self)

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        return await self._cron_turns.submit(msg)

    def pending_cron_job_ids_for_session(self, session_key: str) -> set[str]:
        return self._cron_turns.pending_job_ids_for_session(session_key)

    def _persist_user_message_early(
        self,
        msg: InboundMessage,
        session: Session,
        **kwargs: Any,
    ) -> bool:
        """Persist the triggering user message before the turn starts.

        Returns True if the message was persisted.
        """
        if not turn_continuation.should_persist_user_message(msg.metadata):
            return False
        media_paths = [p for p in (msg.media or []) if isinstance(p, str) and p]
        has_text = isinstance(msg.content, str) and msg.content.strip()
        if has_text or media_paths:
            extra: dict[str, Any] = ({"media": list(media_paths)} if media_paths else {}) | agent_context.session_extra(msg.metadata)
            extra.update(kwargs)
            text = msg.content if isinstance(msg.content, str) else ""
            text_override, cron_extra = cron_history_overrides(msg.metadata)
            if text_override is not None:
                text = text_override
            extra.update(cron_extra)
            session.add_message("user", text, **extra)
            self._mark_pending_user_turn(session)
            self.sessions.save(session)
            return True
        return False

    def _build_initial_messages(
        self,
        msg: InboundMessage,
        session: Session,
        history: list[dict[str, Any]],
        pending_summary: str | None,
        include_memory_recent_history: bool = True,
    ) -> list[dict[str, Any]]:
        """Build the initial message list for the LLM turn."""
        scope = self.workspace_scopes.for_message(msg, session.metadata)
        return self.context.build_messages(
            history=history,
            current_message=image_generation_prompt(msg.content, msg.metadata),
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=self._runtime_chat_id(msg),
            sender_id=msg.sender_id,
            session_summary=pending_summary,
            session_metadata=session.metadata,
            workspace=scope.project_path,
            runtime_state=self,
            inbound_message=msg,
            include_memory_recent_history=include_memory_recent_history,
            session_key=session.key,
            unified_session=self._unified_session,
        )

    async def _dispatch_command_inline(
            self,
            msg: InboundMessage,
            key: str,
            raw: str,
            dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """
        从主事件循环run()内直接执行命令分发，并推送命令处理结果
        :param msg: 用户原始入站消息对象
        :param key: 当前指令唯一业务标识key
        :param raw: 用户发送的原始命令文本
        :param dispatch_fn: 分发处理异步函数，入参CommandContext，返回回复消息或None
        """
        # 构造命令上下文，这里会话session暂时传入None，绑定当前事件循环实例self
        ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
        # 执行传入的分发函数，处理命令逻辑，获取返回的回复消息
        result = await dispatch_fn(ctx)
        if result:
            # 存在回复消息，通过消息总线对外推送这条出站消息
            await self.bus.publish_outbound(result)
        else:
            # 匹配到了对应命令，但处理函数未返回任何回复，打印警告日志
            logger.warning("指令 '{}' 匹配 但 分发返回为None", raw)

    async def _cancel_active_tasks(self, key: str) -> int:
        """
        根据会话唯一标识key，取消并等待该会话下所有运行中的任务与子智能体
        :param key: 会话唯一标识（CommandContext中的ctx.key）
        :return: 已成功取消的任务数量 + 子智能体数量总和
        """
        # 从活跃任务字典中取出当前会话对应的全部任务列表，同时删除该key；无任务则返回空列表
        tasks = self._active_tasks.pop(key, [])
        # 遍历所有任务，统计：任务未完成 且 调用cancel()成功的数量
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        
        for t in tasks:
            # 忽略任务取消异常、其他通用异常，防止等待任务时抛出错误中断流程
            with suppress(asyncio.CancelledError, Exception):
                # 等待任务完整结束，释放资源
                await t
    
        # 调用子智能体管理器，取消同一会话下所有子智能体，返回取消数量
        sub_cancelled = await self.subagents.cancel_by_session(key)
        # 返回普通任务取消数 + 子智能体取消数之和
        return cancelled + sub_cancelled

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """返回用于任务路由、会话中途注入逻辑的实际会话标识key"""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    def _replay_token_budget(self) -> int:
        """Derive a token budget for session history replay from the context window."""
        if self.context_window_tokens <= 0:
            return 0
        max_output = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        try:
            reserved_output = int(max_output)
        except (TypeError, ValueError):
            reserved_output = 4096
        budget = self.context_window_tokens - max(1, reserved_output) - 1024
        return budget if budget > 0 else max(128, self.context_window_tokens // 2)

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
        tools: ToolRegistry | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        self._sync_subagent_runtime_limits()

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
        hook: AgentHook = loop_hook
        if not ephemeral and self._extra_hooks:
            hook = CompositeHook([loop_hook] + self._extra_hooks)

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Drain follow-up messages from the pending queue.

            When no messages are immediately available but sub-agents
            spawned in this dispatch are still running, blocks until at
            least one result arrives (or timeout).  This keeps the runner
            loop alive so subsequent sub-agent completions are consumed
            in-order rather than dispatched separately.
            """
            if pending_queue is None:
                return []

            def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = self._prepare_message_media(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                return {"role": "user", "content": user_content}

            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    items.append(_to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            # Block if nothing drained but sub-agents spawned in this dispatch
            # are still running.  Keeps the runner loop alive so subsequent
            # completions are injected in-order rather than dispatched separately.
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
                    return items
                items.append(_to_user_message(msg))
                while len(items) < limit:
                    try:
                        items.append(_to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items

        active_session_key = session.key if session else session_key
        effective_scope = self.workspace_scopes.for_turn(
            channel=channel,
            message_metadata=metadata,
            session_metadata=session.metadata if session is not None else None,
        )
        request_ctx = RequestContext(
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            session_key=active_session_key,
            metadata=dict(metadata or {}),
        )
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        request_token = bind_request_context(request_ctx)
        workspace_token = bind_workspace_scope(effective_scope)
        # Compute lazily because long_task may create goal metadata during this run.
        def _goal_continue() -> str | None:
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
            result = await self.runner.run(AgentRunSpec(
                initial_messages=initial_messages,
                tools=tools or self.tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                hook=hook,
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True,
                workspace=effective_scope.project_path,
                session_key=session.key if session else None,
                context_window_tokens=self.context_window_tokens,
                context_block_limit=self.context_block_limit,
                provider_retry_mode=self.provider_retry_mode,
                progress_callback=on_progress,
                stream_progress_deltas=on_stream is not None,
                retry_wait_callback=on_retry_wait,
                checkpoint_callback=_checkpoint,
                injection_callback=_drain_pending,
                # Sustained goals may legitimately exceed NANOBOT_LLM_TIMEOUT_S; idle stall
                # is still capped by NANOBOT_STREAM_IDLE_TIMEOUT_S in streaming providers.
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
            reset_workspace_scope(workspace_token)
            reset_request_context(request_token)
            reset_file_states(file_state_token)
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            should_stream = turn_continuation.should_stream_budget_response(
                stop_reason=result.stop_reason,
                pending_queue_available=pending_queue is not None and session is not None,
                session_metadata=session_metadata,
                message_metadata=metadata,
            )
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end and should_stream:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    async def run(self) -> None:
        """
        Agent 主循环：持续从消息总线拉取、消费上行消息
        核心并发设计：
        每条收到的消息单独创建 asyncio.Task 交给 _dispatch 分发处理
        实现效果：不同会话之间并发执行，同一个会话内部消息串行排队执行
        """
        # 标记主循环运行状态
        self._running = True
        # 初始化连接所有MCP服务
        await self._connect_mcp()
        logger.info("Agent 消息主循环已启动")

        # 主循环，运行标识为True时持续轮询消息
        while self._running:
            
            try:
                # 阻塞拉取总线上行消息，设置1秒超时
                # 超时作用：定时执行过期清理、内存压缩逻辑
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            
            # 捕获1秒超时异常，执行自动压缩、过期数据清理
            except asyncio.TimeoutError:
                
                self.auto_compact.check_expired(
                    self._schedule_background,
                    active_session_keys=self._pending_queues.keys(),
                )
                # 清理完成，回到循环开头继续拉消息
                continue
            
            # 捕获任务取消异常，用于优雅关闭程序
            except asyncio.CancelledError:
                # 区分真实任务取消信号与第三方集成泄漏的无效取消信号
                # 如果是正常关闭或当前任务被主动取消，向上抛出完成退出流程
                if not self._running or asyncio.current_task().cancelling():
                    raise
                # 其他无关取消信号直接忽略，继续循环
                continue
            
            # 捕获所有未知异常，单条消息出错不中断整个主循环
            except Exception as e:
                logger.warning("拉取上行消息发生异常: {}, 继续运行...", e)
                continue

            # 去除消息首尾空白字符
            raw = msg.content.strip()
            
            # 计算当前消息实际归属会话标识（统一会话合并场景会转换key）
            effective_key = self._effective_session_key(msg)
            # logger.info("effective_key{}",effective_key)
            # 处理运行时控制指令（如重启、关闭、调试控制类指令）
            if await agent_context.handle_runtime_control(self, msg, self.tools):
                # 控制指令处理完毕，跳过后续业务分发逻辑
                continue

            # 判断是否为高优先级指令（紧急指令优先同步执行，不进队列）
            if self.commands.is_priority(raw):
                # logger.info("self.commands.is_priority{}",raw)
                await self._dispatch_command_inline(
                    msg, effective_key, raw,
                    self.commands.dispatch_priority,
                )
                continue

            # 校验是否为定时轮询消息，若对应会话正在活跃则延迟本次定时任务
            if self._cron_turns.defer_if_active(
                msg,
                session_key=effective_key,
                active_session_keys=self._pending_queues.keys(),
            ):
                logger.info(
                    "会话 {} 当前活跃，延后执行定时轮询任务",
                    effective_key,
                )
                continue

            # 判断：该会话已有正在执行的消息任务（存在待处理消息队列）
            # 新消息不新建任务，丢进当前会话队列排队，保证同会话串行执行
            if effective_key in self._pending_queues:
                # 可即时分发的普通指令不走队列，同步直接处理
                if self.commands.is_dispatchable_command(raw):
                    await self._dispatch_command_inline(
                        msg, effective_key, raw,
                        self.commands.dispatch,
                    )
                    continue
                
                # 普通业务消息，准备放入会话排队队列
                pending_msg = msg
                # 如果实际会话key和原始消息key不一致，生成新消息对象覆盖会话标识
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                
                try:
                    # 非阻塞写入队列
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    # 队列已满，降级策略：放弃排队，新建独立任务处理
                    logger.warning(
                        "会话 {} 消息等待队列已满，降级创建独立任务处理",
                        effective_key,
                    )
                else:
                    logger.info(
                        "将后续消息路由至会话 {} 的等待队列排队处理",
                        effective_key,
                    )
                    # 入队成功，跳过新建任务流程
                    continue

            # ==============================
            # 该会话无正在运行的任务，创建全新异步Task处理本条消息
            # ==============================
            task = asyncio.create_task(self._dispatch(msg))
            # 记录当前会话所有活跃任务
            self._active_tasks.setdefault(effective_key, []).append(task)
            
            # 任务完成回调：自动从会话活跃任务列表移除已结束任务，防止内存泄漏
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """
        分发并处理单条入站消息的完整主流程：
        1. 获取会话独占锁：保证同一个会话的消息串行执行，避免上下文错乱
        2. 恢复会话历史 / 压缩超长对话上下文
        3. 组装本次对话执行上下文，调用大模型LLM生成回复
        4. 循环执行工具调用（ReAct智能工具调用循环，比如调用UE MCP操作引擎）
        5. 持久化对话记录，向外推送回复消息给客户端
        """
        # 计算当前消息真正生效的会话唯一标识
        session_key = self._effective_session_key(msg)
        # 如果计算出的会话标识和消息自带的不一致，复制一份新消息覆盖会话key
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)

        # 获取会话锁：每个会话对应一把独立异步锁，同一会话消息排队串行执行
        # setdefault：字典不存在该key则新建asyncio.Lock，存在则直接取出已有锁
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())

        # 并发限流上下文：配置了并发信号量就用信号量限制同时执行任务数；无配置则空上下文不限制并发
        gate = self._concurrency_gate or nullcontext()

        # 局部临时队列变量，初始值None；仅当前本次消息分发流程内有效
        # 类型注解：变量只能是异步队列对象 或 None
        pending: asyncio.Queue | None = None
        try:
            # 同时持有会话锁+并发限流闸门，进入消息处理状态机（恢复上下文→压缩历史→执行对话→结束）
            async with lock, gate:
                # 注释：只有当前持有会话锁的任务，才有权创建、管理本次会话的工具消息临时队列
                # 初始化最大容量20的异步队列，用于存放本轮对话中途产生的子消息/工具回调消息
                pending = asyncio.Queue(maxsize=20)
                # 将当前会话与临时队列绑定存入全局队列字典，供其他逻辑读取
                self._pending_queues[session_key] = pending

                try:
                    # 流式输出回调、流式结束回调，默认None（不开启流式）
                    on_stream = on_stream_end = None
                    # 判断客户端元数据是否声明需要流式分段输出（Codex/网页客户端实时打字效果）
                    if msg.metadata.get("_wants_stream"):
                        # 生成当前流式会话唯一ID：会话标识+高精度时间戳
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        # 流式分段序号，每结束一段自增
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            """拼接当前分段完整流式ID"""
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            """流式片段推送回调：发送单段增量文字给客户端"""
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True  # 标记本条是流式增量片段
                            meta["_stream_id"] = _current_stream_id()  # 绑定所属流式会话分段ID
                            # 通过消息总线向外推送增量文本消息
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            """流式结束回调：通知客户端本轮流式输出完成"""
                            nonlocal stream_segment  # 引用外层函数的分段序号变量
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True  # 标记流式输出结束
                            meta["_resuming"] = resuming  # 是否为中断后恢复输出
                            meta["_stream_id"] = _current_stream_id()
                            # 推送空内容结束标记报文
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1  # 分段序号+1，准备下一轮流式输出

                    # 执行消息核心处理逻辑：LLM对话+工具调用循环
                    # 传入流式回调、本轮会话专用临时队列
                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )

                    # 记录最终回复要推送的渠道、会话ID
                    completed_channel = msg.channel
                    completed_chat_id = msg.chat_id
                    if response is not None:
                        # 存在回复报文，推送到消息总线发给客户端
                        await self.bus.publish_outbound(response)
                        # 更新实际输出渠道与会话ID（回复可能自动切换渠道）
                        completed_channel = response.channel
                        completed_chat_id = response.chat_id
                    elif msg.channel == "cli":
                        # 命令行渠道无回复时，推送空报文标记本轮对话结束
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))

                    # 判断是否存在内部续对话标记（工具未执行完、需要继续交互）
                    continuing = turn_continuation.internal_continuation_pending(msg.metadata)
                    if not continuing:
                        # 无后续续对话，触发本轮对话完成事件
                        await self._runtime_events().turn_completed(
                            channel=completed_channel,
                            chat_id=completed_chat_id,
                            session_key=session_key,
                            metadata=msg.metadata,
                        )
                    # 定时任务模块标记本轮对话正常完成
                    self._cron_turns.complete(msg, response=response)

                except asyncio.CancelledError:
                    # 捕获任务取消异常（用户发送/stop、强制终止MCP任务）
                    self._cron_turns.complete(
                        msg,
                        error=asyncio.CancelledError(),
                    )
                    logger.info("会话 {} 的任务被用户取消", session_key)

                    # 注释：任务中断时保留已执行的工具结果、助手回复，避免用户丢失上下文
                    # 工具执行过程中已自动写入会话快照，这里把快照加载到对话历史，下次对话可见
                    try:
                        key = self._effective_session_key(msg)
                        session = self.sessions.get_or_create(key)
                        # 恢复中断前的运行时快照
                        if self._restore_runtime_checkpoint(session):
                            self._clear_pending_user_turn(session)
                            self.sessions.save(session)
                            logger.info(
                                "已为被取消的会话 {} 恢复中断前的对话上下文快照",
                                key,
                            )
                    except Exception:
                        # 恢复快照失败仅打印调试日志，不阻断程序
                        logger.debug(
                            "无法为被取消会话 {} 恢复上下文快照",
                            session_key,
                            exc_info=True,
                        )
                    # 通知前端本轮对话已结束（前端 turn_end 事件触发停止按钮消失）
                    await self._runtime_events().turn_completed(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        session_key=session_key,
                        metadata=msg.metadata,
                    )
                    # 重新抛出取消异常，上层流程感知任务终止
                    raise

                except Exception as exc:
                    # 捕获所有未知运行异常
                    logger.exception("处理会话 {} 的消息时发生未知错误", session_key)
                    # 推送错误提示回复给客户端
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="抱歉，处理你的请求时出现了异常。",
                    ))
                    # 无续对话则触发对话完成事件
                    if not turn_continuation.internal_continuation_pending(msg.metadata):
                        await self._runtime_events().turn_completed(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            session_key=session_key,
                            metadata=msg.metadata,
                        )
                    # 定时模块标记本轮对话异常结束
                    self._cron_turns.complete(msg, error=exc)

                finally:
                    # 本轮对话收尾清理：处理队列中未消费完的遗留消息
                    # 注释：队列内剩余未处理消息重新投递至消息总线，作为全新入站消息重新处理，防止消息丢失
                    # 仅删除当前任务创建的队列；其他等待锁的并发任务不能抢占清理权限
                    queue = None
                    # 校验全局队列字典里绑定的队列是不是当前pending，防止多任务错乱
                    if self._pending_queues.get(session_key) is pending:
                        queue = self._pending_queues.pop(session_key, None)
                    else:
                        queue = pending

                    if queue is not None:
                        leftover = 0
                        # 循环取出队列所有残留消息
                        while True:
                            try:
                                item = queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            # 残留消息重新投递进入消息总线
                            await self.bus.publish_inbound(item)
                            leftover += 1
                        # 存在残留消息打印日志告知数量
                        if leftover:
                            logger.info(
                                "为会话 {} 重新投递 {} 条未处理完的遗留消息",
                                leftover, session_key,
                            )

                    # 无内部续对话，更新会话状态为空闲、清空临时对话标记
                    if not turn_continuation.internal_continuation_pending(msg.metadata):
                        await self._runtime_events().run_status_changed(
                            msg, session_key, "idle"
                        )
                        self._runtime_events().clear_turn(session_key)
                    # 执行下一条延迟排队任务
                    await self._cron_turns.publish_next_deferred(session_key)
        finally:
            # 仅当本轮流程未创建临时队列（pending全程为None）时执行空闲状态更新
            if pending is None:
                await self._runtime_events().run_status_changed(
                    msg, session_key, "idle"
                )
                self._runtime_events().clear_turn(session_key)
                await self._cron_turns.publish_next_deferred(session_key)

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        # 运行状态标志
        self._running = False
        logger.info("Agent loop stopping")

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
        _hist_kwargs: dict[str, Any] = {
            "max_messages": self._max_messages,
            "max_tokens": self._replay_token_budget(),
            "include_timestamps": True,
        }
        history = session.get_history(**_hist_kwargs)
        current_role = "assistant" if is_subagent else "user"
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

    async def _process_message(
            self,
            msg: InboundMessage,  # 入站消息对象
            session_key: str | None = None,  # 会话唯一标识，可为空
            on_progress: Callable[..., Awaitable[None]] | None = None,  # 进度异步回调函数（可选）
            on_stream: Callable[[str], Awaitable[None]] | None = None,  # 流式输出异步回调：入参为字符串分片（可选）
            on_stream_end: Callable[..., Awaitable[None]] | None = None,  # 流式结束异步回调（可选）
            pending_queue: asyncio.Queue | None = None,  # 待处理任务异步队列（可选）
            ephemeral: bool = False,  # 是否临时会话（临时会话不持久化上下文，默认关闭）
            tools: ToolRegistry | None = None,  # 工具注册管理器（函数调用工具集，可选）
        ) -> OutboundMessage | None:  # 返回出站响应消息，处理异常时可返回空
        # 刷新底层大模型服务商配置快照（更新模型/密钥/限流等配置）
        self._refresh_provider_snapshot()
        
        # 判断消息渠道：系统内置指令消息
        if msg.channel == "system":
            # 分流调用系统消息专用处理逻辑
            return await self._process_system_message(
                msg,
                session_key=session_key,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                pending_queue=pending_queue,
            )

        # 优先使用传入的会话ID，无则从消息体内读取会话ID
        key = session_key or msg.session_key
        # 记录当前时间戳（秒级，用于耗时统计）
        t0 = time.time()
        # 构建单次会话轮次上下文对象
        
        ctx = TurnContext(
            msg=msg,  # 原始入站消息
            session=None,  # 会话实例（初始为空，后续状态机加载）
            session_key=key,  # 会话标识
            state=TurnState.RESTORE,  # 状态机初始状态：恢复会话上下文
            turn_id=f"{key}:{time.time_ns()}",  # 本轮唯一ID：会话ID+纳秒级时间戳，防重复
            turn_wall_started_at=t0,  # 本轮请求开始时间戳
            # 从消息元数据读取续跑启动时间（断点续传/长会话续跑场景）
            visible_run_started_at=turn_continuation.internal_continuation_run_started_at(
                msg.metadata,
            ),
            on_progress=on_progress,  # 进度回调注入上下文
            on_stream=on_stream,  # 流式分片回调注入上下文
            on_stream_end=on_stream_end,  # 流式结束回调注入上下文
            pending_queue=pending_queue,  # 异步任务队列注入上下文
            ephemeral=ephemeral,  # 临时会话标记
            tools=tools,  # 工具调用注册表注入上下文
        )

        # 状态机主循环：未到达完成状态则持续流转
        while ctx.state is not TurnState.DONE:
            # 根据当前状态名称拼接对应处理方法名（如RESTORE → _state_restore）
            handler_name = f"_state_{ctx.state.name.lower()}"
            logger.info("状态机主循环{}",handler_name)
            # 获取当前状态对应的处理函数
            handler = getattr(self, handler_name, None)
            # 无对应状态处理器，抛出运行时异常
            if handler is None:
                raise RuntimeError(f"缺少 {ctx.state} 状态对应的处理函数")

            # 高精度计时起点（用于计算单状态耗时，毫秒精度）
            t0 = time.perf_counter()
            try:
                # 执行当前状态逻辑，返回状态流转事件标识
                event = await handler(ctx)
            except Exception:
                # 捕获任意异常，计算当前状态执行耗时（毫秒）
                duration = (time.perf_counter() - t0) * 1000
                # 写入状态追踪日志：标记执行失败
                ctx.trace.append(
                    StateTraceEntry(
                        state=ctx.state,  # 当前出错状态
                        started_at=t0,  # 状态开始时间
                        duration_ms=duration,  # 执行耗时
                        event="",  # 无正常流转事件
                        error="exception",  # 错误类型：程序异常
                    )
                )
                # 重新抛出异常，中断整个会话处理
                raise

            # 正常执行完成，计算当前状态耗时
            duration = (time.perf_counter() - t0) * 1000
            # 记录本次状态执行轨迹（用于排查耗时、流程链路）
            ctx.trace.append(
                StateTraceEntry(
                    state=ctx.state,
                    started_at=t0,
                    duration_ms=duration,
                    event=event,  # 本次产生的流转事件
                )
            )
            # 调试日志：打印轮次ID、当前状态、耗时、触发事件
            logger.debug(
                "[本轮会话 {}] 状态 {} 执行耗时 {:.1f}ms → 触发事件 {}",
                ctx.turn_id,
                ctx.state.name,
                duration,
                event,
            )

            # 从状态流转映射表，查询当前状态+事件对应的下一个状态
            next_state = self._TRANSITIONS.get((ctx.state, event))
            # 无匹配流转规则，抛出异常（流程非法）
            if next_state is None:
                raise RuntimeError(
                    f"[本轮会话 {ctx.turn_id}] 不存在流转规则：从 {ctx.state} "
                    f"触发事件 {event!r} 无目标状态"
                )
            # 更新上下文状态，进入下一轮循环处理
            ctx.state = next_state

        # 状态机全部流转完成，打印调试日志：记录总共执行了多少个状态节点
        logger.debug(
            "[本轮会话 {}] 会话处理完成，共执行 {} 个状态节点",
            ctx.turn_id,
            len(ctx.trace),
        )
        # 返回最终生成的出站响应消息
        return ctx.outbound

    def _assemble_outbound(
        self,
        msg: InboundMessage,
        final_content: str,
        all_msgs: list[dict[str, Any]],
        stop_reason: str,
        had_injections: bool,
        on_stream: Callable[[str], Awaitable[None]] | None,
        *,
        turn_latency_ms: int | None = None,
    ) -> OutboundMessage | None:
        """Assemble the final outbound message from turn results."""
        # MessageTool suppression
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason not in {"error", "tool_error"}:
            meta["_streamed"] = True
        if turn_latency_ms is not None:
            meta["latency_ms"] = int(turn_latency_ms)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    async def _state_restore(self, ctx: TurnContext) -> TurnState:
        """
        恢复会话检查点 / 用户未完成对话轮次；同时提取消息内附件文档
        功能：加载上次中断的会话状态、恢复待处理用户对话、预处理消息媒体资源
        参数 ctx: TurnContext - 当前一轮对话的上下文对象，封装消息、会话、渠道等全部上下文数据
        返回 TurnState - 对话轮次状态对象（这里代码实际return字符串"ok"，存在类型标注不一致）

        _state_restore(入参: TurnContext ctx)
        ├─ 步骤1：提取当前消息 msg = ctx.msg
        ├─ 步骤2：判断消息是否携带媒体附件 msg.media
        │  ├─ 分支A：存在媒体
        │  │  ├─ 调用 _prepare_message_media 处理文本+媒体 → 得到 new_content、image_only
        │  │  ├─ dataclasses.replace 生成新消息，覆盖 ctx.msg
        │  │  └─ 更新本地变量 msg = 处理后的新消息
        │  └─ 分支B：无媒体 → 跳过媒体处理
        ├─ 步骤3：生成日志预览文本（截取前80字符，超长加省略号）
        ├─ 步骤4：打印日志：渠道ID、发送人、消息预览
        ├─ 步骤5：校验会话对象 ctx.session 是否为空
        │  ├─ 分支A：session为空
        │  │  └─ 通过 session_key 获取/新建持久会话，赋值给 ctx.session
        │  └─ 分支B：session已存在 → 跳过创建
        ├─ 步骤6：触发运行时事件 session_turn_started（标记新一轮对话开始）
        ├─ 步骤7：持久化当前消息对应的工作空间作用域到会话
        ├─ 步骤8：恢复运行时断点检查点 _restore_runtime_checkpoint
        │  ├─ 分支A：恢复后状态变更（返回True）
        │  │  └─ 保存会话 self.sessions.save(ctx.session)
        │  └─ 分支B：无变更 → 不保存
        ├─ 步骤9：恢复未完成用户对话轮次 _restore_pending_user_turn
        │  ├─ 分支A：恢复后状态变更（返回True）
        │  │  └─ 保存会话 self.sessions.save(ctx.session)
        │  └─ 分支B：无变更 → 不保存
        └─ 步骤10：返回字符串 "ok"
            补充：类型注解标注返回 TurnState，实际返回字符串，存在类型不匹配问题
        """
        # 取出上下文里的用户消息对象
        msg = ctx.msg

        # 如果消息携带图片/文件等媒体附件，执行预处理
        if msg.media:
            # 处理消息正文+媒体资源：转换媒体格式、剥离纯图片资源
            new_content, image_only = self._prepare_message_media(msg.content, msg.media)
            # 复制一份全新消息对象，替换处理后的正文与媒体字段（dataclasses.replace 不可变数据更新）
            ctx.msg = dataclasses.replace(msg, content=new_content, media=image_only)
            # 同步更新本地msg变量为处理后的新消息
            msg = ctx.msg

        # 截取消息前80字符做日志预览，过长则末尾拼接省略号
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        # 打印日志：渠道ID、发送者ID、消息预览内容
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # 注释说明：上层调用函数 _process_message 理论上已经加载过会话
        # 此处做兼容兜底：防止该函数被单独调用、会话未初始化的场景
        # Session = 用户持久会话，存储历史对话、记忆、检查点、配置
        logger.info("ctx.session start")
        if ctx.session is None:
            logger.info("ctx.session is None")
            # 根据会话唯一标识，获取已有会话；不存在则新建空会话
            ctx.session = self.sessions.get_or_create(ctx.session_key)
        logger.info("ctx.session end")
        # 触发运行时事件：标记当前会话开启一轮新对话
        await self._runtime_events().session_turn_started(msg, ctx.session_key)
        # 将本次消息关联的工作空间作用域持久化存入会话（区分多项目/多文件夹隔离）
        self.workspace_scopes.persist_message_scope(ctx.session, msg)

        # 1. 恢复运行时断点检查点（Agent中途中断、工具执行一半的上下文）
        # 函数返回True代表状态发生变更，需要落地保存会话
        if self._restore_runtime_checkpoint(ctx.session):
            logger.info("_restore_runtime_checkpoint")
            self.sessions.save(ctx.session)
        
        # 2. 恢复用户未完成的对话轮次（比如上一轮AI回复一半中断、等待用户续聊）
        # 状态变更则持久化会话数据到磁盘/数据库
        if self._restore_pending_user_turn(ctx.session):
            logger.info("_restore_pending_user_turn")
            self.sessions.save(ctx.session)

        # 函数标注返回TurnState，但实际返回字符串"ok"，属于代码小bug/标注疏漏
        return "ok"
    
    def _prepare_message_media(self, content: str, media: list[str]) -> tuple[str, list[str]]:
        """预处理消息正文与附件文件列表
        :param content: 消息文本内容
        :param media: 媒体/附件文件路径/标识列表
        :return: 处理后的(新文本内容, 过滤后的附件列表)二元组
        """
        # 判断是否需要提取文档内文字
        if self._should_extract_document_text():
            # 提取附件文档中的文本，并更新正文、过滤附件列表后返回
            return extract_documents(content, media)
        # 不提取文档文字：仅标记引用非图片类附件，原样处理正文和附件
        return reference_non_image_attachments(content, media)

    def _should_extract_document_text(self) -> bool:
        if self.channels_config is None:
            return True
        return self.channels_config.extract_document_text

    async def _state_compact(self, ctx: TurnContext) -> str:
        ctx.session, pending = self.auto_compact.prepare_session(ctx.session, ctx.session_key)
        ctx.pending_summary = pending
        return "ok"

    async def _state_command(self, ctx: TurnContext) -> str:
        raw = ctx.msg.content.strip()
        cmd_ctx = CommandContext(
            msg=ctx.msg, session=ctx.session, key=ctx.session_key, raw=raw, loop=self
        )
        result = await self.commands.dispatch(cmd_ctx)
        if result is not None:
            ctx.outbound = result
            # Shortcut commands skip BUILD and SAVE, so we must persist the
            # turn here so WebUI history hydration after _turn_end sees the
            # message.  Mark messages with _command so get_history can filter
            # them out of LLM context.  /new is excluded because it
            # intentionally clears the session.
            if raw.lower() != "/new":
                ctx.user_persisted_early = self._persist_user_message_early(
                    ctx.msg, ctx.session, _command=True
                )
                ctx.session.add_message(
                    "assistant", result.content, _command=True
                )
                self.sessions.save(ctx.session)
                self._clear_pending_user_turn(ctx.session)
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

        _hist_kwargs: dict[str, Any] = {
            "max_messages": self._max_messages,
            "max_tokens": self._replay_token_budget(),
            "include_timestamps": True,
        }
        ctx.history = ctx.session.get_history(**_hist_kwargs)
        self._runtime_events().record_turn_runtime(
            ctx.session_key,
            self.llm_runtime(),
        )

        ctx.initial_messages = self._build_initial_messages(
            ctx.msg,
            ctx.session,
            ctx.history,
            ctx.pending_summary,
            include_memory_recent_history=not ctx.ephemeral,
        )
        ctx.user_persisted_early = self._persist_user_message_early(
            ctx.msg, ctx.session
        )

        if ctx.on_progress is None:
            ctx.on_progress = await self._build_bus_progress_callback(ctx.msg)
        if ctx.on_retry_wait is None:
            ctx.on_retry_wait = await self._build_retry_wait_callback(ctx.msg)

        return "ok"

    async def _state_run(self, ctx: TurnContext) -> str:
        if ctx.visible_run_started_at is None:
            ctx.visible_run_started_at = time.time()
        await self._runtime_events().run_status_changed(
            ctx.msg,
            ctx.session_key,
            "running",
            started_at=ctx.visible_run_started_at,
        )
        result = await self._run_agent_loop(
            ctx.initial_messages,
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
            tools=ctx.tools,
        )
        final_content, tools_used, all_msgs, stop_reason, had_injections = result
        ctx.final_content = final_content
        ctx.tools_used = tools_used
        ctx.all_messages = all_msgs
        ctx.stop_reason = stop_reason
        ctx.had_injections = had_injections
        await turn_continuation.maybe_continue_turn(ctx)
        return "ok"

    async def _state_save(self, ctx: TurnContext) -> str:
        turn_continuation.prepare_save_boundary(ctx)

        if (
            (ctx.final_content is None or not ctx.final_content.strip())
            and not ctx.suppress_response
        ):
            ctx.final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        latency_started_at = (
            ctx.visible_run_started_at
            if turn_continuation.internal_continuation_inbound(ctx.msg.metadata)
            and ctx.visible_run_started_at is not None
            else ctx.turn_wall_started_at
        )
        ctx.turn_latency_ms = max(0, int((time.time() - latency_started_at) * 1000))
        self._save_turn(
            ctx.session, ctx.all_messages, ctx.save_skip,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        self._runtime_events().record_turn_latency(
            ctx.session_key,
            ctx.turn_latency_ms,
        )
        if not ctx.ephemeral:
            ctx.session.enforce_file_cap(
                on_archive=partial(self.context.memory.raw_archive, session_key=ctx.session_key)
            )
            self._schedule_background(
                self.consolidator.maybe_consolidate_by_tokens(
                    ctx.session,
                    replay_max_messages=self._max_messages,
                )
            )
        self._clear_pending_user_turn(ctx.session)
        self._clear_runtime_checkpoint(ctx.session)
        self.sessions.save(ctx.session)
        return "ok"

    async def _state_respond(self, ctx: TurnContext) -> str:
        if ctx.suppress_response:
            ctx.outbound = None
            return "ok"
        ctx.outbound = self._assemble_outbound(
            ctx.msg,
            ctx.final_content,
            ctx.all_messages,
            ctx.stop_reason,
            ctx.had_injections,
            ctx.on_stream,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        if ctx.ephemeral and ctx.outbound is not None:
            ctx.outbound.metadata["_stop_reason"] = ctx.stop_reason
        return "ok"

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(
        self,
        session: Session,
        messages: list[dict],
        skip: int,
        *,
        turn_latency_ms: int | None = None,
    ) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        declared_tool_call_ids = {
            str(tc["id"])
            for m in session.messages
            if m.get("role") == "assistant"
            for tc in m.get("tool_calls") or []
            if isinstance(tc, dict) and tc.get("id")
        }
        last_assistant_idx: int | None = None
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                tool_call_id = entry.get("tool_call_id")
                if not tool_call_id or str(tool_call_id) not in declared_tool_call_ids:
                    # Undeclared tool results corrupt future provider requests.
                    logger.warning(
                        "Dropping orphaned tool result {} from session {} during persistence",
                        tool_call_id or "(missing id)",
                        session.key,
                    )
                    continue
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        # Preserve the tool_call/result pair after block filtering.
                        filtered = [
                            {"type": "text", "text": "[tool result omitted during persistence]"}
                        ]
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and ContextBuilder._RUNTIME_CONTEXT_TAG in content:
                    # Strip the runtime-context block appended at the end.
                    tag_pos = content.find(ContextBuilder._RUNTIME_CONTEXT_TAG)
                    before = content[:tag_pos].rstrip("\n ")
                    if before:
                        entry["content"] = before
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
            if role == "assistant":
                last_assistant_idx = len(session.messages) - 1
                declared_tool_call_ids.update(
                    str(tc["id"])
                    for tc in entry.get("tool_calls") or []
                    if isinstance(tc, dict) and tc.get("id")
                )
        if turn_latency_ms is not None and last_assistant_idx is not None:
            session.messages[last_assistant_idx]["latency_ms"] = int(turn_latency_ms)
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
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
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

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
        """
        恢复程序崩溃前仅保存了用户消息、未生成助手回复的会话轮次
        返回布尔值：True表示执行了恢复逻辑，False表示无需处理
        """
        from datetime import datetime

        # 如果会话元数据中不存在待处理用户轮次标记，直接返回无需恢复
        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        # 校验会话最后一条消息是否为用户发言（说明中断在用户发消息后、AI回复前）
        if session.messages and session.messages[-1].get("role") == "user":
            # 追加一条系统错误提示的助手消息，标记本次对话因程序中断失败
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            # 更新会话最后修改时间戳
            session.updated_at = datetime.now()

        # 清除会话元数据里标记待处理用户轮次的标识
        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        ephemeral: bool = False,
        tools: ToolRegistry | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id,
            content=content, media=media or [],
        )
        # Share the dispatch lock so direct calls serialize with bus turns.
        # 获取 session 锁：同一 session 的消息串行执行
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        try:
        # 持有锁时执行状态机（RESTORE -> COMPACT -> ... -> DONE）
            async with lock:
                kwargs: dict[str, Any] = {
                    "session_key": session_key,
                    "on_progress": on_progress,
                    "on_stream": on_stream,
                    "on_stream_end": on_stream_end,
                    "ephemeral": ephemeral,
                }
                if tools is not None:
                    kwargs["tools"] = tools
                return await self._process_message(
                    msg,
                    **kwargs,
                )
        finally:
            await self._runtime_events().run_status_changed(msg, session_key, "idle")
            self._runtime_events().clear_turn(session_key)
