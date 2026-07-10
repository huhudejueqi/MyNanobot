"""Agent 生命周期钩子 — 与 my-bot 保持一致。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMResponse, ToolCallRequest


@dataclass(slots=True)
class AgentHookContext:
    """每轮迭代的状态快照。"""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    streamed_content: bool = False
    streamed_reasoning: bool = False
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    session_key: str | None = None


@dataclass(slots=True)
class AgentRunHookContext:
    """整个 run 结束时的状态快照。"""

    messages: list[dict[str, Any]]
    final_content: str | None = None
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False
    exception: BaseException | None = None


class AgentHook:
    """Agent 生命周期钩子基类，所有方法默认空操作。

    子类覆盖需要的生命周期方法即可。
    """

    def __init__(self, reraise: bool = False) -> None:
        self._reraise = reraise

    def wants_streaming(self) -> bool:
        """返回 True 表示此钩子期望流式输出。"""
        return False

    # ── run 级别 ──

    async def before_run(self, ctx: AgentRunHookContext) -> None:
        pass

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        pass

    async def on_error(self, ctx: AgentRunHookContext) -> None:
        pass

    async def on_finally(self, ctx: AgentRunHookContext) -> None:
        pass

    # ── 迭代级别 ──

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        pass

    async def _before_iteration(self, ctx: AgentHookContext) -> None:
        """内部调用 before_iteration，不覆盖此方法。"""
        await self.before_iteration(ctx)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        pass

    async def before_execute_tools(self, ctx: AgentHookContext | None = None) -> None:
        pass

    # ── 流式 ──

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, ctx: AgentHookContext, *, resuming: bool) -> None:
        pass

    # ── 推理 ──

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        pass

    async def emit_reasoning_end(self) -> None:
        pass

    # ── 内容后处理 ──

    def finalize_content(
        self, ctx: AgentHookContext, content: str | None
    ) -> str | None:
        return content


class CompositeHook(AgentHook):
    """组合钩子：依次调用所有子钩子，单个异常不影响其他钩子。"""
    # 相当于所有 hook 方法调用时，loop_hook 和每个 run_hooks 都会收到同样的调用：
    # on_stream(delta) 被调用
    #     ├── loop_hook.on_stream(delta)     → 推流式给前端
    #     ├── _DebugHook.on_stream(delta)    → 打日志
#     └── 其他自定义 hook.on_stream(delta)
    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    async def _for_each_hook_safe(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            if getattr(h, "_reraise", False):
                await getattr(h, method_name)(*args, **kwargs)
                continue

            try:
                await getattr(h, method_name)(*args, **kwargs)
            except Exception:
                logger.exception("AgentHook.{} error in {}", method_name, type(h).__name__)


    async def before_run(self, ctx: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("before_run", ctx)

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("after_run", ctx)

    async def on_error(self, ctx: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("on_error", ctx)

    async def on_finally(self, ctx: AgentRunHookContext) -> None:
        await self._for_each_hook_safe("on_finally", ctx)

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        await self._for_each_hook_safe("before_iteration", ctx)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        await self._for_each_hook_safe("after_iteration", ctx)

    async def before_execute_tools(self, ctx: AgentHookContext | None = None) -> None:
        await self._for_each_hook_safe("before_execute_tools", ctx)

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        await self._for_each_hook_safe("on_stream", ctx, delta)

    async def on_stream_end(self, ctx: AgentHookContext, *, resuming: bool) -> None:
        await self._for_each_hook_safe("on_stream_end", ctx, resuming=resuming)

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        await self._for_each_hook_safe("emit_reasoning", reasoning_content)

    async def emit_reasoning_end(self) -> None:
        await self._for_each_hook_safe("emit_reasoning_end")

    def finalize_content(
        self, ctx: AgentHookContext, content: str | None
    ) -> str | None:
        for h in self._hooks:
            content = h.finalize_content(ctx, content)
        return content


class SDKCaptureHook(AgentHook):
    """捕获工具调用信息的钩子。"""

    def __init__(self) -> None:
        super().__init__()
        self.tools_used: list[str] = []
        self.messages: list[dict[str, Any]] = []

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        self.tools_used = list(ctx.tools_used)
        self.messages = list(ctx.messages)
