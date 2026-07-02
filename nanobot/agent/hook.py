"""Agent 生命周期钩子。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import logging

logger = logging.getLogger("nanobot.agent.hook")

from nanobot.providers.base import LLMResponse, ToolCallRequest


@dataclass
class AgentHookContext:
    """每轮迭代的状态快照。"""

    iteration: int
    messages: list[dict[str, Any]]
    response: LLMResponse | None = None
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_results: list[Any] = field(default_factory=list)
    streamed_content: bool = False
    final_content: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    session_key: str | None = None


@dataclass
class AgentRunHookContext:
    """整个 run 结束时的状态快照。"""

    messages: list[dict[str, Any]]
    final_content: str | None = None
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None


class AgentHook:
    """钩子基类，所有方法默认空操作。"""

    async def before_run(self, ctx: AgentRunHookContext) -> None:
        pass

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        pass

    async def on_error(self, ctx: AgentRunHookContext) -> None:
        pass

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        pass

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        pass

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        pass

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, ctx: AgentHookContext, *, resuming: bool) -> None:
        pass


class CompositeHook(AgentHook):
    """组合钩子：依次调用所有子钩子，单个异常不影响其他钩子。"""

    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        self._hooks = list(hooks)

    async def _call_all(self, method: str, *args: Any, **kwargs: Any) -> None:
        for h in self._hooks:
            try:
                await getattr(h, method)(*args, **kwargs)
            except Exception:
                logger.exception("Hook %s.%s 异常", type(h).__name__, method)

    async def before_run(self, ctx: AgentRunHookContext) -> None:
        await self._call_all("before_run", ctx)

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        await self._call_all("after_run", ctx)

    async def on_error(self, ctx: AgentRunHookContext) -> None:
        await self._call_all("on_error", ctx)

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        await self._call_all("before_iteration", ctx)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        await self._call_all("after_iteration", ctx)

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        await self._call_all("before_execute_tools", ctx)

    async def on_stream(self, ctx: AgentHookContext, delta: str) -> None:
        await self._call_all("on_stream", ctx, delta)

    async def on_stream_end(self, ctx: AgentHookContext, *, resuming: bool) -> None:
        await self._call_all("on_stream_end", ctx, resuming=resuming)
