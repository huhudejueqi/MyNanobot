"""AgentRunner：LLM 多轮 ReAct 循环。

LLM 调用 → 解析回复 →
  ├── 有 tool_calls → 执行工具 → 拼接结果 → 继续 LLM 调用
  └── 无 tool_calls → 返回最终回复

流式策略（与原版一致）：
- 首轮 LLM 调用（尚无工具调用时）：若 on_stream 非空，走 chat_stream() 逐 chunk 推送
- 后续轮次（工具调用后的续调）：一律走 chat() 非流式
- 工具调用回合不流式，只流式最终 text 回复
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext, CompositeHook
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, LLMResponse, StreamDelta

logger = logging.getLogger("nanobot.agent.runner")


@dataclass
class AgentRunResult:
    """一次 Agent 运行的结果。"""

    final_content: str = ""
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    stop_reason: str = ""
    had_tool_calls: bool = False


def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """构建 provider-safe 的 assistant 消息，可选推理内容。"""
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content is not None or thinking_blocks:
        msg["reasoning_content"] = reasoning_content if reasoning_content is not None else ""
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg


class AgentRunner:
    """执行 LLM 多轮 ReAct 循环的核心引擎。

    流式逻辑：
    - 首轮（尚未产生工具调用）：若 on_stream 非空，调用 chat_stream()，
      逐 delta 通过 on_stream 回调推送到客户端，同时收集完整回复。
    - 如果首轮流式检测到 finish_reason="tool_calls"（末帧 StreamDelta.tool_calls
      携带完整工具调用列表），进入工具执行分支。
    - 工具执行后，后续迭代走 chat() 非流式，避免 websocket 乱序。
    - 纯 text 回复的首轮流式结束后，调用 on_stream_end(resuming=False) 通知渲染结束。
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_iterations: int = 10,
        on_progress=None,
        on_stream=None,
        on_stream_end=None,
        hook: AgentHook | None = None,
    ):
        self.provider = provider
        self.max_iterations = max_iterations
        self.on_progress = on_progress
        self.on_stream = on_stream
        self.on_stream_end = on_stream_end
        self.hook = hook or AgentHook()
        self._session_key: str | None = None

    async def run(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        model: str | None = None,
    ) -> AgentRunResult:
        """Agent 运行入口：hook 包装层，委托 _run_core 执行 ReAct 循环。

        ┌────────────────────────────────────────────────┐
        │  run()                                         │
        ├────────────────────────────────────────────────┤
        │                                                │
        │  run_hctx = AgentRunHookContext(messages)       │
        │  hook.before_run(run_hctx)                      │
        │       │                                         │
        │       ▼                                         │
        │  result = _run_core(spec, hook, messages)       │
        │       │                                         │
        │       ├── 正常 ──► hook.after_run() ──► return  │
        │       └── 异常 ──► hook.on_error() ──► return   │
        │                                                │
        │  finally:                                       │
        │    hook.after_run() （未调用时）                 │
        │                                                │
        └────────────────────────────────────────────────┘
        """
        run_hctx = AgentRunHookContext(messages=messages)
        await self.hook.before_run(run_hctx)

        if not self.on_progress:
            async def noop(*a, **kw): pass
            self.on_progress = noop

        result = await self._run_core(messages, tools, model)
        run_hctx.final_content = result.final_content
        run_hctx.tools_used = result.tools_used
        run_hctx.stop_reason = result.stop_reason
        if result.stop_reason == "error":
            run_hctx.error = result.final_content
            await self.hook.on_error(run_hctx)
        else:
            await self.hook.after_run(run_hctx)
        return result

    async def _run_core(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        model: str | None = None,
    ) -> AgentRunResult:
        """ReAct 核心循环：LLM 调用 ↔ 工具执行，直到完成或达到最大迭代次数。

        ┌──────────────────────────────────────────────────────┐
        │  _run_core()                                         │
        ├──────────────────────────────────────────────────────┤
        │                                                      │
        │  for iteration in range(max_iterations):             │
        │    hook.before_iteration()                           │
        │       │                                              │
        │       ▼                                              │
        │    LLM call（首轮流式 chat_stream / 后续 chat）      │
        │       │                                              │
        │       ├── finish_reason="error" ──► 返回 error       │
        │       │                                              │
        │       ├── 有 tool_calls:                              │
        │       │    hook.before_execute_tools()                │
        │       │    执行工具 → 追加 tool_result               │
        │       │    hook.after_iteration()                     │
        │       │    continue                                   │
        │       │                                              │
        │       └── 无 tool_calls:                              │
        │            hook.after_iteration()                     │
        │            返回最终回复                                │
        │                                                      │
        │  max_iterations 耗尽 → 返回 fallback                  │
        │                                                      │
        └──────────────────────────────────────────────────────┘
        """
        result = AgentRunResult()
        for iteration in range(self.max_iterations):
            logger.info("ReAct 迭代 %d/%d, messages=%d, stream=%s",
                        iteration + 1, self.max_iterations, len(messages),
                        self.on_stream is not None and iteration == 0)

            # ── hook: before_iteration ──
            hctx = AgentHookContext(iteration=iteration, messages=messages,
                                    session_key=self._session_key)
            await self.hook.before_iteration(hctx)

            # ── 打印本轮消息上下文 ──
            for i, m in enumerate(messages):
                role = m.get("role", "?")
                content_preview = str(m.get("content", ""))
                tc = m.get("tool_calls")
                tc_hint = f", tool_calls={len(tc)}" if tc else ""
                logger.info("  messages[%d] role=%s content=%s%s", i, role, content_preview, tc_hint)

            # ── LLM 调用 ──
            if iteration == 0 and self.on_stream is not None:
                logger.info(f"_chat_stream_with_tools messages ={messages}")
                response = await self._chat_stream_with_tools(messages, tools, model)
            else:
                logger.info(f"await self.provider.chat messages ={messages}")
                response = await self.provider.chat(
                    messages=messages, model=model,
                    tools=tools.get_definitions() if tools.tool_names else None,
                )

            # ── 错误处理 ──
            if response.finish_reason == "error":
                err = response.usage.get("error", "unknown")
                result.final_content = f"[LLM 调用失败: {err}]"
                result.stop_reason = "error"
                return result

            # ── 工具调用分支 ──
            if response.tool_calls:
                logger.info("response.tool_calls")
                result.had_tool_calls = True
                tool_calls_openai = []
                for tc in response.tool_calls:
                    result.tools_used.append(tc.name)
                    tool_calls_openai.append({
                        "id": tc.id, "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments if isinstance(tc.arguments, str)
                            else json.dumps(tc.arguments),
                        },
                    })
                messages.append(build_assistant_message(
                    response.content, tool_calls=tool_calls_openai,
                ))

                hctx.tool_calls = response.tool_calls
                hctx.response = response
                await self.hook.before_execute_tools(hctx)

                await self.on_progress(f"执行工具: {result.tools_used[-1]}")
                for tc in response.tool_calls:
                    exc_result = await tools.execute(tc.name, tc.arguments)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": exc_result,
                    })
                    logger.info("工具 %s 返回 %d 字符", tc.name, len(exc_result))

                hctx.tool_results = [m.get("content", "") for m in messages if m.get("role") == "tool"]
                hctx.usage = response.usage or {}
                await self.hook.after_iteration(hctx)
                continue

            # ── 无工具调用：最终回复 ──
            messages.append(build_assistant_message(response.content))
            result.final_content = response.content or ""
            result.stop_reason = response.finish_reason
            result.all_messages = messages

            hctx.tool_results = []
            hctx.usage = response.usage or {}
            hctx.response = response
            hctx.final_content = result.final_content
            hctx.stop_reason = result.stop_reason
            await self.hook.after_iteration(hctx)
            return result

        # 超过最大迭代次数
        logger.warning("ReAct 循环达到最大迭代次数 %d", self.max_iterations)
        result.final_content = messages[-1].get("content", "") or "[达到最大工具调用次数]"
        result.stop_reason = "max_iterations"
        return result

    async def _chat_stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        model: str | None,
    ) -> LLMResponse:
        """流式调用 LLM + 可选工具，逐 delta 推送 on_stream。

        chat_stream() 返回 AsyncIterator[StreamDelta]：
        - 普通 text chunk：content 非空
        - 末帧 finish_reason="stop"：结束
        - 末帧 finish_reason="tool_calls"：StreamDelta.tool_calls 内含工具列表

        此方法收集所有 deltas，重新组装为 LLMResponse 返回，
        与 chat() 的调用方接口保持一致。
        """
        buf: list[str] = []
        tool_calls: list[Any] = []
        final_finish_reason: str = "stop"

        try:
            async for delta in self.provider.chat_stream(
                messages=messages,
                model=model,
                tools=tools.get_definitions() if tools.tool_names else None,
            ):
                # 推送普通文本 delta
                if delta.content:
                    buf.append(delta.content)
                    await self.on_stream(delta.content)

                # 检测工具调用（末帧携带）
                if delta.tool_calls:
                    tool_calls = delta.tool_calls

                # 记录结束原因
                if delta.finish_reason:
                    final_finish_reason = delta.finish_reason
        except Exception:
            logger.exception("chat_stream 异常")
            return LLMResponse(
                content="".join(buf) if buf else None,
                finish_reason="error",
                usage={"error": "stream_exception"},
            )

        content = "".join(buf) if buf else None

        # 超时 / 错误处理（与 _state_run 原有逻辑一致）
        if final_finish_reason in ("timeout", "error"):
            if not content:
                err = final_finish_reason
                content = f"[LLM 调用失败: {err}]"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=final_finish_reason,
        )
