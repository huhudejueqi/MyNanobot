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
        """执行 ReAct 循环。

        首轮 LLM 调用尝试流式（若 on_stream 非空），
        工具后续轮次走非流式 chat()。
        """
        result = AgentRunResult()
        run_hctx = AgentRunHookContext(messages=messages)
        await self.hook.before_run(run_hctx)
        if not self.on_progress:
            async def noop(*a, **kw): pass
            self.on_progress = noop

        iteration = 0
        while iteration < self.max_iterations:
            logger.info("ReAct 迭代 %d/%d, messages=%d, stream=%s",
                        iteration + 1, self.max_iterations, len(messages),
                        self.on_stream is not None and iteration == 0)
            
            # ── hook: before_iteration ──
            hctx = AgentHookContext(iteration=iteration, messages=messages,
                                    session_key=getattr(self, '_session_key', None))
            await self.hook.before_iteration(hctx)

            # ── 首次调用且要求流式：走 chat_stream() ──
            if iteration == 0 and self.on_stream is not None:
                response = await self._chat_stream_with_tools(
                    messages, tools, model,
                )
            else:
                # 非首轮或非流式模式：走普通 chat()
                response = await self.provider.chat(
                    messages=messages,
                    model=model,
                    tools=tools.get_definitions() if tools.tool_names else None,
                )

            # ── 错误处理 ──
            if response.finish_reason == "error":
                err = response.usage.get("error", "unknown")
                result.final_content = f"[LLM 调用失败: {err}]"
                result.stop_reason = "error"
                run_hctx.error = result.final_content
                await self.hook.on_error(run_hctx)
                return result

            # ── 构建 assistant 消息 ──
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.content:
                assistant_msg["content"] = response.content
            else:
                assistant_msg["content"] = None

            # ── 处理工具调用 ──
            if response.tool_calls:
                result.had_tool_calls = True
                tool_calls_openai = []
                for tc in response.tool_calls:
                    result.tools_used.append(tc.name)
                    tool_calls_openai.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments if isinstance(tc.arguments, str)
                            else json.dumps(tc.arguments),
                        },
                    })
                assistant_msg["tool_calls"] = tool_calls_openai
                messages.append(assistant_msg)

                # ── hook: before_execute_tools ──
                hctx.tool_calls = response.tool_calls
                hctx.response = response
                await self.hook.before_execute_tools(hctx)
                
                # 执行每个工具
                await self.on_progress(f"执行工具: {result.tools_used[-1]}")
                for tc in response.tool_calls:
                    exc_result = await tools.execute(tc.name, tc.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": exc_result,
                    })
                    logger.info("工具 %s 返回 %d 字符", tc.name, len(exc_result))

                iteration += 1
                # ── hook: after_iteration ──
                hctx.tool_results = [m.get("content", "") for m in messages if m.get("role") == "tool"]
                hctx.usage = response.usage or {}
                await self.hook.after_iteration(hctx)

                continue  # 继续下一轮 ReAct 迭代（后续轮次不流式）

            # ── 无工具调用：最终回复 ──
            messages.append(assistant_msg)
            result.final_content = response.content or ""
            result.stop_reason = response.finish_reason
            result.all_messages = messages

            # ── hook: after_iteration ──
            hctx.tool_results = []
            hctx.usage = response.usage or {}
            hctx.response = response
            hctx.final_content = result.final_content
            hctx.stop_reason = result.stop_reason
            await self.hook.after_iteration(hctx)

            # ── hook: after_run ──
            run_hctx.final_content = result.final_content
            run_hctx.tools_used = result.tools_used
            run_hctx.stop_reason = result.stop_reason
            await self.hook.after_run(run_hctx)

            # 首轮流式结束时通知渲染器

            return result

        # 超过最大迭代次数
        logger.warning("ReAct 循环达到最大迭代次数 %d", self.max_iterations)
        result.final_content = messages[-1].get("content", "") or "[达到最大工具调用次数]"
        result.stop_reason = "max_iterations"
#             await self.on_stream_end(resuming=False)
        run_hctx.final_content = result.final_content
        run_hctx.tools_used = result.tools_used
        run_hctx.stop_reason = result.stop_reason
        await self.hook.after_run(run_hctx)
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
