"""OpenAI 兼容 API 的 LLM provider 实现。

支持所有使用 OpenAI Chat Completions 格式的 API，包括：
- DeepSeek API（api_base 设为 https://api.deepseek.com）
- 标准 OpenAI API（默认）
- 本地部署的 vLLM、Ollama 等

参考项目中使用了多种 provider 实现（Anthropic、Bedrock 等），
这里我们用一个通用的 OpenAI 兼容实现覆盖大多数场景。
"""

from typing import Any
import logging
import httpx

from collections.abc import AsyncIterator

from nanobot.providers.base import LLMProvider, LLMResponse, StreamDelta, ToolCallRequest
logger = logging.getLogger("nanobot.agent.loop")

class OpenAICompatProvider(LLMProvider):
    """兼容 OpenAI API 格式的 LLM provider。

    通过配置不同的 api_base 来支持各种 OpenAI 兼容服务：
    - https://api.openai.com/v1 — 官方 OpenAI
    - https://api.deepseek.com — DeepSeek
    - http://localhost:8000/v1 — 本地 vLLM
    - http://localhost:11434/v1 — Ollama
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
    ):
        super().__init__(api_key=api_key, api_base=api_base)

    def get_default_model(self) -> str:
        return "gpt-4o"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.get_default_model()

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            timeout = httpx.Timeout(connect=120.0, read=300.0, write=120.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
                resp = await client.post(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": "timeout"},
            )
        except httpx.ConnectError:
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": "connection_failed"},
            )
        except httpx.HTTPStatusError as e:
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": f"http_{e.response.status_code}"},
            )
        except Exception as e:
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": str(e)[:100]},
            )

        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content")
        tool_calls_data = message.get("tool_calls")
        tool_calls: list[ToolCallRequest] = []

        if tool_calls_data:
            for tc in tool_calls_data:
                fn = tc["function"]
                arguments = fn.get("arguments", "")
                tool_calls.append(
                    ToolCallRequest(
                        id=tc["id"],
                        name=fn["name"],
                        arguments=arguments,
                    )
                )

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamDelta]:
        """流式聊天补全，使用 SSE 逐 chunk 返回。

        支持流式模式下的工具调用检测：
        如果 finish_reason="tool_calls"，末帧的 tool_calls 字段
        携带完整的工具调用列表。
        """
        import json as _json

        model = model or self.get_default_model()

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base.rstrip('/')}/chat/completions"

        try:
            timeout = httpx.Timeout(connect=120.0, read=300.0, write=120.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    resp.raise_for_status()

                    # 流式 tool calls 需要跨 chunk 积累
                    # 每个 tool_call 由 index 标识，不同 chunk 补充同一 index 的参数
                    # 格式：首个 chunk 带 id/name，后续 chunk 只带 arguments 增量
                    pending_tool_calls: dict[int, dict[str, Any]] = {}

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break

                        try:
                            chunk = _json.loads(payload)
                        except _json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [{}])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        finish = choices[0].get("finish_reason")

                        content_text = delta.get("content") or ""

                        # 处理流式 tool_calls：在 delta 中逐步构建
                        tc_delta = delta.get("tool_calls")
                        if tc_delta is not None:
                            for item in tc_delta:
                                idx = item.get("index", 0)
                                if idx not in pending_tool_calls:
                                    pending_tool_calls[idx] = {
                                        "id": item.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": "",
                                            "arguments": "",
                                        },
                                    }
                                tc = pending_tool_calls[idx]
                                func = item.get("function", {})
                                if item.get("id"):
                                    tc["id"] = item["id"]
                                if func.get("name"):
                                    tc["function"]["name"] = func["name"]
                                if func.get("arguments"):
                                    tc["function"]["arguments"] += func["arguments"]

                        # 如果是末帧且 finish_reason="tool_calls"，将累积的工具调用输出
                        if finish == "tool_calls" and pending_tool_calls:
                            tool_calls: list[ToolCallRequest] = []
                            for idx in sorted(pending_tool_calls.keys()):
                                tc = pending_tool_calls[idx]
                                fn = tc.get("function", {})
                                tool_calls.append(ToolCallRequest(
                                    id=tc.get("id", ""),
                                    name=fn.get("name", ""),
                                    arguments=fn.get("arguments", ""),
                                ))
                            yield StreamDelta(
                                content="",
                                finish_reason="tool_calls",
                                tool_calls=tool_calls,
                            )
                            pending_tool_calls.clear()
                            continue

                        # 有内容或有结束原因时才 yield
                        if content_text or finish:
                            yield StreamDelta(content=content_text, finish_reason=finish)

        except httpx.TimeoutException:
            yield StreamDelta(content="", finish_reason="timeout")
        except (httpx.ConnectError, httpx.HTTPStatusError):
            yield StreamDelta(content="", finish_reason="error")
        except Exception:
            yield StreamDelta(content="", finish_reason="error")
