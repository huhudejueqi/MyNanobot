"""OpenAI 兼容 API 的 LLM provider 实现。

支持所有使用 OpenAI Chat Completions 格式的 API，包括：
- DeepSeek API（api_base 设为 https://api.deepseek.com）
- 标准 OpenAI API（默认）
- 本地部署的 vLLM、Ollama 等

参考项目中使用了多种 provider 实现（Anthropic、Bedrock 等），
这里我们用一个通用的 OpenAI 兼容实现覆盖大多数场景。
"""

from typing import Any

import httpx

from collections.abc import AsyncIterator

from nanobot.providers.base import LLMProvider, LLMResponse, StreamDelta, ToolCallRequest


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
        """初始化 provider。

        Args:
            api_key: API 密钥（可选，如 Ollama 本地部署不需要）
            api_base: API 基础地址，默认为 OpenAI 官方
        """
        super().__init__(api_key=api_key, api_base=api_base)

    def get_default_model(self) -> str:
        """返回默认模型名称。

        注意：DeepSeek 场景下应由配置指定模型名，
        此默认值仅作为降级保底。
        """
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
        """发送聊天补全请求到 OpenAI 兼容 API。

        构建符合 OpenAI Chat Completions 格式的请求体，
        使用 httpx 发送异步 HTTP 请求，并统一错误处理。
        """
        model = model or self.get_default_model()

        # 构建请求体，与 OpenAI API 格式保持一致
        body: dict[str, Any] = {
            "model": model,  # 模型名称
            "messages": messages,  # 消息列表
            "max_tokens": max_tokens,  # 最大生成 token 数
            "temperature": temperature,  # 采样温度
        }
        # 如果有工具定义，附加到请求中
        if tools:
            body["tools"] = tools
        # 如果有工具选择策略，附加到请求中
        if tool_choice:
            body["tool_choice"] = tool_choice

        # 构建请求头
        headers = {
            "Content-Type": "application/json",
        }
        # 有 API key 时添加认证头（本地模型可能不需要）
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # 分段超时：连接 30s，读取 300s
            timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
                resp = await client.post(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=body,
                )
                # 检查 HTTP 状态码，4xx/5xx 抛 HTTPStatusError，被外层 except 捕获
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            # 请求超时，返回自定义错误
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": "timeout"},  # 标记为超时错误
            )
        except httpx.ConnectError:
            # 连接失败（如网络不通、地址错误）
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": "connection_failed"},  # 标记为连接失败
            )
        except httpx.HTTPStatusError as e:
            # API 返回非 2xx 状态码
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": f"http_{e.response.status_code}"},
            )
        except Exception as e:
            # 其他非预期错误
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": str(e)[:100]},
            )

        # 解析响应体
        # OpenAI 格式: data.choices[0].message.{content,tool_calls}
        choice = data["choices"][0]
        message = choice.get("message", {})

        # 提取文本回复
        content = message.get("content")
        # 提取工具调用
        tool_calls_data = message.get("tool_calls")
        tool_calls: list[ToolCallRequest] = []

        if tool_calls_data:
            # 遍历所有工具调用请求，解析为统一的 ToolCallRequest
            for tc in tool_calls_data:
                fn = tc["function"]
                # 参数可能是 JSON 字符串或已经是 dict
                arguments = fn.get("arguments", "")
                tool_calls.append(
                    ToolCallRequest(
                        id=tc["id"],  # 工具调用 ID
                        name=fn["name"],  # 工具名称
                        arguments=arguments,  # 参数
                    )
                )

        # 提取 token 用量统计
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
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamDelta]:
        """流式聊天补全，使用 SSE（Server-Sent Events）逐 chunk 返回。

        与 chat() 的区别：
        - chat() 发送 stream=False，等服务端生成完才一次性返回完整响应
        - chat_stream() 发送 stream=True，服务端边生成边推送 SSE 事件，
          每个事件携带一小段文本增量（delta），调用方可以逐 chunk 渲染，
          用户体验为"打字机效果"

        SSE 协议格式（服务端返回的每一行）：
            data: {"choices":[{"delta":{"content":"你好"},"finish_reason":null}]}

            data: {"choices":[{"delta":{"content":"！"},"finish_reason":null}]}

            data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

            data: [DONE]


        Yields:
            StreamDelta: 每次 yield 一个增量片段，包含 content 和 finish_reason
        """
        import json as _json

        model = model or self.get_default_model()

        # 构建请求体：关键区别是 stream=True
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,  # 告诉服务端用 SSE 流式返回
        }

        # 请求头：与非流式请求一致
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base.rstrip('/')}/chat/completions"

        try:
            # 分段超时：连接 30s，chunk 间等待 300s（模型可能思考很久才开始输出）
            timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False) as client:
                # client.stream() 不会一次性下载整个响应体，
                # 而是建立连接后通过 aiter_lines() 逐行读取 SSE 事件
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    # 检查 HTTP 状态码，4xx/5xx 直接抛异常，不走后面的解析
                    resp.raise_for_status()

                    # 逐行读取 SSE 事件流
                    async for line in resp.aiter_lines():
                        # SSE 格式：每行以 "data: " 前缀开头
                        # 空行和非 data 行（如 event:、id:）跳过
                        if not line or not line.startswith("data: "):
                            continue

                        # 去掉 "data: " 前缀，得到 JSON 载荷
                        payload = line[6:].strip()

                        # 服务端发送 "[DONE]" 表示流式输出结束
                        if payload == "[DONE]":
                            break

                        try:
                            # 解析 JSON 载荷，提取增量文本和结束原因
                            chunk = _json.loads(payload)

                            # OpenAI 格式：choices[0].delta.content 是本次增量文本
                            # choices[0].delta 是本次变化的部分（非完整消息）
                            delta = chunk.get("choices", [{}])[0].get("delta", {})

                            # finish_reason 非空时表示生成结束（stop/tool_calls 等）
                            # 只有最后一个 chunk 会携带此字段
                            finish = chunk["choices"][0].get("finish_reason")

                            # 提取增量文本，可能为空（如首个 chunk 只含 role 信息）
                            content_text = delta.get("content") or ""

                            # 有内容或有结束原因时才 yield，过滤空 chunk
                            if content_text or finish:
                                yield StreamDelta(content=content_text, finish_reason=finish)

                        except (_json.JSONDecodeError, KeyError, IndexError):
                            # 解析失败（格式异常、字段缺失）静默跳过，不中断流
                            continue

        except httpx.TimeoutException:
            # 请求超时（120 秒内没有新 chunk 到达）
            yield StreamDelta(content="", finish_reason="timeout")
        except (httpx.ConnectError, httpx.HTTPStatusError):
            # 连接失败 或 服务端返回非 2xx 状态码
            yield StreamDelta(content="", finish_reason="error")
        except Exception:
            # 其他未预期的异常，兜底处理
            yield StreamDelta(content="", finish_reason="error")
