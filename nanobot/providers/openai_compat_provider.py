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

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


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
            # 发送 HTTP 请求，超时 120 秒
            async with httpx.AsyncClient(timeout=120.0, proxy=None, trust_env=False) as client:
                resp = await client.post(
                    f"{self.api_base.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=body,
                )
                # 检查 HTTP 状态码
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
