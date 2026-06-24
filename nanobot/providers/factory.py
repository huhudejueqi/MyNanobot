"""LLM Provider 工厂与快照。

负责根据配置创建 Provider 实例，并提供不可变的 ProviderSnapshot。
快照（Snapshot）模式是参考项目中的关键设计：
将 provider、model、context_window 等信息打包成不可变对象，
方便在运行时安全切换模型配置而不会影响正在进行的请求。

典型用法：
    snapshot = build_provider_snapshot(config)
    # snapshot 后续可传递给 AgentLoop.apply_snapshot()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.config.loader import Config, ProviderConfig, load_config
from nanobot.providers.base import LLMProvider


@dataclass(frozen=True)
class ProviderSnapshot:
    """Provider 快照：不可变的数据传输对象。

    将 provider、model 等信息冻结在一个对象中，
    确保运行时切换配置时不会出现中间状态不一致的问题。
    参考项目中 AgentLoop 使用此快照来管理模型切换。

    Attributes:
        provider: 已实例化的 LLM provider
        model: 使用的模型名称
        context_window_tokens: 模型上下文窗口大小
        signature: 配置签名，用于判断配置是否发生变化
    """

    provider: LLMProvider  # 已初始化的 LLM provider 实例
    model: str  # 模型名称，如 deepseek-v4-flash
    context_window_tokens: int  # 上下文窗口大小
    signature: tuple[object, ...]  # 配置签名元组，用于检测变更


def _provider_signature(config: Config) -> tuple[object, ...]:
    """从 Config 中提取影响 provider 行为的字段签名。

    签名用于判断 provider 配置是否发生了变化。
    如果签名不变，可以复用已有的 provider 实例。
    参考项目中 AgentLoop 用此签名来做热切换检测。
    """
    name, pcfg = config.get_active_provider()
    return (
        config.agents.model,  # 模型名称
        name,  # 服务商名称
        pcfg.api_key,  # API 密钥
        pcfg.api_base,  # API 地址
        config.agents.max_tokens,  # 最大 token 数
        config.agents.temperature,  # 温度
        config.agents.context_window_tokens,  # 上下文窗口
    )


def _make_provider(config: Config) -> LLMProvider:
    """根据 Config 创建 LLM provider 实例。

    根据 config 中配置的服务商名称选择对应的 provider 实现。
    目前支持 deepseek 和通用 OpenAI 兼容格式。
    后续可在此扩展对 Anthropic、Google 等 provider 的支持。
    """
    name, pcfg = config.get_active_provider()
    api_key = pcfg.api_key  # 用户 API 密钥
    api_base = pcfg.api_base  # API 基础地址

    if name == "deepseek":
        # DeepSeek 使用 OpenAI 兼容协议，走同一个 provider
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        return OpenAICompatProvider(
            api_key=api_key,
            api_base=api_base or "https://api.deepseek.com",
        )

    # 默认走 OpenAI 兼容（支持 OpenAI、DeepSeek、vLLM、Ollama 等）
    from nanobot.providers.openai_compat_provider import OpenAICompatProvider

    return OpenAICompatProvider(
        api_key=api_key,
        api_base=api_base or "https://api.openai.com/v1",
    )


def build_provider_snapshot(config: Config) -> ProviderSnapshot:
    """从 Config 构建 ProviderSnapshot。

    这是从配置到快照的标准转换流程：
    Config → Provider 实例 → ProviderSnapshot

    Args:
        config: 已加载的配置对象

    Returns:
        包含已初始化 provider 的不可变快照
    """
    provider = _make_provider(config)
    model = config.agents.model

    return ProviderSnapshot(
        provider=provider,
        model=model,
        context_window_tokens=config.agents.context_window_tokens,
        signature=_provider_signature(config),
    )


def load_provider_snapshot(config_path: Path | None = None) -> ProviderSnapshot:
    """从 ~/.nanobot/config.json 加载 ProviderSnapshot。

    便捷方法，适用于快速启动场景。
    内部调用 load_config() 再调用 build_provider_snapshot()。

    Args:
        config_path: 可选的配置文件路径，不传则走默认路径

    Returns:
        包含已初始化 provider 的不可变快照
    """
    config = load_config(config_path)
    return build_provider_snapshot(config)
