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

from nanobot.config.schema import Config, InlineFallbackConfig, ModelPresetConfig
from nanobot.providers.base import LLMProvider
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.registry import create_dynamic_spec, find_by_name

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

def _resolve_model_preset(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> ModelPresetConfig:
    """
    解析模型预设配置（二选一逻辑）
    优先级：直接传入的 preset 对象 > 通过名称从 config 加载预设
    """
    return preset if preset is not None else config.resolve_preset(preset_name)

def _make_provider_core(
    config:Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
    model: str|None = None,
) -> LLMProvider:
    """Create a plain LLM provider without failover wrapping.

    Resolve preset ──► detect backend ──► validate creds ──► instantiate
                                                                                    ┌──────────────┐
                                          ┌──── openai_codex ──────────────────►│OpenAICodex   │
                                          │                                    └──────────────┘
                                          ├──── azure_openai ──────────────────►┌──────────────┐
                                          │                                    │AzureOpenAI   │
                                          │                                    └──────────────┘
                                          ├──── github_copilot ─────────────────►┌──────────────┐
                                          │                                    │GitHubCopilot │
        config.resolve_preset() ──► backend detect ──┼──── anthropic ────────────────────►┌──────────────┐
        config.get_provider_name()                   │                                    │Anthropic     │
        config.get_provider()                        │                                    └──────────────┘
        registry.find_by_name()                      ├──── bedrock ──────────────────────►┌──────────────┐
                                                    │                                    │Bedrock       │
                                                    │                                    └──────────────┘
                                                    └──── openai_compat (else) ──────────►┌────────────────┐
                                                                                         │OpenAICompat    │
                                                                                         └────────────────┘

    Before instantiation, validates api_key / api_base for the detected backend.
    After instantiation, attaches generation settings (temperature, max_tokens, etc.)
    from the resolved preset.
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    model = model or resolved.model
    provider_name = config.get_provider_name(model, preset=resolved)
    p = config.get_provider(model, preset=resolved)
    spec = find_by_name(provider_name) if provider_name else None
    if provider_name and not spec and p:
        if not p.api_base:
            raise ValueError(f"Provider '{provider_name}' requires api_base in config.")
        spec = create_dynamic_spec(provider_name)
    if spec and spec.is_transcription_only:
        raise ValueError(f"Provider '{provider_name}' only supports transcription.")
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not p.api_base:
            raise ValueError("Azure OpenAI requires api_base in config.")
    elif (
        backend == "openai_compat"
        and spec
        and spec.is_direct
        and not spec.default_api_base
        and not (p and p.api_base)
    ):
        raise ValueError(f"Provider '{provider_name}' requires api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key or "",
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "github_copilot":
        from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model, preset=resolved),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    elif backend == "bedrock":
        from nanobot.providers.bedrock_provider import BedrockProvider

        provider = BedrockProvider(
            api_key=p.api_key if p else None,
            api_base=p.api_base if p else None,
            default_model=model,
            region=getattr(p, "region", None) if p else None,
            profile=getattr(p, "profile", None) if p else None,
            extra_body=p.extra_body if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model, preset=resolved),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
            extra_body=p.extra_body if p else None,
            api_type=p.api_type if p and provider_name == "openai" else "auto",
            extra_query=p.extra_query if p else None,
        )

    provider.generation = resolved.to_generation_settings()
    return provider

def provider_signature(
    config: Config,
    *,
    preset_name: str | None = None,
    preset: ModelPresetConfig | None = None,
) -> tuple[object, ...]:
    """从 Config 中提取影响 provider 行为的字段签名。

    签名用于判断 provider 配置是否发生了变化。
    如果签名不变，可以复用已有的 provider 实例。
    参考项目中 AgentLoop 用此签名来做热切换检测。
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    p = config.get_provider(resolved.model, preset=resolved)
    fallback_presets = _resolve_fallback_presets(config, resolved)

    def _fallback_signature(fallback: ModelPresetConfig) -> tuple[object, ...]:
        fp = config.get_provider(fallback.model, preset=fallback)
        return (
            fallback.model,
            fallback.provider,
            config.get_provider_name(fallback.model, preset=fallback),
            config.get_api_key(fallback.model, preset=fallback),
            config.get_api_base(fallback.model, preset=fallback),
            fp.extra_headers if fp else None,
            fp.extra_body if fp else None,
            fp.api_type if fp else "auto",
            fp.extra_query if fp else None,
            getattr(fp, "region", None) if fp else None,
            getattr(fp, "profile", None) if fp else None,
            fallback.max_tokens,
            fallback.temperature,
            fallback.reasoning_effort,
            fallback.context_window_tokens,
        )

    return (
        resolved.model,
        resolved.provider,
        config.get_provider_name(resolved.model, preset=resolved),
        config.get_api_key(resolved.model, preset=resolved),
        config.get_api_base(resolved.model, preset=resolved),
        p.extra_headers if p else None,
        p.extra_body if p else None,
        p.api_type if p else "auto",
        p.extra_query if p else None,
        getattr(p, "region", None) if p else None,
        getattr(p, "profile", None) if p else None,
        resolved.max_tokens,
        resolved.temperature,
        resolved.reasoning_effort,
        resolved.context_window_tokens,
        tuple(_fallback_signature(fallback) for fallback in fallback_presets),
    )



# def _make_provider(config: Config) -> LLMProvider:
#     """根据 Config 创建 LLM provider 实例。

#     根据 config 中配置的服务商名称选择对应的 provider 实现。
#     目前支持 deepseek 和通用 OpenAI 兼容格式。
#     后续可在此扩展对 Anthropic、Google 等 provider 的支持。
#     """
#     name, pcfg = config.get_active_provider()
#     api_key = pcfg.api_key  # 用户 API 密钥
#     api_base = pcfg.api_base  # API 基础地址

#     if name == "deepseek":
#         # DeepSeek 使用 OpenAI 兼容协议，走同一个 provider
#         from nanobot.providers.openai_compat_provider import OpenAICompatProvider

#         return OpenAICompatProvider(
#             api_key=api_key,
#             api_base=api_base or "https://api.deepseek.com",
#         )

#     # 默认走 OpenAI 兼容（支持 OpenAI、DeepSeek、vLLM、Ollama 等）
#     from nanobot.providers.openai_compat_provider import OpenAICompatProvider

#     return OpenAICompatProvider(
#         api_key=api_key,
#         api_base=api_base or "https://api.openai.com/v1",
#     )


def build_provider_snapshot(
    config: Config,
    *,
    preset_name:str|None=None,
    preset:ModelPresetConfig|None = None
                            
) -> ProviderSnapshot:
    """从 Config 构建 ProviderSnapshot。

    这是从配置到快照的标准转换流程：
    Config → Provider 实例 → ProviderSnapshot

    Args:
        config: 已加载的配置对象

    Returns:
        包含已初始化 provider 的不可变快照
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    # provider = _make_provider(config)
    # model = config.agents.defaults.model
    fallback_windows = [
            fallback.context_window_tokens
            for fallback in _resolve_fallback_presets(config, resolved)
        ]
    return ProviderSnapshot(
        provider=make_provider(config, preset=resolved),
        model=resolved.model,
        context_window_tokens=min([resolved.context_window_tokens, *fallback_windows]),
        signature=provider_signature(config, preset=resolved),
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

def _inline_fallback_preset(
    primary: ModelPresetConfig,
    fallback: InlineFallbackConfig,
) -> ModelPresetConfig:
    return ModelPresetConfig(
        model=fallback.model,
        provider=fallback.provider,
        max_tokens=fallback.max_tokens if fallback.max_tokens is not None else primary.max_tokens,
        context_window_tokens=(
            fallback.context_window_tokens
            if fallback.context_window_tokens is not None
            else primary.context_window_tokens
        ),
        temperature=(
            fallback.temperature if fallback.temperature is not None else primary.temperature
        ),
        reasoning_effort=fallback.reasoning_effort,
    )

def _resolve_fallback_presets(config: Config, primary: ModelPresetConfig) -> list[ModelPresetConfig]:
    presets: list[ModelPresetConfig] = []
    for fallback in config.agents.defaults.fallback_models:
        if isinstance(fallback, str):
            presets.append(config.model_presets[fallback])
        else:
            presets.append(_inline_fallback_preset(primary, fallback))
    return presets


def make_provider(
    config:Config,
    *,
    preset_name: str|None = None,
    preset: ModelPresetConfig| None = None,
    model: str | None = None,
) -> LLMProvider:
    """
    根据配置创建对应的大模型服务提供者。

    若传入了 model 参数，该值会覆盖从预设解析得到的模型名称——该逻辑用于故障降级流程，
    为兜底备选模型创建对应的服务提供者实例。
    """
    resolved = _resolve_model_preset(config, preset_name=preset_name, preset=preset)
    provider = _make_provider_core(config, preset_name=preset_name, preset=preset, model=model)
    fallback_presets = _resolve_fallback_presets(config, resolved)

    if fallback_presets:
        provider = FallbackProvider(
            primary=provider,
            fallback_presets=fallback_presets,
            provider_factory=lambda fb: _make_provider_core(
                config, preset_name=preset_name, preset=fb
            ),
        )

    return provider