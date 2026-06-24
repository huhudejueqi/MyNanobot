"""配置加载工具。

负责从 ~/.nanobot/config.json 读取并解析配置，
提供 AgentDefaults、ProviderConfig、Config 三个数据类。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 默认配置文件路径：用户家目录下的 .nanobot/config.json
# 参考项目也使用同样的默认路径
_DEFAULT_CONFIG_PATH = Path.home() / ".nanobot" / "config.json"


@dataclass
class AgentDefaults:
    """Agent 默认配置。

    属性对应 config.json 中 agents.defaults 下的字段。
    字段名同时支持 camelCase（参考项目格式）和 snake_case。
    """

    model: str = "gpt-4o"                      # 使用的模型名称，如 deepseek-v4-flash
    provider: str = "openai"                    # 服务商名称，如 deepseek、openai
    max_tokens: int = 8192                       # 每次请求的最大 token 数
    context_window_tokens: int = 128_000         # 模型上下文窗口大小
    temperature: float = 0.7                     # 采样温度，越低越确定


@dataclass
class ProviderConfig:
    """Provider 连接配置。

    对应 config.json 中 providers.<name> 下的字段。
    config.json 中使用 camelCase（apiKey），我们的 loader 同时兼容两种风格。
    """

    api_key: str | None = None       # API 密钥
    api_base: str | None = None      # API 基础地址，如 https://api.deepseek.com


@dataclass
class Config:
    """MyNanobot 顶层配置。

    聚合 agents 默认参数和所有 providers 的连接信息，
    同时保留原始 JSON 字典供后续扩展使用。
    """

    agents: AgentDefaults = field(default_factory=AgentDefaults)           # Agent 默认参数
    providers: dict[str, ProviderConfig] = field(default_factory=dict)     # 所有服务商配置
    raw: dict[str, Any] = field(default_factory=dict)                      # 原始 JSON 数据

    def get_active_provider(self) -> tuple[str, ProviderConfig]:
        """返回当前活跃的 (provider_name, provider_config) 对。

        根据 agents.defaults.provider 字段的值，
        到 providers 字典中查找对应的连接信息。
        如果找不到对应配置，返回一个空 ProviderConfig。
        """
        name = self.agents.provider
        cfg = self.providers.get(name, ProviderConfig())
        return name, cfg


# ---- 加载器 ----

def get_config_path() -> Path:
    """返回 ~/.nanobot/config.json 的路径。"""
    return _DEFAULT_CONFIG_PATH


def load_config(config_path: Path | None = None) -> Config:
    """从 ~/.nanobot/config.json 加载配置。

    1. 如果文件不存在，返回全部走默认值的 Config
    2. JSON 解析支持 camelCase 和 snake_case 两种字段名
    3. providers 字典中只提取 apiKey/apiBase，其余忽略

    Args:
        config_path: 可选的配置文件路径，不传则走默认 ~/.nanobot/config.json

    Returns:
        解析后的 Config 实例
    """
    path = config_path or get_config_path()

    # 文件不存在时返回默认配置，不报错
    if not path.exists():
        return Config()

    # 读取并解析 JSON
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 提取 agents.defaults 字段
    agents_data = data.get("agents", {}).get("defaults", {})
    # 提取 providers 字典
    providers_raw = data.get("providers", {})

    # 构建 AgentDefaults，同时兼容 camelCase 和 snake_case
    agents = AgentDefaults(
        model=agents_data.get("model", "gpt-4o"),
        provider=agents_data.get("provider", "openai"),
        # config.json 中用的是 maxTokens（camelCase）
        max_tokens=agents_data.get("maxTokens", agents_data.get("max_tokens", 8192)),
        # config.json 中用的是 contextWindowTokens（camelCase）
        context_window_tokens=agents_data.get(
            "contextWindowTokens",
            agents_data.get("context_window_tokens", 128_000),
        ),
        temperature=agents_data.get("temperature", 0.7),
    )

    # 遍历所有 providers，提取 apiKey/apiBase
    providers: dict[str, ProviderConfig] = {}
    for name, pdata in providers_raw.items():
        providers[name] = ProviderConfig(
            # config.json 中用的是 apiKey（camelCase），同时兼容 api_key
            api_key=pdata.get("apiKey") or pdata.get("api_key"),
            api_base=pdata.get("apiBase") or pdata.get("api_base"),
        )

    return Config(
        agents=agents,
        providers=providers,
        raw=data,
    )
