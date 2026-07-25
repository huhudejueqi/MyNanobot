"""配置加载工具。

负责从 ~/.nanobot/config.json 读取并解析配置，
提供 AgentDefaults、ProviderConfig、Config 三个数据类。
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from nanobot.config.schema import Config, _resolve_tool_config_refs

# 默认配置文件路径：用户家目录下的 .nanobot/config.json
# 参考项目也使用同样的默认路径
_DEFAULT_CONFIG_PATH = Path.home() / ".nanobot" / "config.json"

# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None
_schema_refs_ready = False

def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


# @dataclass
# class AgentDefaults:
#     """Agent 默认配置。

#     属性对应 config.json 中 agents.defaults 下的字段。
#     字段名同时支持 camelCase（参考项目格式）和 snake_case。
#     """

#     model: str = "gpt-4o"                      # 使用的模型名称，如 deepseek-v4-flash
#     provider: str = "openai"                    # 服务商名称，如 deepseek、openai
#     max_tokens: int = 8192                       # 每次请求的最大 token 数
#     context_window_tokens: int = 128_000         # 模型上下文窗口大小
#     temperature: float = 0.7                     # 采样温度，越低越确定

#     # 以下字段匹配参考项目结构，供 CLI / loop 使用
#     model_preset: str | None = None              # 活跃 preset 名称
#     timezone: str = "UTC"                        # IANA 时区
#     bot_name: str = "nanobot"                    # CLI 中显示的 bot 名称
#     bot_icon: str = ""                           # CLI 中显示的图标
#     unified_session: bool = False                # 是否跨频道共享 session
#     consolidation_ratio: float = 0.5             # 记忆压缩目标比例
#     disabled_skills: list[str] = field(default_factory=list)


# @dataclass
# class AgentsConfig:
#     """Agent 配置容器，匹配参考项目 agents.* 结构。"""

#     defaults: AgentDefaults = field(default_factory=AgentDefaults)


# @dataclass
# class ProviderConfig:
#     """Provider 连接配置。

#     对应 config.json 中 providers.<name> 下的字段。
#     config.json 中使用 camelCase（apiKey），我们的 loader 同时兼容两种风格。
#     """

#     api_key: str | None = None       # API 密钥
#     api_base: str | None = None      # API 基础地址，如 https://api.deepseek.com


# @dataclass
# class Config:
#     """MyNanobot 顶层配置。

#     聚合 agents 默认参数和所有 providers 的连接信息，
#     同时保留原始 JSON 字典供后续扩展使用。
#     """

#     agents: AgentsConfig = field(default_factory=AgentsConfig)              # Agent 配置
#     providers: dict[str, ProviderConfig] = field(default_factory=dict)       # 所有服务商配置
#     raw: dict[str, Any] = field(default_factory=dict)                        # 原始 JSON 数据

#     def get_active_provider(self) -> tuple[str, ProviderConfig]:
#         """返回当前活跃的 (provider_name, provider_config) 对。

#         根据 agents.defaults.provider 字段的值，
#         到 providers 字典中查找对应的连接信息。
#         如果找不到对应配置，返回一个空 ProviderConfig。
#         """
#         name = self.agents.defaults.provider
#         cfg = self.providers.get(name, ProviderConfig())
#         return name, cfg


# # ---- 加载器 ----

def get_config_path() -> Path:
    """获取配置的路径"""
    if _current_config_path:
        return _current_config_path
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
    global _schema_refs_ready
    if not _schema_refs_ready:
        _resolve_tool_config_refs()
        _schema_refs_ready = True

    path = config_path or get_config_path()
    config = Config()
    if path.exists():
        try:
            with open(path,encoding="utf-8") as f:
                data = json.load(f)
            data = _migrate_config(data)
            config = Config.model_validate(data)
        except (json.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            raise ValueError(f"Failed to load config from {path}: {e}") from e

    _apply_ssrf_whitelist(config)
    return config

def _apply_ssrf_whitelist(config: Config) -> None:
    """从配置文件加载 SSRF 白名单，并应用至网络安全模块"""
    from nanobot.security.network import configure_ssrf_whitelist

    configure_ssrf_whitelist(config.tools.ssrf_whitelist)
    
    # # 文件不存在时返回默认配置，不报错
    # if not path.exists():
    #     return Config()

    # # 读取并解析 JSON
    # with open(path, encoding="utf-8") as f:
    #     data = json.load(f)

    # # 提取 agents.defaults 字段
    # agents_data = data.get("agents", {}).get("defaults", {})
    # # 提取 providers 字典
    # providers_raw = data.get("providers", {})

    # # 构建 AgentDefaults，同时兼容 camelCase 和 snake_case
    # defaults = AgentDefaults(
    #     model=agents_data.get("model", "gpt-4o"),
    #     provider=agents_data.get("provider", "openai"),
    #     # config.json 中用的是 maxTokens（camelCase）
    #     max_tokens=agents_data.get("maxTokens", agents_data.get("max_tokens", 8192)),
    #     # config.json 中用的是 contextWindowTokens（camelCase）
    #     context_window_tokens=agents_data.get(
    #         "contextWindowTokens",
    #         agents_data.get("context_window_tokens", 128_000),
    #     ),
    #     temperature=agents_data.get("temperature", 0.7),
    #     model_preset=agents_data.get("modelPreset", agents_data.get("model_preset")),
    #     timezone=agents_data.get("timezone", "UTC"),
    #     bot_name=agents_data.get("botName", agents_data.get("bot_name", "nanobot")),
    #     bot_icon=agents_data.get("botIcon", agents_data.get("bot_icon", "")),
    #     unified_session=agents_data.get("unifiedSession", agents_data.get("unified_session", False)),
    #     consolidation_ratio=agents_data.get(
    #         "consolidationRatio", agents_data.get("consolidation_ratio", 0.5)
    #     ),
    #     disabled_skills=agents_data.get("disabledSkills", agents_data.get("disabled_skills", [])),
    # )

    # # 遍历所有 providers，提取 apiKey/apiBase
    # providers: dict[str, ProviderConfig] = {}
    # for name, pdata in providers_raw.items():
    #     providers[name] = ProviderConfig(
    #         # config.json 中用的是 apiKey（camelCase），同时兼容 api_key
    #         api_key=pdata.get("apiKey") or pdata.get("api_key"),
    #         api_base=pdata.get("apiBase") or pdata.get("api_base"),
    #     )

    # return Config(
    #     agents=AgentsConfig(defaults=defaults),
    #     providers=providers,
    #     raw=data,
    # )


# ---- 环境变量插值 ----

def save_config(config: Config, config_path: Path | None = None) -> None:
    """保存配置到文件。"""
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json", by_alias=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _env_replace(m: re.Match) -> str:
    name = m.group(1)
    val = os.environ.get(name)
    if val is None:
        raise ValueError(f"环境变量 ${name} 未设置")
    return val


def resolve_config_env_vars(config: Config) -> Config:
    """Return *config* with ``${VAR}`` env-var references resolved.

    Walks in place so extra fields not on Config survive;
    returns the same instance when no references are present.
    Raises ``ValueError`` if a referenced variable is not set.
    """
    return _resolve_in_place(config)


def _resolve_in_place(obj: Any) -> Any:
    if isinstance(obj, str):
        new = _ENV_REF_PATTERN.sub(_env_replace, obj)
        return new if new != obj else obj
    if isinstance(obj, dict):
        resolved = {k: _resolve_in_place(v) for k, v in obj.items()}
        return resolved if any(resolved[k] is not obj[k] for k in obj) else obj
    if isinstance(obj, list):
        resolved = [_resolve_in_place(v) for v in obj]
        return resolved if any(nv is not ov for nv, ov in zip(resolved, obj)) else obj
    # Walk dataclass fields
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import fields as dc_fields

        changed = False
        updates: dict[str, Any] = {}
        for f in dc_fields(obj):
            old = getattr(obj, f.name)
            new = _resolve_in_place(old)
            if new is not old:
                updates[f.name] = new
                changed = True
        if not changed:
            return obj
        return type(obj)(**{**{f.name: getattr(obj, f.name) for f in dc_fields(obj)}, **updates})
    return obj

def _migrate_config(data: dict) -> dict:
    """将旧版配置格式迁移为当前新版格式"""
    # 将 tools.exec.restrictToWorkspace 配置项迁移至 tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    # 如果旧执行配置里存在 restrictToWorkspace，且顶层tools下无该配置，则迁移并删除旧字段
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")

    # 将 tools.myEnabled / tools.mySet 迁移至 tools.my.enable、tools.my.allowSet
    # 说明：最初My工具上线时采用平铺式顶层配置键；现在把它们收拢到my子配置中，
    # 让 web、exec、my 三类配置结构保持统一，同时预留后续扩展字段的空间
    if "myEnabled" in tools or "mySet" in tools:
        # 不存在my子配置则自动创建空字典
        my_cfg = tools.setdefault("my", {})
        # 迁移myEnabled到my.enable，迁移后删除原顶层键
        if "myEnabled" in tools and "enable" not in my_cfg:
            my_cfg["enable"] = tools.pop("myEnabled")
        else:
            tools.pop("myEnabled", None)
        # 迁移mySet到my.allowSet，迁移后删除原顶层键
        if "mySet" in tools and "allowSet" not in my_cfg:
            my_cfg["allowSet"] = tools.pop("mySet")
        else:
            tools.pop("mySet", None)

    return data
