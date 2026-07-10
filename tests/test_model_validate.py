"""演示 Pydantic model_validate 的核心用法。"""

from nanobot.config.schema import AgentDefaults, Config, ModelPresetConfig


def test_basic_parse_from_dict():
    """从 dict 解析单个模型，不存在的字段用默认值。"""
    data = {"model": "gpt-4o", "maxTokens": 16384}
    cfg = AgentDefaults.model_validate(data)

    assert cfg.model == "gpt-4o"
    assert cfg.max_tokens == 16384
    # 没传的字段走 schema 默认值
    assert cfg.temperature == 0.1
    assert cfg.provider == "auto"


def test_camelCase_alias():
    """config.json 里的 camelCase 自动映射到 Python snake_case。"""
    data = {"maxTokens": 4096, "contextWindowTokens": 65536}
    cfg = AgentDefaults.model_validate(data)

    assert cfg.max_tokens == 4096
    assert cfg.context_window_tokens == 65536


def test_snake_case_also_works():
    """Python 风格的 snake_case 也认。"""
    data = {"max_tokens": 9999, "context_window_tokens": 88888}
    cfg = AgentDefaults.model_validate(data)

    assert cfg.max_tokens == 9999
    assert cfg.context_window_tokens == 88888


def test_full_config_tree():
    """Config.model_validate 递归解析整棵树。"""
    data = {
        "agents": {
            "defaults": {
                "model": "deepseek-chat",
                "provider": "deepseek",
                "maxTokens": 4096,
                "temperature": 0.7,
                "timezone": "Asia/Shanghai",
            }
        },
        "providers": {
            "deepseek": {
                "apiKey": "sk-xxx",
                "apiBase": "https://api.deepseek.com",
            }
        },
        "modelPresets": {
            "fast": {
                "model": "deepseek/deepseek-chat",
                "maxTokens": 2048,
                "temperature": 0.5,
            }
        },
    }
    config = Config.model_validate(data)

    # agents.defaults
    defaults = config.agents.defaults
    assert defaults.model == "deepseek-chat"
    assert defaults.provider == "deepseek"
    assert defaults.max_tokens == 4096
    assert defaults.temperature == 0.7
    assert defaults.timezone == "Asia/Shanghai"
    assert defaults.bot_name == "nanobot"  # 没传，走默认

    # providers
    assert config.providers.deepseek.api_key == "sk-xxx"
    assert config.providers.deepseek.api_base == "https://api.deepseek.com"

    # modelPresets
    assert "fast" in config.model_presets
    assert config.model_presets["fast"].model == "deepseek/deepseek-chat"
    assert config.model_presets["fast"].max_tokens == 2048


def test_type_validation():
    """类型不对会抛 ValidationError。"""
    import pydantic

    try:
        AgentDefaults.model_validate({"maxTokens": "not_a_number"})
        assert False, "应该抛异常"
    except pydantic.ValidationError as e:
        assert "maxTokens" in str(e)


def test_extra_field_ignored_by_default():
    """多余的字段默认被忽略（BaseSettings 行为）。"""
    cfg = AgentDefaults.model_validate({"model": "gpt-4o", "someUnknownField": "blah"})
    assert cfg.model == "gpt-4o"


def test_model_preset_validation():
    """用 / 前缀自动推导 provider。"""
    preset = ModelPresetConfig.model_validate({
        "model": "anthropic/claude-sonnet-4-5",
        "provider": "auto",
    })
    assert preset.model == "anthropic/claude-sonnet-4-5"
    assert preset.provider == "auto"
