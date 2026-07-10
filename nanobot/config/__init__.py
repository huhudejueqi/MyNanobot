"""配置模块，读取 ~/.nanobot/config.json。"""

from nanobot.config.loader import load_config, get_config_path

__all__ = ["load_config", "get_config_path"]
