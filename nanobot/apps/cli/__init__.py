"""统一应用域对应的命令行应用适配器。"""

from nanobot.apps.cli.service import (
    CliAppError,
    CliAppManager,
    CliAppsRuntimeConfig,
)

__all__ = [
    "CliAppError",
    "CliAppManager",
    "CliAppsRuntimeConfig",
]
