"""Shell 执行工具。"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import current_request_session_key
from nanobot.agent.tools.exec_session import (
    DEFAULT_EXEC_SESSION_MANAGER,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_YIELD_MS,
    MAX_OUTPUT_CHARS,
    MAX_YIELD_MS,
    clamp_session_int,
    format_session_poll,
)
from nanobot.agent.tools.sandbox import wrap_command
from nanobot.agent.tools.schema import (
    BooleanSchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.config.paths import get_media_dir
from nanobot.config_base import Base
from nanobot.security.workspace_access import current_scope_allows_loopback, current_tool_workspace
from nanobot.security.workspace_policy import is_path_within

_IS_WINDOWS = sys.platform == "win32"


class ShellTool(Tool):
    """在本地执行 shell 命令并返回输出。"""

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "在本地系统执行 shell 命令，返回标准输出和标准错误。适用于运行脚本、操作文件、查询系统信息等。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间(秒)，默认 30",
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str, timeout: int = 30) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"[命令执行超时 ({timeout}s)]"

            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            result = f"退出码: {proc.returncode}"
            if out:
                result += f"\nSTDOUT:\n{out[:8000]}"
            if err:
                result += f"\nSTDERR:\n{err[:2000]}"
            return result
        except Exception as e:
            return f"[shell 执行失败: {e!s}]"
            
class ExecToolConfig(Base):
    """Shell exec tool configuration."""
    enable: bool = True
    timeout: int = Field(default=60, ge=0)  # Hard timeout (s); 0 = no limit. Not capped by the per-call max.
    path_prepend: str = ""
    path_append: str = ""
    sandbox: str = ""
    allowed_env_keys: list[str] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)
    deny_patterns: list[str] = Field(default_factory=list)
