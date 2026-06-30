"""Shell 执行工具。"""

import asyncio
from typing import Any

from nanobot.agent.tools.base import Tool


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
