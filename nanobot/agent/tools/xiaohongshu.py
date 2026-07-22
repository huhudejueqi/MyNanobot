"""小红书 CLI 工具 — 包装 xiaohongshu-skills 的 CLI 命令。

依赖：
  - ~/workspace/xiaohongshu-skills/ 项目已 clone
  - uv 已安装（~/.local/bin/uv）
  - Chrome 扩展已连接（bridge server 会自动启动）
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    NumberSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

XHS_PROJECT = Path.home() / "workspace/xiaohongshu-skills"
UV_BIN = Path.home() / ".local/bin/uv"
CLI_SCRIPT = XHS_PROJECT / "scripts/cli.py"

_SUBCOMMANDS = [
    "check-login", "search-feeds", "get-feed-detail", "list-feeds",
    "user-profile", "like-feed", "favorite-feed", "post-comment",
    "reply-comment",
]

_SUBCOMMAND_HELP = {
    "check-login": "检查小红书登录状态",
    "search-feeds": "搜索笔记（--keyword 关键词, --sort 排序, --note-type 类型）",
    "get-feed-detail": "查看笔记详情（--feed-id, --xsec-token）",
    "list-feeds": "首页推荐流",
    "user-profile": "用户主页（--user-id）",
    "like-feed": "点赞/取消（--feed-id, --xsec-token, --cancel）",
    "favorite-feed": "收藏/取消（--feed-id, --xsec-token, --cancel）",
    "post-comment": "评论（--feed-id, --xsec-token, --content）",
    "reply-comment": "回复评论（--feed-id, --xsec-token, --comment-id, --content）",
}


@tool_parameters(
    tool_parameters_schema(
        subcommand=StringSchema(
            f"子命令: {', '.join(_SUBCOMMANDS)}",
            enum=_SUBCOMMANDS,
        ),
        args=StringSchema(
            "CLI 参数，如 '--keyword \"北京旅游\" --sort hot'",
        ),
        required=["subcommand"],
    ),
)
class XiaohongshuTool(Tool):
    """操作小红书：搜索笔记、查看详情、点赞收藏等。

    需要 Chrome 浏览器运行中并已安装扩展，首次使用会自动启动 bridge server。
    """

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls()

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return XHS_PROJECT.exists() and UV_BIN.exists()

    @property
    def name(self) -> str:
        return "xiaohongshu"

    @property
    def description(self) -> str:
        return (
            "搜索小红书内容、查看笔记详情、点赞收藏等。"
            "子命令: " + ", ".join(
                f"{k}({v})" for k, v in _SUBCOMMAND_HELP.items()
            )
        )

    async def execute(self, subcommand: str, args: str = "", **kwargs: Any) -> str:
        if subcommand not in _SUBCOMMANDS:
            return f"Error: 不支持的子命令 '{subcommand}'，可用: {', '.join(_SUBCOMMANDS)}"

        cmd = [
            str(UV_BIN), "run", "python", str(CLI_SCRIPT),
            subcommand,
            *shlex.split(args),
        ]

        env = os.environ.copy()
        env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"

        try:
            result = subprocess.run(
                cmd,
                cwd=str(XHS_PROJECT),
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return "Error: 操作超时（120s）"

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            return f"Error ({result.returncode}): {stderr}"

        # Try to parse JSON output
        stdout = result.stdout.strip()
        try:
            data = json.loads(stdout)
            return json.dumps(data, ensure_ascii=False, indent=2)[:5000]
        except (json.JSONDecodeError, ValueError):
            return stdout[:3000]
