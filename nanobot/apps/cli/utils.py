"""供智能体主循环与配置界面共用的命令行应用工具函数集。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """从元数据中提取 CLI 应用附件的持久化会话参数。"""
    cli_apps = metadata.get("cli_apps") if isinstance(metadata, Mapping) else None
    return {"cli_apps": cli_apps} if isinstance(cli_apps, list) and cli_apps else {}


def runtime_lines(message: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """返回当前轮次中模型可见的 CLI 应用标注信息。"""
    if skip:
        return []
    text = message.content if isinstance(getattr(message, "content", None), str) else ""
    metadata = message.metadata if isinstance(getattr(message, "metadata", None), Mapping) else None
    return _cli_app_runtime_lines(text, metadata, workspace)


def _cli_app_runtime_lines(
    text: str,
    metadata: Mapping[str, Any] | None,
    workspace: Path,
) -> list[str]:
    structured = metadata.get("cli_apps") if isinstance(metadata, Mapping) else None
    if isinstance(structured, list):
        mentions = [
            item for item in structured
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        ]
        if mentions:
            return [
                "CLI App 附件: "
                f"@{str(item['name']).strip().lower()} "
                f"（已安装；工具=run_cli_app；"
                f"入口点={str(item.get('entry_point') or 'unknown')}；"
                f"技能=skills/cli-app-{str(item['name']).strip().lower()}/SKILL.md）。"
                "如有需要请读取对应的技能文件，然后使用 `run_cli_app` 运行此应用；不要用 Shell 绕过。"
                for item in mentions
                if str(item.get("name") or "").strip()
            ]
    if "@" not in text:
        return []
    try:
        from nanobot.apps.cli import CliAppManager

        mentions = CliAppManager(workspace=workspace).mentioned_installed_apps(text)
    except Exception:
        return []
    return [
        "CLI App 提及: "
        f"@{item['name']} "
        f"（已安装；工具={item['tool']}；"
        f"入口点={item['entry_point'] or 'unknown'}；"
        f"技能={item['skill']}）。"
        "如有需要请读取对应的技能文件，然后使用 `run_cli_app` 运行此应用；不要用 Shell 绕过。"
        for item in mentions
    ]
