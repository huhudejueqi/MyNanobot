"""重启通知辅助工具：通过环境变量在进程间传递重启完成消息。

重启流程：
  1. 旧进程在退出前调用 set_restart_notice_to_env() 写入环境变量
  2. 新进程启动后调用 consume_restart_notice_from_env() 读取并清除
  3. should_show_cli_restart_notice() 判断是否在当前 CLI 会话中展示

这样重启完成后 CLI 或频道可以显示一条"重启完成，耗时 2.3s"，
用户知道 bot 已经恢复运行。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

# ── 环境变量名 ────────────────────────────────────────────────────
RESTART_NOTIFY_CHANNEL_ENV = "NANOBOT_RESTART_NOTIFY_CHANNEL"       # 频道标识
RESTART_NOTIFY_CHAT_ID_ENV = "NANOBOT_RESTART_NOTIFY_CHAT_ID"       # 对话 ID
RESTART_NOTIFY_METADATA_ENV = "NANOBOT_RESTART_NOTIFY_METADATA"     # 附加元数据（JSON）
RESTART_STARTED_AT_ENV = "NANOBOT_RESTART_STARTED_AT"               # 开始重启的时间戳


@dataclass(frozen=True)
class RestartNotice:
    """重启通知的数据结构。

    属性：
      channel:       频道类型（如 "cli"、"telegram"）
      chat_id:       频道内对话 ID
      started_at_raw: 开始重启的时间戳字符串
      metadata:      附加元数据字典
    """
    channel: str
    chat_id: str
    started_at_raw: str
    metadata: dict[str, Any] = field(default_factory=dict)


def format_restart_completed_message(started_at_raw: str) -> str:
    """构建重启完成通知文本，如果提供了开始时间则附带耗时。

    参数：
      started_at_raw: 开始重启的时间戳字符串

    返回：
      通知文本，如 "Restart completed in 2.3s."
    """
    elapsed_suffix = ""
    if started_at_raw:
        with suppress(ValueError):
            elapsed_s = max(0.0, time.time() - float(started_at_raw))
            elapsed_suffix = f" in {elapsed_s:.1f}s"
    return f"Restart completed{elapsed_suffix}."


def set_restart_notice_to_env(
    *, channel: str, chat_id: str, metadata: dict[str, Any] | None = None,
) -> None:
    """将重启通知信息写入环境变量，供新进程读取。

    应在旧进程退出前调用。

    参数：
      channel:  频道标识
      chat_id:  对话 ID
      metadata: 附加元数据（可选）
    """
    os.environ[RESTART_NOTIFY_CHANNEL_ENV] = channel
    os.environ[RESTART_NOTIFY_CHAT_ID_ENV] = chat_id
    os.environ[RESTART_STARTED_AT_ENV] = str(time.time())
    if metadata:
        try:
            os.environ[RESTART_NOTIFY_METADATA_ENV] = json.dumps(metadata, default=str)
        except (TypeError, ValueError):
            os.environ.pop(RESTART_NOTIFY_METADATA_ENV, None)
    else:
        os.environ.pop(RESTART_NOTIFY_METADATA_ENV, None)


def consume_restart_notice_from_env() -> RestartNotice | None:
    """读取并清除环境变量中的重启通知。

    新进程启动时调用一次，读取后即清除环境变量，避免多次触发。

    返回：
      有通知时返回 RestartNotice，没有时返回 None
    """
    channel = os.environ.pop(RESTART_NOTIFY_CHANNEL_ENV, "").strip()
    chat_id = os.environ.pop(RESTART_NOTIFY_CHAT_ID_ENV, "").strip()
    started_at_raw = os.environ.pop(RESTART_STARTED_AT_ENV, "").strip()
    metadata_raw = os.environ.pop(RESTART_NOTIFY_METADATA_ENV, "").strip()
    if not (channel and chat_id):
        return None
    metadata: dict[str, Any] = {}
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            metadata = parsed
    return RestartNotice(
        channel=channel,
        chat_id=chat_id,
        started_at_raw=started_at_raw,
        metadata=metadata,
    )


def should_show_cli_restart_notice(notice: RestartNotice, session_id: str) -> bool:
    """判断重启通知是否应在当前 CLI 会话中展示。

    规则：只展示给触发重启的那个 CLI 会话。

    参数：
      notice:     重启通知
      session_id: 当前 CLI 会话 ID

    返回：
      True 表示应展示
    """
    if notice.channel != "cli":
        return False
    if ":" in session_id:
        _, cli_chat_id = session_id.split(":", 1)
    else:
        cli_chat_id = session_id
    return not notice.chat_id or notice.chat_id == cli_chat_id
