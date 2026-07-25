"""消息总线的事件类型。"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# OutboundMessage.metadata 可选键，用于存放结构化、不依赖渠道的UI交互载荷
OUTBOUND_META_AGENT_UI = "_agent_ui"

# 仅内部使用的入站消息元数据标识，供进程内通信渠道使用
INBOUND_META_RUNTIME_CONTROL = "_runtime_control"
RUNTIME_CONTROL_ACK = "_ack"
RUNTIME_CONTROL_MCP_RELOAD = "mcp_reload"


@dataclass
class InboundMessage:
    """从聊天频道收到的入站消息。"""

    channel: str       # 频道标识，如 telegram、discord、cli
    sender_id: str     # 发送者唯一 ID
    chat_id: str       # 会话/群组 ID
    content: str       # 消息文本
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)        # 媒体资源 URL 列表
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展数据
    session_key_override: str | None = None  # 可选：手动指定会话唯一标识

    @property
    def session_key(self) -> str:
        """会话唯一标识，默认按「频道:会话ID」拼接。"""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """需要发送到聊天频道的出站回复。"""

    channel: str                    # 目标频道
    chat_id: str                    # 目标会话 ID
    content: str                    # 回复文本
    reply_to: str | None = None     # 回复的原消息 ID
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    buttons: list[list[str]] = field(default_factory=list)  # 按钮布局
