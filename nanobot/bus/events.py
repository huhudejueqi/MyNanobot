"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# OutboundMessage.metadata 可选键，用于存放结构化、不依赖渠道的UI交互载荷
# 【通俗解释】
# 渠道：Codex、Claude、Discord、网页面板等不同客户端；
# UI交互载荷：用来渲染按钮、可视化面板的附加配置数据；
# 不依赖渠道：数据格式通用，高级客户端渲染界面，简易客户端直接忽略不报错；
# 【举例】metadata={"_agent_ui":{"kind":"scene_panel","buttons":["打开UE","重新生成场景"]}}
# 对应值必须支持JSON序列化，且至少包含 kind 字段；功能完善的客户端会渲染这份UI数据，
# 其他简易通信渠道可直接忽略不认识的字段，不会影响基础文字消息展示
OUTBOUND_META_AGENT_UI = "_agent_ui"

# 仅内部使用的入站消息元数据标识，供进程内通信渠道使用
# 作用：无需经过用户会话，直接通知Agent主循环更新运行时状态
INBOUND_META_RUNTIME_CONTROL = "_runtime_control"
# 运行时控制指令：确认应答
RUNTIME_CONTROL_ACK = "_ack"
# 运行时控制指令：重载MCP服务/配置
RUNTIME_CONTROL_MCP_RELOAD = "mcp_reload"

@dataclass
class InboundMessage:
    """从聊天通信渠道接收的消息（入站消息）"""

    channel: str  # 通信渠道标识：如 telegram、discord、slack、whatsapp
    sender_id: str  # 发送者唯一用户ID
    chat_id: str  # 会话/群组频道唯一ID
    content: str  # 消息文本内容
    timestamp: datetime = field(default_factory=datetime.now)  # 消息时间戳，默认创建对象时自动赋值当前时间
    media: list[str] = field(default_factory=list)  # 媒体资源URL列表（图片/文件等）
    metadata: dict[str, Any] = field(default_factory=dict)  # 渠道自定义扩展数据字典
    session_key_override: str | None = None  # 可选：手动指定会话唯一标识，用于覆盖默认生成规则

    @property
    def session_key(self) -> str:
        """会话唯一标识，用于区分独立对话上下文"""
        # 优先使用手动指定的会话key，无自定义则按「渠道:会话ID」拼接生成
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """需要发送至聊天渠道的消息（出站回复消息）

    metadata 字典可承载多种扩展数据：
    1. 路由信息：message_id 消息ID等
    2. 追踪标记：_progress 进度标识等
    3. 富UI载荷：可存放 OUTBOUND_META_AGENT_UI 结构化数据，专供网页客户端渲染交互界面
    【通俗解释】富UI载荷是通用界面配置，高级客户端显示按钮/面板，简易客户端直接忽略该数据
    【举例】生成场景完成后携带UI面板：
    OutboundMessage(
        channel="codex",
        chat_id="xxx",
        content="森林营地场景生成完成",
        metadata={
            "_agent_ui": {
                "kind": "scene_panel",
                "scene_name": "林间营地",
                "buttons": ["打开UE查看", "修改灯光", "重新生成"]
            }
        }
    )
    非网页类客户端会自动忽略无法识别的metadata字段
    """

    channel: str  # 目标发送渠道
    chat_id: str  # 目标会话/群组ID
    content: str  # 回复文本内容
    reply_to: str | None = None  # 需回复的原消息ID，无则为空
    media: list[str] = field(default_factory=list)  # 附带媒体文件URL列表
    metadata: dict[str, Any] = field(default_factory=dict)  # 扩展元数据（路由、进度、富UI数据等）
    buttons: list[list[str]] = field(default_factory=list)  # 交互按钮二维数组，内层为同一行按钮集合