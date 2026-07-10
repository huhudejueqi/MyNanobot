"""会话管理模块：对话历史记录的存储、加载、截断和持久化。

核心职责：
  - Session 数据类：单次对话的消息列表 + 元数据 + 合并偏移
  - SessionManager 管理器：所有 Session 的 CRUD、磁盘 JSONL 持久化、缓存、修复、fork

数据存储格式：
  磁盘文件为 .jsonl（JSON Lines），首行是 _type=metadata 的元数据记录，
  后续每行一条对话消息。文件路径 <workspace>/sessions/<safe_key>.jsonl。

典型使用流程：
  manager = SessionManager(workspace_path)
  session = manager.get_or_create("telegram:12345")
  session.add_message("user", "你好")
  session.add_message("assistant", "你好！")
  history = session.get_history(max_messages=10)
  manager.save(session)
"""

import json
import os
import re
import shutil
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ── 项目内部模块 ──────────────────────────────────────────────────────
from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import (
    ensure_dir,                    # 确保目录存在，不存在则创建
    estimate_message_tokens,       # 估算一条消息的 token 数
    find_legal_message_start,      # 找到合法消息序列的起始索引（跳过孤立的 tool_result）
    image_placeholder_text,        # 图片路径 → 面包屑占位文本
    safe_filename,                 # 文件名安全化（移除危险字符）
    strip_think,                   # 移除模型 推理标签
)
from nanobot.utils.subagent_channel_display import scrub_subagent_announce_body

# ═══════════════════════════════════════════════════════════════════════
#  全局常量
# ═══════════════════════════════════════════════════════════════════════

# 单个会话文件最大消息条数（软上限，超出后归档旧消息）
FILE_MAX_MESSAGES = 2000

# 正则：匹配消息头部的时间戳前缀行，如 "[Message Time: 2026-07-04T16:05:12]\n"
_MESSAGE_TIME_PREFIX_RE = re.compile(r"^\[Message Time: [^\]]+\]\n?")

# 正则：匹配本地图片面包屑行，如 "[image: /tmp/photo.png]"
_LOCAL_IMAGE_BREADCRUMB_RE = re.compile(r"^\[image: (?:/|~)[^\]]+\]\s*$")

# 正则：匹配模型可能会模仿输出的工具调用回显行，如 "generate_image(...)" 或 "message(...)"
_TOOL_CALL_ECHO_RE = re.compile(r'^\s*(?:generate_image|message)\([^)]*\)\s*$')

# 会话列表中单条消息预览的最大字符数
_SESSION_PREVIEW_MAX_CHARS = 120

# 扫描会话文件预览时最多读取的记录数（防止超大文件导致 OOM）
_SESSION_LIST_PREVIEW_MAX_RECORDS = 200

# 扫描会话文件预览时最多读取的字符数
_SESSION_LIST_PREVIEW_MAX_CHARS = 1_000_000

# fork 会话时需要删除的挥发元数据字段名集合
# 这些字段是运行时临时状态（goal/checkpoint/title），fork 新会话不应继承
_FORK_VOLATILE_METADATA_KEYS = {
    "goal_state",
    "pending_user_turn",
    "runtime_checkpoint",
    "thread_goal",
    "title",
    "title_user_edited",
}


# ═══════════════════════════════════════════════════════════════════════
#  辅助函数（模块内部使用）
# ═══════════════════════════════════════════════════════════════════════

def _sanitize_assistant_replay_text(content: str) -> str:
    """清洗助手消息中的内部回放标记，防止模型在后续对话中模仿输出这些标记。

    问题的背景：
      程序在运行过程中会在消息内容中插入一些内部元数据（时间戳、图片面包屑、
      工具调用回显行等）。这些标记对程序正常运行是有用的，但如果它们出现在
      被回放作为历史示例的 assistant 消息中，模型就会把它们当成对话范本，
      导致模型在后续回复中主动生成类似的元数据标记 —— 这会造成对话污染。

    本函数清除三类标记：
      1. 消息头部的时间戳前缀，如 "[Message Time: 2026-07-04T16:05:12]\n"
      2. 本地图片面包屑行，如 "[image: /tmp/photo.png]"
      3. 工具调用回显行，如 "generate_image(...)" 或 "message(...)"

    参数：
      content: 原始助手消息文本

    返回：
      清洗后的文本，不含上述三类内部标记
    """
    # 移除消息头部的时间戳前缀（只移除第一处出现，避免过度删除）
    content = _MESSAGE_TIME_PREFIX_RE.sub("", content, count=1)
    # 按行拆分，逐行过滤掉图片面包屑和工具调用回显行
    lines = [
        line
        for line in content.splitlines()
        if not _LOCAL_IMAGE_BREADCRUMB_RE.match(line)
        and not _TOOL_CALL_ECHO_RE.match(line)
    ]
    # 用换行符重新拼接，首尾去空白
    return "\n".join(lines).strip()


def _text_preview(content: Any) -> str:
    """将消息内容截取为会话列表用的简短预览文本。

    功能：
      - 兼容纯字符串和多模态消息块数组两种输入格式
      - 自动清洗内部回放标记（时间戳、面包屑等脚手架文字）
      - 压缩连续空白字符为一个空格
      - 超长自动截断（配置常量 _SESSION_PREVIEW_MAX_CHARS=120）

    多模态格式说明：
      content 可能是 [{"type": "text", "text": "你好"}, {"type": "image_url", ...}]
      这种列表结构，本函数只提取 type=text 的文本块拼接为预览。

    参数：
      content: 消息的 content 字段，str 或 list[dict]

    返回：
      精简后的预览文本字符串；无法处理时返回空字符串
    """
    # ── 分支1：纯字符串 ──
    if isinstance(content, str):
        text = content
    # ── 分支2：多模态消息块列表 ──
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = " ".join(parts)
    # ── 分支3：无法处理的类型 ──
    else:
        return ""

    # 清洗内部回放元数据
    text = _sanitize_assistant_replay_text(text)
    # 压缩连续空白（换行/制表/多空格 → 单个空格），首尾去空白
    text = re.sub(r"\s+", " ", text).strip()
    # 超长截断，末尾加省略号
    if len(text) > _SESSION_PREVIEW_MAX_CHARS:
        text = text[: _SESSION_PREVIEW_MAX_CHARS - 1].rstrip() + "…"
    return text


def _message_preview_text(message: dict[str, Any]) -> str:
    """从单条消息字典生成会话列表用的预览文本。

    特殊处理：
      如果消息是子代理注入结果（injected_event == "subagent_result"），
      会额外调用 scrub_subagent_announce_body() 来精简子代理的大段内容，
      避免列表预览被整块子代理输出撑爆。

    参数：
      message: 消息字典，通常包含 role / content 等字段

    返回：
      精简后的预览文本
    """
    content: Any = message.get("content")
    # ── 子代理注入消息的特殊精简 ──
    if message.get("injected_event") == "subagent_result" and isinstance(content, str):
        content = scrub_subagent_announce_body(content)
    # 统一走通用文本预览逻辑
    return _text_preview(content)


def _metadata_title(metadata: Any) -> str:
    """从会话元数据字典中提取会话标题，自动清洗模型推理标签。

    规则：
      - 非 dict 类型 → 返回空字符串
      - title 非字符串 → 返回空字符串
      - title_user_edited == True → 用户手动编辑的标题，原样返回（保留用户原始输入）
      - 其他情况 → 自动生成的 AI 标题，清除 /<thought> 推理标签后返回

    参数：
      metadata: 会话元数据字典

    返回：
      清洗后的标题字符串；无标题时返回空字符串
    """
    if not isinstance(metadata, dict):
        return ""
    title = metadata.get("title")
    if not isinstance(title, str):
        return ""
    # 用户手动编辑过的标题，保留原始内容不动
    if metadata.get("title_user_edited") is True:
        return title
    # AI 自动生成的标题，清除模型推理标签
    return strip_think(title)


# ═══════════════════════════════════════════════════════════════════════
#  Session 数据类 — 存储单次对话的所有数据
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Session:
    """单次会话：存储对话历史、元数据、记忆合并偏移量。

    每个 Session 实例对应一个独立的对话（如某个 Telegram 群聊、某个 WebUI 聊天），
    通过 SessionManager.get_or_create(key) 获取。

    关键字段：
      key              - 会话唯一标识，格式为 "channel:id"，如 "telegram:12345"
      messages         - 消息列表，按时间顺序追加，每元素为一条消息字典
      created_at       - 会话创建时间
      updated_at       - 最后更新时间
      metadata         - 扩展元数据字典（标题、工作空间、goal 状态等）
      last_consolidated - 已合并到记忆文件的条数偏移量

    数据流示意：
      用户输入 → add_message("user", ...) 
              → 调用 LLM
              → add_message("assistant", ...)
              → get_history() 获取处理后的历史（截断、过滤、注入面包屑）
    """

    key: str                                                    # 会话唯一标识，格式 channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)       # 消息列表，按时间顺序追加，每行一条 JSONL
    created_at: datetime = field(default_factory=datetime.now)         # 会话创建时间
    updated_at: datetime = field(default_factory=datetime.now)         # 最后更新时间
    metadata: dict[str, Any] = field(default_factory=dict)             # 扩展元数据（标题、工作空间、goal状态等）
    last_consolidated: int = 0                                          # 已合并到记忆文件的条数偏移量

    def __post_init__(self) -> None:
        """数据类初始化完成后自动执行，校验偏移量合法性，修复损坏数据。

        触发场景：
          - 从磁盘加载 Session 时，metadata 中的 last_consolidated 可能已损坏
          - 损坏原因：旧版本 bug、手动编辑文件、数据迁移不完整等

        非法情况：
          - 存储类型是 bool：历史数据可能把 True/False 错误存成了偏移量
          - 存储类型不是 int：完全不合法
          - 数值超过消息总数或小于 0：会隐藏全部历史记录

        处理方式：
          任何一种非法情况，直接重置为 0，从头读取全部消息。
        """
        if (
            isinstance(self.last_consolidated, bool)            # 场景1：布尔值（历史遗留脏数据）
            or not isinstance(self.last_consolidated, int)      # 场景2：非整数类型
            or not 0 <= self.last_consolidated <= len(self.messages)  # 场景3：越界
        ):
            self.last_consolidated = 0

    @staticmethod
    def _annotate_message_time(message: dict[str, Any], content: Any) -> Any:
        """为消息内容添加时间戳前缀，帮助模型进行相对时间推理。

        设计考量：
          这条时间戳是消息实际产生的时间，而非当前调用 get_history 的时间。
          这样模型可以看到"用户昨天说了一句话"，从而做出更合理的上下文判断。

        安全限制：
          只给 user 角色的消息添加时间戳，不给 assistant 添加。
          原因是：如果每条 assistant 消息也有时间戳前缀，模型会通过上下文学习
          模仿在回复开头也加上 [Message Time: ...] 前缀，这会把内部元数据
          泄露给用户可看到的回复内容。

        参数：
          message: 原始消息字典
          content: 消息内容（可能是字符串或其他格式）

        返回：
          添加了时间戳前缀的内容；如无条件则返回原内容
        """
        timestamp = message.get("timestamp")
        # 没有时间戳或内容不是纯文本 → 直接返回，不加前缀
        if not timestamp or not isinstance(content, str):
            return content
        role = message.get("role")
        # 只给 user 角色加时间戳（assistant 加了会诱导模型模仿）
        if role != "user":
            return content
        # 在用户消息前插入时间戳行
        return f"[Message Time: {timestamp}]\n{content}"

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """追加一条消息到会话末尾。

        每条消息会附带当前时间的 ISO 格式时间戳，便于后续 get_history()
        进行相对时间推理和会话时间线展示。

        参数：
          role:    消息角色，通常是 "user"、"assistant"、"system" 或 "tool"
          content: 消息文本内容
          **kwargs: 额外的消息字段，如：
                    - timestamp: 自定义时间戳（不传则自动生成）
                    - tool_calls: assistant 的工具调用列表
                    - tool_call_id: tool 消息关联的调用 ID
                    - media: 用户上传的媒体文件路径列表
                    - _command: 内部命令标记（会被 get_history 过滤）
                    - reasoning_content: 模型推理过程文本
                    - thinking_blocks: 分块推理内容
                    以及其他需要随消息一起存储的元数据

        示例：
          >>> session.add_message("user", "你好")
          >>> session.add_message("assistant", "有什么可以帮你的？",
          ...     reasoning_content="用户问好，回复问候")
        """
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(
        self,
        max_messages: int = 120,
        *,
        max_tokens: int = 0,
        include_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        """获取加工后的历史消息列表，供 LLM 调用时使用。

        这是整个 Session 最核心的方法，负责将原始消息列表处理为 LLM 可用的格式。
        处理管线如下：

        get_history(max_messages, max_tokens, include_timestamps)
        ├─ 1. 取未合并部分 messages[last_consolidated:]
        ├─ 2. 按 max_messages 截尾（只保留末尾 N 条）
        ├─ 3. 对齐到 user turn 开头（如果 user 前一条是 _channel_delivery 则一并保留）
        ├─ 4. 丢弃前端孤立的 tool_result
        └─ 5-7. 逐条处理消息:
            ├─ 过滤掉 _command 内部命令
            ├─ 注入面包屑（media → [image:路径], cli_apps → [CLI App], mcp_presets → [MCP Preset]）
            ├─ 可选注入时间戳前缀
            └─ 跳过空内容的 assistant（除非有 tool_calls / reasoning_content）
        └─ 8. 如果 max_tokens > 0，从尾部开始按 token 预算剪裁
            ├─ 重新对齐到 user turn 开头
            └─ 丢弃新产生的孤立 tool_result

        参数：
          max_messages:      最大返回消息条数，默认 120
          max_tokens:        token 预算上限（0 表示不限制），超出从尾部剪裁
          include_timestamps: 是否在 user 消息前添加 [Message Time: ...] 时间戳前缀

        返回：
          处理后的消息字典列表，可直接传入 LLM API 的 messages 参数
        """
        # ── 步骤1：取未合并部分 ──────────────────────────────────────
        # last_consolidated 是已合并到长期记忆的偏移量，之前的部分不需要给 LLM
        unconsolidated = self.messages[self.last_consolidated:]

        # ── 步骤2：按 max_messages 截尾（只保留末尾 N 条） ───────────
        max_messages = max_messages if max_messages > 0 else 120
        sliced = unconsolidated[-max_messages:]

        # ── 步骤3：对齐到 user turn 开头 ──────────────────────────────
        # 避免从 assistant 消息中间开始，导致 LLM 看到不完整的对话段
        # 特殊处理：如果 user 消息前面紧跟着一条 _channel_delivery（频道主动推送，
        # 如 Newsletter 推送），则把那条推送也包含进来，保证上下文完整
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                start = i
                if i > 0 and sliced[i - 1].get("_channel_delivery"):
                    start = i - 1
                sliced = sliced[start:]
                break

        # ── 步骤4：丢弃前端孤立的 tool_result ────────────────────────
        # tool_result 必须对应前面某条 assistant 消息的 tool_call，
        # 如果截断后只剩下 tool_result 而没有对应的 tool_call，则丢弃
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        # ── 步骤5-7：逐条处理消息 ─────────────────────────────────────
        out: list[dict[str, Any]] = []
        for message in sliced:
            # ── 5a. 过滤内部命令消息 ──
            if message.get("_command"):
                continue

            content = message.get("content", "")
            role = message.get("role")

            # ── 5b. 清洗 assistant 消息中的内部回放标记 ──
            if role == "assistant" and isinstance(content, str):
                content = _sanitize_assistant_replay_text(content)

            # ── 6. 注入面包屑：让 LLM 知道图片/CLI/MCP 附件曾经存在 ──
            # 设计目的：附件（图片、CLI 工具、MCP 预设）在被序列化为 JSONL 时，
            # 真正的二进制数据不会存入消息 content。回放时如果用户消息只有文字，
            # LLM 就不知道之前有过附件。面包屑文本让 LLM 至少知道附件存在。
            #
            # 比如用户发了 "看看这张图" + 一张图片，JSONL 存的是
            # {content: "看看这张图", media: ["/tmp/photo.png"]}
            # 回放时 get_history 会注入成：
            # {content: "看看这张图\n[image: /tmp/photo.png]"}

            # ── 图片面包屑 ──
            media = message.get("media")
            if role == "user" and isinstance(media, list) and media and isinstance(content, str):
                breadcrumbs = "\n".join(
                    image_placeholder_text(p) for p in media if isinstance(p, str) and p
                )
                content = f"{content}\n{breadcrumbs}" if content else breadcrumbs

            # ── CLI 应用面包屑 ──
            cli_apps = message.get("cli_apps")
            if role == "user" and isinstance(cli_apps, list) and cli_apps and isinstance(content, str):
                cli_lines: list[str] = []
                for item in cli_apps[:8]:  # 最多 8 个，防止面包屑过长
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    if not name:
                        continue
                    entry = str(item.get("entry_point") or "unknown").strip() or "unknown"
                    cli_lines.append(
                        f"[CLI App Attachment: @{name}; tool=run_cli_app; entry_point={entry}; "
                        f"skill=skills/cli-app-{name}/SKILL.md]"
                    )
                if cli_lines:
                    breadcrumbs = "\n".join(cli_lines)
                    content = f"{content}\n{breadcrumbs}" if content else breadcrumbs

            # ── MCP 预设面包屑 ──
            mcp_presets = message.get("mcp_presets")
            if (
                role == "user"
                and isinstance(mcp_presets, list)
                and mcp_presets
                and isinstance(content, str)
            ):
                mcp_lines: list[str] = []
                for item in mcp_presets[:8]:  # 最多 8 个
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    if not name:
                        continue
                    transport = str(item.get("transport") or "mcp").strip() or "mcp"
                    mcp_lines.append(
                        f"[MCP Preset Attachment: @{name}; tool_prefix=mcp_{name}_; "
                        f"transport={transport}]"
                    )
                if mcp_lines:
                    breadcrumbs = "\n".join(mcp_lines)
                    content = f"{content}\n{breadcrumbs}" if content else breadcrumbs

            # ── 7a. 可选注入时间戳 ──
            if include_timestamps:
                content = self._annotate_message_time(message, content)

            # ── 7b. 跳过空内容的 assistant ──
            # 有些 assistant 消息内容为空，只有 tool_calls（工具调用请求），
            # 这些消息如果不跳过，会被 LLM 当成一条只有 content="" 的冗余消息
            # 但如果它有 reasoning_content / thinking_blocks，即使内容为空也要保留
            if role == "assistant" and isinstance(content, str) and not content.strip():
                if not any(key in message for key in ("tool_calls", "reasoning_content", "thinking_blocks")):
                    continue

            # ── 构造输出消息字典 ──
            entry: dict[str, Any] = {"role": message["role"], "content": content}
            # 保留 LLM 需要但非每次必有的字段
            for key in ("tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)

        # ── 步骤8：按 token 预算从尾部剪裁 ────────────────────────────
        if max_tokens > 0 and out:
            kept: list[dict[str, Any]] = []
            used = 0
            # 从尾部向前遍历，累加 token 直到超过预算
            for message in reversed(out):
                tokens = estimate_message_tokens(message)
                if kept and used + tokens > max_tokens:
                    break
                kept.append(message)
                used += tokens
            kept.reverse()

            # ── 8a. 剪裁后重新对齐到 user turn 开头 ──
            first_user = next((i for i, m in enumerate(kept) if m.get("role") == "user"), None)
            if first_user is not None:
                kept = kept[first_user:]
            else:
                # 预算太紧，可能只剩下 assistant 尾巴
                # 从原始列表中找回最近的 user turn
                recovered_user = next(
                    (i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"),
                    None,
                )
                if recovered_user is not None:
                    kept = out[recovered_user:]

            # ── 8b. 丢弃新产生的孤立 tool_result ──
            # cut 边界处可能有 tool_result 没有对应的 tool_call
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
            out = kept

        return out

    def clear(self) -> None:
        """清空所有消息，重置会话到初始状态。

        效果：
          - 清空 messages 列表
          - last_consolidated 重置为 0
          - updated_at 更新为当前时间
          - 删除 _last_summary 缓存（如果有）
        """
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()
        self.metadata.pop("_last_summary", None)

    def retain_recent_legal_suffix(
        self,
        max_messages: int,
        *,
        extend_to_user: bool = False,
    ) -> tuple[list[dict], int]:
        """保留最近 N 条合法对话片段，丢弃前面的旧消息。

        这个方法是 enforce_file_cap 的核心实现，用于当消息数量超过上限时，
        丢弃旧消息只保留最近的部分。它比简单的切片更智能：
          - 确保开头是合法的消息类型（user 开头，无孤立 tool_result）
          - 可选扩展到最近的 user turn（避免从 assistant 中间截断）
          - 返回被丢弃的消息列表，供调用方归档

        参数：
          max_messages: 要保留的最大消息条数
          extend_to_user: True 时尝试扩展到最近一条 user 消息开头

        返回：
          (dropped, already_consolidated_count)
          - dropped: 被丢弃的消息列表（按原始顺序）
          - already_consolidated_count: 丢弃消息中属于已合并部分的数量
            （这些消息已在 last_consolidated 之前，不需要再次归档到记忆文件）
        """
        # ── max_messages <= 0：全部丢弃 ──
        if max_messages <= 0:
            dropped = list(self.messages)
            lc = self.last_consolidated
            self.clear()
            return dropped, min(lc, len(dropped))

        # ── 没超限，不需要丢弃 ──
        if len(self.messages) <= max_messages:
            return [], 0

        original = list(self.messages)
        before_lc = self.last_consolidated

        # ── 计算保留的起始索引 ──
        start_idx = max(0, len(self.messages) - max_messages)
        if extend_to_user:
            # 往前找最近的 user 消息作为起始点
            start_idx = next(
                (i for i in range(start_idx, -1, -1) if self.messages[i].get("role") == "user"),
                start_idx,
            )

        retained = self.messages[start_idx:]

        # ── 确保保留段以 user 消息开头 ──
        first_user = next((i for i, m in enumerate(retained) if m.get("role") == "user"), None)
        if first_user is not None:
            retained = retained[first_user:]
        elif not extend_to_user:
            # 截尾后如果全是 assistant/tool，从全量会话中找最近的 user 并以其为起点
            latest_user = next(
                (i for i in range(len(self.messages) - 1, -1, -1)
                 if self.messages[i].get("role") == "user"),
                None,
            )
            if latest_user is not None:
                retained = self.messages[latest_user: latest_user + max_messages]

        # ── 丢弃保留段前端的孤立 tool_result ──
        start = find_legal_message_start(retained)
        if start:
            retained = retained[start:]

        # ── 硬上限保证（除非调用方要求扩展到 user turn） ──
        if not extend_to_user and len(retained) > max_messages:
            retained = retained[-max_messages:]
            start = find_legal_message_start(retained)
            if start:
                retained = retained[start:]

        # ── 计算实际丢弃的消息（通过对象 id 比对，确保不重复不遗漏） ──
        retained_ids = set(id(m) for m in retained)
        dropped = [m for m in original if id(m) not in retained_ids]

        # 统计丢弃消息中有多少条在合并区域内（这些不需要再次归档）
        already_consolidated = sum(
            1 for i, m in enumerate(original)
            if i < before_lc and id(m) not in retained_ids
        )

        # 计算新的 last_consolidated（保留段中原来在合并区域内的消息数）
        new_lc = sum(
            1 for i, m in enumerate(original)
            if i < before_lc and id(m) in retained_ids
        )

        # ── 更新状态 ──
        self.messages = retained
        self.last_consolidated = new_lc
        self.updated_at = datetime.now()
        return dropped, already_consolidated

    def enforce_file_cap(
        self,
        on_archive: Any = None,
        limit: int = FILE_MAX_MESSAGES,
    ) -> None:
        """检查消息总数是否超过上限，超出则归档旧消息并截断。

        这是会话文件大小的"安全阀"，在每次 save() 时自动调用（由 SessionManager 触发）。
        当消息数超过 limit 时：
          1. 调用 retain_recent_legal_suffix() 截断到 limit 条
          2. 已合并部分的旧消息不需要重复归档
          3. 未合并部分的旧消息通过 on_archive 回调传给调用方处理

        参数：
          on_archive: 回调函数，接收被丢弃的消息列表作为参数
                      （通常是写入记忆文件或同步到外部存储）
          limit:      消息总数上限，默认 FILE_MAX_MESSAGES (2000)
        """
        if limit <= 0 or len(self.messages) <= limit:
            return

        dropped, already_consolidated = self.retain_recent_legal_suffix(limit)
        if not dropped:
            return

        # 只归档未合并部分的旧消息
        archive_chunk = dropped[already_consolidated:]
        if archive_chunk and on_archive:
            on_archive(archive_chunk)

        logger.info(
            "Session file cap hit for {}: dropped {}, raw-archived {}, kept {}",
            self.key,
            len(dropped),
            len(archive_chunk),
            len(self.messages),
        )


# ═══════════════════════════════════════════════════════════════════════
#  SessionManager 类 — 管理所有 Session 的生命周期
# ═══════════════════════════════════════════════════════════════════════

class SessionManager:
    """会话管理器：负责所有 Session 的创建、加载、缓存、持久化和删除。

    核心职责：
      - get_or_create(key):        获取或创建 Session（内存缓存 + 磁盘加载）
      - save(session):             保存 Session 到磁盘 JSONL 文件
      - delete_session(key):       删除会话文件和缓存
      - fork_session_before_user_index(): 基于已有会话创建分支（fork）
      - list_sessions():           列出所有会话（含预览和元数据）
      - flush_all():               关闭前持久化所有缓存会话
      - read_session_file():       只读方式加载会话文件（不缓存）

    文件存储结构：
      <workspace>/sessions/
        telegram_12345.jsonl    ← 首行 metadata，后续每行一条消息
        discord_67890.jsonl

    缓存机制：
      get_or_create 的结果会缓存到 self._cache 字典，后续同 key 请求直接返回。
      save() 会同时更新磁盘和缓存。
      invalidate() 可主动清除缓存。

    数据安全：
      save() 使用原子写入（先写 .tmp，再 os.replace），避免写入中断导致文件损坏。
      fsync=True 时同步刷盘，适用于优雅关闭场景。
    """

    def __init__(self, workspace: Path):
        """初始化会话管理器。

        参数：
          workspace: 工作区目录路径，会话 JSONL 文件保存在 <workspace>/sessions/ 下
        """
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        # 旧版全局会话目录（~/.nanobot/sessions/），用于迁移数据
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        # 内存缓存：key → Session 对象
        self._cache: dict[str, Session] = {}

    @staticmethod
    def safe_key(key: str) -> str:
        """将会话 key 转换为安全的文件名。

        key 格式为 "channel:id"，例如 "telegram:12345"。
        文件名中的冒号会转义为下划线，其他特殊字符也会被 safe_filename 处理。

        这个方法是公开的，供 HTTP 处理器等外部调用方生成稳定的文件名映射。

        参数：
          key: 原始会话 key，如 "telegram:12345"

        返回：
          安全的文件名字符串（不含路径），如 "telegram_12345"
        """
        return safe_filename(key.replace(":", "_"))

    def _get_session_path(self, key: str) -> Path:
        """构造会话 JSONL 文件的完整路径。

        参数：
          key: 会话标识

        返回：
          Path 对象，如 <workspace>/sessions/telegram_12345.jsonl
        """
        return self.sessions_dir / f"{self.safe_key(key)}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """构造旧版全局会话目录中的文件路径。

        用于在加载新路径文件不存在时，自动从旧路径迁移数据。

        参数：
          key: 会话标识

        返回：
          Path 对象，如 ~/.nanobot/sessions/telegram_12345.jsonl
        """
        return self.legacy_sessions_dir / f"{self.safe_key(key)}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """获取已有会话，不存在则创建新的空会话。

        查找优先级：内存缓存 → 磁盘文件 → 新建

        参数：
          key: 会话唯一标识，格式 "channel:id"

        返回：
          Session 实例
        """
        # 1. 内存缓存命中 → 直接返回
        if key in self._cache:
            return self._cache[key]

        # 2. 尝试从磁盘加载
        session = self._load(key)
        if session is None:
            # 3. 磁盘也不存在 → 创建全新空 Session
            session = Session(key=key)

        # 4. 写入缓存后返回
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """从磁盘 JSONL 文件加载会话。

        加载流程：
          1. 拼接新路径 <workspace>/sessions/<key>.jsonl
          2. 如果新路径不存在，尝试旧路径 <legacy_dir>/<key>.jsonl
             旧路径有文件则自动移动到新路径（迁移操作）
          3. 新旧都不存在 → 返回 None
          4. 读取文件，首行 _type=metadata 解析元数据，其余行解析为消息
          5. 构造 Session 对象返回
          6. 解析失败 → 尝试 _repair() 修复

        参数：
          key: 会话标识

        返回：
          成功返回 Session 对象，文件不存在或修复失败返回 None
        """
        path = self._get_session_path(key)

        # ── 尝试从旧版路径迁移 ──
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        # 路径仍然不存在 → 返回 None
        if not path.exists():
            return None

        # ── 读取 JSONL 文件 ──
        try:
            messages = []
            metadata = {}
            created_at = None
            updated_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    # 首行 _type=metadata 是元数据记录
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        # 其余行是消息记录
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            # 解析失败 → 尝试修复
            logger.warning("Failed to load session {}: {}", key, e)
            repaired = self._repair(key)
            if repaired is not None:
                logger.info("Recovered session {} from corrupt file ({} messages)", key, len(repaired.messages))
            return repaired

    def _repair(self, key: str) -> Session | None:
        """尝试从损坏的 JSONL 文件中恢复会话数据。

        损坏的可能原因：
          - 磁盘写入中断导致部分行不完整
          - JSON 格式错误（某一行解析失败）
          - 元数据丢失或不完整

        恢复策略：
          逐行解析，跳过所有 JSON 格式错误的行。
          如果能读取到至少一条消息或元数据，就返回恢复后的 Session。
          如果完全无法读取，返回 None。

        参数：
          key: 会话标识

        返回：
          成功返回恢复后的 Session，失败返回 None
        """
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0
            skipped = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # 这一行损坏了，跳过
                        skipped += 1
                        continue

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        if data.get("created_at"):
                            with suppress(ValueError, TypeError):
                                created_at = datetime.fromisoformat(data["created_at"])
                        if data.get("updated_at"):
                            with suppress(ValueError, TypeError):
                                updated_at = datetime.fromisoformat(data["updated_at"])
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            if skipped:
                logger.warning("Skipped {} corrupt lines in session {}", skipped, key)

            # 如果没有任何数据可恢复，返回 None
            if not messages and not metadata:
                return None

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Repair failed for session {}: {}", key, e)
            return None

    @staticmethod
    def _session_payload(session: Session) -> dict[str, Any]:
        """将会话对象序列化为字典，供只读接口返回。

        参数：
          session: Session 实例

        返回：
          包含所有会话数据的字典
        """
        return {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "messages": session.messages,
        }

    def save(self, session: Session, *, fsync: bool = False) -> None:
        """将会话持久化到磁盘 JSONL 文件。

        写入策略（原子写入）：
          1. 先写临时文件 <key>.jsonl.tmp
          2. 写入完成后通过 os.replace() 原子替换目标文件
          3. 如果写入过程中程序崩溃，.tmp 文件在下次启动时被忽略
             （只有 .jsonl 文件才是有效的）

        文件格式：
          第一行：{"_type": "metadata", "key": ..., "created_at": ..., ...}
          后续行：{"role": "user", "content": "你好", "timestamp": "..."}
          后续行：{"role": "assistant", "content": "回复", ...}

        fsync 参数：
          默认 False（操作系统页缓存即可满足正常运行需求）。
          在优雅关闭时需要设为 True，强制刷盘确保数据不丢，
          特别是文件系统使用 write-back 缓存时（如 rclone VFS、NFS、FUSE 挂载）。

        参数：
          session: 要保存的 Session 实例
          fsync:   True 时执行 fsync 刷盘，适用于优雅关闭流程
        """
        path = self._get_session_path(session.key)
        tmp_path = path.with_suffix(".jsonl.tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # ── 写入元数据行（首行） ──
                metadata_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "last_consolidated": session.last_consolidated#整数索引，记录上一轮记忆合并到了哪条消息
                }
                f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")

                # ── 逐条写入消息 ──
                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

                # ── 刷盘（可选） ──
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())

            # ── 原子替换 ──
            os.replace(tmp_path, path)

            # ── 目录 fsync（可选） ──
            if fsync:
                # Windows 上以 O_RDONLY 打开目录会报 PermissionError，跳过
                with suppress(PermissionError):
                    fd = os.open(str(path.parent), os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)

        except BaseException:
            # 写入过程中出错，清理临时文件
            tmp_path.unlink(missing_ok=True)
            raise

        # ── 更新缓存 ──
        self._cache[session.key] = session

    def flush_all(self) -> int:
        """关闭前持久化所有缓存会话（带 fsync 刷盘）。

        在整个程序关闭时调用，确保所有尚未持久化的缓存数据都写入磁盘。
        每个会话独立 try-except，单个会话写入失败不影响其他会话。

        返回：
          成功刷盘的会话数量
        """
        flushed = 0
        for key, session in list(self._cache.items()):
            try:
                self.save(session, fsync=True)
                flushed += 1
            except Exception:
                logger.warning("Failed to flush session {}", key, exc_info=True)
        return flushed

    def invalidate(self, key: str) -> None:
        """从内存缓存中移除指定会话。

        移除缓存不会删除磁盘文件。
        下次 get_or_create 时会重新从磁盘加载。

        参数：
          key: 会话标识
        """
        self._cache.pop(key, None)

    def delete_session(self, key: str) -> bool:
        """删除会话文件和缓存。

        从磁盘删除 JSONL 文件，同时从内存缓存中移除。
        删除后该会话的所有数据不可恢复。

        参数：
          key: 会话标识

        返回：
          True 表示文件存在并被删除，False 表示文件不存在
        """
        path = self._get_session_path(key)
        self.invalidate(key)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.warning("Failed to delete session file {}: {}", path, e)
            return False

    def fork_session_before_user_index(
        self,
        source_key: str,
        target_key: str,
        before_user_index: int,
    ) -> Session | None:
        """基于源会话，在指定用户消息之前创建分支会话。

        这个操作用于"回退到某条消息之前"的场景：
          - WebUI 用户想从某个历史节点重新开始对话
          - 用户想保留之前的上下文但换个方向继续

        参数：
          source_key:   源会话标识
          target_key:   目标（新）会话标识
          before_user_index: 用户消息索引（从 0 开始）
            - 0 表示"第一条用户消息之前"（复制空会话）
            - N 表示"第 N+1 条用户消息之前"（复制前 N 条用户消息及之前的对话）
            - 等于总用户消息数时复制整个会话

        分支时自动移除的挥发元数据：
          goal_state / pending_user_turn / runtime_checkpoint /
          thread_goal / title / title_user_edited

        返回：
          成功返回新创建的 Session，失败返回 None
        """
        if before_user_index < 0:
            return None

        # ── 获取源会话 ──
        source = self._cache.get(source_key) or self._load(source_key)
        if source is None:
            return None

        # ── 遍历到指定索引位置，复制消息 ──
        copied: list[dict[str, Any]] = []
        user_index = 0
        found_target = False
        for message in source.messages:
            if message.get("role") == "user":
                if user_index == before_user_index:
                    found_target = True
                    break
                user_index += 1
            copied.append(deepcopy(message))

        # 如果 before_user_index 等于总 user 消息数，也认为是合法目标
        if user_index == before_user_index:
            found_target = True

        if not found_target:
            return None

        # ── 复制元数据，移除挥发字段 ──
        metadata = deepcopy(source.metadata)
        for key in _FORK_VOLATILE_METADATA_KEYS:
            metadata.pop(key, None)

        # ── 处理合并偏移量 ──
        last_consolidated = min(source.last_consolidated, len(copied))
        if source.last_consolidated > len(copied):
            # 如果合并点超出了复制范围，清除摘要缓存
            metadata.pop("_last_summary", None)
            last_consolidated = 0

        # ── 创建新会话 ──
        now = datetime.now()
        target = Session(
            key=target_key,
            messages=copied,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            last_consolidated=last_consolidated,
        )
        self.save(target, fsync=True)
        return target

    def read_session_file(self, key: str) -> dict[str, Any] | None:
        """只读方式加载会话数据（不缓存，不修改内存状态）。

        与 get_or_create 的区别：
          get_or_create 会缓存 Session 对象到内存，适合频繁读写。
          read_session_file 每次从磁盘读取，适合只读展示场景（如 WebUI 查看会话）。

        返回的字典结构：
          {
            "key": "telegram:12345",
            "created_at": "2026-07-04T10:00:00",
            "updated_at": "2026-07-04T16:00:00",
            "metadata": {...},
            "messages": [{...}, {...}]
          }

        参数：
          key: 会话标识

        返回：
          包含会话数据的字典，文件不存在或解析失败返回 None
        """
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: str | None = None
            updated_at: str | None = None
            stored_key: str | None = None

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = data.get("created_at")
                        updated_at = data.get("updated_at")
                        stored_key = data.get("key")
                    else:
                        messages.append(data)

            return {
                "key": stored_key or key,
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": metadata,
                "messages": messages,
            }
        except Exception as e:
            logger.warning("Failed to read session {}: {}", key, e)
            repaired = self._repair(key)
            if repaired is not None:
                logger.info("Recovered read-only session view {} from corrupt file", key)
                return self._session_payload(repaired)
            return None

    def read_session_metadata(self, key: str) -> dict[str, Any] | None:
        """只读方式加载会话的元数据部分（跳过消息列表）。

        相比 read_session_file 更轻量，因为不加载全部消息。
        适用于 WebUI 中只需要展示会话列表标题/时间，不需要全部消息的场景。

        返回的字典结构：
          {
            "key": "telegram:12345",
            "created_at": "2026-07-04T10:00:00",
            "updated_at": "2026-07-04T16:00:00",
            "metadata": {...},
          }

        参数：
          key: 会话标识

        返回：
          包含元数据的字典，文件不存在或解析失败返回 None
        """
        path = self._get_session_path(key)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") != "metadata":
                        # 首行不是 metadata（文件格式异常），直接返回 None
                        return None
                    metadata = data.get("metadata", {})
                    return {
                        "key": data.get("key") or key,
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "metadata": metadata if isinstance(metadata, dict) else {},
                    }
            return None
        except Exception as e:
            logger.warning("Failed to read session metadata {}: {}", key, e)
            repaired = self._repair(key)
            if repaired is not None:
                logger.info("Recovered read-only session metadata {} from corrupt file", key)
                return {
                    "key": repaired.key,
                    "created_at": repaired.created_at.isoformat(),
                    "updated_at": repaired.updated_at.isoformat(),
                    "metadata": repaired.metadata,
                }
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出 sessions 目录下的所有会话。

        返回按更新时间倒序排列的会话列表，每条包含：
          - key: 会话标识
          - created_at: 创建时间
          - updated_at: 最后更新时间
          - title: 会话标题（从 metadata 提取）
          - preview: 消息预览片段
          - path: 文件路径

        性能优化：
          - 最多读取前 _SESSION_LIST_PREVIEW_MAX_RECORDS 条消息（默认 200）
          - 最多读取 _SESSION_LIST_PREVIEW_MAX_CHARS 个字符（默认 1MB）
          - 超出限制的文件跳过后续扫描

        参数：
          无

        返回：
          会话信息字典列表，按 updated_at 降序排列
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            fallback_key = path.stem.replace("_", ":", 1)
            try:
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            metadata = data.get("metadata", {})
                            title = _metadata_title(metadata)

                            # ── 扫描预览文本 ──
                            preview = ""
                            fallback_preview = ""
                            scanned_records = 0
                            scanned_chars = 0

                            for line in f:
                                if not line.strip():
                                    continue
                                scanned_records += 1
                                scanned_chars += len(line)

                                # 超出扫描限制 → 跳过剩余内容
                                if (
                                    scanned_records > _SESSION_LIST_PREVIEW_MAX_RECORDS
                                    or scanned_chars > _SESSION_LIST_PREVIEW_MAX_CHARS
                                ):
                                    break

                                item = json.loads(line)
                                if item.get("_type") == "metadata":
                                    continue

                                text = _message_preview_text(item)
                                if not text:
                                    continue

                                # 优先取 user 消息作为预览
                                if item.get("role") == "user":
                                    preview = text
                                    break

                                # 如果没有 user 消息，fallback 到 assistant 消息
                                if not fallback_preview and item.get("role") == "assistant":
                                    fallback_preview = text

                            preview = preview or fallback_preview

                            sessions.append(
                                {
                                    "key": key,
                                    "created_at": data.get("created_at"),
                                    "updated_at": data.get("updated_at"),
                                    "title": title,
                                    "preview": preview,
                                    "path": str(path),
                                }
                            )
            except Exception:
                # 文件损坏 → 尝试修复并获取预览
                repaired = self._repair(fallback_key)
                if repaired is not None:
                    sessions.append(
                        {
                            "key": repaired.key,
                            "created_at": repaired.created_at.isoformat(),
                            "updated_at": repaired.updated_at.isoformat(),
                            "title": _metadata_title(repaired.metadata),
                            "preview": next(
                                (
                                    text
                                    for msg in repaired.messages
                                    if (text := _message_preview_text(msg))
                                ),
                                "",
                            ),
                            "path": str(path),
                        }
                    )
                continue

        # ── 按更新时间降序排列（最新的在前） ──
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
