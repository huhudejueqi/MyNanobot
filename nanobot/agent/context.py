"""上下文构建器：组装历史消息、技能提示等上下文信息。

典型用法：
  >>> builder = ContextBuilder(workspace_path)
  >>> system_prompt = builder.build_system_prompt(skill_names=["weather", "cron"])
  >>> print(system_prompt)
  # 输出组装好的完整系统提示，包含身份、记忆和技能说明
"""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.agent.tools import mcp as mcp_tools
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.apps.cli import utils as cli_app_utils
from nanobot.bus.events import InboundMessage
from nanobot.session.goal_state import goal_state_runtime_lines
from nanobot.agent.tools.todo import todo_state_runtime_lines
from nanobot.utils.helpers import (
    current_time_str,
    detect_image_mime,
    load_bundled_template,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template
from loguru import logger


def session_extra(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """返回当前轮次附带的持久化能力参数（CLI 应用 + MCP 服务器的合并结果）。

    从会话元数据中提取 CLI 应用和 MCP 服务器的持久化配置，
    合并为一个字典返回，供后续消息处理链路使用。

    参数：
      metadata: 会话元数据字典，可能包含 cli_apps、mcp_presets 等字段

    返回：
      合并后的能力参数字典

    示例：
      >>> meta = {"cli_apps": [{"name": "gh"}], "mcp_presets": [{"name": "weather"}]}
      >>> extra = session_extra(meta)
      >>> extra  # 返回 cli 和 mcp 两部分的合并结果
      {'cli_apps': [...], 'mcp_presets': [...]}
    """
    return cli_app_utils.session_extra(metadata) | mcp_tools.session_extra(metadata)


def runtime_lines(state: Any, msg: Any, workspace: Path, *, skip: bool = False) -> list[str]:
    """返回当前轮次附带的运行时标注行（CLI 应用 + MCP 服务器状态），这些行会展示给 LLM 可见。

    收集当前可用的 CLI 应用列表和已连接/已配置的 MCP 服务器信息，
    返回字符串列表，每行是一个标注，最终拼接到 LLM 的上下文中。

    参数：
      state:     Agent 运行时状态，包含 _mcp_servers / _mcp_stacks 等字段
      msg:       当前输入消息对象
      workspace: 工作区目录路径
      skip:      True 时跳过某些耗时检查（如 CLI 应用可用性检测）

    返回：
      标注行列表，每行是一个描述性字符串

    示例：
      >>> lines = runtime_lines(state, msg, workspace)
      >>> for line in lines:
      ...     print(line)
      [CLI App] gh (GitHub CLI)
      [MCP] weather (connected)
    """
    return [
        *cli_app_utils.runtime_lines(msg, workspace, skip=skip),
        *mcp_tools.runtime_lines(
            msg,
            configured_server_names=set(state._mcp_servers),
            connected_server_names=set(state._mcp_stacks),
            skip=skip,
        ),
    ]


async def connect_mcp(state: Any, tools: ToolRegistry) -> None:
    await mcp_tools.connect_missing_servers(state, tools)


async def handle_runtime_control(state: Any, msg: InboundMessage, tools: ToolRegistry) -> bool:
    return await mcp_tools.handle_runtime_control(state, msg, tools)


class ContextBuilder:
    """上下文构建器：负责组装发给 LLM 的系统提示（身份、记忆、技能、摘要等）。

    这是 Agent 的核心组件之一，负责收集所有上下文信息并组装成
    发给 LLM 的系统提示（system prompt），包括：
      - Agent 身份定义（名称、角色、平台信息）
      - 引导文件（AGENTS.md / SOUL.md / USER.md）
      - 工具契约说明
      - 长期记忆（MEMORY.md）
      - Always 技能（始终激活的技能，如 memory、my）
      - 技能摘要列表
      - 最近历史事件摘要
      - 当前会话摘要（如果有）
      - 目标状态信息（如果有活跃目标）

    参数：
      workspace:       工作区目录路径，用于定位 skills/、memory/ 等目录
      disabled_skills: 禁用的技能名称列表，这些技能不会出现在系统提示中

    示例：
      >>> builder = ContextBuilder(workspace)
      >>> prompt = builder.build_system_prompt(
      ...     skill_names=["weather", "csv-tool"],
      ...     channel="telegram",
      ... )
      >>> print(prompt[:200])  # 查看系统提示的前 200 个字符

      >>> # 带会话摘要的构建
      >>> prompt2 = builder.build_system_prompt(
      ...     skill_names=["weather"],
      ...     session_summary="用户正在查询最近一周的天气数据",
      ... )
    """
    BOOTSTRAP_FILES = ["AGENTS.md","SOUL.md","USER.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _MAX_RECENT_HISTORY = 50
    _MAX_HISTORY_CHARS = 32_000  # 近期历史区块大小的硬性上限
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(self, workspace: Path, timezone: str | None = None, disabled_skills: list[str] | None = None):
        """初始化上下文构建器。

        参数：
          workspace:       工作区目录路径
          disabled_skills: 禁用的技能名称列表

        示例：
          >>> builder = ContextBuilder(Path("/home/user/.nanobot/workspace"))
          >>> builder = ContextBuilder(workspace, disabled_skills=["tmux", "github"])
        """
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace, disabled_skills=set(disabled_skills) if disabled_skills else None)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        session_summary: str | None = None,
        workspace: Path | None = None,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
    ) -> str:
        """构建 LLM 的 System Prompt，逐层组装身份、记忆、技能、历史等上下文。

        组装顺序（从顶到底）：
        ┌─────────────────────────────────────────────────────────┐
        │ ① _get_identity()                                       │
        │   ├─ templates/agent/identity.md（身份模板）             │
        │   ├─ 工作区路径 workspace_path                           │
        │   ├─ 运行时信息 runtime（OS / Python 版本）               │
        │   └─ 平台安全策略 platform_policy                        │
        ├─────────────────────────────────────────────────────────┤
        │ ② _load_bootstrap_files()                             │
        │   读取工作区根目录下的引导文件：                       │
        │   SOUL.md（人格）、USER.md（用户画像）、             │
        │   MEMORY.md（长期记忆）                               │
        ├─────────────────────────────────────────────────────────┤
        │ ③ tool_contract.md（工具契约模板）                    │
        │   定义 tool call 的格式规范                            │
        ├─────────────────────────────────────────────────────────┤
        │ ④ # Memory（长期记忆上下文）                          │
        │   从 memory/MEMORY.md 读取，跳过未自定义的模板内容    │
        ├─────────────────────────────────────────────────────────┤
        │ ⑤ # Active Skills（始终激活的技能）                   │
        │   always: true 的 skill 完整内容直接内联               │
        ├─────────────────────────────────────────────────────────┤
        │ ⑥ # Skills（技能清单）                                │
        │   其余 skill 的摘要列表，LLM 按需 read_file 读取      │
        ├─────────────────────────────────────────────────────────┤
        │ ⑦ # Recent History（未处理的 Dream 历史摘要）          │
        │   history.jsonl 中未被 Dream 消费的条目，最多 30 条   │
        ├─────────────────────────────────────────────────────────┤
        │ ⑧ [Archived Context Summary]（压缩上下文摘要）         │
        │   上一轮 consolidate 产生的会话摘要（如存在）          │
        └─────────────────────────────────────────────────────────┘

        各层之间以 --- 分隔，最终拼接为一条 system message。
        """
        root = workspace or self.workspace
        # ① 身份模板 + 工作区路径 + 运行时 + 平台安全策略
        parts = [self._get_identity(channel=channel, workspace=root)]

        # ② 工作区引导文件：SOUL.md / USER.md / MEMORY.md
        bootstrap = self._load_bootstrap_files(root)
        if bootstrap:
            parts.append(bootstrap)

        # ③ 工具调用契约（JSON Schema 格式规范）
        parts.append(render_template("agent/tool_contract.md"))

        # ④ 长期记忆（跳过模板默认内容）
        memory = self.memory.get_memory_context()
        if memory and not self._is_template_content(self.memory.read_memory(), "memory/MEMORY.md"):
            parts.append(f"# Memory\n\n{memory}")

        # ⑤ always=true 的技能：完整内容内联到 System Prompt
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # ⑥ 非 always 技能：仅列出摘要，LLM 按需 read_file 读取
        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(render_template("agent/skills_section.md", skills_summary=skills_summary))

        # ⑦ history.jsonl 中的未处理历史（Dream 尚未消费的条目）
        if include_memory_recent_history:
            entries = self.memory.read_recent_history_for_prompt(
                since_cursor=self.memory.get_last_dream_cursor(),
                session_key=session_key,
                unified_session=unified_session,
            )
            if entries:
                capped = entries[-self._MAX_RECENT_HISTORY:]
                history_text = "\n".join(
                    f"- [{e['timestamp']}] {e['content']}" for e in capped
                )
                history_text = truncate_text(history_text, self._MAX_HISTORY_CHARS)
                parts.append("# Recent History\n\n" + history_text)

        # ⑧ 上一轮 consolidate 的会话摘要
        if session_summary:
            parts.append(f"[Archived Context Summary]\n\n{session_summary}")

        return "\n\n---\n\n".join(parts)


    def _get_identity(self,channel: str | None = None, workspace: Path | None = None) -> str:
        root = workspace or self.workspace
        workspace_path = str(root.expanduser().resolve()) # 标准项目转
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=runtime,
            platform_policy=render_template("agent/platform_policy.md", system=system),
            channel=channel or "",
        )
    
    @staticmethod
    def _build_runtime_context(
        channel: str|None,
        chat_id: str|None,
        timezone: str|None = None,
        sender_id: str | None = None,
        supplemental_lines: Sequence[str] | None = None,
    )->str:
        "构建不可信的运行时元数据块，追加在用户内容之后。"
        lines =[f"Current Time:{current_time_str(timezone)}"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if sender_id:
            lines += [f"Sender ID: {sender_id}"]
        if supplemental_lines:
            lines.extend(supplemental_lines)
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n" + ContextBuilder._RUNTIME_CONTEXT_END


    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)
    
    def _load_bootstrap_files(self,workspace:Path|None = None):

        #从工作空间拿所有的
        parts = []
        root = workspace or self.workspace
        for filename in self.BOOTSTRAP_FILES:
            file_path = root/filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""
    @staticmethod
    def _is_template_content(content: str, template_path: str) -> bool:
        """判断传入内容是否与内置模板完全一致（即用户未做自定义修改）。"""
        # 读取内置模板文件
        tpl = load_bundled_template(template_path)
        if tpl is not None:
            # 去除首尾空白后对比内容
            return content.strip() == tpl.strip()
        # 模板读取失败，返回False
        return False
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
        sender_id: str | None = None,
        session_summary: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        current_runtime_lines: Sequence[str] | None = None,
        workspace: Path | None = None,
        runtime_state: Any | None = None,
        inbound_message: Any | None = None,
        skip_runtime_lines: bool = False,
        include_memory_recent_history: bool = True,
        session_key: str | None = None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        root = workspace or self.workspace
        logger.debug(r"Runtime context: goal_lines={},/n todo_lines={}",
                     goal_state_runtime_lines(session_metadata),
                     todo_state_runtime_lines(session_metadata))
        extra = [
            *goal_state_runtime_lines(session_metadata),
            *todo_state_runtime_lines(session_metadata),
        ]
        if runtime_state is not None and inbound_message is not None:
            extra.extend(runtime_lines(runtime_state, inbound_message, root, skip=skip_runtime_lines))
        if current_runtime_lines:
            extra.extend(line for line in current_runtime_lines if line)
        runtime_ctx = self._build_runtime_context(
            channel,
            chat_id,
            self.timezone,
            sender_id=sender_id,
            supplemental_lines=extra or None,
        )
        user_content = self._build_user_content(current_message, media)

        # Merge runtime context and user content into a single user message
        # to avoid consecutive same-role messages that some providers reject.
        # Runtime context is appended to keep the user-content prefix stable
        # for prompt-cache hits (the context changes every turn due to time).
        if isinstance(user_content, str):
            merged = f"{user_content}\n\n{runtime_ctx}"
        else:
            merged = user_content + [{"type": "text", "text": runtime_ctx}]
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    skill_names,
                    channel=channel,
                    session_summary=session_summary,
                    workspace=root,
                    include_memory_recent_history=include_memory_recent_history,
                    session_key=session_key,
                    unified_session=unified_session,
                ),
            },
            *history,
        ]
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue
            raw = p.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(p)},
            })

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    
