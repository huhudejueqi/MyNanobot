"""TodoWrite 工具 — LLM 自己管理的多步骤任务清单。

LLM 通过调用 ``todo`` 工具来跟踪自己的进度，每次状态变更时传入**完整的**任务列表。
状态持久化在 ``session.metadata["todo_state"]`` 中，不会被 compaction 或重启冲掉。

参考 learn-claude-code s03 的 TodoWrite 模式。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.schema import (
    ArraySchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager

# session.metadata 中存储 todo 状态的 key
TODO_STATE_KEY = "todo_state"

# 合法的状态值集合
_VALID_STATUSES = frozenset({"pending", "in_progress", "completed"})
# 单次最多允许的任务数
_MAX_ITEMS = 20


def todo_state_runtime_lines(metadata: dict[str, Any] | None) -> list[str]:
    """生成 Runtime Context 中显示活跃 todo 的文本行。

    如果有活跃的 todo 列表，返回格式化的进度文本，每轮对话自动注入给 LLM 看。
    如果没有活跃 todo 或元数据为空，返回空列表。
    """
    if not metadata:
        return []
    blob = metadata.get(TODO_STATE_KEY)
    if not isinstance(blob, dict):
        return []
    items = blob.get("items")
    if not isinstance(items, list) or not items:
        return []
    out = ["Todos:"]
    for item in items:
        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
            item.get("status", ""), "[?]"
        )
        out.append(f"  {marker} #{item['id']}: {item['text']}")
    done = sum(1 for t in items if t.get("status") == "completed")
    out.append(f"  ({done}/{len(items)} completed)")
    return out


def _render(items: list[dict[str, str]]) -> str:
    """把任务列表渲染成 LLM 易读的文本格式。

    格式：
      [ ] #1: 安装依赖
      [>] #2: 写测试
      [x] #3: 运行测试
      (2/3 completed)
    """
    if not items:
        return "No todos."
    lines = []
    for item in items:
        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
            item["status"], "[?]"
        )
        lines.append(f"{marker} #{item['id']}: {item['text']}")
    done = sum(1 for t in items if t["status"] == "completed")
    lines.append(f"\n({done}/{len(items)} completed)")
    return "\n".join(lines)


@tool_parameters(
    tool_parameters_schema(
        items=ArraySchema(
            items=ObjectSchema(
                id=StringSchema("任务唯一标识。"),
                text=StringSchema("任务描述。"),
                status=StringSchema(
                    "状态：pending / in_progress / completed。",
                    enum=["pending", "in_progress", "completed"],
                ),
                required=["id", "text", "status"],
            ),
            description="完整的任务列表（每次调用都传 ALL 条目，不是只传变更的）。",
        ),
        required=["items"],
    )
)
class TodoTool(Tool, ContextAware):
    """计划和跟踪多步骤任务的进度。

    LLM 每次状态变更时调用此工具，传入**完整**的任务列表。
    同一时间最多只能有一个任务处于 ``in_progress`` 状态。
    """

    def __init__(self, sessions: SessionManager) -> None:
        self._sessions = sessions
        # 通过 ContextVar 保存当前请求的路由上下文（channel / chat_id / session_key）
        self._request_ctx: ContextVar[RequestContext | None] = ContextVar(
            "TodoTool_request_ctx", default=None,
        )

    # -- ContextAware 接口 ---------------------------------------------------

    def set_context(self, ctx: RequestContext) -> None:
        """保存当前请求的路由信息，后续 _get_session() 用。"""
        self._request_ctx.set(ctx)

    # -- Tool 接口 -----------------------------------------------------------

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        """ToolLoader 工厂方法：从 ToolContext 中提取 SessionManager。"""
        return cls(sessions=getattr(ctx, "sessions"))

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        """只有配置了 SessionManager 时才启用此工具。"""
        return getattr(ctx, "sessions", None) is not None

    @property
    def name(self) -> str:
        return "todo"

    @property
    def description(self) -> str:
        return (
            "更新多步骤工作的任务清单。"
            "每次调用传**完整**列表（id + text + status）。"
            "用来规划步骤、展示进度、保持方向。"
            "同一时间只能有一个任务 'in_progress'。"
        )

    async def execute(self, items: list[dict[str, Any]], **kwargs: Any) -> str:
        """执行 todo 更新：校验 → 持久化 → 返回渲染文本。"""
        # -- 校验输入 --
        if not isinstance(items, list):
            return "Error: items must be a list."
        if len(items) > _MAX_ITEMS:
            return f"Error: at most {_MAX_ITEMS} todos allowed."

        validated: list[dict[str, str]] = []
        in_progress_count = 0
        for i, item in enumerate(items):
            raw_id = item.get("id", "")
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            if not raw_id:
                return f"Error: item {i}: id required."
            if not text:
                return f"Error: item #{raw_id}: text required."
            if status not in _VALID_STATUSES:
                return (
                    f"Error: item #{raw_id}: invalid status '{status}'. "
                    f"Must be one of: pending, in_progress, completed."
                )
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": str(raw_id), "text": text, "status": status})

        # 校验：同一时间只能有一个进行中的任务
        if in_progress_count > 1:
            return "Error: only one task can be 'in_progress' at a time."

        # -- 持久化到 session metadata --
        sess = self._get_session()
        if sess is None:
            return "Error: todo requires an active chat session."
        sess.metadata[TODO_STATE_KEY] = {"items": validated}
        self._sessions.save(sess)

        return _render(validated)

    # -- 工具方法 ------------------------------------------------------------

    def _get_session(self):
        """从 ContextVar 中取出当前请求的 session_key，获取 session 对象。"""
        rc = self._request_ctx.get()
        if rc is None:
            return None
        return self._sessions.get_or_create(rc.session_key)
