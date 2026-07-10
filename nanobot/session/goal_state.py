# 用于持续性目标（如 long_task / complete_goal）的会话元数据辅助工具
#
# 工具会设置 metadata[GOAL_STATE_KEY]。读取逻辑兼容旧会话遗留键 thread_goal。
# 调用方可直接使用 goal_state_runtime_lines、goal_state_ws_blob、runner_wall_llm_timeout_s，
# 无需导入各工具的实现代码。

from __future__ import annotations
import json
from typing import Any, Mapping, MutableMapping

from nanobot.session.manager import SessionManager

GOAL_STATE_KEY = "goal_state"
# 旧版本构建产物会将同一份 JSON 二进制数据存储在该键名下。
_LEGACY_GOAL_STATE_SESSION_KEY = "thread_goal"
_MAX_OBJECTIVE_IN_RUNTIME = 4000
_MAX_OBJECTIVE_WS = 600

def _session_goal_raw(metadata:Mapping[str,Any]|None)->Any:
    if not metadata:
        return None
    if GOAL_STATE_KEY in metadata:
        return metadata.get(GOAL_STATE_KEY)
    return metadata.get(_LEGACY_GOAL_STATE_SESSION_KEY)

def discard_legacy_goal_state_key(metadata: MutableMapping[str, Any]) -> None:
    """在写入逻辑迁移至常量 GOAL_STATE_KEY 后，移除遗留的元数据键。"""
    metadata.pop(_LEGACY_GOAL_STATE_SESSION_KEY, None)

def goal_state_raw(metadata: Mapping[str, Any] | None) -> Any:
    """读取会话目标数据块，优先读取常量 GOAL_STATE_KEY，无则读取遗留键。"""
    return _session_goal_raw(metadata)

def sustained_goal_active(metadata: Mapping[str, Any] | None) -> bool:
    """当前会话存在进行中的持续性目标时返回 True（用于 long_task 任务状态记录）。"""
    goal = parse_goal_state(goal_state_raw(metadata))
    return isinstance(goal, dict) and goal.get("status") == "active"

def sustained_goal_turn(
    metadata: Mapping[str, Any] | None,
    *,
    message_metadata: Mapping[str, Any] | None = None,
) -> bool:
    """当本轮交互需要采用持续性目标运行限制时返回 True。"""
    if sustained_goal_active(metadata):
        return True
    if not message_metadata:
        return False
    return str(message_metadata.get("original_command") or "").strip() == "/goal"

def parse_goal_state(blob: Any) -> dict[str, Any] | None:
    if blob is None:
        return None
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, str):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def goal_state_runtime_lines(metadata: Mapping[str, Any] | None) -> list[str]:
    """当存在激活状态的持续性目标时，追加至运行时上下文块内的文本内容。"""
    # 无会话元数据，直接返回空列表
    if not metadata:
        return []
    # 读取并解析原始目标状态数据
    goal = parse_goal_state(_session_goal_raw(metadata))
    # 目标非字典 / 状态不为active，无目标上下文，返回空
    if not isinstance(goal, dict) or goal.get("status") != "active":
        return []
    # 提取任务目标描述文本，空值置空字符串并去除首尾空格
    objective = str(goal.get("objective") or "").strip()
    # 存在活跃目标但无任务描述，返回固定提示文本
    if not objective:
        return ["Goal: active (no objective text stored)."]
    # 目标文本超出运行时最大字符限制，截断并添加省略标记
    if len(objective) > _MAX_OBJECTIVE_IN_RUNTIME:
        objective = objective[:_MAX_OBJECTIVE_IN_RUNTIME].rstrip() + "\n… (truncated)"
    # 初始化输出列表，写入目标标题与截断后的任务描述
    out = ["Goal (active):", objective]
    # 提取前端展示摘要
    hint = str(goal.get("ui_summary") or "").strip()
    # 摘要非空则追加到上下文列表
    if hint:
        out.append(f"Summary: {hint}")
    return out

def goal_state_ws_blob(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """用于WebSocket的goal_state事件、兼容JSON格式的快照数据（每条消息帧仅携带单个对话ID）。"""
    goal = parse_goal_state(_session_goal_raw(metadata)) if metadata else None
    if isinstance(goal, dict) and goal.get("status") == "active":
        objective = str(goal.get("objective") or "").strip()
        if len(objective) > _MAX_OBJECTIVE_WS:
            objective = objective[:_MAX_OBJECTIVE_WS].rstrip() + "…"
        summary = str(goal.get("ui_summary") or "").strip()[:120]
        blob: dict[str, Any] = {"active": True}
        if summary:
            blob["ui_summary"] = summary
        if objective:
            blob["objective"] = objective
        return blob
    return {"active": False}


def runner_wall_llm_timeout_s(
    sessions: SessionManager,
    session_key: str | None,
    *,
    metadata: Mapping[str, Any] | None = None,
    message_metadata: Mapping[str, Any] | None = None,
) -> float | None:
    """流式输出大模型时，AgentRunner 执行器的全局硬超时上限。

    若本轮为持续性目标交互，返回 0.0，表示关闭请求外层的 asyncio.wait_for 超时控制；
    返回 None 则使用环境配置 NANOBOT_LLM_TIMEOUT_S 默认超时值。
    若调用方已持有本轮会话的内存元数据，可直接传入 metadata 以省去会话查询开销。
    """
    meta: Mapping[str, Any] | None = metadata
    if meta is None and session_key:
        meta = sessions.get_or_create(session_key).metadata
    return 0.0 if sustained_goal_turn(meta, message_metadata=message_metadata) else None