"""
结构化进度事件通用工具函数，供各类智能体运行时共用
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from nanobot.agent.hook import AgentHookContext


def on_progress_accepts_tool_events(cb: Callable[..., Any]) -> bool:
    """
    判断进度回调函数是否支持接收 tool_events 参数
    :param cb: 待检测的进度回调函数
    :return: 支持返回 True，不支持返回 False
    """
    return _on_progress_accepts(cb, "tool_events")


def on_progress_accepts_file_edit_events(cb: Callable[..., Any]) -> bool:
    """
    判断进度回调函数是否支持接收 file_edit_events 参数
    :param cb: 待检测的进度回调函数
    :return: 支持返回 True，不支持返回 False
    """
    return _on_progress_accepts(cb, "file_edit_events")


def _on_progress_accepts(cb: Callable[..., Any], name: str) -> bool:
    """
    通用检测逻辑：判断回调函数是否包含指定命名参数
    两种情况视为支持：1. 函数存在**kwargs可变关键字参数；2. 形参列表包含目标参数名
    :param cb: 待检测回调函数
    :param name: 需要检测的参数名
    :return: 存在该参数返回 True，否则 False
    """
    try:
        # 解析函数签名，获取形参信息
        sig = inspect.signature(cb)
    except (TypeError, ValueError):
        # 无法解析函数签名（非标准可调用对象），判定不支持
        return False
    # 函数包含**kwargs可变关键字参数，可兼容任意额外参数
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    # 形参列表中存在目标参数名
    return name in sig.parameters


async def invoke_on_progress(
    on_progress: Callable[..., Awaitable[None]],
    content: str,
    *,
    tool_hint: bool = False,
    tool_events: list[dict[str, Any]] | None = None,
) -> None:
    """
    执行通用进度回调，自动兼容是否携带工具事件列表
    :param on_progress: 异步进度回调函数
    :param content: 进度文本内容
    :param tool_hint: 是否展示工具调用提示标记，仅关键字传参
    :param tool_events: 工具调用事件列表，仅关键字传参，可为空
    """
    # 存在工具事件且回调支持tool_events参数，传入完整事件数据
    if tool_events and on_progress_accepts_tool_events(on_progress):
        await on_progress(content, tool_hint=tool_hint, tool_events=tool_events)
        return
    # 不支持工具事件参数，仅传递基础进度内容
    await on_progress(content, tool_hint=tool_hint)


async def invoke_file_edit_progress(
    on_progress: Callable[..., Awaitable[None]],
    file_edit_events: list[dict[str, Any]],
) -> None:
    """
    执行文件编辑进度回调，仅当存在文件编辑事件且回调支持对应参数时触发
    :param on_progress: 异步进度回调函数
    :param file_edit_events: 文件编辑事件列表
    """
    # 无编辑事件 / 回调不支持file_edit_events参数，直接退出不执行回调
    if not file_edit_events or not on_progress_accepts_file_edit_events(on_progress):
        return
    # 触发文件编辑进度回调，空文本+文件事件列表
    await on_progress("", file_edit_events=file_edit_events)


def _tool_event_arguments(tool_call: Any) -> dict[str, Any]:
    """
    提取工具调用入参，做类型安全校验，确保返回字典
    :param tool_call: 工具调用对象实例
    :return: 工具参数字典，无参数则返回空字典
    """
    # 获取工具调用的arguments属性，不存在则赋值空字典
    arguments = getattr(tool_call, "arguments", {}) or {}
    # 校验类型，非字典统一返回空字典避免解析报错
    return arguments if isinstance(arguments, dict) else {}


def build_tool_event_start_payload(tool_call: Any) -> dict[str, Any]:
    """
    构建【工具调用开始】阶段事件载荷
    :param tool_call: 当前执行的工具调用对象
    :return: 标准化工具启动事件字典
    """
    return {
        "version": 1,          # 事件数据结构版本号
        "phase": "start",      # 阶段标识：工具开始执行
        "call_id": str(getattr(tool_call, "id", "") or ""),  # 工具调用唯一ID
        "name": getattr(tool_call, "name", ""),              # 工具名称
        "arguments": _tool_event_arguments(tool_call),       # 工具入参
        "result": None,        # 执行结果（启动阶段无结果）
        "error": None,         # 错误信息（启动阶段无错误）
        "files": [],           # 工具产出文件列表
        "embeds": [],          # 富媒体嵌入内容列表
    }


def tool_event_result_extras(result: Any) -> tuple[list[Any], list[Any]]:
    """
    从工具执行结果中提取产出文件、嵌入内容列表
    :param result: 工具返回结果对象
    :return: (文件列表, 嵌入内容列表)，非字典/无对应字段返回空列表
    """
    # 结果非字典，无附加数据，返回双空列表
    if not isinstance(result, dict):
        return [], []
    # 安全读取files字段，非列表则置空
    files = result.get("files") if isinstance(result.get("files"), list) else []
    # 安全读取embeds字段，非列表则置空
    embeds = result.get("embeds") if isinstance(result.get("embeds"), list) else []
    return files, embeds


def build_tool_event_finish_payloads(context: AgentHookContext) -> list[dict[str, Any]]:
    """
    批量构建工具调用结束事件载荷，区分执行成功(end)与失败(error)
    :param context: 智能体钩子上下文，存储工具调用、返回结果、事件记录
    :return: 标准化工具结束事件载荷列表，一次调用对应一条载荷
    """
    payloads: list[dict[str, Any]] = []
    # 取三者最小长度，保证调用、结果、事件一一对应，防止数组越界
    count = min(len(context.tool_calls), len(context.tool_results), len(context.tool_events))

    for idx in range(count):
        tool_call = context.tool_calls[idx]    # 单条工具调用记录
        result = context.tool_results[idx]     # 对应工具执行结果
        # 读取对应事件，非字典则初始化为空字典
        event = context.tool_events[idx] if isinstance(context.tool_events[idx], dict) else {}
        status = event.get("status")
        # 根据执行状态区分阶段：ok=正常结束，其余为异常报错
        phase = "end" if status == "ok" else "error"
        # 提取工具输出的文件与嵌入内容
        files, embeds = tool_event_result_extras(result)

        # 初始化基础事件载荷
        payload = {
            "version": 1,
            "phase": phase,
            "call_id": str(getattr(tool_call, "id", "") or ""),
            "name": getattr(tool_call, "name", ""),
            "arguments": _tool_event_arguments(tool_call),
            "result": result if phase == "end" else None,  # 失败清空结果字段
            "error": None,
            "files": files,
            "embeds": embeds,
        }

        # 工具执行失败，填充错误信息
        if phase == "error":
            # 结果为非空字符串，直接作为错误描述
            if isinstance(result, str) and result.strip():
                payload["error"] = result.strip()
            else:
                # 优先取事件详情，兜底默认报错文案
                payload["error"] = str(event.get("detail") or "Tool execution failed")
        payloads.append(payload)
    return payloads