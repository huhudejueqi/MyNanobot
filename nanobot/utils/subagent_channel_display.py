"""
去除内部子代理注入的脚手架模板内容，仅保留面向用户展示渠道的可读内容。

持久化存储的子代理通知完整格式参考 agent/subagent_announce.md：
包含头部标识、完整 Task 任务分配（供给模型上下文使用）、Result 执行结果，
以及末尾仅给模型使用的 Summarize… 总结指令。
对外展示渠道（内嵌网页前端、会话预览）仅需展示头部标识 + 截断后的结果正文。
"""
from __future__ import annotations

from typing import Any

# 限制Result结果区块最大字符长度，保证WebSocket会话回放可读性；
# 完整原始文本仍保存在磁盘供大模型完整复现对话，仅修改WebSocket向外输出的副本
_SUBAGENT_CHANNEL_RESULT_MAX_CHARS = 800


def scrub_subagent_announce_body(content: str) -> str:
    """
    处理完整子代理通知原始文本，返回适配前端渠道展示的精简文本
    """
    # 统一换行符并去除首尾空白
    stripped = content.replace("\r\n", "\n").strip()
    lines = stripped.splitlines()
    header = ""
    # 提取头部 [Subagent 开头的标识行
    if lines and lines[0].startswith("[Subagent"):
        header = lines[0].strip()

    lower = stripped.lower()
    # 查找结果区块标记：\nresult:\n
    key = "\nresult:\n"
    ri = lower.find(key)
    # 未找到带换行的标记则查找单行标记 \nresult:
    if ri == -1:
        key = "\nresult:"
        ri = lower.find(key)
    # 不存在Result区块，直接返回头部或原始文本
    if ri == -1:
        return header if header else stripped

    # 截取Result标记后的全部内容，并去除开头空白
    after = stripped[ri + len(key) :].lstrip()
    # 模型专属总结指令标记
    summ_marker = "summarize this naturally"
    si = after.lower().find(summ_marker)
    # 截断掉末尾给模型的总结提示文本
    if si != -1:
        after = after[:si].rstrip()

    body = after.strip()
    limit = _SUBAGENT_CHANNEL_RESULT_MAX_CHARS
    # 超过字符上限则截断并添加省略号
    if limit and len(body) > limit:
        body = body[: limit - 1].rstrip() + "…"
    # 同时存在头部和结果正文，拼接返回
    if header and body:
        return f"{header}\n\n{body}"
    # 优先级：头部 > 处理后的结果正文 > 原始文本
    return header or body or stripped


def scrub_subagent_messages_for_channel(messages: list[dict[str, Any]]) -> None:
    """
    就地修改消息字典列表：若消息携带子代理注入事件，则清洗其content内容
    """
    for msg in messages:
        # 非字典消息直接跳过
        if not isinstance(msg, dict):
            continue
        # 仅处理注入事件为 subagent_result 的消息
        if msg.get("injected_event") != "subagent_result":
            continue
        raw = msg.get("content")
        # 内容为空或非字符串直接跳过
        if not isinstance(raw, str) or not raw.strip():
            continue
        # 覆盖替换为清洗后的精简展示文本
        msg["content"] = scrub_subagent_announce_body(raw)