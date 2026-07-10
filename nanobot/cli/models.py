"""模型信息查询辅助工具（供 onboard 向导和 CLI 使用）。

当前 litellm 正在替换中，模型数据库/自动补全临时禁用。
所有公共函数签名保持不变，确保调用方不受影响。
"""

from __future__ import annotations

from typing import Any


def get_all_models() -> list[str]:
    """返回所有已知模型的名称列表。当前返回空列表。"""
    return []


def find_model_info(model_name: str) -> dict[str, Any] | None:
    """查询指定模型的详细信息。当前返回 None。"""
    return None


def get_model_context_limit(model: str, provider: str = "auto") -> int | None:
    """获取模型的上下文窗口大小。当前返回 None。"""
    return None


def get_model_suggestions(_partial: str, provider: str = "auto", limit: int = 20) -> list[str]:
    """根据输入部分匹配推荐模型。当前返回空列表。"""
    return []


def format_token_count(tokens: int) -> str:
    """格式化 token 数为可读字符串（如 200000 → '200,000'）。"""
    return f"{tokens:,}"
