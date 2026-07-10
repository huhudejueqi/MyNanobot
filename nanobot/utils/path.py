"""
路径缩写工具类，用于界面展示时缩短超长文件路径/URL
"""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse


def abbreviate_path(path: str, max_len: int = 40) -> str:
    """
    缩短文件路径或网址，保留文件名与关键层级目录
    缩写策略：
    1. 路径长度未超限则直接原样返回
    2. 用户家目录统一替换为 ~/
    3. 从后往前保留文件名+上层目录，直至用尽字符配额
    4. 省略的前缀部分用 …/ 标识
    :param path: 原始文件路径或URL字符串
    :param max_len: 输出字符串最大允许长度
    :return: 缩写后的路径文本
    """
    if not path:
        return path

    # 匹配 http/https 链接，走URL专属缩写逻辑
    if re.match(r"https?://", path):
        return _abbreviate_url(path, max_len)

    # 统一路径分隔符为 /，兼容Windows反斜杠
    normalized = path.replace("\\", "/")

    # 将系统家目录替换为波浪号简写 ~
    home = os.path.expanduser("~").replace("\\", "/")
    if normalized.startswith(home + "/"):
        normalized = "~" + normalized[len(home):]
    elif normalized == home:
        normalized = "~"

    # 归一化、替换家目录后长度达标，直接返回
    if len(normalized) <= max_len:
        return normalized

    # 去除末尾斜杠，按斜杠切分为目录分段列表
    parts = normalized.rstrip("/").split("/")
    # 只有单一段落，直接截断并添加省略号
    if len(parts) <= 1:
        return normalized[:max_len - 1] + "\u2026"

    # 固定保留末尾文件名
    basename = parts[-1]
    # 字符配额：总长度 - 省略前缀…/ 2字符 - 分隔符/ - 文件名字符数，预留拼接空间
    budget = max_len - len(basename) - 3

    kept: list[str] = []
    # 倒序遍历文件名以外的上层目录
    for seg in reversed(parts[:-1]):
        seg_cost = len(seg) + 1  # 当前目录名 + 分隔符/ 占用长度
        if not kept:
            # 还未存入任何目录，当前段长度未超剩余配额则保存
            if seg_cost <= budget:
                kept.append(seg)
                budget -= seg_cost
        else:
            # 已有保存目录，叠加当前段长度仍有配额则继续保存
            if seg_cost <= budget:
                kept.append(seg)
                budget -= seg_cost
            else:
                break

    # 还原目录正序
    kept.reverse()
    if kept:
        # 存在保留的中间目录：…/目录1/目录2/文件名
        return "\u2026/" + "/".join(kept) + "/" + basename
    # 无保留中间目录，仅保留省略前缀+文件名
    return "\u2026/" + basename


def _abbreviate_url(url: str, max_len: int = 40) -> str:
    """
    URL专属缩写函数，保留域名与末尾文件名
    :param url: 原始http/https链接
    :param max_len: 输出最大长度限制
    :return: 缩短后的URL展示文本
    """
    # 长度未超限直接返回原链接
    if len(url) <= max_len:
        return url

    parsed = urlparse(url)
    domain = parsed.netloc       # 域名，如 example.com
    path_part = parsed.path      # 链接路径部分，如 /api/v2/file.json

    # 拆分路径，提取末尾文件名
    segments = path_part.rstrip("/").split("/")
    basename = segments[-1] if segments else ""

    # 路径无文件，直接整体截断链接
    if not basename:
        return url[: max_len - 1] + "\u2026"

    # 计算可用于中间目录的剩余字符配额（预留…/、分隔符空间）
    budget = max_len - len(domain) - len(basename) - 4
    # 域名+文件名已超出最大长度，截断文件名展示
    if budget < 0:
        trunc_len = max_len - len(domain) - 5
        return domain + "/\u2026/" + (basename[:trunc_len] if trunc_len > 0 else "")

    kept: list[str] = []
    # 倒序遍历路径中间目录，填充剩余字符配额
    for seg in reversed(segments[:-1]):
        seg_cost = len(seg) + 1
        if seg_cost <= budget:
            kept.append(seg)
            budget -= seg_cost
        else:
            break

    # 还原目录正序拼接
    kept.reverse()
    if kept:
        return domain + "/\u2026/" + "/".join(kept) + "/" + basename
    # 无可用中间目录，仅域名+省略号+文件名
    return domain + "/\u2026/" + basename