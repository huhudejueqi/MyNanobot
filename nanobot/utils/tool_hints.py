
"""
工具提示格式化模块，用于生成简洁、易读的工具调用展示文本
"""
from __future__ import annotations

import re

from nanobot.utils.path import abbreviate_path

# 工具格式化注册表：key=工具名称
# value=(优先级参数名列表, 展示模板, 是否路径类参数, 是否命令类参数)
_TOOL_FORMATS: dict[str, tuple[list[str], str, bool, bool]] = {
    "read_file":  (["path", "file_path"],              "read {}",     True,  False),
    "write_file": (["path", "file_path"],              "write {}",    True,  False),
    "edit":       (["file_path", "path"],              "edit {}",     True,  False),
    "find_files": (["query", "glob", "path"],           "find {}",     False, False),
    "grep":       (["pattern"],                        'grep "{}"',   False, False),
    "exec":       (["command"],                        "$ {}",        False, True),
    "list_exec_sessions": ([],                          "exec sessions", False, False),
    "web_search": (["query"],                          'search "{}"', False, False),
    "web_fetch":  (["url"],                            "fetch {}",    True,  False),
    "list_dir":   (["path"],                           "ls {}",       True,  False),
}

# 正则匹配shell命令中内嵌的文件路径，支持带空格的单/双引号路径、无引号裸路径
_PATH_IN_CMD_RE = re.compile(
    r'"(?P<double>(?:[A-Za-z]:[/\\]|~/|/)[^"]+)"'
    r"|'(?P<single>(?:[A-Za-z]:[/\\]|~/|/)[^']+)'"
    r"|(?P<bare>(?:[A-Za-z]:[/\\]|~/|(?<=\s)/)[^\s;&|<>\"']+)"
)


def format_tool_hints(tool_calls: list, max_length: int = 40) -> str:
    """
    将工具调用列表格式化为简洁可读的提示文本，自动对超长路径做缩写处理
    :param tool_calls: 工具调用对象列表
    :param max_length: 单条提示文本最大长度限制
    :return: 拼接完成的工具调用提示字符串，多条工具用逗号分隔，重复调用标注次数
    """
    if not tool_calls:
        return ""

    formatted = []
    # 遍历所有工具调用，分别格式化
    for tc in tool_calls:
        fmt = _TOOL_FORMATS.get(tc.name)
        if fmt:
            # 存在预设格式化规则，走标准格式化
            formatted.append(_fmt_known(tc, fmt, max_length))
        elif tc.name.startswith("mcp_"):
            # MCP 外部工具单独格式化
            formatted.append(_fmt_mcp(tc, max_length))
        else:
            # 无预设规则的未知工具，使用兜底格式化
            formatted.append(_fmt_fallback(tc, max_length))

    # 合并连续重复的工具提示，统计重复次数
    hints = []
    for hint in formatted:
        if hints and hints[-1][0] == hint:
            # 和上一条提示完全相同，计数+1
            hints[-1] = (hint, hints[-1][1] + 1)
        else:
            hints.append((hint, 1))

    # 拼接输出：重复多次则追加 ×次数，单次直接展示文本
    return ", ".join(
        f"{h} \u00d7 {c}" if c > 1 else h for h, c in hints
    )


def _get_args(tc) -> dict:
    """
    安全提取工具调用的参数字典，兼容参数为None/列表/字典等异常格式
    :param tc: 单个工具调用对象
    :return: 标准化后的参数字典，解析失败返回空字典
    """
    if tc.arguments is None:
        return {}
    # 参数是列表时取第一个元素作为参数字典
    if isinstance(tc.arguments, list):
        return tc.arguments[0] if tc.arguments else {}
    # 参数原生为字典直接返回
    if isinstance(tc.arguments, dict):
        return tc.arguments
    # 其他非法类型返回空字典
    return {}


def _extract_arg(tc, key_args: list[str]) -> str | None:
    """
    按优先级列表提取第一个非空字符串参数值
    :param tc: 工具调用对象
    :param key_args: 优先级从高到低的参数名列表
    :return: 匹配到的字符串参数值，无有效参数返回 None
    """
    args = _get_args(tc)
    if not isinstance(args, dict):
        return None
    # 优先按预设参数名顺序查找
    for key in key_args:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    # 预设参数全部无值，遍历所有参数取第一个非空字符串
    for val in args.values():
        if isinstance(val, str) and val:
            return val
    return None


def _fmt_known(tc, fmt: tuple, max_length: int = 40) -> str:
    """
    格式化已注册、存在预设模板的标准工具
    :param tc: 工具调用对象
    :param fmt: 注册表内对应工具的格式化元组
    :param max_length: 文本最大长度限制
    :return: 处理完成的工具展示文本
    """
    # 无参数且模板无占位符，直接返回模板文本
    if not fmt[0] and "{}" not in fmt[1]:
        return fmt[1]
    val = _extract_arg(tc, fmt[0])
    # 未提取到有效参数，直接返回工具名
    if val is None:
        return tc.name
    # 判断是否为路径，执行路径缩写
    if fmt[2]:
        val = abbreviate_path(val, max_len=max_length)
    # 判断是否为shell命令，对命令内路径做缩写并截断
    elif fmt[3]:
        val = _abbreviate_command(val, max_len=max_length)
    # 将处理后的参数填充进模板
    return fmt[1].format(val)


def _abbreviate_command(cmd: str, max_len: int = 40) -> str:
    """
    处理shell命令字符串：先缩写内部所有文件路径，再整体截断超长文本
    :param cmd: 原始命令字符串
    :param max_len: 命令整体最大展示长度
    :return: 缩写并截断后的命令文本
    """
    path_max = max(max_len // 2, 25)

    def _replace_path(match: re.Match[str]) -> str:
        """正则回调：匹配到路径后执行缩写，保留原有引号格式"""
        if match.group("double") is not None:
            return f'"{abbreviate_path(match.group("double"), max_len=path_max)}"'
        if match.group("single") is not None:
            return f"'{abbreviate_path(match.group('single'), max_len=path_max)}'"
        return abbreviate_path(match.group("bare"), max_len=path_max)

    # 替换命令中所有匹配到的路径
    abbreviated = _PATH_IN_CMD_RE.sub(_replace_path, cmd)
    # 长度未超限直接返回
    if len(abbreviated) <= max_len:
        return abbreviated
    # 超长截断，末尾添加省略号
    return abbreviated[:max_len - 1] + "\u2026"


def _fmt_mcp(tc, max_length: int = 40) -> str:
    """
    格式化MCP外部工具，输出格式为 server::tool(参数)
    :param tc: MCP工具调用对象
    :param max_length: 参数文本最大长度限制
    :return: MCP工具展示字符串
    """
    name = tc.name
    # 分割服务名与工具名，分隔符优先双下划线
    if "__" in name:
        parts = name.split("__", 1)
        server = parts[0].removeprefix("mcp_")
        tool = parts[1]
    else:
        rest = name.removeprefix("mcp_")
        parts = rest.split("_", 1)
        server = parts[0] if parts else rest
        tool = parts[1] if len(parts) > 1 else ""
    # 无法拆分出工具名，直接返回原始名称
    if not tool:
        return name
    args = _get_args(tc)
    # 取第一个非空字符串参数
    val = next((v for v in args.values() if isinstance(v, str) and v), None)
    # 无参数只展示服务::工具名
    if val is None:
        return f"{server}::{tool}"
    # 存在参数，缩写后拼接展示
    return f'{server}::{tool}("{abbreviate_path(val, max_length)}")'


def _fmt_fallback(tc, max_length: int = 40) -> str:
    """
    兜底格式化函数：用于无预设模板的未知工具
    :param tc: 工具调用对象
    :param max_length: 参数最大展示长度
    :return: 通用工具展示文本
    """
    args = _get_args(tc)
    val = next(iter(args.values()), None) if isinstance(args, dict) else None
    # 无有效字符串参数，仅返回工具名
    if not isinstance(val, str):
        return tc.name
    # 参数超长则缩写，否则直接原样展示
    if len(val) > max_length:
        return f'{tc.name}("{abbreviate_path(val, max_length)}")'
    else:
        return f'{tc.name}("{val}")'