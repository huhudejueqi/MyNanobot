# 工具集函数注册地
"""nanobot 通用工具函数集"""

import base64       # Base64编码解码，用于图片多模态消息
import json         # JSON序列化/反序列化，处理工具调用、消息结构
import re           # 正则表达式，清洗模型推理标签、脏标签
import shutil       # 文件/文件夹递归删除，清理过期工具输出缓存
import time         # 时间戳、过期时间计算
import uuid         # 生成唯一临时文件名，原子写入防文件损坏
from contextlib import suppress  # 静默捕获异常，忽略指定报错
from datetime import datetime    # 标准时间格式化
from pathlib import Path         # 跨平台文件路径对象
from typing import Any           # 通用任意类型注解

import tiktoken     # OpenAI官方分词库，计算消息Token消耗
from loguru import logger        # 日志打印组件

import tiktoken
from loguru import logger

def strip_think(text:str)->str:
    """
        移除思考块、未闭合的尾部标记，以及部分模型（典型如 Ollama 部署的 Gemma 4）偶尔输出的分词器层级模板残留内容。
        处理范围包含：
        格式完整的 ...、<thought>...</thought> 思考块。
        流式输出中只输出开头、全程未闭合的思考标记块。
        格式错误的起始标签（缺少末尾 >），例如 <think广场…。
        模型有时会在标签名称后直接拼接对外展示内容，无分隔符；若不做清洗，文本里会残留原始 <think 字符串。
        仅出现在文本首尾的 Harmony 通道标记 <channel|> / <|channel|>；
        仅首尾清理是保守设计，避免误删除用户 / 助手正常提及该标记的正文内容。
        仅出现在文本最开头或最末尾的孤立闭合标签 `` / </thought>，设计逻辑同上。
        流式分片截断产生的残缺尾部控制标签，例如 <thi、<thin、<tho。
        该清洗逻辑会在内容持久化存入历史记忆（memory.py）前执行；
        第 4、5 条仅清理文本两端标记是刻意设计：如果全文无差别清除这类标记，会悄悄篡改用户或助手正常讨论这些标签的对话内容。
    """
    #优先处理格式完整的标签块
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"^\s*<think>[\s\S]*$", "", text)
    text = re.sub(r"<thought>[\s\S]*?</thought>", "", text)
    text = re.sub(r"^\s*<thought>[\s\S]*$", "", text)
    # 处理格式残缺的起始标签：形如 <think / <thought，且紧跟的下一个字符
    # 不属于合法标签 / 标识符的延续字符。这里手动枚举 ASCII 标签合法字符（大小写字母、数字、下划线_、横杠-、冒号:）以及>、/；
    # 此处不能直接用\w，因为 Python 正则默认开启 Unicode 匹配模式，\w会匹配中文字符，
    # 会导致无法修复 <think广场… 这类标签泄漏问题。
    # text1 = "<think广场今天去哪玩"
    # res1 = re.sub(r"<think(?![A-Za-z0-9_\-:>/])", "", text1)
    # 输出：广场今天去哪玩
    # <think 被删掉
    text = re.sub(r"<think(?![A-Za-z0-9_\-:>/])", "", text)
    text = re.sub(r"<thought(?![A-Za-z0-9_\-:>/])", "", text)
    # 仅清理文本首尾孤立的闭合标签
    text = re.sub(r"^\s*</think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*$", "", text)
    text = re.sub(r"^\s*</thought>\s*", "", text)
    text = re.sub(r"\s*</thought>\s*$", "", text)
    # Edge-only channel markers (harmony / Gemma 4 variant leaks).
    text = re.sub(r"^\s*<\|?channel\|?>\s*", "", text)
    # Stream chunks may end in the middle of a control tag. Strip only known
    # control-token prefixes at the very end.
    partial_control_tag = (
        r"</?(?:t|th|thi|thin|think|tho|thou|thoug|though|thought)>?"
        r"|<\|?(?:c|ch|cha|chan|chann|channe|channel)(?:\|?>?)?"
    )
    text = re.sub(rf"(?:{partial_control_tag})$", "", text)
    text = re.sub(r"^\s*<\|?$", "", text)
    return text.strip()


def extract_think(text: str) -> tuple[str | None, str]:
    """
    从内嵌的 `` / `<thought>` 标签块中提取思考内容。

    返回格式：(思考文本, 清洗后的正文文本)。
    仅提取**完整成对闭合**的标签内容；
    流式输出那种只开标签、无闭合的残缺片段，只会在清洗文本中直接删掉，不会提取出来；
    残缺标签的清理逻辑由 strip_think 函数统一处理。
    """
    #     raw = """
    # 我先分析需求
    # 用户问上海武汉薪资对比
    # 实际答案：上海工资更高
    # """
    # think_content, clean_ans = extract_think(raw)
    # print(think_content)
    # # 输出：
    # # 我先分析需求
    # #
    # # 用户问上海武汉薪资对比

    # print(clean_ans)
    # # 输出：实际答案：上海工资更高

    parts: list[str] = []
    for m in re.finditer(r"<think>([\s\S]*?)</think>", text):
        parts.append(m.group(1).strip())
    for m in re.finditer(r"<thought>([\s\S]*?)</thought>", text):
        parts.append(m.group(1).strip())
    thinking = "\n\n".join(parts) if parts else None
    return thinking, strip_think(text)

class IncrementalThinkExtractor:
    """带状态的流式  增量提取器
    适用场景：模型流式分段输出，需要实时增量推送推理内容，不重复推送已输出内容
    内部记录已推送游标，保证运行器/钩子共用同一套状态。
    """
    __slots__ = ("_emitted",)  # 限定实例变量，节省内存

    def __init__(self) -> None:
        self._emitted = ""  # 记录已经推送过的推理文本

    def reset(self) -> None:
        """重置推送记录，用于新一轮对话"""
        self._emitted = ""

    async def feed(self, buf: str, emit: Any) -> bool:
        """传入当前分片文本，推送新增推理内容
        参数：
            buf：当前流式分片字符串
            emit：异步回调函数，接收字符串（通常是hook.emit_reasoning，推送推理内容给前端）
        返回：
            True=本次有新推理内容推送；False=无新增内容
        """
        thinking, _ = extract_think(buf)
        # 无推理 / 推理内容和已推送完全一致，直接返回
        if not thinking or thinking == self._emitted:
            return False
        # 截取未推送的新增推理片段
        new = thinking[len(self._emitted):].strip()
        self._emitted = thinking
        if not new:
            return False
        # 异步推送新增推理文本
        await emit(new)
        return True
    
def extract_reasoning(
    reasoning_content: str | None,
    thinking_blocks: list[dict[str, Any]] | None,
    content: str | None,
) -> tuple[str | None, str | None]:
    """从模型响应中统一提取推理内容，并返回清洗后的正文
    返回：(推理文本, 移除推理标签后的干净消息正文)
    多厂商推理字段兼容，优先级从高到低：
    1. 独立 reasoning_content 字段（DeepSeek-R1、Kimi、MiMo、OpenAI推理模型、AWS Bedrock）
    2. Anthropic Claude 专用 thinking_blocks 推理块数组
    3. 消息content正文内内嵌的 /<thought> 标签

    同一响应仅取最高优先级来源；低优先级字段会被忽略，但正文里的标签仍会被清洗干净，不会展示给用户。
    """
    # 优先级1：独立推理字段
    if reasoning_content:
        return reasoning_content, strip_think(content) if content else content
    # 优先级2：Claude 推理块数组
    if thinking_blocks:
        parts = [
            tb.get("thinking", "")
            for tb in thinking_blocks
            if isinstance(tb, dict) and tb.get("type") == "thinking"
        ]
        joined = "\n\n".join(p for p in parts if p)
        return (joined or None), strip_think(content) if content else content
    # 优先级3：正文内嵌推理标签
    if content:
        return extract_think(content)
    # 无任何推理内容
    return None, content

def detect_image_mime(data: bytes) -> str | None:
    """通过二进制文件头魔数识别图片类型，不依赖文件后缀名
    返回标准MIME字符串，无法识别返回None
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None

def build_image_content_blocks(
    raw: bytes, mime: str, path: str, label: str
) -> list[dict[str, Any]]:
    """构建标准大模型兼容的图片多模态消息结构，附带文本说明标签
    返回包含图片base64块+说明文字的消息数组
    """
    b64 = base64.b64encode(raw).decode()
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
            "_meta": {"path": path},  # 内部元数据：原始图片路径
        },
        {"type": "text", "text": label},
    ]

def ensure_dir(path: Path) -> Path:
    """确保路径存在，返回."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def timestamp() -> str:
    """返回当前ISO格式标准时间字符串"""
    return datetime.now().isoformat()

def current_time_str(timezone: str | None = None) -> str:
    """生成带时区、星期、时区偏移的完整人类可读时间字符串"""
    from zoneinfo import ZoneInfo

    try:
        tz = ZoneInfo(timezone) if timezone else None
    except (KeyError, Exception):
        tz = None

    now = datetime.now(tz=tz) if tz else datetime.now().astimezone()
    offset = now.strftime("%z")
    offset_fmt = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    tz_name = timezone or (time.strftime("%Z") or "UTC")
    return f"{now.strftime('%Y-%m-%d %H:%M (%A)')} ({tz_name}, UTC{offset_fmt})"

# 系统非法路径字符正则（Windows/通用文件路径禁止字符）
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')
# 工具结果预览最大字符长度
_TOOL_RESULT_PREVIEW_CHARS = 1200
# 工具输出缓存根目录
_TOOL_RESULTS_DIR = ".nanobot/tool-results"
# 工具文件保留时长：7天，单位秒
_TOOL_RESULT_RETENTION_SECS = 7 * 24 * 60 * 60
# 最多保留多少个会话缓存文件夹
_TOOL_RESULT_MAX_BUCKETS = 32

def safe_filename(name: str) -> str:
    """替换路径非法字符为下划线，生成安全文件名"""
    return _UNSAFE_CHARS.sub("_", name).strip()

def image_placeholder_text(path: str | None, *, empty: str = "[image]") -> str:
    """生成图片占位文字，日志/历史记录中替代二进制图片"""
    return f"[image: {path}]" if path else empty

def truncate_text(text: str, max_chars: int) -> str:
    """按最大字符截断文本，超长添加截断提示后缀"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"

def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    """查找合法消息起始下标：过滤掉工具调用与工具返回不配对的残缺上下文
    逻辑：助手发起工具调用会记录ID，后续工具返回必须存在对应ID；
    出现无匹配ID的工具返回消息时，当前及之前所有消息作废，从下一条重新开始。
    返回合法消息起始索引，仅截取 messages[start:] 送入模型。
    """
    declared: set[str] = set()  # 存储助手已发起的工具调用ID
    start = 0
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            # 助手消息：记录所有工具调用ID
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    declared.add(str(tc["id"]))
        elif role == "tool":
            # 工具返回消息
            tid = msg.get("tool_call_id")
            # 工具ID不存在于已记录集合 → 上下文断裂
            if tid and str(tid) not in declared:
                start = i + 1
                declared.clear()
    return start

def stringify_text_blocks(content: list[dict[str, Any]]) -> str | None:
    """解析多模态消息数组，仅提取type=text的文本内容拼接；存在非文本块返回None"""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            return None
        if block.get("type") != "text":
            return None
        text = block.get("text")
        if not isinstance(text, str):
            return None
        parts.append(text)
    return "\n".join(parts)

def _render_tool_result_reference(
    filepath: Path,
    *,
    original_size: int,
    preview: str,
    truncated_preview: bool,
) -> str:
    """生成给模型看的提示文本：超长工具输出已存入文件，附带预览片段"""
    result = (
        f"[工具输出已持久化保存]\n"
        f"完整输出文件路径: {filepath}\n"
        f"原始文本总字符数: {original_size}\n"
        f"内容预览:\n{preview}"
    )
    if truncated_preview:
        result += "\n...\n(如需完整内容，请读取保存的文件)"
    return result

def _bucket_mtime(path: Path) -> float:
    """获取目录最后修改时间，读取失败返回0"""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
    
def _cleanup_tool_result_buckets(root: Path, current_bucket: Path) -> None:
    """清理过期/超限会话缓存目录：
    1. 删除超过7天的旧文件夹
    2. 文件夹总数超过上限时，删除修改时间最早的目录
    """
    # 筛选同目录下其他会话文件夹，排除当前会话
    siblings = [path for path in root.iterdir() if path.is_dir() and path != current_bucket]
    cutoff = time.time() - _TOOL_RESULT_RETENTION_SECS
    # 删除过期目录
    for path in siblings:
        if _bucket_mtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
    keep = max(_TOOL_RESULT_MAX_BUCKETS - 1, 0)
    siblings = [path for path in siblings if path.exists()]
    if len(siblings) <= keep:
        return
    # 按修改时间倒序，删除最早的多余文件夹
    siblings.sort(key=_bucket_mtime, reverse=True)
    for path in siblings[keep:]:
        shutil.rmtree(path, ignore_errors=True)

def _write_text_atomic(path: Path, content: str) -> None:
    """原子写入：先写临时文件，成功后重命名覆盖原文件，避免中途断电文件损坏"""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    finally:
        # 清理临时文件
        if tmp.exists():
            tmp.unlink(missing_ok=True)

def maybe_persist_tool_result(
    workspace: Path | None,
    session_key: str | None,
    tool_call_id: str,
    content: Any,
    *,
    max_chars: int,
) -> Any:
    """工具返回内容过长时，将完整内容写入本地文件，返回文件引用提示文本；
    内容较短则原样返回不存储。
    参数：
        workspace：工作区根目录
        session_key：会话唯一标识，区分不同会话缓存
        tool_call_id：本次工具调用ID，作为文件名
        content：工具原始输出内容
        max_chars：触发持久化的字符阈值
    返回：原始内容 / 文件引用提示文本
    """
    if workspace is None or max_chars <= 0:
        return content

    text_payload: str | None = None
    suffix = "txt"
    # 纯文本直接赋值
    if isinstance(content, str):
        text_payload = content
    # 多模态文本块数组，拼接为纯文本
    elif isinstance(content, list):
        text_payload = stringify_text_blocks(content)
        if text_payload is None:
            return content
        suffix = "json"
    # 非文本/非文本数组不处理，直接返回原内容
    else:
        return content

    # 未超过字符阈值，无需存储
    if len(text_payload) <= max_chars:
        return content

    # 创建缓存根目录
    root = ensure_dir(workspace / _TOOL_RESULTS_DIR)
    # 创建当前会话缓存文件夹
    bucket = ensure_dir(root / safe_filename(session_key or "default"))
    # 清理过期缓存文件夹
    try:
        _cleanup_tool_result_buckets(root, bucket)
    except Exception:
        logger.exception("清理过期工具缓存目录失败: {}", root)
    # 生成安全文件名
    path = bucket / f"{safe_filename(tool_call_id)}.{suffix}"
    # 文件不存在才写入
    if not path.exists():
        if suffix == "json" and isinstance(content, list):
            _write_text_atomic(path, json.dumps(content, ensure_ascii=False, indent=2))
        else:
            _write_text_atomic(path, text_payload)
    # 截取预览片段
    preview = text_payload[:_TOOL_RESULT_PREVIEW_CHARS]
    # 返回文件引用提示文本
    return _render_tool_result_reference(
        path,
        original_size=len(text_payload),
        preview=preview,
        truncated_preview=len(text_payload) > _TOOL_RESULT_PREVIEW_CHARS,
    )

def split_message(content: str, max_len: int = 2000) -> list[str]:
    """将超长文本切分为多段，优先按换行、空格分割，适配Discord单条2000字符限制
    参数：
        content：待分割文本
        max_len：单段最大字符长度，默认2000
    返回：分段字符串列表
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # 优先在换行处截断，其次空格，无分隔符则硬切
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks

def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """构建兼容各大模型厂商标准的助手消息结构，自动携带推理、工具调用字段"""
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    # 推理字段兼容
    if reasoning_content is not None or thinking_blocks:
        msg["reasoning_content"] = reasoning_content if reasoning_content is not None else ""
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg

def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """使用tiktoken批量估算整套对话提示词总Token
    统计范围：消息正文、工具调用、推理内容、name/tool_call_id字段，附加每条消息固定占位Token
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text", "")
                        if txt:
                            parts.append(txt)
            # 工具调用序列化计入统计
            tc = msg.get("tool_calls")
            if tc:
                parts.append(json.dumps(tc, ensure_ascii=False))
            # 推理内容计入
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc:
                parts.append(rc)
            # name、tool_call_id字段计入
            for key in ("name", "tool_call_id"):
                value = msg.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)
        # 工具描述列表也计入Token
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        # 每条消息固定4个占位Token
        per_message_overhead = len(messages) * 4
        return len(enc.encode("\n".join(parts))) + per_message_overhead
    except Exception:
        return 0
    
def estimate_message_tokens(message: dict[str, Any]) -> int:
    """估算单条持久化对话消息占用的提示词总token数量。
    参数：
        message: 单条对话消息字典
    返回：
        int：该消息预估token总数
    """
    # 获取消息主体内容 content 字段
    content = message.get("content")
    # 收集所有需要参与token计算的文本片段
    parts: list[str] = []

    # 分支1：内容为纯文本字符串
    if isinstance(content, str):
        parts.append(content)
    # 分支2：内容为多模态片段列表（文本、图片、文件等混合）
    elif isinstance(content, list):
        for part in content:
            # 如果片段是文本类型，单独提取文字
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            # 图片/媒体等非文本片段，整体转为JSON字符串参与统计
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    # 分支3：内容既不是字符串也不是列表，且不为空，整体序列化JSON
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))

    # 额外提取 name、tool_call_id 两个字符串字段加入文本列表
    for key in ("name", "tool_call_id"):
        value = message.get(key)
        # 仅非空字符串才计入统计
        if isinstance(value, str) and value:
            parts.append(value)

    # 如果存在工具调用数组，整体转JSON加入统计
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))

    # 提取模型推理思考内容 reasoning_content
    rc = message.get("reasoning_content")
    if isinstance(rc, str) and rc:
        parts.append(rc)

    # 将所有文本片段用换行拼接成完整待分词字符串
    payload = "\n".join(parts)
    # 无任何文本时，直接返回基础占位token 4
    if not payload:
        return 4

    try:
        # 加载 gpt3.5/gpt4 标准分词器 cl100k_base
        enc = tiktoken.get_encoding("cl100k_base")
        # 分词计算token数，额外+4做消息元数据占位，最小不低于4
        return max(4, len(enc.encode(payload)) + 4)
    except Exception:
        # 分词器加载失败降级方案：粗略估算，每4个字符约1个token，再加4基础占位
        return max(4, len(payload) // 4 + 4)
    
def estimate_prompt_tokens_chain(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    """分层计算Token：优先调用模型厂商原生Token统计接口，失败则使用本地tiktoken估算
    返回格式：(token数值, 计算来源标识)
    """
    provider_counter = getattr(provider, "estimate_prompt_tokens", None)
    if callable(provider_counter):
        with suppress(Exception):
            tokens, source = provider_counter(messages, tools, model)
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens), str(source or "provider_counter")
    # 厂商接口失效，使用本地tiktoken
    estimated = estimate_prompt_tokens(messages, tools)
    if estimated > 0:
        return int(estimated), "tiktoken"
    return 0, "none"

def build_status_content(
    *,
    version: str,
    model: str,
    start_time: float,
    last_usage: dict[str, int],
    context_window_tokens: int,
    session_msg_count: int,
    context_tokens_estimate: int,
    search_usage_text: str | None = None,
    active_task_count: int = 0,
    max_completion_tokens: int = 8192,
) -> str:
    """生成人类可读的运行状态快照文本，用于日志/前端状态展示
    参数：
        search_usage_text：可选、预格式化的联网搜索消耗文本，追加到状态末尾
    """
    # 计算运行时长
    uptime_s = int(time.time() - start_time)
    uptime = (
        f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m"
        if uptime_s >= 3600
        else f"{uptime_s // 60}m {uptime_s % 60}s"
    )
    last_in = last_usage.get("prompt_tokens", 0)
    last_out = last_usage.get("completion_tokens", 0)
    cached = last_usage.get("cached_tokens", 0)
    ctx_total = max(context_window_tokens, 0)
    # 可用上下文预算 = 总窗口 - 最大输出长度 - 安全预留缓冲区
    ctx_budget = max(ctx_total - int(max_completion_tokens) - 1024, 1)
    ctx_pct = min(int((context_tokens_estimate / ctx_budget) * 100), 999) if ctx_budget > 0 else 0
    ctx_used_str = (
        f"{context_tokens_estimate // 1000}k"
        if context_tokens_estimate >= 1000
        else str(context_tokens_estimate)
    )
    ctx_total_str = f"{ctx_total // 1000}k" if ctx_total > 0 else "n/a"
    token_line = f"\U0001f4ca Token消耗: {last_in} 输入 / {last_out} 输出"
    if cached and last_in:
        token_line += f" ({cached * 100 // last_in}% 缓存命中)"
    lines = [
        f"\U0001f408 nanobot v{version}",
        f"\U0001f9e0 当前模型: {model}",
        token_line,
        f"\U0001f4da 上下文占用: {ctx_used_str}/{ctx_total_str} (输入预算 {ctx_pct}%)",
        f"\U0001f4ac 会话消息总数: {session_msg_count}",
        f"\u23f1 运行时长: {uptime}",
        f"\u26a1 活跃任务数: {active_task_count}",
    ]
    if search_usage_text:
        lines.append(search_usage_text)
    return "\n".join(lines)

def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """同步程序内置模板文件到用户工作区，仅创建缺失文件，不会覆盖用户已修改文件
    返回本次新建的文件相对路径列表
    """
    from importlib.resources import files as pkg_files

    try:
        tpl = pkg_files("nanobot") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        content = src.read_text(encoding="utf-8") if src else ""
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    # 同步根目录md模板
    for item in tpl.iterdir():
        if item.name.endswith(".md") and not item.name.startswith("."):
            _write(item, workspace / item.name)
    # 同步记忆模板、初始化历史文件
    _write(tpl / "memory" / "MEMORY.md", workspace / "memory" / "MEMORY.md")
    _write(None, workspace / "memory" / "history.jsonl")
    # 创建技能文件夹
    (workspace / "skills").mkdir(exist_ok=True)

    # 控制台打印新建文件提示
    if added and not silent:
        from rich.console import Console
        for name in added:
            Console().print(f"  [dim]已创建文件 {name}[/dim]")

    # 初始化记忆目录Git版本管理
    try:
        from nanobot.utils.gitstore import GitStore
        gs = GitStore(
            workspace,
            tracked_files=[
                "SOUL.md",
                "USER.md",
                "memory/MEMORY.md",
            ],
        )
        gs.init()
    except Exception:
        logger.exception("初始化记忆目录Git仓库失败: {}", workspace)

    return added

def load_bundled_template(template_name: str) -> str | None:
    """读取nanobot包内置模板文件文本内容，读取失败返回None"""
    from importlib.resources import files as pkg_files

    with suppress(Exception):
        tpl = pkg_files("nanobot") / "templates" / template_name
        if tpl.is_file():
            return tpl.read_text(encoding="utf-8")
    return None
