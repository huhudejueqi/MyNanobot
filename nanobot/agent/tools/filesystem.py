"""文件读写工具。"""

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class ReadFileTool(Tool):
    """读取文件内容。"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取指定文件的内容。支持文本文件和代码文件，二进制文件将提示无法读取。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径，支持绝对路径和相对路径",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"[文件不存在: {p}]"
        if not p.is_file():
            return f"[不是文件: {p}]"
        try:
            content = p.read_bytes()
            # 尝试 UTF-8 解码
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                return f"[二进制文件，无法读取文本内容，大小: {len(content)} bytes]"
            if len(text) > 10000:
                text = text[:10000] + f"\n... (截断，共 {len(text)} 字符)"
            return text
        except Exception as e:
            return f"[读取失败: {e!s}]"


class WriteFileTool(Tool):
    """写入文件内容。"""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "将内容写入指定文件。如果文件已存在则覆盖。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> str:
        p = Path(path).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"[已写入 {len(content)} 字符到 {p}]"
        except Exception as e:
            return f"[写入失败: {e!s}]"
