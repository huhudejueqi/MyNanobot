"""Jinja2 模板加载与渲染工具：管理 agent 系统提示模板。

模板文件存放位置：
  <nanobot>/templates/agent/       ← agent 系统提示的主模板
  <nanobot>/templates/agent/_snippets/  ← 可复用的共享片段（通过 {% include %} 引用）

典型用法：
  在 Python 代码中调用：
    render_template("agent/identity.md", channel="telegram")
    render_template("agent/skills_section.md", skills_summary="...")

  在 Jinja2 模板中引用片段：
    {% include 'agent/_snippets/tool_intro.md' %}

为什么用 Jinja2 而非 f-string：
  - 模板与代码分离，非开发者也能编辑 prompt 内容
  - 支持条件判断、循环、include 嵌套等复杂渲染逻辑
  - 带 LRU 缓存的 Environment，重复渲染零开销
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

# ── 模板根目录 ─────────────────────────────────────────────────────
# 相对于当前文件 utils/prompt_templates.py，向上两级到 nanobot/，
# 再取 templates/ 目录。
# 最终路径形如：<project>/nanobot/templates/
_TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


@lru_cache
def _environment() -> Environment:
    """创建并缓存 Jinja2 环境实例（仅创建一次，后续返回缓存）。

    lru_cache 装饰器确保整个进程生命周期内只初始化一次 Environment，
    避免每次调用 render_template 都重新扫描文件系统。

    Jinja2 配置说明：
      - autoescape=False: 不自动 HTML 转义（系统提示是纯文本，非 HTML）
      - trim_blocks=True:  移除 {%%} 控制块后面的换行符，使模板输出更干净
      - lstrip_blocks=True: 移除 {%%} 前面的空白，避免缩进导致多余空格
    """
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """渲染指定名称的 Jinja2 模板文件，返回渲染后的文本。

    参数：
      name:   模板名称（相对于 templates/ 的路径，如 "agent/identity.md"）
      strip:  True 时去除末尾换行，适用于需要嵌入单行字符串的场景
      **kwargs: 传递给模板的变量，在模板中用 {{ variable_name }} 引用

    返回：
      渲染后的文本字符串

    示例：
      >>> render_template("agent/identity.md", channel="telegram")
      '你是 Nanobot，运行在 Telegram 频道上……'

      >>> render_template("agent/skills_section.md",
      ...     skills_summary="- **weather** — 天气查询")
      '# Skills\\n\\n- **weather** — 天气查询'

    注意：
      - 模板不存在时 Jinja2 会抛出 TemplateNotFound 异常
      - kwargs 中的变量名必须与模板中的 {{ }} 占位符一致
    """
    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text
