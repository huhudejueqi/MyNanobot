"""WebUI 技能数据接口：为前端提供安全的技能列表和详情。

核心功能：
  - webui_skills_payload():      返回所有技能的列表（不含文件路径等敏感信息）
  - webui_skill_detail_payload(): 返回单个技能的详细信息（含原始 Markdown 和依赖状态）

安全设计：
  _skill_payload() 只返回 name / description / source / available 等安全字段，
  不暴露后端文件路径。只有调用 detail 接口时才返回 raw_markdown。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.skills import SkillsLoader


def webui_skills_payload(
    workspace_path: Path,
    *,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any]:
    """生成 WebUI 技能列表的响应数据。

    从 SkillsLoader 获取所有技能，排序规则：
      工作区技能（workspace）排在前面，内置技能（builtin）排在后面。
    去掉了文件路径等后端敏感信息，前端只看到 name / description / source / available。

    参数：
      workspace_path:  工作区目录路径
      disabled_skills: 禁用的技能名称集合

    返回：
      {"skills": [{name, description, source, available, unavailable_reason}, ...]}
    """
    loader = SkillsLoader(workspace_path, disabled_skills=disabled_skills)
    entries = sorted(
        loader.list_skills(filter_unavailable=False),
        # 先按来源排序（workspace 在前），再按名称排序
        key=lambda entry: (entry.get("source") != "workspace", entry["name"]),
    )
    return {"skills": [_skill_payload(loader, entry) for entry in entries]}


def webui_skill_detail_payload(
    workspace_path: Path,
    name: str,
    *,
    disabled_skills: set[str] | None = None,
) -> dict[str, Any] | None:
    """生成 WebUI 技能详情的响应数据。

    相比列表接口，详情接口额外返回：
      - requirements: 依赖需求（bins / env 及当前缺失项）
      - raw_markdown: 技能的完整 SKILL.md 原始内容

    参数：
      workspace_path:  工作区目录路径
      name:            技能名称
      disabled_skills: 禁用的技能名称集合

    返回：
      技能详情的字典，技能不存在时返回 None
    """
    loader = SkillsLoader(workspace_path, disabled_skills=disabled_skills)
    entries = loader.list_skills(filter_unavailable=False)
    entry = next((item for item in entries if item["name"] == name), None)
    if entry is None:
        return None
    return {
        **_skill_payload(loader, entry),
        "requirements": loader.get_skill_requirements(name),
        "raw_markdown": loader.load_skill(name) or "",
    }


def _skill_payload(loader: SkillsLoader, entry: dict[str, str]) -> dict[str, Any]:
    """将单条技能条目转换为前端安全的响应负载。

    参数：
      loader: SkillsLoader 实例（用于获取元数据和可用性）
      entry:  技能信息字典，包含 name / path / source

    返回：
      前端可用的技能信息字典（不含文件路径）
    """
    name = entry["name"]
    metadata = loader.get_skill_metadata(name)
    available, unavailable_reason = loader.get_skill_availability(name)
    return {
        "name": name,
        "description": _description(metadata, name),
        "source": entry.get("source", "unknown"),
        "available": available,
        "unavailable_reason": unavailable_reason,
    }


def _description(metadata: dict[str, Any] | None, fallback: str) -> str:
    """从技能元数据中提取描述文本。

    参数：
      metadata: 技能的 frontmatter 元数据
      fallback: 降级使用的默认文本（通常是技能名）

    返回：
      描述文本字符串
    """
    if metadata is None:
        return fallback
    value = metadata.get("description")
    return value.strip() if isinstance(value, str) and value.strip() else fallback
