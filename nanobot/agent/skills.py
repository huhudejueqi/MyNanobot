"""技能（Skill）加载器：管理 agent 可用的技能文件。

技能是一些 Markdown 文件（SKILL.md），教导 agent 如何使用特定工具或执行特定任务。
每个技能是一个独立的目录，根目录下包含 SKILL.md 文件，可选的 YAML frontmatter
头部元数据定义了技能的描述、依赖需求等信息。

技能来源（按优先级降序）：
  1. 工作区技能  <workspace>/skills/<name>/SKILL.md
  2. 内置技能    <nanobot>/skills/<name>/SKILL.md

工作区技能会覆盖同名的内置技能（同名时内置技能被隐藏），方便用户自定义。

典型使用流程：
  loader = SkillsLoader(workspace_path)
  all_skills = loader.list_skills()
  content = loader.load_skill("weather")
  summary = loader.build_skills_summary()
"""

import json
import os
import re
import shutil
from pathlib import Path

import yaml

# ── 全局常量 ────────────────────────────────────────────────────────

# 内置技能目录：相对于当前文件路径 <nanobot>/skills/
BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# 正则：匹配 SKILL.md 开头的 YAML frontmatter，如：
#   ---
#   description: 天气查询技能
#   ---
#   实际内容...
# 支持 CRLF 换行，group(1) 捕获 YAML 主体
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)


class SkillsLoader:
    """技能加载器：负责技能的列举、加载、元数据解析、依赖检查。

    核心职责：
      - list_skills():           列举所有可用技能（含来源和依赖状态）
      - load_skill(name):        加载指定技能的 SKILL.md 内容
      - load_skills_for_context(): 加载多个技能内容，拼接为 agent 上下文格式
      - build_skills_summary():  生成技能摘要 Markdown，用于渐进式加载提示
      - get_skill_metadata():    解析技能文件的 YAML frontmatter 元数据
      - get_always_skills():     获取标记为 always=true 且依赖满足的技能
      - get_skill_availability(): 检查技能的 bin/env 依赖是否满足

    技能文件格式（SKILL.md）：
      ```markdown
      ---
      description: 某个有用的技能
      metadata:
        nanobot:
          always: true
          requires:
            bins: ["git", "curl"]
            env: ["GITHUB_TOKEN"]
      ---
      技能的具体指令内容……
      ```

    always 标记：
      如果 YAML 元数据中 metadata.nanobot.always 为 true，且依赖满足，
      该技能会自动包含在 agent 的系统提示中（无需用户显式启用）。
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        disabled_skills: set[str] | None = None,
    ):
        """初始化技能加载器。

        参数：
          workspace:         工作区目录路径（用于查找 <workspace>/skills/）
          builtin_skills_dir: 内置技能目录路径，默认使用 <nanobot>/skills/
          disabled_skills:   禁用的技能名称集合，这些技能不会被包含在任何结果中
        """
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"                     # 工作区技能目录
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR   # 内置技能目录
        self.disabled_skills = disabled_skills or set()                  # 禁用的技能集合

    def _skill_entries_from_dir(
        self,
        base: Path,
        source: str,
        *,
        skip_names: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """扫描指定目录，收集所有有效的技能条目。

        规则：
          - 只查一层目录（不递归）
          - 子目录内必须有 SKILL.md 才认为是有效技能
          - 可跳过的名称列表（用于工作区技能覆盖内置技能）

        参数：
          base:       要扫描的根目录
          source:     来源标记，如 "workspace" 或 "builtin"
          skip_names: 要跳过的技能名称集合

        返回：
          技能信息字典列表，每个字典包含 name / path / source 三个字段
        """
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """列举所有可用的技能。

        返回顺序：工作区技能在前（可覆盖内置），内置技能在后。

        参数：
          filter_unavailable: True 时过滤掉依赖不满足的技能

        返回：
          技能信息字典列表，每项有 name / path / source 字段
        """
        # 1. 收集工作区技能
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        workspace_names = {entry["name"] for entry in skills}

        # 2. 收集内置技能（同名的工作区技能会覆盖内置）
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(
                    self.builtin_skills, "builtin", skip_names=workspace_names
                )
            )

        # 3. 排除禁用的技能
        if self.disabled_skills:
            skills = [s for s in skills if s["name"] not in self.disabled_skills]

        # 4. 可选过滤：只返回依赖满足的技能
        if filter_unavailable:
            return [
                skill for skill in skills
                if self._check_requirements(self._get_skill_meta(skill["name"]))
            ]
        return skills

    def load_skill(self, name: str) -> str | None:
        """按名称加载技能的 SKILL.md 完整内容。

        查找优先级：
          1. 工作区技能 <workspace>/skills/<name>/SKILL.md
          2. 内置技能 <nanobot>/skills/<name>/SKILL.md

        参数：
          name: 技能名称（即技能目录名）

        返回：
          SKILL.md 的完整文本内容，技能不存在返回 None
        """
        roots = [self.workspace_skills]
        if self.builtin_skills:
            roots.append(self.builtin_skills)
        for root in roots:
            path = root / name / "SKILL.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """加载多个技能内容，拼接为 agent 上下文格式。

        每个技能以 "### Skill: <名称>" 为标题，内容经过 frontmatter 清洗
        （去掉 YAML 头部元数据，只保留纯 Markdown 指令）。
        多个技能之间用 "---" 分隔。

        参数：
          skill_names: 要加载的技能名称列表

        返回：
          拼接好的 Markdown 文本；如果所有技能都不存在返回空字符串
        """
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self, exclude: set[str] | None = None) -> str:
        """生成所有技能的摘要信息（名称、描述、路径、可用状态）。

        这个摘要用于渐进式加载场景：agent 可以先看摘要了解有哪些技能可选，
        需要详细信息时再通过 read_file 读取具体的 SKILL.md 文件。

        不可用的技能会标注缺少的依赖（如 missing CLI: git, missing ENV: GITHUB_TOKEN）。

        参数：
          exclude: 要从摘要中排除的技能名称集合

        返回：
          Markdown 格式的技能列表文本；无可用技能时返回空字符串
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        lines: list[str] = []
        for entry in all_skills:
            skill_name = entry["name"]
            if exclude and skill_name in exclude:
                continue
            meta = self._get_skill_meta(skill_name)
            available = self._check_requirements(meta)
            desc = self._get_skill_description(skill_name)
            if available:
                lines.append(f"- **{skill_name}** — {desc}  `{entry['path']}`")
            else:
                missing = self._get_missing_requirements(meta)
                suffix = f" (unavailable: {missing})" if missing else " (unavailable)"
                lines.append(f"- **{skill_name}** — {desc}{suffix}  `{entry['path']}`")
        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """获取技能缺少的依赖描述文本。

        检查两类依赖：
          - bins: 需要的系统命令（如 git, curl），用 shutil.which 检查
          - env:  需要的环境变量（如 GITHUB_TOKEN），用 os.environ 检查

        参数：
          skill_meta: 技能的 nanobot 元数据字典（已从 frontmatter 解析）

        返回：
          缺失依赖的描述字符串，如 "CLI: git, ENV: GITHUB_TOKEN"；
          无缺失时返回空字符串
        """
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        missing_parts = []
        for command_name in required_bins:
            if not shutil.which(command_name):
                missing_parts.append(f"CLI: {command_name}")
        for env_name in required_env_vars:
            if not os.environ.get(env_name):
                missing_parts.append(f"ENV: {env_name}")
        return ", ".join(missing_parts)

    def get_skill_availability(self, name: str) -> tuple[bool, str]:
        """查询指定技能的可用状态。

        参数：
          name: 技能名称

        返回：
          (available, reason) 元组：
          - available: True 表示可用，False 表示依赖不满足
          - reason: 不可用的原因描述；可用时为空字符串
        """
        meta = self._get_skill_meta(name)
        available = self._check_requirements(meta)
        return available, "" if available else self._get_missing_requirements(meta)

    def get_skill_requirements(self, name: str) -> dict[str, list[str]]:
        """获取技能的依赖需求及当前系统中缺失项。

        参数：
          name: 技能名称

        返回：
          字典结构：
          {
            "bins": ["git", "curl"],          # 需要的系统命令列表
            "env": ["GITHUB_TOKEN"],           # 需要的环境变量列表
            "missing_bins": ["git"],            # 缺失的系统命令
            "missing_env": ["GITHUB_TOKEN"],    # 缺失的环境变量
          }
        """
        requires = self._get_skill_meta(name).get("requires", {})
        bins = [str(value) for value in requires.get("bins", [])]
        env = [str(value) for value in requires.get("env", [])]
        return {
            "bins": bins,
            "env": env,
            "missing_bins": [value for value in bins if not shutil.which(value)],
            "missing_env": [value for value in env if not os.environ.get(value)],
        }

    def _get_skill_description(self, name: str) -> str:
        """获取技能的中文描述文本，从 YAML frontmatter 的 description 字段提取。

        参数：
          name: 技能名称

        返回：
          描述文本；如果元数据中没有 description，返回技能名作为降级
        """
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # 降级：直接用技能名

    def _strip_frontmatter(self, content: str) -> str:
        """移除 Markdown 文件开头的 YAML frontmatter 部分。

        frontmatter 格式：
          ---
          description: 技能描述
          ---
          实际内容……

        只移除首次出现的 frontmatter 块，后续内容中的 --- 不受影响。

        参数：
          content: 原始 Markdown 文本

        返回：
          去掉 frontmatter 后的纯内容文本；如果没有 frontmatter 则返回原内容
        """
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: object) -> dict:
        """从 frontmatter 的 metadata 字段中提取 nanobot/openclaw 专用配置。

        支持两种格式：
          - 直接是 dict（yaml.safe_load 已解析好）
          - 是 JSON 字符串（需要反序列化）

        字段格式：
          metadata:
            nanobot:
              always: true
              requires:
                bins: ["git"]
                env: ["TOKEN"]

        或者兼容旧版本：
          metadata:
            openclaw:
              always: true
              requires:
                bins: ["git"]

        参数：
          raw: frontmatter 中 metadata 字段的值

        返回：
          提取出的 nanobot 配置字典；解析失败返回空字典
        """
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        # 先读 nanobot 命名空间，再兼容旧版 openclaw 命名空间
        payload = data.get("nanobot", data.get("openclaw", {}))
        return payload if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """检查技能的依赖是否满足。

        检查两类依赖：
          - bins: 系统命令是否可用（通过 shutil.which 检测）
          - env:  环境变量是否已设置（通过 os.environ.get 检测）

        参数：
          skill_meta: 技能的 nanobot 元数据字典

        返回：
          True 表示所有依赖满足，False 表示至少有一项不满足
        """
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return (
            all(shutil.which(cmd) for cmd in required_bins)
            and all(os.environ.get(var) for var in required_env_vars)
        )

    def _get_skill_meta(self, name: str) -> dict:
        """获取技能的 nanobot 元数据，从 frontmatter 解析并缓存。

        这是内部快捷方法，相当于：
          get_skill_metadata(name) → 提取 metadata 字段 → _parse_nanobot_metadata()

        参数：
          name: 技能名称

        返回：
          nanobot 元数据字典；无元数据时返回空字典
        """
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        """获取标记为 always=true 且依赖满足的技能名称列表。

        always 技能会自动包含在 agent 的系统提示中，无需用户显式启用。
        典型用途：一些基础能力技能（如文件系统操作）应该始终可用。

        返回：
          技能名称列表（仅包含依赖满足的）
        """
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")  # 兼容：直接在根级别的 always 标记
            )
        ]

    def get_skill_metadata(self, name: str) -> dict | None:
        """解析并返回技能文件 SKILL.md 的 YAML frontmatter 元数据。

        frontmatter 是 SKILL.md 文件开头的 YAML 块：
          ---
          description: 技能描述
          always: true
          metadata:
            nanobot:
              requires:
                bins: ["git"]
                env: ["TOKEN"]
          ---
          实际内容……

        参数：
          name: 技能名称

        返回：
          解析后的元数据字典；没有 frontmatter 或解析失败返回 None
        """
        content = self.load_skill(name)
        if not content or not content.startswith("---"):
            return None
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if not match:
            return None
        try:
            parsed = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(parsed, dict):
            return None
        # yaml.safe_load 返回原生类型（int, bool, list 等），保持原样
        metadata: dict[str, object] = {}
        for key, value in parsed.items():
            metadata[str(key)] = value
        return metadata
