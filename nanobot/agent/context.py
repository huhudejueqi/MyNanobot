"""上下文构建器：组装历史消息、技能提示等上下文信息。"""

from pathlib import Path


class ContextBuilder:
    """构建 Agent 的 LLM 上下文。"""

    def __init__(self, workspace: Path, disabled_skills: list[str] | None = None):
        self.workspace = workspace
        self.disabled_skills = disabled_skills or []
