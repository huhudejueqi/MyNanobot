"""工作区路径边界检查工具。

这些辅助函数是应用层的守卫，确保路径操作在允许的工作区范围内进行。
它们不能替代操作系统级别的沙箱机制。
"""

from nanobot.security.workspace_policy import (
    WORKSPACE_BOUNDARY_NOTE,
    WorkspaceBoundaryError,
    is_path_allowed,
    is_path_within,
    require_path_within,
    resolve_allowed_path,
    resolve_path,
)

__all__ = [
    "WORKSPACE_BOUNDARY_NOTE",
    "WorkspaceBoundaryError",
    "is_path_allowed",
    "is_path_within",
    "require_path_within",
    "resolve_allowed_path",
    "resolve_path",
]
