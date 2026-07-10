"""工作区路径边界检查工具。

这些辅助函数是应用层的守卫，确保路径操作在允许的工作区范围内进行。
它们不能替代操作系统级别的沙箱机制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

WORKSPACE_BOUNDARY_NOTE = (
    " （此为硬性策略边界，非临时性故障；"
    "请勿用 Shell 技巧或替代工具重试，"
    "若该资源确实必要，请询问用户如何处理）"
)


class WorkspaceBoundaryError(PermissionError):
    """当请求路径超出了允许的工作区边界时抛出。"""


def resolve_path(path: str | Path, workspace: str | Path | None = None, *, strict: bool = False) -> Path:
    """解析路径，若设置了 workspace，相对路径将基于 workspace 进行拼接。"""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and workspace is not None:
        candidate = Path(workspace).expanduser() / candidate
    return candidate.resolve(strict=strict)


def is_path_within(path: str | Path, root: str | Path) -> bool:
    """当 path 解析后位于 root 内部或为其子路径时返回 True。"""
    try:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        resolved_root = Path(root).expanduser().resolve(strict=False)
        resolved_path.relative_to(resolved_root)
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def is_path_allowed(path: str | Path, roots: Iterable[str | Path]) -> bool:
    """当 path 位于任意一个允许的根目录内时返回 True。"""
    return any(is_path_within(path, root) for root in roots)


def require_path_within(
    path: str | Path,
    root: str | Path,
    *,
    message: str | None = None,
) -> Path:
    """解析路径并强制其必须在 root 内部。"""
    resolved = Path(path).expanduser().resolve(strict=False)
    if not is_path_within(resolved, root):
        raise WorkspaceBoundaryError(
            message
            or f"路径 {path} 超出了允许的目录 {Path(root).expanduser()}"
            + WORKSPACE_BOUNDARY_NOTE
        )
    return resolved


def resolve_allowed_path(
    path: str | Path,
    *,
    workspace: str | Path | None = None,
    allowed_root: str | Path | None = None,
    extra_allowed_roots: Iterable[str | Path] | None = None,
    strict: bool = False,
) -> Path:
    """解析路径并在配置了允许根目录时强制路径包含在其中。"""
    if allowed_root is None:
        return resolve_path(path, workspace, strict=strict) if strict else resolve_path(path, workspace)

    roots = [allowed_root, *(extra_allowed_roots or [])]
    resolved = resolve_path(path, workspace, strict=False)
    if not is_path_allowed(resolved, roots):
        raise WorkspaceBoundaryError(
            f"路径 {path} 超出了允许的目录 {Path(allowed_root).expanduser()}"
            + WORKSPACE_BOUNDARY_NOTE
        )
    if strict:
        return resolve_path(path, workspace, strict=True)
    return resolved
