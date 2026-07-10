from __future__ import annotations
from pathlib import Path

from nanobot.utils.helpers import ensure_dir

def get_config_path()-> Path:
    """
    获取配置文件路径（延迟导入，用于打破循环依赖）。

    函数调用时才委托 nanobot.config.loader.get_config_path 执行逻辑，
    以此保证启动阶段导入当前模块时不会触发循环导入问题。
    """
    from nanobot.config.loader import get_config_path as _loader_get_config_path
    return _loader_get_config_path()

def get_data_dir() -> Path:
    """返回当前实例对应的运行时数据根目录路径"""
    return ensure_dir(get_config_path().parent)


def get_runtime_subdir(name: str) -> Path:
    """返回实例数据目录下指定名称的运行时子目录路径"""
    return ensure_dir(get_data_dir() / name)

def get_media_dir(channel: str | None = None) -> Path:
    """返回媒体文件存储目录，可按频道独立分文件夹存放资源。
    参数：
        channel: 频道标识字符串，不传/传None则使用媒体根目录，传入则创建对应频道子目录
    返回：Path 对象，媒体文件夹路径（已确保文件夹存在）
    """
    # 获取运行目录下的 media 基础文件夹路径
    base = get_runtime_subdir("media")
    # 若传入频道名：拼接频道子目录并自动创建；无频道则直接返回媒体根目录
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """获取定时任务（cron）数据存储目录。
    存放定时任务配置、任务执行记录、定时缓存文件等。
    """
    return get_runtime_subdir("cron")


def get_logs_dir() -> Path:
    """获取程序日志文件存放目录。
    所有运行日志、报错日志、操作日志均保存在此文件夹。
    """
    return get_runtime_subdir("logs")


def get_webui_dir() -> Path:
    """获取 WebUI 前端专用持久化数据目录。
    仅存放网页界面展示用的持久化会话/线程 JSON 文件，不存业务媒体资源。
    """
    return get_runtime_subdir("webui")

def get_workspace_path(workspace: str | None = None) -> Path:
    """解析并创建机器人代理工作区目录，返回路径对象。
    参数：
        workspace: 自定义工作区路径字符串，不传则使用程序默认工作区
    返回：
        Path 工作区路径（已自动创建目录）
    """
    # 传入自定义路径：解析用户家目录符号~；未传入则使用默认路径 ~/.nanobot/workspace
    path = Path(workspace).expanduser() if workspace else Path.home() / ".nanobot" / "workspace"
    # 自动创建目录并返回路径
    return ensure_dir(path)


def is_default_workspace(workspace: str | Path | None) -> bool:
    """判断传入的工作区路径是否为 Nanobot 默认工作目录。
    参数：
        workspace: 待判断的工作区路径（字符串/Path对象/空）
    返回：
        布尔值，True=默认工作区，False=自定义工作区
    """
    # 解析传入的工作区路径，为空则取默认路径
    current = Path(workspace).expanduser() if workspace is not None else Path.home() / ".nanobot" / "workspace"
    # 定义程序内置默认工作区路径
    default = Path.home() / ".nanobot" / "workspace"
    # 标准化两条路径后对比是否完全一致（无需文件真实存在）
    return current.resolve(strict=False) == default.resolve(strict=False)


def get_cli_history_path() -> Path:
    """返回命令行交互共用历史记录文件路径。"""
    return Path.home() / ".nanobot" / "history" / "cli_history"


def get_bridge_install_dir() -> Path:
    """返回 WhatsApp 通信桥接服务的公共安装目录。"""
    return Path.home() / ".nanobot" / "bridge"


def get_legacy_sessions_dir() -> Path:
    """返回旧版全局会话存储目录，用于数据迁移降级兜底方案。"""
    return Path.home() / ".nanobot" / "sessions"