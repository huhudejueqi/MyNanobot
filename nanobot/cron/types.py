"""Cron 定时任务类型定义：计划、负载、运行记录、任务状态。"""
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CronSchedule:
    """定时任务的调度定义。"""
    kind: Literal["at", "every", "cron"]
    at_ms: int | None = None        # "at" 类型：时间戳（毫秒）
    every_ms: int | None = None     # "every" 类型：间隔（毫秒）
    expr: str | None = None         # "cron" 类型：cron 表达式，如 "0 9 * * *"
    tz: str | None = None            # 时区


@dataclass
class CronPayload:
    """任务触发时要执行的内容。"""
    kind: Literal["system_event", "agent_turn"] = "agent_turn"
    message: str = ""
    deliver: bool = False             # 是否投递到频道
    channel: str | None = None        # 目标频道
    to: str | None = None             # 目标接收者
    channel_meta: dict[str, Any] = field(default_factory=dict)
    session_key: str | None = None    # 原始会话 key
    origin_channel: str | None = None
    origin_chat_id: str | None = None
    origin_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CronRunRecord:
    """单次任务执行记录。"""
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None


@dataclass
class CronJobState:
    """任务的运行时状态。"""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)


@dataclass
class CronJob:
    """一个完整的定时任务定义。"""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False

    @classmethod
    def from_dict(cls, kwargs: dict):
        """从字典反序列化 CronJob（递归处理嵌套 dataclass）。"""
        state_kwargs = dict(kwargs.get("state", {}))
        state_kwargs["run_history"] = [
            record if isinstance(record, CronRunRecord) else CronRunRecord(**record)
            for record in state_kwargs.get("run_history", [])
        ]
        kwargs["schedule"] = CronSchedule(**kwargs.get("schedule", {"kind": "every"}))
        kwargs["payload"] = CronPayload(**kwargs.get("payload", {}))
        kwargs["state"] = CronJobState(**state_kwargs)
        return cls(**kwargs)


@dataclass
class CronStore:
    """定时任务的持久化存储。"""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
