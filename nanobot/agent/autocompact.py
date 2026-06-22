"""Auto compact: proactive compression of idle sessions to reduce token cost and latency."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Coroutine

from loguru import logger

from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.agent.memory import Consolidator


class AutoCompact:
    _RECENT_SUFFIX_MESSAGES = 8
    _INTERNAL_SESSION_PREFIXES = ("dream:",)# Dream 记忆合并功能运行时，会创建一个临时会话，session key 长这样：dream:<timestamp>

    def __init__(self, sessions: SessionManager, consolidator: Consolidator,
                 session_ttl_minutes: int = 0):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set[str] = set()
        self._summaries: dict[str, tuple[str, datetime]] = {}

    def _is_expired(self, ts: datetime | str | None,
                    now: datetime | None = None) -> bool:
        if self._ttl <= 0 or not ts:
            return False
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ((now or datetime.now()) - ts).total_seconds() >= self._ttl * 60

    @staticmethod
    def _format_summary(text: str, last_active: datetime) -> str:
        return f"Previous conversation summary (last active {last_active.isoformat()}):\n{text}"

    @classmethod
    def _is_internal_session(cls, key: str) -> bool:
        return key.startswith(cls._INTERNAL_SESSION_PREFIXES)

    def check_expired(self, schedule_background: Callable[[Coroutine], None],
                      active_session_keys: Collection[str] = ()) -> None:
        """Schedule archival for idle sessions, skipping those with in-flight agent tasks."""
        now = datetime.now()
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            if not key or self._is_internal_session(key) or key in self._archiving:
                continue
            if key in active_session_keys:
                continue
            if self._is_expired(info.get("updated_at"), now):
                self._archiving.add(key)
                schedule_background(self._archive(key))

    async def _archive(self, key: str) -> None:
        if self._is_internal_session(key):
            self._archiving.discard(key)
            return
        try:
            summary = await self.consolidator.compact_idle_session(
                key, self._RECENT_SUFFIX_MESSAGES,
            )
            if summary and summary != "(nothing)":
                session = self.sessions.get_or_create(key)
                meta = session.metadata.get("_last_summary")
                if isinstance(meta, dict):
                    self._summaries[key] = (
                        meta["text"],
                        datetime.fromisoformat(meta["last_active"]),
                    )
        except Exception:
            logger.exception("Auto-compact: failed for {}", key)
        finally:
            self._archiving.discard(key)

    def prepare_session(self, session: Session, key: str) -> tuple[Session, str | None]:
        """
        预处理会话对象，完成会话过期/归档清理、重载、摘要读取逻辑
        :param session: 当前传入的会话实例
        :param key: 会话唯一标识key
        :return: 二元组 (处理后的会话对象, 格式化后的摘要字符串/None)
        """
        # dream: 这类内部会话不走自动归档和摘要缓存，下面的 discard/pop 只是兜底清理残留
        if self._is_internal_session(key):
            # check_expired 已跳过 dream:，但以防其他路径漏了
            self._archiving.discard(key)
            self._summaries.pop(key, None)
            # 内部会话不留摘要、不入归档
            return session, None

        # 分支2：会话处于归档状态 或 会话已过期，自动重载最新会话数据
        if key in self._archiving or self._is_expired(session.updated_at):
            # 打印日志：自动整理会话，触发重载；打印当前是否归档
            logger.info("Auto-compact: reloading session {} (archiving={})", key, key in self._archiving)
            # 从持久层重新加载/创建会话，覆盖旧session对象
            session = self.sessions.get_or_create(key)

        # 热路径：进程未重启，摘要存在内存缓存 _summaries 中，读取速度快
        # 取出内存缓存里的摘要记录，同时删除缓存（一次性消费）
        entry = self._summaries.pop(key, None)
        if entry:
            # entry 元组：(摘要文本, 最后活跃时间)，格式化后返回
            return session, self._format_summary(entry[0], entry[1])

        # 冷路径：进程重启，内存缓存丢失，从会话持久化元数据读取摘要
        meta = session.metadata.get("_last_summary")
        # 校验元数据是字典格式（合法存储结构）
        if isinstance(meta, dict):
            # 从元数据取出文本、ISO格式时间字符串，转datetime后格式化摘要
            return session, self._format_summary(meta["text"], datetime.fromisoformat(meta["last_active"]))

        # 内存无缓存、元数据也无摘要记录，返回空摘要
        return session, None
