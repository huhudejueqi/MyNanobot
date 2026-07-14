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
    _INTERNAL_SESSION_PREFIXES = ("dream:",)

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
        return f"Previous conversation summary (last active {last_active.isoformat()}).\n{text}"

    @classmethod
    def _is_internal_session(cls, key: str) -> bool:
        return key.startswith(cls._INTERNAL_SESSION_PREFIXES)

    def check_expired(self, schedule_background: Callable[[Coroutine], None],
                      active_session_keys: Collection[str] = ()) -> None:
        """扫描所有 session，把超时不活跃的归档成摘要，释放内存。

        跳过条件（满足任一即跳过）：
        - key 为空
        - 内部 session（dream: 前缀）
        - 正在归档中的（_archiving）
        - 正在被 agent 处理中的（active_session_keys）
        - 未达到 TTL 过期时间
        """
        now = datetime.now()
        # 遍历所有已持久化的 session 元信息列表
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            # 跳过无效 key、内部 session、以及已在归档队列中的
            if not key or self._is_internal_session(key) or key in self._archiving:
                continue
            # 跳过正在被 agent 处理中的 session（防止在对话中途归档）
            if key in active_session_keys:
                continue
            # 检查是否超过 TTL，没超就跳过
            if self._is_expired(info.get("updated_at"), now):
                self._archiving.add(key)  # 标记为正在归档，防止重复调度
                # 把归档任务丢到事件循环后台跑，不等它完成
                schedule_background(self._archive(key))

    async def _archive(self, key: str) -> None:
        """异步归档一个过期 session：压缩历史记录为摘要，存入 session 元数据。"""
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
        """预处理会话，返回 (session, 摘要文本)。

        按优先级尝试：
        1. 内部 session（如 dream:）→ 不走归档和摘要，直接返回
        2. session 在归档中或已过期 → 从持久层重载最新数据
        3. 热路径：摘要还在内存 _summaries 缓存 → 直接返回
        4. 冷路径：从 session.metadata["_last_summary"] 读取
        5. 都没有 → 返回 None
        """
        if self._is_internal_session(key):
            self._archiving.discard(key)
            self._summaries.pop(key, None)
            return session, None
        if key in self._archiving or self._is_expired(session.updated_at):
            logger.info("Auto-compact: reloading session {} (archiving={})", key, key in self._archiving)
            session = self.sessions.get_or_create(key)
        # Hot path: summary from in-memory dict (process hasn't restarted).
        entry = self._summaries.pop(key, None)
        if entry:
            return session, self._format_summary(entry[0], entry[1])
        # Cold path: summary persisted in session metadata (process restarted).
        meta = session.metadata.get("_last_summary")
        if isinstance(meta, dict):
            return session, self._format_summary(meta["text"], datetime.fromisoformat(meta["last_active"]))
        return session, None
