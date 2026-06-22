"""Minimal command routing table for slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from nanobot.session.manager import Session

Handler = Callable[["CommandContext"], Awaitable["OutboundMessage | None"]]


@dataclass
class CommandContext:
    """Everything a command handler needs to produce a response."""

    msg: InboundMessage
    session: Session | None
    key: str
    raw: str
    args: str = ""
    loop: Any = None


class CommandRouter:
    """基于纯字典实现的命令路由分发器

    按优先级顺序分为三层匹配规则，校验顺序从上到下：
      1. 高优先级命令 priority —— 完全匹配，**不占用分发锁**，优先执行
         示例：/stop、/restart 这类紧急启停指令
      2. 精确匹配命令 exact —— 完全匹配，需要进入分发锁后执行
      3. 前缀匹配命令 prefix —— 按最长前缀优先匹配（长前缀优先命中）
         示例："/team " 开头的一系列子命令
    """

    def __init__(self) -> None:
        # 存储高优先级命令: key=命令字符串, value=对应处理函数
        self._priority: dict[str, Handler] = {}
        # 存储普通精确匹配命令
        self._exact: dict[str, Handler] = {}
        # 存储前缀匹配列表，元素元组(前缀字符串, 处理函数)
        self._prefix: list[tuple[str, Handler]] = []

    def priority(self, cmd: str, handler: Handler) -> None:
        """注册高优先级完全匹配命令
        :param cmd: 触发命令文本
        :param handler: 该命令对应的异步处理函数
        """
        self._priority[cmd] = handler

    def exact(self, cmd: str, handler: Handler) -> None:
        """注册普通精确匹配命令
        :param cmd: 完整匹配的命令字符串
        :param handler: 命令处理函数
        """
        self._exact[cmd] = handler

    def prefix(self, pfx: str, handler: Handler) -> None:
        """注册前缀匹配命令
        注册后自动按前缀长度倒序排序，保证长前缀优先匹配
        :param pfx: 命令前缀
        :param handler: 匹配该前缀后的处理函数
        """
        self._prefix.append((pfx, handler))
        # 按前缀字符串长度从大到小排序，实现最长前缀优先
        self._prefix.sort(key=lambda p: len(p[0]), reverse=True)

    def is_priority(self, text: str) -> bool:
        """判断输入文本是否属于高优先级命令
        :param text: 用户原始输入文本
        :return: 是高优先级命令返回True，否则False
        """
        # 去除首尾空格、转小写后匹配高优先级命令字典
        return text.strip().lower() in self._priority

    def is_dispatchable_command(self, text: str) -> bool:
        """校验输入是否能匹配【非高优先级】命令（精确匹配 / 前缀匹配）
        仅判断普通指令，不检查高优先级指令
        返回True则代表dispatch()一定能找到对应处理函数
        :param text: 用户输入文本
        :return: 存在可分发普通命令返回True
        """
        cmd = text.strip().lower()
        # 先判断是否精确命中普通命令
        if cmd in self._exact:
            return True
        # 遍历前缀列表，判断是否以任一前缀开头
        for pfx, _ in self._prefix:
            if cmd.startswith(pfx):
                return True
        return False

    async def dispatch_priority(self, ctx: CommandContext) -> OutboundMessage | None:
        """执行高优先级命令分发
        从主逻辑run()中调用，执行时不加分发锁，优先处理紧急指令
        :param ctx: 命令上下文对象，包含原始输入、用户信息等
        :return: 命令执行后的回复消息；无匹配命令返回None
        """
        handler = self._priority.get(ctx.raw.lower())
        if handler:
            return await handler(ctx)
        return None

    async def dispatch(self, ctx: CommandContext) -> OutboundMessage | None:
        """普通命令分发逻辑：先精确匹配，再前缀匹配
        无匹配命令返回None
        :param ctx: 命令上下文对象
        :return: 命令处理后的回复消息，无匹配返回None
        """
        cmd = ctx.raw.lower()

        # 第一步：尝试精确匹配普通命令
        if handler := self._exact.get(cmd):
            return await handler(ctx)

        # 第二步：遍历前缀列表，匹配最长前缀
        for pfx, handler in self._prefix:
            if cmd.startswith(pfx):
                # 截取前缀后面的内容存入args，作为命令参数
                ctx.args = ctx.raw[len(pfx):]
                return await handler(ctx)

        # 精确、前缀均无匹配，返回空
        return None
