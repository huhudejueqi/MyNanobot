"""异步消息队列，解耦频道和 Agent 核心。"""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """异步消息总线。

    频道将消息放入 inbound 队列，
    Agent 处理后把响应放入 outbound 队列。
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """频道调用此方法，将消息发给 Agent。"""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Agent 调用此方法，等待下一条消息。"""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Agent 处理完后调用此方法，将响应发回频道。"""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """频道调用此方法，等待下一条响应。"""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
