"""Async message queue for decoupled channel-agent communication."""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """异步消息总线，解耦频道和 Agent 核心。

    Channel 把消息丢进 inbound 队列，
    Agent 处理后将响应丢进 outbound 队列。
    """

    def __init__(self):
        # 入站队列：Channel -> Agent
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        # 出站队列：Agent -> Channel
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Channel 调用此方法，将消息发给 Agent。"""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """AgentLoop.run() 调用此方法，等待下一条消息。"""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Agent 处理完调用此方法，将响应发回 Channel。"""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Channel 调用此方法，等待下一条响应。"""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
