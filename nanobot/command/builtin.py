"""Simplified built-in slash commands for MyNanobot."""

from __future__ import annotations

from datetime import datetime

from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter


BUILTIN_COMMAND_SPECS: tuple = ()


def register_builtin_commands(router: CommandRouter) -> None:
    """Register simplified built-in commands."""

    async def _ping(ctx: CommandContext) -> OutboundMessage | None:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content="pong",
        )

    async def _time(ctx: CommandContext) -> OutboundMessage | None:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=f"当前时间: {datetime.now()}",
        )

    async def _version(ctx: CommandContext) -> OutboundMessage | None:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="MyNanobot v0.1.0",
        )

    async def _help(ctx: CommandContext) -> OutboundMessage | None:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="可用命令: /ping, /time, /version, /help, /status",
        )

    async def _status(ctx: CommandContext) -> OutboundMessage | None:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="MyNanobot 运行正常",
        )

    router.exact("/ping", _ping)
    router.exact("/time", _time)
    router.exact("/version", _version)
    router.exact("/help", _help)
    router.exact("/status", _status)
