"""Simplified built-in slash commands for MyNanobot."""

from __future__ import annotations

from datetime import datetime

from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.pairing import PAIRING_COMMAND_META_KEY, handle_pairing_command


BUILTIN_COMMAND_SPECS: tuple = ()

def builtin_command_palette() -> list[dict[str, str]]:
    """Return structured command metadata for UI command palettes."""
    return [spec.as_dict() for spec in BUILTIN_COMMAND_SPECS]


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

    async def _pairing(ctx: CommandContext) -> OutboundMessage | None:
        reply = handle_pairing_command(ctx.msg.channel, ctx.args)
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=reply, metadata={PAIRING_COMMAND_META_KEY: True},
        )

    async def _new(ctx: CommandContext) -> OutboundMessage | None:
        """Clear session history and start fresh."""
        loop = ctx.loop
        session = ctx.session or loop.sessions.get_or_create(ctx.key)
        session.clear()
        loop.sessions.save(session)
        loop.sessions.invalidate(session.key)
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="会话已清空，重新开始。", metadata=dict(ctx.msg.metadata or {}),
        )

    router.exact("/ping", _ping)
    router.exact("/time", _time)
    router.exact("/version", _version)
    router.exact("/help", _help)
    router.exact("/status", _status)
    router.exact("/pairing", _pairing)
    router.prefix("/pairing ", _pairing)
    router.exact("/new", _new)
