"""CLI 交互终端运行器。

职责：
- 管理 CLI 交互循环（输入解析、命令分发）
- 处理流式输出（Rich Live 逐 chunk 渲染 markdown）
- 附件解析（@filepath 语法）
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cli import AsyncCli, CliConfig
from nanobot.cli.stream import StreamRenderer


# ── 附件解析 ──────────────────────────────────────────────────────────

def parse_media_input(text: str) -> tuple[str, list[str], list[str]]:
    """解析 @filepath 语法，返回 (cleaned_text, media_list, warnings)。

    语法：
        @/path/to/file       简单路径
        @"/path with spaces"  含空格的路径用引号包裹

    @ 前面必须是行首或空格，避免误匹配邮箱地址等。
    文件不存在时加入 warnings 列表。
    """
    media: list[str] = []
    warnings: list[str] = []
    pattern = r'(?:^|\s)@(?:("[^"]+")|(\S+))'

    def _replacer(m: re.Match) -> str:
        prefix = m.group(0)[0] if m.group(0)[0] == ' ' else ''
        path = m.group(1).strip('"') if m.group(1) else m.group(2)
        if os.path.isfile(path):
            media.append(os.path.abspath(path))
        else:
            warnings.append(f'文件不存在: {path}')
        return prefix

    cleaned = re.sub(pattern, _replacer, text).strip()
    cleaned = ' '.join(cleaned.split())
    return cleaned, media, warnings


# ── CLI 命令 ─────────────────────────────────────────────────────────

CLI_COMMANDS = [
    "/new", "/list", "/switch", "/help", "/exit",
    "/ping", "/time", "/version",
]


def tab_completer(text: str) -> list[str]:
    if not text.startswith("/"):
        return []
    return [cmd for cmd in CLI_COMMANDS if cmd.startswith(text)]


# ── Banner ───────────────────────────────────────────────────────────

def print_banner() -> None:
    print("=" * 50)
    print("  MyNanobot - 个人 AI 助手")
    print("=" * 50)
    print("  输入消息开始对话")
    print("  @/路径   附加文件（如 @/tmp/test.png）")
    print("  /new     新建对话")
    print("  /list    列出所有对话")
    print("  /switch  切换到指定对话")
    print("  ↑↓ 历史  Tab 补全")
    print("  /exit    退出")
    print()


# ── 主运行函数 ────────────────────────────────────────────────────────

async def run_cli(agent: AgentLoop, *, log_file: Path | None = None) -> None:
    """启动 CLI 交互终端。

    Args:
        agent: 已初始化的 AgentLoop 实例
        log_file: 日志文件路径，用于 banner 显示
    """
    await agent.start()

    # 会话管理
    session_ts = int(datetime.now().timestamp())
    chat_index = 0

    def current_chat_id() -> str:
        return f"chat_{chat_index}_{session_ts}"

    print_banner()
    if log_file:
        print(f"  日志: {log_file}")
    print(f"  当前对话: {current_chat_id()}")
    print()

    cli_config = CliConfig(
        prompt=">>> ",
        history_file=str(Path.home() / ".nanobot" / "cli_history"),
        history_max=1000,
        auto_save_history=True,
        completer=tab_completer,
    )

    try:
        async with AsyncCli(cli_config) as cli:
            agent.on_llm_start = _make_on_llm_start()
            agent.on_llm_end = _make_on_llm_end()

            while True:

                text = await cli.readline()
                text = text.strip()
                if not text:
                    continue

                # ── 内置命令 ──
                if text.lower() in ("/exit", "/quit", ":q", "/q"):
                    break

                if text == "/new":
                    chat_index += 1
                    cli.output(f"[已切换到新对话: {current_chat_id()}]")
                    continue

                if text == "/list":
                    cli.output(f"当前对话: {current_chat_id()}")
                    if chat_index > 0:
                        cli.output(f"历史对话: 0 ~ {chat_index - 1}")
                    else:
                        cli.output("暂无历史对话")
                    continue

                if text.startswith("/switch"):
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2:
                        cli.output("用法: /switch <对话序号>")
                        continue
                    try:
                        idx = int(parts[1])
                        if idx < 0:
                            cli.output("序号不能为负")
                        else:
                            chat_index = idx
                            cli.output(f"[已切换到对话: {current_chat_id()}]")
                    except ValueError:
                        cli.output("请输入数字序号")
                    continue

                if text == "/help":
                    cli.output(
                        "可用命令: /new, /list, /switch <n>, "
                        "/ping, /time, /version, /help, /exit"
                    )
                    cli.output('附件: @/path/to/file 或 @"含空格的路径"')
                    cli.output("键盘: ↑↓ 历史  ←→ 光标  Tab 补全  Ctrl+D 退出")
                    continue

                # ── 解析附件 ──
                content, media, warnings = parse_media_input(text)
                for w in warnings:
                    cli.output(f"[警告] {w}")
                if not content and not media:
                    continue

                # ── 发送消息并等待回复 ──
                turn_done = asyncio.Event()
                turn_content: list[str] = []
                turn_metadata: dict[str, Any] = {}

                async def _consume_outbound() -> None:
                    """消费 bus 上的 outbound 消息，处理流式 delta 和最终回复。"""
                    while not turn_done.is_set():
                        try:
                            msg = await asyncio.wait_for(
                                agent.bus.consume_outbound(), timeout=1.0,
                            )
                        except asyncio.TimeoutError:
                            continue

                        meta = msg.metadata or {}

                        if meta.get("_stream_delta"):
                            stream_started = True
                            # raw 终端下直接输出，不用 Rich Live
                            write = msg.content.replace("\n", "\r\n")
                            sys.stdout.write(write)
                            sys.stdout.flush()
                            continue

                        if meta.get("_stream_end"):
                            if stream_started:
                                sys.stdout.write("\r\n")
                                sys.stdout.flush()
                            continue

                        if meta.get("_streamed"):
                            turn_done.set()
                            continue

                        if msg.content:
                            turn_content.append(msg.content)
                            turn_metadata.update(meta)
                        turn_done.set()

                consumer = asyncio.create_task(_consume_outbound())

                await agent.bus.publish_inbound(InboundMessage(
                    channel="cli",
                    sender_id=f"cli_user_{session_ts}",
                    chat_id=current_chat_id(),
                    content=content,
                    media=media,
                    metadata={"_wants_stream": True},
                ))

                await turn_done.wait()

                # 非流式回复：用 Rich 渲染
                if turn_content:
                    from rich.console import Console
                    from rich.markdown import Markdown
                    c = Console(file=sys.stdout)
                    c.print()
                    c.print("[cyan]🤖 MyNanobot[/cyan]")
                    c.print(Markdown(turn_content[0]))
                    c.print()

                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await agent.stop()
        print("\r\nMyNanobot 已停止。")


def _make_on_llm_start():
    async def on_llm_start():
        pass  # StreamRenderer 的 spinner 已经在显示 "正在思考..."
    return on_llm_start


def _make_on_llm_end():
    async def on_llm_end():
        pass
    return on_llm_end
