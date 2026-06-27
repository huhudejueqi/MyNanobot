"""MyNanobot CLI 交互终端，支持多会话切换。"""

import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.cli import AsyncCli, CliConfig

# ── 日志：写入文件，不干扰终端 ──
log_file = Path.home() / ".nanobot" / "agent.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(str(log_file), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger("nanobot").setLevel(logging.INFO)
logging.getLogger("nanobot").addHandler(_fh)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


CLI_COMMANDS = ["/new", "/list", "/switch", "/help", "/exit",
                "/ping", "/time", "/version"]

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


def tab_completer(text: str) -> list[str]:
    if not text.startswith("/"):
        return []
    return [cmd for cmd in CLI_COMMANDS if cmd.startswith(text)]


def print_banner():
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


async def main():
    # ── 初始化 AgentLoop ──
    print("读取 ~/.nanobot/config.json...")
    agent = AgentLoop.from_config()
    print(f"模型: {agent.model}, provider: {type(agent.provider).__name__}")
    print(f"日志: {log_file}")
    print()

    await agent.start()

    # ── 会话管理 ──
    session_ts = int(datetime.now().timestamp())
    chat_index = 0

    def current_chat_id() -> str:
        return f"chat_{chat_index}_{session_ts}"

    print_banner()
    print(f"  当前对话: {current_chat_id()}")
    print()

    # ── CLI ──
    cli_config = CliConfig(
        prompt=">>> ",
        history_file=str(Path.home() / ".nanobot" / "cli_history"),
        history_max=1000,
        auto_save_history=True,
        completer=tab_completer,
    )

    try:
        async with AsyncCli(cli_config) as cli:
            # LLM 调用前后回调：显示等待提示
            async def on_llm_start():
                sys.stdout.write("\r[思考中...]\r\n")
                sys.stdout.flush()

            async def on_llm_end():
                pass

            agent.on_llm_start = on_llm_start
            agent.on_llm_end = on_llm_end

            while True:
                text = await cli.readline()
                text = text.strip()
                if not text:
                    continue

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

                # 解析 @filepath 附件语法
                content, media, warnings = parse_media_input(text)
                for w in warnings:
                    cli.output(f"[警告] {w}")
                if not content and not media:
                    continue

                await agent.bus.publish_inbound(InboundMessage(
                    channel="cli",
                    sender_id=f"cli_user_{session_ts}",
                    chat_id=current_chat_id(),
                    content=content,
                    media=media,
                ))
                reply = await agent.bus.consume_outbound()
                cli.output(reply.content)

    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        await agent.stop()
        print("\nMyNanobot 已停止。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n退出。")
