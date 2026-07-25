"""MyNanobot CLI 入口。

只负责加载配置、初始化 AgentLoop、启动 CLI 交互终端。
所有 CLI 逻辑在 nanobot/cli/runner.py 中。
"""

import asyncio
import concurrent.futures
import logging, os
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.agent.loop import AgentLoop
from nanobot.cli.runner import run_cli
from nanobot.config.loader import load_config, resolve_config_env_vars


def setup_logging() -> Path:
    """配置日志：写入文件，不干扰终端。"""
    log_file = Path.home() / ".nanobot" / "agent.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_level = os.environ.get("NANOBOT_LOG_LEVEL", "INFO").upper()
    from loguru import logger
    logger.remove()                          # 关掉 loguru 默认的 stderr 输出
    logger.add(str(log_file),                # 输出到文件
             level=log_level, format="{time:YYYY-MM-DD HH:mm:ss.SSS} [{level}][{name}] {message}",
             encoding="utf-8", rotation="10 MB", retention=3)
    from nanobot.utils.logging_bridge import redirect_lib_logging
    redirect_lib_logging("nanobot", level=log_level)  # 把标准 logging 的 nanobot.* 都导入 loguru
    logging.getLogger("nanobot").setLevel(getattr(logging, log_level, logging.INFO))  # logger 自身也要放行
    redirect_lib_logging("httpx", level="WARNING")
    redirect_lib_logging("httpcore", level="WARNING")
    return log_file


_MCP_SERVERS: list[subprocess.Popen] = []


def _start_mcp_servers() -> None:
    """启动配置中标记为 auto_start 的 HTTP MCP 服务。"""
    import json, subprocess
    cfg_path = Path.home() / ".nanobot" / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    servers = cfg.get("tools", {}).get("mcpServers", {})
    python = sys.executable
    project = Path(__file__).parent
    for name, svc in servers.items():
        if not svc.get("auto_start"):
            continue
        if svc.get("command"):
            args = [svc["command"]] + svc.get("args", [])
        elif svc.get("url") and "127.0.0.1" in svc["url"]:
            # 本地 HTTP 服务，需要启动脚本
            # 脚本命名: mcp_sse_server.py → auto_start 里配 sse-test
            # 去掉 -test 后缀查找
            base = name.replace("-test", "").replace("_test", "")
            script = project / "tests" / f"mcp_{base}_server.py"
            if not script.exists():
                script = project / "tests" / f"mcp_{name}_server.py"
            if not script.exists():
                logging.getLogger("nanobot.agent.loop").warning(
                    "auto_start MCP '%s': no startup script found", name
                )
                continue
            args = [python, str(script)]
        else:
            continue
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        _MCP_SERVERS.append(proc)
        logging.getLogger("nanobot.agent.loop").info(
            "auto_start MCP '%s': PID %d", name, proc.pid
        )


def _stop_mcp_servers() -> None:
    for proc in _MCP_SERVERS:
        proc.terminate()
    for proc in _MCP_SERVERS:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _send_and_exit(agent: AgentLoop, message: str) -> None:
    """发送一条消息，等待回复后打印并退出。"""
    from nanobot.bus.events import InboundMessage, OutboundMessage
    from datetime import datetime

    # 设空回调，避免 _state_run 的 finally 块报 AttributeError
    async def _noop(): pass
    agent.on_llm_end = _noop

    await agent.start()
    await asyncio.sleep(0.5)

    turn_done = asyncio.Event()
    result_content = ""

    async def _consume() -> None:
        nonlocal result_content
        while not turn_done.is_set():
            try:
                msg = await asyncio.wait_for(
                    agent.bus.consume_outbound(), timeout=60.0,
                )
            except asyncio.TimeoutError:
                turn_done.set()
                break
            meta = msg.metadata or {}
            if meta.get("_stream_delta"):
                print(msg.content, end="", flush=True)
                continue
            if meta.get("_stream_end"):
                print(flush=True)
                continue
            if msg.content:
                result_content = msg.content
                turn_done.set()

    consumer = asyncio.create_task(_consume())
    await agent.bus.publish_inbound(InboundMessage(
        channel="cli",
        sender_id=f"cli_user_{int(asyncio.get_running_loop().time())}",
        chat_id="chat_send",
        content=message,
        metadata={"_wants_stream": True},
    ))
    await asyncio.wait_for(turn_done.wait(), timeout=120.0)
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    if result_content:
        from rich.console import Console
        from rich.markdown import Markdown
        Console(file=sys.stdout).print()
        Console(file=sys.stdout).print("[cyan]🤖 MyNanobot[/cyan]")
        Console(file=sys.stdout).print(Markdown(result_content))
        Console(file=sys.stdout).print()
    await agent.stop()


def main():
    log_file = setup_logging()
    _start_mcp_servers()
    config = resolve_config_env_vars(load_config())

    print("读取 ~/.nanobot/config.json...")
    agent = AgentLoop.from_config(config)
    print(f"模型: {agent.model}, provider: {type(agent.provider).__name__}")

    # 注册 token / cost 统计钩子
    from nanobot.utils.pricing import CostTrackingHook
    agent._extra_hooks = [CostTrackingHook(model=agent.model)]

    # ── --send 模式：发送一条消息然后退出 ──
    if len(sys.argv) > 2 and sys.argv[1] == "--send":
        asyncio.run(_send_and_exit(agent, sys.argv[2]))
        _stop_mcp_servers()
        return

    try:
        asyncio.run(run_cli(agent, log_file=log_file))
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        # anyio 的 cancel_scope 在 asyncio.run() shutdown 时会报不兼容错误，
        # 这是 mcp 库的已知问题，不影响正常退出，静默忽略。
        _stop_mcp_servers()


if __name__ == "__main__":
    main()
