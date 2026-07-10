"""MyNanobot CLI 入口。

只负责加载配置、初始化 AgentLoop、启动 CLI 交互终端。
所有 CLI 逻辑在 nanobot/cli/runner.py 中。
"""

import asyncio
import logging, os
import subprocess
import sys
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
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log_level = os.environ.get("NANOBOT_LOG_LEVEL", "INFO").upper()
    logging.getLogger("nanobot").setLevel(getattr(logging, log_level, logging.INFO))
    logging.getLogger("nanobot").addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
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


def main():
    log_file = setup_logging()
    _start_mcp_servers()
    config = resolve_config_env_vars(load_config())

    print("读取 ~/.nanobot/config.json...")
    agent = AgentLoop.from_config(config)
    print(f"模型: {agent.model}, provider: {type(agent.provider).__name__}")

    try:
        asyncio.run(run_cli(agent, log_file=log_file))
    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        _stop_mcp_servers()


if __name__ == "__main__":
    main()
