"""MyNanobot CLI 入口。

只负责加载配置、初始化 AgentLoop、启动 CLI 交互终端。
所有 CLI 逻辑在 nanobot/cli/runner.py 中。
"""

import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.agent.loop import AgentLoop
from nanobot.cli.runner import run_cli


def setup_logging() -> Path:
    """配置日志：写入文件，不干扰终端。"""
    log_file = Path.home() / ".nanobot" / "agent.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger("nanobot").setLevel(logging.INFO)
    logging.getLogger("nanobot").addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_file


def main():
    log_file = setup_logging()

    print("读取 ~/.nanobot/config.json...")
    agent = AgentLoop.from_config()
    print(f"模型: {agent.model}, provider: {type(agent.provider).__name__}")

    try:
        asyncio.run(run_cli(agent, log_file=log_file))
    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
