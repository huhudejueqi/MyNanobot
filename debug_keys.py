"""方向键诊断工具——跑这个看看按键解析是否正常。"""
#!/usr/bin/env python3
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from nanobot.cli.cli_reader import AsyncCli, CliConfig

async def main():
    config = CliConfig(
        prompt="按方向键（上/下/左/右）看看解析结果，按 Ctrl+C 退出\n>>> ",
        history_file="/tmp/debug_history.txt",
    )

    # 预先填充测试历史
    config.completer = lambda t: ["/ping", "/new", "/help", "/version"]

    try:
        async with AsyncCli(config) as cli:
            while True:
                try:
                    line = await cli.readline()
                    if not line:
                        continue
                    cli.output(f"收到输入: [{line}]")
                    # 手动添加到历史（方便方向键测试）
                    cli._add_to_history(line)
                except KeyboardInterrupt:
                    break
    except KeyboardInterrupt:
        pass
    print("\n诊断结束")

if __name__ == "__main__":
    asyncio.run(main())
