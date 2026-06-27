"""绕过 bus 直接测试 loop 的流式能力。

直接调 _process_message，用 on_stream 回调接收 chunk，
排查是 provider 的问题还是 bus 传输的问题。
"""

import asyncio
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.providers.openai_compat_provider import OpenAICompatProvider


async def test_direct_stream():
    """绕过 bus，直接调 _process_message 测试流式。"""
    config_path = Path.home() / ".nanobot" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    provider_cfg = config["providers"]["xiaomiMimo"]
    provider = OpenAICompatProvider(
        api_key=provider_cfg["apiKey"],
        api_base=provider_cfg["apiBase"],
    )
    model = config["agents"]["defaults"]["model"]

    from nanobot.bus.queue import MessageBus
    bus = MessageBus()
    workspace = Path.home() / ".nanobot" / "workspace"

    agent = AgentLoop(
        bus=bus, provider=provider,
        workspace=workspace, model=model,
    )

    msg = InboundMessage(
        channel="cli", sender_id="test", chat_id="test_chat",
        content="你好，用一句话介绍你自己",
        metadata={"_wants_stream": True},
    )

    print(f"模型: {model}")
    print(f"模式: 直接调 _process_message（绕过 bus）")
    print("-" * 50)

    # 直接用 on_stream 回调，不经过 bus
    chunks = 0
    full_text = []
    t0 = time.time()

    async def on_delta(delta: str):
        nonlocal chunks
        chunks += 1
        sys.stdout.write(delta)
        sys.stdout.flush()
        full_text.append(delta)

    async def on_end(*, resuming: bool = False):
        pass

    outbound = await agent._process_message(
        msg, session_key="test:direct",
        on_stream=on_delta, on_stream_end=on_end,
    )

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 50}")
    print(f"总 chunk:  {chunks}")
    print(f"总字符:    {len(''.join(full_text))}")
    print(f"耗时:      {elapsed:.1f}s")
    print(f"outbound:  {outbound.content[:50] if outbound else 'None'}...")
    print(f"metadata:  {outbound.metadata if outbound else {}}")


if __name__ == "__main__":
    asyncio.run(test_direct_stream())
