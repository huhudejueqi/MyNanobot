"""mimo-v2.5-pro 流式/非流式测试脚本。

从 ~/.nanobot/config.json 读取 xiaomiMimo 配置，
分别测试 chat() 和 chat_stream() 两种模式。
"""

import asyncio
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def load_mimo_config() -> tuple[str, str, str]:
    """从配置文件读取 xiaomiMimo 的 apiKey、apiBase、model。"""
    config_path = Path.home() / ".nanobot" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    provider_cfg = config["providers"]["xiaomiMimo"]
    api_key = provider_cfg["apiKey"]
    api_base = provider_cfg["apiBase"]
    model = config["agents"]["defaults"]["model"]  # mimo-v2.5-pro

    return api_key, api_base, model


async def test_stream(api_key: str, api_base: str, model: str):
    """测试流式输出。"""
    provider = OpenAICompatProvider(api_key=api_key, api_base=api_base)
    messages = [{"role": "user", "content": "你好，用一句话介绍你自己"}]

    print(f"[流式] 模型: {model}")
    print(f"[流式] API:  {api_base}")
    print("-" * 50)

    t0 = time.time()
    chunks = 0
    full_text = []

    async for delta in provider.chat_stream(messages=messages, model=model):
        chunks += 1
        if delta.content:
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            full_text.append(delta.content)
        if delta.finish_reason:
            elapsed = time.time() - t0
            print(f"\n\n{'=' * 50}")
            print(f"结束原因:  {delta.finish_reason}")
            print(f"总 chunk:  {chunks}")
            print(f"总字符:    {len(''.join(full_text))}")
            print(f"耗时:      {elapsed:.1f}s")
            return True

    elapsed = time.time() - t0
    print(f"\n\n[警告] 流结束但没收到 finish_reason")
    print(f"总 chunk:  {chunks}, 总字符: {len(''.join(full_text))}, 耗时: {elapsed:.1f}s")
    return False


async def test_non_stream(api_key: str, api_base: str, model: str):
    """测试非流式输出。"""
    provider = OpenAICompatProvider(api_key=api_key, api_base=api_base)
    messages = [{"role": "user", "content": "你好，用一句话介绍你自己"}]

    print(f"\n[非流式] 模型: {model}")
    print(f"[非流式] API:  {api_base}")
    print("-" * 50)

    t0 = time.time()
    response = await provider.chat(messages=messages, model=model)
    elapsed = time.time() - t0

    if response.finish_reason == "error":
        print(f"错误:  {response.usage}")
        return False

    print(f"内容:    {response.content}")
    print(f"原因:    {response.finish_reason}")
    print(f"耗时:    {elapsed:.1f}s")
    return True


async def main():
    api_key, api_base, model = load_mimo_config()
    print(f"配置加载成功: {model} @ {api_base}\n")

    print("=" * 50)
    print("测试 1: 非流式 chat()")
    print("=" * 50)
    ok1 = await test_non_stream(api_key, api_base, model)

    print("\n")
    print("=" * 50)
    print("测试 2: 流式 chat_stream()")
    print("=" * 50)
    ok2 = await test_stream(api_key, api_base, model)

    print(f"\n{'=' * 50}")
    print(f"结果: 非流式={'✓' if ok1 else '✗'}, 流式={'✓' if ok2 else '✗'}")


if __name__ == "__main__":
    asyncio.run(main())
