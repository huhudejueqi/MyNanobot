"""SSE 流式传输测试脚本。

测试 chat_stream() 是否能正常逐 chunk 返回。
使用本地 Ollama (qwen3.5:9b) 或其他 OpenAI 兼容 API。
"""

import asyncio
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


async def test_stream():
    """测试流式输出。"""
    # 本地 Ollama 配置，按需修改
    provider = OpenAICompatProvider(
        api_key=None,
        api_base="http://localhost:11434/v1",
    )
    model = "qwen3:0.6b"  # 用小模型测试，速度快

    messages = [
        {"role": "user", "content": "用三句话介绍一下你自己"},
    ]

    print(f"模型: {model}")
    print(f"API:  {provider.api_base}")
    print("-" * 40)
    print("开始流式输出：\n")

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
            print(f"\n\n{'=' * 40}")
            print(f"结束原因:  {delta.finish_reason}")
            print(f"总 chunk:  {chunks}")
            print(f"总字符:    {len(''.join(full_text))}")
            print(f"耗时:      {elapsed:.1f}s")
            return

    print("\n\n[警告] 流结束但没收到 finish_reason")


async def test_non_stream():
    """对比非流式输出。"""
    provider = OpenAICompatProvider(
        api_key=None,
        api_base="http://localhost:11434/v1",
    )
    model = "qwen3:0.6b"

    messages = [
        {"role": "user", "content": "用三句话介绍一下你自己"},
    ]

    print(f"\n{'=' * 40}")
    print("非流式对比：")
    print(f"模型: {model}")
    print("-" * 40)

    t0 = time.time()
    response = await provider.chat(messages=messages, model=model)
    elapsed = time.time() - t0

    print(f"内容:      {response.content[:100]}...")
    print(f"结束原因:  {response.finish_reason}")
    print(f"耗时:      {elapsed:.1f}s")


async def main():
    try:
        await test_stream()
    except Exception as e:
        print(f"\n流式测试失败: {e}")

    try:
        await test_non_stream()
    except Exception as e:
        print(f"\n非流式测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
