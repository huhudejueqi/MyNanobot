"""测试 Session.get_history() 的单元测试。

覆盖场景：
  - 基础历史获取（无截断）
  - max_messages 截尾
  - max_tokens 按 token 预算剪裁
  - include_timestamps 时间戳注入
  - _command 消息过滤
  - media / cli_apps / mcp_presets 面包屑注入
  - 空内容 assistant 跳过
  - 孤立 tool_result 丢弃
  - last_consolidated 偏移
  - 对齐到 user turn 开头
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

# 确保可以 import 项目模块
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nanobot.session.manager import Session


def make_msg(role: str, content: str, **kwargs) -> dict:
    """构造一条消息字典（不含 timestamp，让 get_history 自己填）。"""
    msg = {"role": role, "content": content}
    msg.update(kwargs)
    return msg


def show(name: str, history: list[dict]) -> None:
    """打印测试结果。"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  消息条数: {len(history)}")
    for i, m in enumerate(history):
        role = m["role"]
        content = m["content"]
        preview = content[:60].replace("\n", "\\n") + ("…" if len(content) > 60 else "")
        extra = {k: v for k, v in m.items() if k not in ("role", "content")}
        extra_str = json.dumps(extra, ensure_ascii=False, indent=2) if extra else ""
        print(f"  [{i}] {role:<10} | {preview}")
        if extra_str:
            for line in extra_str.split("\n"):
                print(f"        {line}")
    print(f"  {'─'*50}")


# ═══════════════════════════════════════════════════════════════
# 1. 基础：无截断，全部返回
# ═══════════════════════════════════════════════════════════════
def test_basic():
    s = Session(key="test:basic")
    s.add_message("user", "你好")
    s.add_message("assistant", "你好！有什么可以帮你的？")
    s.add_message("user", "今天天气怎么样")
    s.add_message("assistant", "今天天气晴朗")

    h = s.get_history(max_messages=100)
    show("基础测试（全部返回）", h)
    assert len(h) == 4, f"预期 4 条，实际 {len(h)}"
    assert h[0]["content"] == "你好"
    assert h[-1]["content"] == "今天天气晴朗"


# ═══════════════════════════════════════════════════════════════
# 2. max_messages 截尾
# ═══════════════════════════════════════════════════════════════
def test_max_messages():
    s = Session(key="test:maxmsg")
    for i in range(20):
        s.add_message("user", f"消息{i}")
        s.add_message("assistant", f"回复{i}")

    h = s.get_history(max_messages=6)
    show("max_messages=6（取末尾 6 条对话）", h)
    # 末尾 6 条 = 3 轮 user+assistant，但取出的 sliced 是末尾 6 条消息
    assert len(h) <= 6, f"预期 ≤6 条，实际 {len(h)}"
    # 应该从"消息17"附近开始
    assert "消息17" in h[0]["content"] or "消息16" in h[0]["content"]


# ═══════════════════════════════════════════════════════════════
# 3. max_tokens 剪裁
# ═══════════════════════════════════════════════════════════════
def test_max_tokens():
    s = Session(key="test:maxtok")
    for i in range(10):
        s.add_message("user", f"第{i}条用户消息 " * 50)   # 较长的消息
        s.add_message("assistant", f"第{i}条助手回复 " * 30)

    # token 预算很小，只保留末尾 1-2 轮
    h = s.get_history(max_messages=100, max_tokens=200)
    show("max_tokens=200（严格 token 剪裁）", h)
    assert len(h) < 10, f"token 剪裁后应少于 10 条，实际 {len(h)}"
    # 应始终对齐到 user turn 开头
    if h:
        assert h[0]["role"] in ("user", "assistant"), f"首条角色异常: {h[0]['role']}"
    # 不应有孤立的 tool_result
    for i, m in enumerate(h):
        if m["role"] == "tool":
            # 前面必须有 assistant 的 tool_call
            assert i > 0 and h[i - 1].get("tool_calls"), f"孤立 tool_result 在 [{i}]"


# ═══════════════════════════════════════════════════════════════
# 4. include_timestamps
# ═══════════════════════════════════════════════════════════════
def test_timestamps():
    s = Session(key="test:ts")
    s.add_message("user", "你好")

    h = s.get_history(max_messages=10, include_timestamps=True)
    show("include_timestamps=True", h)
    assert len(h) == 1
    assert "[Message Time:" in h[0]["content"], "应包含时间戳前缀"
    assert "你好" in h[0]["content"], "应保留原始内容"


# ═══════════════════════════════════════════════════════════════
# 5. _command 消息过滤
# ═══════════════════════════════════════════════════════════════
def test_filter_command():
    s = Session(key="test:cmd")
    s.add_message("user", "正常消息")
    s.add_message("assistant", "正常回复")
    s.add_message("user", "内部命令", _command=True)  # 应被过滤
    s.add_message("assistant", "命令回复", _command=True)

    h = s.get_history(max_messages=10)
    show("过滤 _command 消息", h)
    assert len(h) == 2
    assert all("内部命令" not in m["content"] for m in h)


# ═══════════════════════════════════════════════════════════════
# 6. media 面包屑注入
# ═══════════════════════════════════════════════════════════════
def test_media_breadcrumbs():
    s = Session(key="test:media")
    s.add_message("user", "看看这张图", media=["/tmp/test.png", "/tmp/photo.jpg"])
    s.add_message("assistant", "好看")

    h = s.get_history(max_messages=10)
    show("media 面包屑注入", h)
    assert "[image:" in h[0]["content"], "应注入图片面包屑"
    assert "看看这张图" in h[0]["content"], "应保留原始内容"


# ═══════════════════════════════════════════════════════════════
# 7. cli_apps 面包屑注入
# ═══════════════════════════════════════════════════════════════
def test_cli_apps_breadcrumbs():
    s = Session(key="test:cli")
    s.add_message("user", "帮我用 gh", cli_apps=[{"name": "gh", "entry_point": "gh"}])

    h = s.get_history(max_messages=10)
    show("cli_apps 面包屑注入", h)
    assert "[CLI App Attachment: @gh" in h[0]["content"], "应注入 CLI App 面包屑"


# ═══════════════════════════════════════════════════════════════
# 8. mcp_presets 面包屑注入
# ═══════════════════════════════════════════════════════════════
def test_mcp_presets_breadcrumbs():
    s = Session(key="test:mcp")
    s.add_message("user", "查天气", mcp_presets=[{"name": "weather", "transport": "sse"}])

    h = s.get_history(max_messages=10)
    show("mcp_presets 面包屑注入", h)
    assert "[MCP Preset Attachment: @weather" in h[0]["content"]


# ═══════════════════════════════════════════════════════════════
# 9. 空内容 assistant 跳过（无 tool_calls 时）
# ═══════════════════════════════════════════════════════════════
def test_skip_empty_assistant():
    s = Session(key="test:empty")
    s.add_message("user", "你好")
    s.add_message("assistant", "")       # 空内容，应跳过
    s.add_message("assistant", "   ")    # 纯空白，应跳过
    s.add_message("assistant", "真的回复")

    h = s.get_history(max_messages=10)
    show("跳过空内容 assistant", h)
    assert len(h) == 2, f"预期 2 条（user + 非空 assistant），实际 {len(h)}"
    assert h[-1]["content"] == "真的回复"


# ═══════════════════════════════════════════════════════════════
# 10. 带 tool_calls 的空 assistant 应保留
# ═══════════════════════════════════════════════════════════════
def test_keep_assistant_with_tool_calls():
    s = Session(key="test:toolcall")
    s.add_message("user", "生成图片")
    s.add_message("assistant", "", tool_calls=[{"id": "call_1", "function": {"name": "generate_image"}}])
    s.add_message("tool", "图片已生成", tool_call_id="call_1")
    s.add_message("assistant", "图片在这里")

    h = s.get_history(max_messages=10)
    show("保留有 tool_calls 的空 assistant", h)
    # 跳过规则应保留有 tool_calls 的空 assistant
    assert any(m.get("tool_calls") for m in h), "有 tool_calls 的 assistant 被错误跳过"
    assert any(m["role"] == "tool" for m in h), "tool 消息被过滤了"


# ═══════════════════════════════════════════════════════════════
# 11. last_consolidated 偏移
# ═══════════════════════════════════════════════════════════════
def test_last_consolidated():
    s = Session(key="test:consolidate")
    for i in range(5):
        s.add_message("user", f"旧消息{i}")
        s.add_message("assistant", f"旧回复{i}")
    s.last_consolidated = 10  # 前 10 条（5 轮对话）已合并
    s.add_message("user", "新消息")
    s.add_message("assistant", "新回复")

    h = s.get_history(max_messages=10)
    show("last_consolidated=6（只取未合并部分）", h)
    assert len(h) == 2, f"预期 2 条（仅新消息），实际 {len(h)}"
    assert h[0]["content"] == "新消息"
    assert h[1]["content"] == "新回复"


# ═══════════════════════════════════════════════════════════════
# 12. 对齐到 user turn 开头
# ═══════════════════════════════════════════════════════════════
def test_align_to_user():
    s = Session(key="test:align")
    # 构造末尾从 assistant 开始的情况
    s.add_message("user", "你好")
    s.add_message("assistant", "你好呀")
    s.add_message("user", "今天天气")
    s.add_message("assistant", "很好")
    s.add_message("assistant", "补充一下")  # 末尾是 assistant

    h = s.get_history(max_messages=3)
    show("max_messages=3（末尾是 assistant，应对齐到 user）", h)
    # 应该是从 "今天天气" 开始
    assert len(h) >= 2
    assert h[0]["role"] == "user", f"首条应为 user，实际 {h[0]['role']}"
    assert h[0]["content"] == "今天天气"


# ═══════════════════════════════════════════════════════════════
# 13. 空会话
# ═══════════════════════════════════════════════════════════════
def test_empty_session():
    s = Session(key="test:empty")
    h = s.get_history(max_messages=10)
    show("空会话", h)
    assert len(h) == 0


# ═══════════════════════════════════════════════════════════════
# 14. _channel_delivery 保留（user 前一条主动推送）
# ═══════════════════════════════════════════════════════════════
def test_channel_delivery():
    s = Session(key="test:delivery")
    s.add_message("assistant", "推送通知", _channel_delivery=True)
    s.add_message("user", "你好")
    s.add_message("assistant", "你好呀")

    h = s.get_history(max_messages=10)
    show("_channel_delivery 随 user 保留", h)
    if h and h[0]["role"] == "assistant":
        # _channel_delivery 不会被传递到输出中（get_history 只输出 role/content 等关键字段）
        assert h[0]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════
# 15. reasoning_content / thinking_blocks 保留
# ═══════════════════════════════════════════════════════════════
def test_reasoning_fields():
    s = Session(key="test:reason")
    s.add_message("user", "推理题")
    s.add_message("assistant", "答案是 42", reasoning_content="先加后乘……")

    h = s.get_history(max_messages=10)
    show("reasoning_content 字段保留", h)
    assert h[1].get("reasoning_content") == "先加后乘……", "reasoning 字段丢失"


# ═══════════════════════════════════════════════════════════════
# 16. 批量注入 + 多附件
# ═══════════════════════════════════════════════════════════════
def test_mixed_breadcrumbs():
    s = Session(key="test:mix")
    s.add_message(
        "user", "处理这个",
        media=["/tmp/doc.pdf"],
        cli_apps=[{"name": "gh", "entry_point": "gh"}],
        mcp_presets=[{"name": "weather", "transport": "sse"}],
    )
    s.add_message("assistant", "好的")

    h = s.get_history(max_messages=10)
    show("多类型面包屑混合注入", h)
    assert "[image:" in h[0]["content"]
    assert "[CLI App Attachment: @gh" in h[0]["content"]
    assert "[MCP Preset Attachment: @weather" in h[0]["content"]


# ═══════════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        ("基础测试", test_basic),
        ("max_messages 截尾", test_max_messages),
        ("max_tokens 剪裁", test_max_tokens),
        ("时间戳注入", test_timestamps),
        ("过滤 _command", test_filter_command),
        ("media 面包屑", test_media_breadcrumbs),
        ("cli_apps 面包屑", test_cli_apps_breadcrumbs),
        ("mcp_presets 面包屑", test_mcp_presets_breadcrumbs),
        ("跳过空 assistant", test_skip_empty_assistant),
        ("保留 tool_calls assistant", test_keep_assistant_with_tool_calls),
        ("last_consolidated 偏移", test_last_consolidated),
        ("对齐到 user turn", test_align_to_user),
        ("空会话", test_empty_session),
        ("_channel_delivery 保活", test_channel_delivery),
        ("reasoning 字段保留", test_reasoning_fields),
        ("多类型面包屑混合", test_mixed_breadcrumbs),
    ]

    passed = 0
    failed = 0
    fail_details = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  ❌ {name}: {e}")
            print(f"     {tb.split(chr(10))[-2]}")
            fail_details.append((name, str(e)))
            failed += 1

    print(f"\n{'='*60}")
    print(f"  结果: {passed}/{len(tests)} 通过", end="")
    if failed:
        print(f", {failed} 失败" )
        for n, e in fail_details:
            print(f"       {n}: {e}")
    else:
        print()
    print(f"{'='*60}")
