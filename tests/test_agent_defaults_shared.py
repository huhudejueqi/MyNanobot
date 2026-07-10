"""演示 AgentDefaults() 如何作为共享默认值的单一来源。

原版 loop.py:230 和 subagent.py:91 都通过 AgentDefaults() 拿默认值，
而不是各自写死数字。本测试展示这个模式的效果。
"""
from __future__ import annotations

from nanobot.config.schema import AgentDefaults


class TestAgentDefaultsAsSingleSource:
    """验证所有模块共享同一份默认值定义。"""

    def test_module_level_constant_from_agent_defaults(self):
        """测试中也用 AgentDefaults() 拿默认值，而非硬编码。"""
        print("\n  [1] 测试也用 AgentDefaults() 拿默认值，而非硬编码")
        max_chars = AgentDefaults().max_tool_result_chars
        max_iter = AgentDefaults().max_tool_iterations
        print(f"      _MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars → {max_chars}")
        print(f"      _MAX_TOOL_ITERATIONS   = AgentDefaults().max_tool_iterations   → {max_iter}")
        assert max_chars == 16_000
        assert max_iter == 200

    def test_agent_defaults_are_immutable_fallback_source(self):
        """每次调 AgentDefaults() 都是新实例，不会互相污染。"""
        print("\n  [2] 每次调 AgentDefaults() 都是新实例，不会互相污染")
        d1 = AgentDefaults()
        d2 = AgentDefaults()
        print(f"      d1 is d2 → {d1 is d2}（不同实例）")
        d1.disabled_skills.append("shell")
        print(f"      d1.disabled_skills = {d1.disabled_skills}")
        print(f"      d2.disabled_skills = {d2.disabled_skills}（不受影响）")
        assert d1 is not d2
        assert "shell" in d1.disabled_skills
        assert d2.disabled_skills == []

    def test_loop_and_subagent_share_same_defaults(self):
        """loop.py 和 subagent.py 都依赖 AgentDefaults() 拿 fallback 值。"""
        print("\n  [3] loop.py 和 subagent.py 共享相同的 fallback 值")
        defaults = AgentDefaults()
        loop_max_iter = (
            None if None is not None else defaults.max_tool_iterations
        )
        subagent_max_iter = (
            None if None is not None else defaults.max_tool_iterations
        )
        subagent_max_concurrent = (
            None if None is not None else defaults.max_concurrent_subagents
        )
        print(f"      loop.py:     max_iterations          = defaults.max_tool_iterations          → {loop_max_iter}")
        print(f"      subagent.py: max_iterations          = defaults.max_tool_iterations          → {subagent_max_iter}")
        print(f"      subagent.py: max_concurrent_subagents = defaults.max_concurrent_subagents     → {subagent_max_concurrent}")
        assert loop_max_iter == subagent_max_iter == 200
        assert subagent_max_concurrent == 1

    def test_mimic_loop_constructor_pattern(self):
        """模拟 loop.py 的 __init__ 怎样用 AgentDefaults() 做 fallback。"""
        print("\n  [4] 模拟 loop.__init__ 的 fallback 行为")

        class FakeLoop:
            def __init__(self, max_iterations: int | None = None):
                defaults = AgentDefaults()
                self.max_iterations = (
                    max_iterations
                    if max_iterations is not None
                    else defaults.max_tool_iterations
                )

        loop1 = FakeLoop()
        loop2 = FakeLoop(max_iterations=50)
        print(f"      FakeLoop()                   → max_iterations = {loop1.max_iterations}（fallback 到 AgentDefaults）")
        print(f"      FakeLoop(max_iterations=50)  → max_iterations = {loop2.max_iterations}（用传进来的值）")
        assert loop1.max_iterations == 200
        assert loop2.max_iterations == 50

    def test_what_happens_when_default_changes(self):
        """展示 AgentDefaults 是整个系统的"唯一真相来源"：改 schema 一处，所有调用方自动跟着变。"""
        print("\n  [5] 一处修改，处处生效")

        # loop.py、subagent.py、测试都通过 AgentDefaults() 拿默认值，
        # 而不是各写各的硬编码数字
        loop_fallback = AgentDefaults().max_tool_iterations          # loop.py
        subagent_fallback = AgentDefaults().max_tool_iterations       # subagent.py
        test_constant = AgentDefaults().max_tool_iterations           # 测试
        print(f"      loop.py:     AgentDefaults().max_tool_iterations → {loop_fallback}")
        print(f"      subagent.py: AgentDefaults().max_tool_iterations → {subagent_fallback}")
        print(f"      测试:        AgentDefaults().max_tool_iterations → {test_constant}")
        print(f"      三者值相同：{loop_fallback} == {subagent_fallback} == {test_constant}")
        assert loop_fallback == subagent_fallback == test_constant

        # 证明这个值来自 schema.py 的字段定义，不是硬编码
        raw_default = AgentDefaults.model_fields["max_tool_iterations"].default
        print(f"      源头：AgentDefaults.model_fields['max_tool_iterations'].default = {raw_default}")
        print(f"      schema.py:AgentDefaults 里改这个值，上面三处全部自动跟着变")
