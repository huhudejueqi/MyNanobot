"""SkillsLoader 的单元测试。

覆盖场景：
  - 列表加载：内置技能、工作区技能、禁用技能
  - 技能加载：单个加载、上下文拼接、frontmatter 清洗
  - 元数据解析：YAML frontmatter、nanobot 命名空间、兼容 openclaw
  - 依赖检查：bin 命令检查、环境变量检查、缺失详情
  - 特殊功能：always 技能、build_skills_summary、工作区覆盖
  - 边界：空工作区、不存在技能、损坏 frontmatter、无 frontmatter 的技能
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanobot.agent.skills import SkillsLoader, BUILTIN_SKILLS_DIR


# ═══════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════

def make_skill(base: Path, name: str, description: str = "",
               always: bool = False, requires: dict | None = None,
               body: str = "技能正文内容\n") -> Path:
    """在工作区目录下创建一个测试用的技能。"""
    skill_dir = base / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md_path = skill_dir / "SKILL.md"

    meta = {}
    if description:
        meta["description"] = description
    nano = {}
    if always:
        nano["always"] = True
    if requires:
        nano["requires"] = requires
    if nano:
        meta["metadata"] = {"nanobot": nano}

    frontmatter = "---\n"
    if meta:
        frontmatter += yaml_safe_dump(meta) + "\n"
    frontmatter += "---\n\n"

    md_path.write_text(frontmatter + body, encoding="utf-8")
    return md_path


def yaml_safe_dump(d: dict) -> str:
    """简单的 dict → YAML 序列化（避免引入 PyYAML 依赖）。"""
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    lines.append(f"  {sk}:")
                    for ssk, ssv in sv.items():
                        if isinstance(ssv, list):
                            items = ", ".join(f"\"{i}\"" for i in ssv)
                            lines.append(f"    {ssk}: [{items}]")
                        elif isinstance(ssv, bool):
                            lines.append(f"    {ssk}: {'true' if ssv else 'false'}")
                        else:
                            lines.append(f"    {ssk}: {ssv}")
                elif isinstance(sv, list):
                    items = ", ".join(f"\"{i}\"" for i in sv)
                    lines.append(f"  {sk}: [{items}]")
                elif isinstance(sv, bool):
                    lines.append(f"  {sk}: {'true' if sv else 'false'}")
                else:
                    lines.append(f"  {sk}: {sv}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def show(label: str, loader: SkillsLoader, *,
         list_all: bool = False, always: bool = False, summary: bool = False) -> None:
    """打印测试中间结果。"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    if list_all:
        skills = loader.list_skills(filter_unavailable=False)
        print(f"  技能总数: {len(skills)}")
        for s in skills:
            print(f"    {s['name']:<20} source={s['source']}")
    if always:
        a = loader.get_always_skills()
        print(f"  always 技能: {a}")
    if summary:
        s = loader.build_skills_summary()
        lines = s.split("\n")
        print(f"  摘要 ({len(s)} chars):")
        for line in lines[:4]:
            print(f"    {line}")
        if len(lines) > 4:
            print(f"    ... (共 {len(lines)} 行)")
    print(f"  {'─'*50}")


# ═══════════════════════════════════════════════════════════════════
#  1. 基础：加载内置技能列表
# ═══════════════════════════════════════════════════════════════════
def test_list_builtin_skills():
    """验证能从内置目录加载到技能。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))
        skills = loader.list_skills(filter_unavailable=False)
        assert len(skills) > 0, f"应加载到内置技能，实际 0"
        names = [s["name"] for s in skills]
        assert "weather" in names, f"应包含 weather，实际有 {names}"
        assert "cron" in names, f"应包含 cron"
        show("1. 内置技能列表", loader, list_all=True)


# ═══════════════════════════════════════════════════════════════════
#  2. 加载单个技能内容
# ═══════════════════════════════════════════════════════════════════
def test_load_skill():
    """验证 load_skill 能正确读取 SKILL.md 内容。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        # 正常技能
        content = loader.load_skill("weather")
        assert content is not None, "weather 技能应存在"
        assert "wttr.in" in content, "应包含天气服务描述"
        print(f"  weather 技能: {len(content)} chars")

        # 不存在的技能
        none_content = loader.load_skill("nonexistent_skill")
        assert none_content is None, "不存在的技能应返回 None"
        print("  不存在技能 → None ✓")


# ═══════════════════════════════════════════════════════════════════
#  3. 元数据解析
# ═══════════════════════════════════════════════════════════════════
def test_get_metadata():
    """验证能正确解析技能的 YAML frontmatter。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        meta = loader.get_skill_metadata("weather")
        assert meta is not None, "weather 应有 frontmatter"
        assert meta.get("description"), "应有描述"
        assert "metadata" in meta, "应有 metadata 字段"
        print(f"  weather 元数据: description={meta.get('description')}")
        print(f"                 keys={list(meta.keys())}")

        # 无 frontmatter 的技能（没有 --- 开头的技能）
        meta2 = loader.get_skill_metadata("nonexistent")
        assert meta2 is None, "不存在的技能应返回 None"
        print("  不存在技能 metadata → None ✓")


# ═══════════════════════════════════════════════════════════════════
#  4. frontmatter 清洗
# ═══════════════════════════════════════════════════════════════════
def test_strip_frontmatter():
    """验证 load_skills_for_context 能正确移除 YAML 头部。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        ctx = loader.load_skills_for_context(["weather"])
        assert "### Skill: weather" in ctx
        # frontmatter 中的 YAML 不应出现在上下文中
        assert "---\n" not in ctx, "frontmatter 应在上下文移除"
        # 但技能正文应保留
        assert "wttr.in" in ctx, "技能正文应保留"
        print(f"  weather 上下文: {len(ctx)} chars")
        print(f"  不含 frontmatter: {'---' not in ctx} ✓")


# ═══════════════════════════════════════════════════════════════════
#  5. 依赖检查
# ═══════════════════════════════════════════════════════════════════
def test_requirements_check():
    """验证依赖检查和可用性查询。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        # 依赖满足的技能
        avail, reason = loader.get_skill_availability("weather")
        print(f"  weather: available={avail}, reason={reason!r}")
        # weather 依赖 curl，多数系统都有

        # 查询依赖详情
        reqs = loader.get_skill_requirements("weather")
        assert "bins" in reqs
        assert "missing_bins" in reqs
        assert "missing_env" in reqs
        print(f"  weather requires.bins={reqs['bins']}")
        print(f"  weather missing.bins={reqs['missing_bins']}")

        # 技能不存在时的 availability
        avail2, reason2 = loader.get_skill_availability("no_such_skill")
        # 不存在的技能：_get_skill_meta 返回 {}，_check_requirements 返回 True
        # 因为空依赖视为满足
        assert avail2 is True, f"不存在的技能无依赖应视为可用，实际 {avail2}"
        print(f"  不存在技能: available={avail2} ✓")


# ═══════════════════════════════════════════════════════════════════
#  6. 工作区技能覆盖内置技能
# ═══════════════════════════════════════════════════════════════════
def test_workspace_override():
    """验证工作区技能优先级高于内置技能。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        make_skill(ws, "weather",
                   description="工作区自定义天气",
                   body="这是自定义的工作区版本\n")

        loader = SkillsLoader(ws)

        # 验证列表中有且只有一个 weather（工作区覆盖内置）
        skills = loader.list_skills(filter_unavailable=False)
        weather_entries = [s for s in skills if s["name"] == "weather"]
        assert len(weather_entries) == 1, f"weather 应唯一，实际 {len(weather_entries)}"
        assert weather_entries[0]["source"] == "workspace", f"来源应为 workspace"

        # 验证加载的是工作区版本
        content = loader.load_skill("weather")
        assert content is not None
        assert "自定义的工作区版本" in content
        print("  工作区 weather 已覆盖内置 ✓")
        print(f"  来源: {weather_entries[0]['source']}")
        print(f"  正文包含自定义内容: {'自定义的工作区版本' in content}")


# ═══════════════════════════════════════════════════════════════════
#  7. 禁用技能
# ═══════════════════════════════════════════════════════════════════
def test_disabled_skills():
    """验证禁用技能过滤器。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir), disabled_skills={"weather", "cron"})

        skills = loader.list_skills(filter_unavailable=False)
        names = {s["name"] for s in skills}
        assert "weather" not in names, "weather 应被禁用"
        assert "cron" not in names, "cron 应被禁用"
        print(f"  禁用列表: weather, cron")
        print(f"  当前技能数: {len(skills)}")


# ═══════════════════════════════════════════════════════════════════
#  8. filter_unavailable 过滤
# ═══════════════════════════════════════════════════════════════════
def test_filter_unavailable():
    """验证 filter_unavailable 能过滤依赖不满足的技能。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        all_skills = loader.list_skills(filter_unavailable=False)
        available_skills = loader.list_skills(filter_unavailable=True)

        # 过滤后数量应 ≤ 全部
        assert len(available_skills) <= len(all_skills)
        unavailable_count = len(all_skills) - len(available_skills)
        print(f"  全部: {len(all_skills)}, 可用: {len(available_skills)}")
        print(f"  因依赖不可用: {unavailable_count}")


# ═══════════════════════════════════════════════════════════════════
#  9. build_skills_summary
# ═══════════════════════════════════════════════════════════════════
def test_build_summary():
    """验证 build_skills_summary 生成正确的 Markdown 摘要。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        summary = loader.build_skills_summary()
        assert "**weather**" in summary, "摘要应包含 weather"
        assert summary.count("**") >= 4  # 至少有一个技能
        print(f"  摘要长度: {len(summary)} chars")
        # 打印前几行
        for line in summary.split("\n")[:3]:
            print(f"    {line}")

        # 测试 exclude
        filtered = loader.build_skills_summary(exclude={"weather"})
        assert "weather" not in filtered
        print(f"  排除 weather 后: 摘要包含 weather = {'weather' in filtered}")

        # 空工作区 + exclude 全部
        with tempfile.TemporaryDirectory() as tmpdir2:
            loader2 = SkillsLoader(Path(tmpdir2))
            empty = loader2.build_skills_summary(exclude={"weather", "cron", "memory", "github",
                                                           "tmux", "my", "summarize",
                                                           "update-setup", "skill-creator",
                                                           "image-generation", "long-goal",
                                                           "clawhub"})
            # 如果 exclude 了全部内置技能，应返回空字符串或剩余技能
            # 具体取决于内置技能有哪些
            print(f"  排除全部后摘要长度: {len(empty)} chars")


# ═══════════════════════════════════════════════════════════════════
#  10. always 技能
# ═══════════════════════════════════════════════════════════════════
def test_always_skills():
    """验证 get_always_skills 能正确返回始终加载的技能。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        always = loader.get_always_skills()
        # memory 和 my 有 always: true
        assert "memory" in always, f"memory 应该是 always 技能，实际 {always}"
        assert "my" in always, f"my 应该是 always 技能，实际 {always}"
        # weather 没有 always
        assert "weather" not in always, f"weather 不应是 always 技能"
        show("10. Always 技能", loader, always=True)


# ═══════════════════════════════════════════════════════════════════
#  11. load_skills_for_context 多技能拼接
# ═══════════════════════════════════════════════════════════════════
def test_context_multiple():
    """验证多技能上下文拼接格式。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        ctx = loader.load_skills_for_context(["weather", "cron"])
        assert "### Skill: weather" in ctx
        assert "### Skill: cron" in ctx
        # 两个技能之间应有分隔符
        assert "---" in ctx
        print(f"  多技能上下文: {len(ctx)} chars")
        print(f"  包含 weather: {'weather' in ctx}")
        print(f"  包含 cron: {'cron' in ctx}")

        # 空列表
        ctx_empty = loader.load_skills_for_context([])
        assert ctx_empty == "", "空列表应返回空字符串"
        print(f"  空列表: {repr(ctx_empty)} ✓")


# ═══════════════════════════════════════════════════════════════════
#  12. 空工作区 + 只有内置技能
# ═══════════════════════════════════════════════════════════════════
def test_empty_workspace():
    """验证空工作区仍能加载内置技能。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        # 不创建 skills 目录
        loader = SkillsLoader(ws)

        skills = loader.list_skills(filter_unavailable=False)
        assert len(skills) > 0, "空工作区也应能加载内置技能"
        # 所有技能来源应为 builtin
        assert all(s["source"] == "builtin" for s in skills)
        print(f"  空工作区: {len(skills)} 个内置技能 ✓")


# ═══════════════════════════════════════════════════════════════════
#  13. 自定义 builtin_skills_dir
# ═══════════════════════════════════════════════════════════════════
def test_custom_builtin_dir():
    """验证可以传入自定义的内置技能目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = Path(tmpdir) / "custom_skills"
        make_skill(Path(tmpdir), "my-tool", description="自定义工具", body="用法说明")
        # make_skill 会在 <tmpdir>/skills/ 下创建，需要移动到 custom_dir
        src = Path(tmpdir) / "skills" / "my-tool"
        dst = custom_dir / "my-tool"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

        loader = SkillsLoader(Path(tmpdir) / "empty_ws",
                              builtin_skills_dir=custom_dir)

        skills = loader.list_skills(filter_unavailable=False)
        names = {s["name"] for s in skills}
        assert "my-tool" in names
        assert all(s["source"] == "builtin" for s in skills)
        print(f"  自定义内置目录: my-tool 已加载 ✓")


# ═══════════════════════════════════════════════════════════════════
#  14. 损坏 frontmatter 的容错
# ═══════════════════════════════════════════════════════════════════
def test_corrupted_frontmatter():
    """验证损坏的 frontmatter 不会导致崩溃。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        skill_dir = ws / "skills" / "broken"
        skill_dir.mkdir(parents=True)
        # 写一个 YAML 格式错误的前言
        (skill_dir / "SKILL.md").write_text(
            "---\ninvalid: [unclosed\ndescription: 坏掉的\n---\n\n内容",
            encoding="utf-8",
        )

        loader = SkillsLoader(ws)
        # 元数据解析失败应返回 None
        meta = loader.get_skill_metadata("broken")
        assert meta is None, "损坏的 frontmatter 应返回 None"

        # 但仍然能加载到原始内容
        content = loader.load_skill("broken")
        assert content is not None
        print(f"  损坏 frontmatter: metadata=None ✓, content 可加载 ✓")


# ═══════════════════════════════════════════════════════════════════
#  15. 无 frontmatter 的技能
# ═══════════════════════════════════════════════════════════════════
def test_no_frontmatter():
    """验证没有 frontmatter 的技能也能正常处理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        skill_dir = ws / "skills" / "plain"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# 简单技能\n\n没有元数据。", encoding="utf-8")

        loader = SkillsLoader(ws)

        # metadata 应为 None
        meta = loader.get_skill_metadata("plain")
        assert meta is None

        # load_skills_for_context 应正常工作（只是不洗掉任何东西）
        ctx = loader.load_skills_for_context(["plain"])
        assert '简单技能' in ctx
        print(f'  无 frontmatter: metadata=None, 内容含 "简单技能" ✓')


# ═══════════════════════════════════════════════════════════════════
#  16. _parse_nanobot_metadata 兼容性
# ═══════════════════════════════════════════════════════════════════
def test_parse_metadata_compat():
    """验证 nanobot/openclaw 命名空间兼容。"""
    from nanobot.agent.skills import SkillsLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillsLoader(Path(tmpdir))

        # 测试 dict 输入
        result = loader._parse_nanobot_metadata({"nanobot": {"always": True}})
        assert result.get("always") is True

        # 测试 openclaw 兼容
        result2 = loader._parse_nanobot_metadata({"openclaw": {"always": True}})
        assert result2.get("always") is True

        # 测试 JSON 字符串
        result3 = loader._parse_nanobot_metadata('{"nanobot": {"always": true}}')
        assert result3.get("always") is True

        # 测试无效输入
        assert loader._parse_nanobot_metadata(None) == {}
        assert loader._parse_nanobot_metadata([]) == {}
        assert loader._parse_nanobot_metadata("not json") == {}

        print("  nanobot/openclaw 兼容 ✓")
        print("  JSON 字符串解析 ✓")
        print("  无效输入容错 ✓")


# ═══════════════════════════════════════════════════════════════════
#  运行
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        ("内置技能列表", test_list_builtin_skills),
        ("加载单个技能", test_load_skill),
        ("元数据解析", test_get_metadata),
        ("frontmatter 清洗", test_strip_frontmatter),
        ("依赖检查", test_requirements_check),
        ("工作区覆盖", test_workspace_override),
        ("禁用技能", test_disabled_skills),
        ("可用性过滤", test_filter_unavailable),
        ("技能摘要", test_build_summary),
        ("Always 技能", test_always_skills),
        ("多技能上下文", test_context_multiple),
        ("空工作区", test_empty_workspace),
        ("自定义内置目录", test_custom_builtin_dir),
        ("损坏 frontmatter", test_corrupted_frontmatter),
        ("无 frontmatter", test_no_frontmatter),
        ("元数据兼容性", test_parse_metadata_compat),
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
        print(f", {failed} 失败")
        for n, e in fail_details:
            print(f"       {n}: {e}")
    else:
        print()
    print(f"{'='*60}")
