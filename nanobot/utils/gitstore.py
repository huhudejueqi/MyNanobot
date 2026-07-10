"""Git 版本控制存储：基于 dulwich（纯 Python Git 库）的内存文件版本管理。

核心功能：
  - 在工作区初始化轻量级 Git 仓库
  - 自动检测变更并提交（auto_commit）
  - 查看提交历史（log）
  - 查看文件每行最后修改时间（line_ages，即 git blame）
  - 对比两个版本的差异（diff_commits）
  - 回退到指定版本（revert）

为什么用 Git + dulwich 而非直接备份文件：
  - 增量存储：只保存变更部分，不每次全量复制
  - 天然历史追溯：谁在什么时候改了什么都一清二楚
  - 与 Dream 系统的记忆管理深度集成
  - 纯 Python 实现（dulwich），不依赖 git CLI

典型使用场景：
  - Dream 自动更新 SOUL.md / USER.md 后自动提交
  - Agent 修改 MEMORY.md 后自动提交
  - 用户查看记忆文件的修改历史
  - 回退不满意的修改
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass
class CommitInfo:
    """单条提交信息的展示模型。

    属性：
      sha:       短 SHA（8 位十六进制）
      message:   提交消息
      timestamp: 格式化的时间戳字符串

    用法：
      >>> info = CommitInfo(sha="abc12345", message="fix: 更新用户偏好",
      ...                   timestamp="2026-07-05 14:30")
      >>> print(info.format())
      ## fix: 更新用户偏好
      `abc12345` — 2026-07-05 14:30
      (no file changes)

      >>> print(info.format(diff="@@ -1 +1 @@\n-old\n+new"))
      ## fix: 更新用户偏好
      ...
      ```diff
      @@ -1 +1 @@
      -old
      +new
      ```
    """
    sha: str          # 8 位短 SHA
    message: str      # 提交消息
    timestamp: str    # 格式化的时间戳，如 "2026-07-05 14:30"

    def format(self, diff: str = "") -> str:
        """格式化为可读的提交信息展示文本，可选附带 diff。

        参数：
          diff: 差异文本（如果传入，会放在代码块中展示）

        返回：
          格式化的 Markdown 文本
        """
        header = f"## {self.message.splitlines()[0]}\n`{self.sha}` — {self.timestamp}\n"
        if diff:
            return f"{header}\n```diff\n{diff}\n```"
        return f"{header}\n(no file changes)"


@dataclass
class LineAge:
    """通过 git blame 分析出的单行代码/文本的时效信息。

    用于判断记忆文件中的某条信息是否已经过时。
    Dream 系统会根据这个数据决定是否需要更新某些记忆条目。

    属性：
      age_days: 距离最后一次修改的天数

    用法：
      >>> age = LineAge(age_days=30)
      >>> if age.age_days > 7:
      ...     print("这条信息超过一周未更新，可能需要重新确认")
    """
    age_days: int  # 距离最后一次修改的天数


def _compute_line_ages(annotated) -> list[LineAge]:
    """将 dulwich 的 annotate 结果转换为每行的时效数据列表。

    dulwich.porcelain.annotate() 返回的数据结构是：
      [(commit, tree_entry), line_bytes), ...]
    每个元素对应文件中的一行，包含该行最后一次提交的 commit 对象。

    此函数遍历所有行，计算每行从最后一次修改到现在的天数，
    返回与文件行数等长的 LineAge 列表。

    参数：
      annotated: dulwich.porcelain.annotate() 的原始返回值

    返回：
      LineAge 列表，第 i 个元素对应文件的第 i 行
    """
    now = datetime.now(tz=timezone.utc).date()
    ages: list[LineAge] = []
    for (commit, _tree_entry), _line_bytes in annotated:
        dt = datetime.fromtimestamp(commit.commit_time, tz=timezone.utc).date()
        ages.append(LineAge(age_days=(now - dt).days))
    return ages


class GitStore:
    """Git 版本控制存储：为工作区中的关键文件提供版本管理。

    GitStore 管理的不是整个工作区，而是指定的几个跟踪文件（tracked_files），
    通常是 Dream 系统维护的记忆文件（SOUL.md、USER.md、MEMORY.md 等）。

    .gitignore 策略：
      默认忽略所有文件（/*），只放行跟踪文件所在目录和跟踪文件本身。
      这样即使工作区有其他文件，也不会被意外加入到版本控制中。

    参数：
      workspace:     工作区目录路径
      tracked_files: 要跟踪的文件列表（相对于 workspace 的路径）

    用法：
      >>> store = GitStore(workspace_path,
      ...                  tracked_files=["SOUL.md", "USER.md", "memory/MEMORY.md"])
      >>> store.init()      # 初始化仓库
      >>> store.auto_commit("更新用户偏好")  # 自动提交变更
      >>> store.log(5)       # 查看最近 5 条提交
    """

    def __init__(self, workspace: Path, tracked_files: list[str]):
        """初始化 GitStore。

        参数：
          workspace:     工作区目录路径（在其中创建 .git 仓库）
          tracked_files: 要跟踪的文件路径列表（相对于 workspace）

        用法：
          >>> store = GitStore(Path("/home/user/.nanobot/workspace"),
          ...                  tracked_files=["SOUL.md", "USER.md"])
        """
        self._workspace = workspace
        self._tracked_files = tracked_files

    def is_initialized(self) -> bool:
        """检查工作区是否已经初始化了 Git 仓库。

        检测方式：检查工作区目录下是否存在 .git 目录。

        返回：
          True 表示已初始化，False 表示尚未初始化
        """
        return (self._workspace / ".git").is_dir()

    # ── 初始化 ──────────────────────────────────────────────────────────

    def init(self) -> bool:
        """在工作区初始化 Git 仓库（如果尚未初始化）。

        初始化流程：
          1. 检查是否已初始化 → 已初始化则跳过
          2. 检查是否已在某个外部 Git 仓库内 → 避免嵌套仓库，跳过
          3. 调用 dulwich.porcelain.init() 创建空仓库
          4. 写入 .gitignore（如果已存在则合并，不覆盖用户已有规则）
          5. 创建跟踪文件（如果不存在则创建空文件占位）
          6. 执行首次提交 init: nanobot memory store

        返回：
          True 表示新仓库创建成功，False 表示已存在、在外部仓库内或失败

        用法：
          >>> store = GitStore(workspace, ["SOUL.md", "USER.md"])
          >>> if store.init():
          ...     print("仓库已初始化")
          ... else:
          ...     print("已存在或初始化失败")
        """
        if self.is_initialized():
            return False

        if self._is_inside_git_repo():
            logger.warning(
                "工作区 {} 已在某个 Git 仓库中，跳过内嵌仓库初始化",
                self._workspace,
            )
            return False

        try:
            from dulwich import porcelain

            # 创建空仓库
            porcelain.init(str(self._workspace))

            # ── 处理 .gitignore ──
            gitignore = self._workspace / ".gitignore"
            dream_entries = self._build_gitignore()
            if gitignore.exists():
                # 合并：用户已有的规则不动，只补充缺失的 Dream 条目
                existing = gitignore.read_text(encoding="utf-8")
                existing_lines = set(existing.splitlines())
                new_lines = [
                    line
                    for line in dream_entries.splitlines()
                    if line not in existing_lines
                ]
                if new_lines:
                    merged = existing.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"
                    gitignore.write_text(merged, encoding="utf-8")
            else:
                gitignore.write_text(dream_entries, encoding="utf-8")

            # ── 确保跟踪文件存在（空文件占位） ──
            for rel in self._tracked_files:
                p = self._workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text("", encoding="utf-8")

            # ── 首次提交 ──
            porcelain.add(str(self._workspace), paths=[".gitignore"] + self._tracked_files)
            porcelain.commit(
                str(self._workspace),
                message=b"init: nanobot memory store",
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            logger.info("GitStore 已初始化: {}", self._workspace)
            return True
        except Exception:
            logger.exception("GitStore 初始化失败: {}", self._workspace)
            return False

    # ── 日常操作 ────────────────────────────────────────────────────────

    def auto_commit(self, message: str) -> str | None:
        """自动提交跟踪文件的变更。

        检查跟踪文件是否有未提交的变更，有则执行 add + commit。
        如果没有变更则跳过，不产生空提交。

        参数：
          message: 提交消息，如 "更新用户时区"

        返回：
          8 位短 SHA，无变更或失败返回 None

        用法：
          >>> sha = store.auto_commit("Dream: 更新用户语言偏好")
          >>> if sha:
          ...     print(f"已提交 {sha}")
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich import porcelain

            # 检查是否有变更
            st = porcelain.status(str(self._workspace))
            if not st.unstaged and not any(st.staged.values()):
                return None

            msg_bytes = message.encode("utf-8") if isinstance(message, str) else message
            porcelain.add(str(self._workspace), paths=self._tracked_files)
            sha_bytes = porcelain.commit(
                str(self._workspace),
                message=msg_bytes,
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            if sha_bytes is None:
                return None
            sha = sha_bytes.hex()[:8]
            logger.debug("Git 自动提交: {} ({})", sha, message)
            return sha
        except Exception:
            logger.exception("Git 自动提交失败: {}", message)
            return None

    # ── 内部辅助方法 ────────────────────────────────────────────────────

    def _resolve_sha(self, short_sha: str) -> bytes | None:
        """将短的 SHA 前缀解析为完整的 20 字节 SHA。

        从 HEAD 开始沿着父链回溯，找到第一个以 short_sha 开头的完整 SHA。

        参数：
          short_sha: 短 SHA 前缀（如 "abc12345"）

        返回：
          完整的 SHA 字节序列，找不到返回 None
        """
        try:
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                try:
                    sha = repo.refs[b"HEAD"]
                except KeyError:
                    return None

                while sha:
                    if sha.hex().startswith(short_sha):
                        return sha
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        break
                    sha = commit.parents[0] if commit.parents else None
            return None
        except Exception:
            return None

    def _is_inside_git_repo(self) -> bool:
        """检查工作区是否已经在某个外部 Git 仓库中。

        从工作区目录开始逐级向上查找，直到文件系统根目录。
        如果任意父目录包含 .git 目录或 .git 文件，则说明在外部仓库内。

        .git 文件 vs .git 目录：
          - 普通仓库：.git 是目录
          - Git 子模块 / worktree：.git 是文件（内容指向实际仓库位置）
        所以统一用 .exists() 检查而非 .is_dir()。

        返回：
          True 表示在外部仓库内（此时不应初始化内嵌仓库）
        """
        current = self._workspace.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    def _build_gitignore(self) -> str:
        """生成 .gitignore 内容：只放行跟踪文件和 .gitignore 自身。

        策略：
          /*              → 忽略所有文件
          !<dir>/         → 放行跟踪文件所在的目录
          !<file>         → 放行跟踪文件本身
          !.gitignore     → 放行 .gitignore

        这样即使工作区有大量临时文件，也不会被意外纳入版本控制。

        返回：
          .gitignore 的完整文本内容

        示例输出：
          /*
          !memory/
          !SOUL.md
          !USER.md
          !memory/MEMORY.md
          !.gitignore
        """
        dirs: set[str] = set()
        for f in self._tracked_files:
            parent = str(Path(f).parent)
            if parent != ".":
                dirs.add(parent)
        lines = ["/*"]
        for d in sorted(dirs):
            lines.append(f"!{d}/")
        for f in self._tracked_files:
            lines.append(f"!{f}")
        lines.append("!.gitignore")
        return "\n".join(lines) + "\n"

    # ── 查询接口 ────────────────────────────────────────────────────────

    def log(self, max_entries: int = 20) -> list[CommitInfo]:
        """返回简化的提交历史。

        从 HEAD 开始沿父链回溯，最多返回 max_entries 条提交。
        每条记录包含 8 位短 SHA、提交消息和格式化时间戳。

        参数：
          max_entries: 最多返回的提交数，默认 20

        返回：
          CommitInfo 列表，按时间倒序（最新的在前）

        用法：
          >>> for c in store.log(5):
          ...     print(f"{c.sha} {c.timestamp} {c.message[:50]}")
          abc12345 2026-07-05 14:30 Dream: 更新用户偏好
          def67890 2026-07-04 10:00 init: nanobot memory store
        """
        if not self.is_initialized():
            return []

        try:
            from dulwich.repo import Repo

            entries: list[CommitInfo] = []
            with Repo(str(self._workspace)) as repo:
                try:
                    head = repo.refs[b"HEAD"]
                except KeyError:
                    return []

                sha = head
                while sha and len(entries) < max_entries:
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        break
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(commit.commit_time),
                    )
                    msg = commit.message.decode("utf-8", errors="replace").strip()
                    entries.append(CommitInfo(
                        sha=sha.hex()[:8],
                        message=msg,
                        timestamp=ts,
                    ))
                    sha = commit.parents[0] if commit.parents else None

            return entries
        except Exception:
            logger.exception("Git log 查询失败")
            return []

    def line_ages(self, file_path: str) -> list[LineAge]:
        """通过 git blame 计算跟踪文件每行的最后修改时间（天数）。

        这个数据用于判断记忆文件中的内容是否过时：
          - age_days 很小的行 → 最近更新过的信息，较可信
          - age_days 很大的行 → 很久没动过的信息，可能需要重新确认

        Dream 系统会利用这个数据做记忆维护决策。

        参数：
          file_path: 文件路径（相对于 workspace）

        返回：
          LineAge 列表，长度等于文件行数；
          文件不存在、为空、或分析失败时返回空列表

        用法：
          >>> ages = store.line_ages("SOUL.md")
          >>> for i, age in enumerate(ages[:5], 1):
          ...     print(f"第{i}行: {age.age_days}天前修改")
        """
        if not self.is_initialized():
            return []

        target = self._workspace / file_path
        if not target.exists() or target.stat().st_size == 0:
            return []

        try:
            from dulwich import porcelain

            annotated = porcelain.annotate(str(self._workspace), file_path)
        except Exception:
            logger.exception("Git line_ages 分析失败: {}", file_path)
            return []

        if not annotated:
            return []

        return _compute_line_ages(annotated)

    def diff_commits(self, sha1: str, sha2: str) -> str:
        """对比两个提交之间的差异。

        参数：
          sha1: 第一个提交的短 SHA（旧版本）
          sha2: 第二个提交的短 SHA（新版本）

        返回：
          Unified diff 格式的差异文本；失败时返回空字符串

        用法：
          >>> diff = store.diff_commits("abc12345", "def67890")
          >>> print(diff)
          diff --git a/SOUL.md b/SOUL.md
          @@ -1,3 +1,4 @@
           # Soul
          +新添加的一行
        """
        if not self.is_initialized():
            return ""

        try:
            from dulwich import porcelain

            full1 = self._resolve_sha(sha1)
            full2 = self._resolve_sha(sha2)
            if not full1 or not full2:
                return ""

            out = io.BytesIO()
            porcelain.diff(
                str(self._workspace),
                commit=full1,
                commit2=full2,
                outstream=out,
            )
            return out.getvalue().decode("utf-8", errors="replace")
        except Exception:
            logger.exception("Git diff 对比失败")
            return ""

    def find_commit(self, short_sha: str, max_entries: int = 20) -> CommitInfo | None:
        """通过短 SHA 前缀查找指定的提交。

        参数：
          short_sha:   短 SHA 前缀（如 "abc12345"）
          max_entries: 最多搜索的提交数

        返回：
          CommitInfo 对象，找不到返回 None

        用法：
          >>> commit = store.find_commit("abc12")
          >>> if commit:
          ...     print(f"找到: {commit.sha} {commit.message}")
        """
        for c in self.log(max_entries=max_entries):
            if c.sha.startswith(short_sha):
                return c
        return None

    def show_commit_diff(self, short_sha: str, max_entries: int = 20) -> tuple[CommitInfo, str] | None:
        """查找指定提交并返回其与上一个版本的差异。

        与 find_commit 类似，但额外返回 diff。
        如果是最近的提交（没有下一个），diff 为空字符串。

        参数：
          short_sha:   短 SHA 前缀
          max_entries: 最多搜索的提交数

        返回：
          (CommitInfo, diff_str) 元组，找不到返回 None

        用法：
          >>> result = store.show_commit_diff("abc12")
          >>> if result:
          ...     commit, diff = result
          ...     print(commit.format(diff=diff))
        """
        commits = self.log(max_entries=max_entries)
        for i, c in enumerate(commits):
            if c.sha.startswith(short_sha):
                if i + 1 < len(commits):
                    diff = self.diff_commits(commits[i + 1].sha, c.sha)
                else:
                    diff = ""
                return c, diff
        return None

    # ── 回退 ───────────────────────────────────────────────────────────

    def revert(self, commit: str) -> str | None:
        """回退（撤销）指定提交引入的更改。

        实现方式：
          1. 解析目标提交
          2. 取其父提交的文件树
          3. 从父提交树中恢复所有跟踪文件的内容
          4. 创建一个新的 revert 提交记录这次回退

        注意：不是历史改写（不会删除原提交），而是新建一个"撤销"提交。

        参数：
          commit: 要撤销的提交的短 SHA

        返回：
          新创建的 revert 提交的短 SHA，失败返回 None

        用法：
          >>> new_sha = store.revert("abc12345")
          >>> if new_sha:
          ...     print(f"已回退，新提交: {new_sha}")
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich.repo import Repo

            full_sha = self._resolve_sha(commit)
            if not full_sha:
                logger.warning("回退失败: SHA 未找到: {}", commit)
                return None

            with Repo(str(self._workspace)) as repo:
                commit_obj = repo[full_sha]
                if commit_obj.type_name != b"commit":
                    return None

                if not commit_obj.parents:
                    logger.warning("回退失败: 无法回退初始提交 {}", commit)
                    return None

                # 取父提交的文件树（相当于撤销本次所有更改）
                parent_obj = repo[commit_obj.parents[0]]
                tree = repo[parent_obj.tree]

                restored: list[str] = []
                for filepath in self._tracked_files:
                    content = self._read_blob_from_tree(repo, tree, filepath)
                    if content is not None:
                        dest = self._workspace / filepath
                        dest.write_text(content, encoding="utf-8")
                        restored.append(filepath)

            if not restored:
                return None

            # 提交恢复后的状态
            msg = f"revert: undo {commit}"
            return self.auto_commit(msg)
        except Exception:
            logger.exception("Git 回退失败: {}", commit)
            return None

    @staticmethod
    def _read_blob_from_tree(repo, tree, filepath: str) -> str | None:
        """从 Git 树对象中递归查找文件并读取其文本内容。

        Git 仓库中文件按路径存储在树对象中。此方法沿着路径逐级
        进入子目录（子树），找到对应的 blob（文件内容）后解码为文本。

        参数：
          repo:     Dulwich Repo 对象
          tree:     顶层树对象
          filepath: 相对于仓库根的文件路径，如 "memory/MEMORY.md"

        返回：
          文件内容的 UTF-8 文本，文件不存在或非纯文本返回 None
        """
        parts = Path(filepath).parts
        current = tree
        for part in parts:
            try:
                entry = current[part.encode()]
            except KeyError:
                return None
            obj = repo[entry[1]]
            if obj.type_name == b"blob":
                return obj.data.decode("utf-8", errors="replace")
            if obj.type_name == b"tree":
                current = obj
            else:
                return None
        return None
