"""基于原始终端模式的异步行编辑器。

工作原理：
  1. 用 termios 把终端设为"原始模式"（不缓冲、不回显）
  2. 用 loop.add_reader() 在事件循环中逐字节读取 stdin
  3. 自行解析转义序列（方向键等）并维护行缓冲区
  4. 按 Enter 提交整行，返回给调用者

这种方式完全避开线程，也不需要 readline / prompt_toolkit 等外部依赖。
"""

from __future__ import annotations

import asyncio
import os
import sys
import termios
import tty
import atexit
import signal
from dataclasses import dataclass, field
from typing import Callable


# ── 终端 ANSI 辅助常量 ──────────────────────────────────────────────────

# 用于光标和行更新的 ANSI 转义序列
_SAVE_CURSOR = "\033[s"       # 保存光标位置
_RESTORE_CURSOR = "\033[u"    # 恢复光标位置
_CURSOR_UP = "\033[A"         # 光标上移一行
_CURSOR_DOWN = "\033[B"       # 光标下移一行
_CURSOR_RIGHT = "\033[C"      # 光标右移 n 格
_CURSOR_LEFT = "\033[D"       # 光标左移 n 格
_CLEAR_LINE = "\033[2K"       # 清除整行
_CARRIAGE_RETURN = "\r"       # 回车（回到行首）


# ── 转义序列表 ──────────────────────────────────────────────────────────

# 多字节转义序列的匹配表
# 格式：{(第一个字节, 第二个字节, ...): 含义}
_ESCAPE_SEQUENCES: dict[tuple[int, ...], str] = {
    # 方向键（CSI 序列：ESC [ A/B/C/D）
    (0x1b, 0x5b, 0x41): "up",      # ESC [ A
    (0x1b, 0x5b, 0x42): "down",    # ESC [ B
    (0x1b, 0x5b, 0x43): "right",   # ESC [ C
    # SS3 序列（部分终端用 ESC O 代替 ESC [ 发送方向键）
    (0x1b, 0x4f, 0x41): "up",       # ESC O A
    (0x1b, 0x4f, 0x42): "down",     # ESC O B
    (0x1b, 0x4f, 0x43): "right",    # ESC O C
    (0x1b, 0x4f, 0x44): "left",     # ESC O D
    (0x1b, 0x4f, 0x48): "home",     # ESC O H
    (0x1b, 0x4f, 0x46): "end",      # ESC O F

    (0x1b, 0x5b, 0x44): "left",    # ESC [ D
    # Home / End
    (0x1b, 0x5b, 0x48): "home",    # ESC [ H
    (0x1b, 0x5b, 0x46): "end",     # ESC [ F
    # Delete
    (0x1b, 0x5b, 0x33, 0x7e): "delete",  # ESC [ 3 ~
}


@dataclass
class CliConfig:
    """异步行编辑器的配置参数。"""
    prompt: str = ">>> "                     # 行提示符
    history_file: str | None = None          # 历史持久化文件路径
    history_max: int = 1000                  # 历史记录最大条数
    auto_save_history: bool = True           # 是否自动保存历史到文件
    completer: Callable[[str], list[str]] | None = None  # Tab 补全回调


class AsyncCli:
    """异步命令行编辑器。

    使用方式：
        async with AsyncCli() as cli:
            line = await cli.readline(">>> ")

    会自动管理终端原始模式的进入和退出。
    """

    def __init__(self, config: CliConfig | None = None):
        """初始化异步 CLI 编辑器。"""
        self.config = config or CliConfig()

        # stdin 文件描述符（标准输入）
        self._stdin_fileno = sys.stdin.fileno()

        # 保存原始终端属性，退出时恢复
        self._old_term_attr: list | None = None

        # 事件循环引用
        self._loop: asyncio.AbstractEventLoop | None = None

        # 输入字节队列：add_reader 回调往里放，readline 协程往外取
        self._input_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # 添加和移除 reader 的锁
        self._reader_active = False

        # ── 历史管理 ──
        self._history: list[str] = []         # 历史记录列表（最近的在末尾）
        self._history_index: int = 0          # 当前浏览的历史位置
        self._history_loaded = False          # 是否已从文件加载

        # ── 行编辑状态（每个 readline 调用独立） ──
        self._buffer: list[str] = []          # 当前输入字符列表
        self._cursor: int = 0                 # 光标在 buffer 中的位置

        # ── Ctrl+C 处理 ──
        self._interrupted = False             # 是否收到 Ctrl+C

        self._reading = False                  # 是否正在 readline 等待中
    # ── 上下文管理器：进入/退出原始模式 ──

    async def __aenter__(self) -> "AsyncCli":
        """进入异步上下文：设置原始终端模式 + 注册 stdin reader。"""
        self._enter_raw_mode()
        await self._start_reader()
        if self.config.history_file:
            self._load_history()
        return self

    async def __aexit__(self, *args) -> None:
        """退出异步上下文：恢复终端 + 保存历史 + 移除 reader。"""
        if self.config.history_file and self.config.auto_save_history:
            self._save_history()
        await self._stop_reader()
        self._exit_raw_mode()

    def _enter_raw_mode(self) -> None:
        """把终端设为原始模式（不缓存、不回显、逐字节读取）。"""
        try:
            fd = self._stdin_fileno
            self._old_term_attr = termios.tcgetattr(fd)
            tty.setraw(fd)  # 设置原始模式
            # 注册 atexit 恢复终端：即使程序异常退出也能恢复
            atexit.register(self._exit_raw_mode)
        except (termios.error, OSError):
            # 如果 stdin 不是 TTY（比如 pipe 输入），不做特殊处理
            self._old_term_attr = None

    def _exit_raw_mode(self) -> None:
        """恢复终端到进入原始模式前的设置。"""
        if self._old_term_attr is not None:
            try:
                termios.tcsetattr(
                    self._stdin_fileno, termios.TCSANOW, self._old_term_attr
                )
            except (termios.error, OSError):
                pass
            self._old_term_attr = None

    async def _start_reader(self) -> None:
        """注册 stdin 的异步读取回调到事件循环。"""
        self._loop = asyncio.get_running_loop()
        if not self._reader_active:
            self._loop.add_reader(self._stdin_fileno, self._on_stdin_data)
            self._reader_active = True

    async def _stop_reader(self) -> None:
        """从事件循环中移除 stdin 读取回调。"""
        if self._reader_active and self._loop is not None:
            self._loop.remove_reader(self._stdin_fileno)
            self._reader_active = False

    def _on_stdin_data(self) -> None:
        """add_reader 回调：stdin 有数据可读时触发。

        读取所有可用字节，逐个放入 asyncio.Queue 供 readline 协程消费。
        """
        try:
            data = os.read(self._stdin_fileno, 4096)
        except (OSError, ValueError):
            # stdin 关闭或无数据可读
            return
        if not data:
            # EOF（Ctrl+D 在原始模式下表现为空读取）
            self._input_queue.put_nowait(b"")
            return
        for byte in data:
            # 每个字节作为一个独立的 bytes 对象入队
            self._input_queue.put_nowait(bytes([byte]))

    async def _read_byte(self) -> bytes:
        """异步读取一个字节。

        如果收到 Ctrl+C（SIGINT），抛出 KeyboardInterrupt。
        如果收到 EOF（Ctrl+D），返回空 bytes。
        """
        byte = await self._input_queue.get()
        if byte == b"\x03":
            # Ctrl+C → KeyboardInterrupt
            raise KeyboardInterrupt()
        if byte == b"":
            # Ctrl+D → EOF
            return b""
        return byte

    # ── 转义序列解析 ──

    async def _read_key(self) -> str:
        """异步读取一个按键，返回统一的事件名或字符。

        返回值:
          - 可打印字符: 直接返回该字符
          - 功能键如 "up", "down", "left", "right", "home", "end", "delete"
          - "backspace", "enter", "tab"
          - "eof" (Ctrl+D)
          - "unknown" (无法识别的序列)
        """
        byte = await self._read_byte()
        if byte == b"":
            return "eof"

        code = byte[0]

        # ── 单字节控制字符 ──
        if code == 0x0d or code == 0x0a:
            # Enter（CR 或 LF）
            return "enter"
        if code == 0x7f or code == 0x08:
            # Backspace（DEL 或 BS）
            return "backspace"
        if code == 0x09:
            # Tab
            return "tab"
        if code == 0x04:
            # Ctrl+D
            return "eof"

        # ── 可打印 ASCII 字符 ──
        if 0x20 <= code <= 0x7e:
            return chr(code)

        # ── 多字节 UTF-8 / 扩展字符 ──
        if code >= 0x80:
            # 尝试读取完整的 UTF-8 序列（最多 4 字节）
            extra_needed = 0
            if (code & 0xe0) == 0xc0:      # 2 字节 UTF-8
                extra_needed = 1
            elif (code & 0xf0) == 0xe0:     # 3 字节 UTF-8
                extra_needed = 2
            elif (code & 0xf8) == 0xf0:     # 4 字节 UTF-8
                extra_needed = 3
            else:
                return "unknown"

            utf8_bytes = bytearray([code])
            for _ in range(extra_needed):
                try:
                    next_byte = await asyncio.wait_for(self._read_byte(), timeout=0.1)
                except asyncio.TimeoutError:
                    return "unknown"
                utf8_bytes.append(next_byte[0])

            try:
                return utf8_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return "unknown"

        # ── ESC 序列（方向键等） ──
        if code == 0x1b:
            return await self._read_escape_sequence()

        return "unknown"

    async def _read_escape_sequence(self) -> str:
        """读取 ESC 之后的多字节转义序列。"""
        try:
            # 读取 ESC 后的下一个字节
            next_byte = await asyncio.wait_for(self._read_byte(), timeout=0.1)
        except asyncio.TimeoutError:
            # 单独的 ESC 键
            return "unknown"

        if next_byte == b"":
            return "unknown"

        seq = [0x1b, next_byte[0]]

        # CSI 序列（以 [ 开头）：继续读取直到遇到终止符 0x40-0x7e
        if next_byte[0] == 0x5b:  # '['
            while True:
                try:
                    b = await asyncio.wait_for(self._read_byte(), timeout=0.1)
                except asyncio.TimeoutError:
                    break
                if b == b"":
                    break
                seq.append(b[0])
                if 0x40 <= b[0] <= 0x7e:
                    break

        # SS3 序列（以 O 开头）：下一个字节就是终止符，直接查表
        elif next_byte[0] == 0x4f:  # 'O'
            # SS3 序列：ESC O + 一个字节的命令字符，直接读完查表
            try:
                b = await asyncio.wait_for(self._read_byte(), timeout=0.1)
                if b:
                    seq.append(b[0])
            except asyncio.TimeoutError:
                pass
        # 查表匹配（同时支持 CSI 和 SS3 序列）
        key = _ESCAPE_SEQUENCES.get(tuple(seq), "unknown")
        return key

    # ── 行渲染 ──

    def _render_line(self, prompt: str) -> None:
        """重新绘制当前行（提示符 + 输入缓冲）。

        使用 \r 回到行首再重绘，覆盖旧内容。
        """
        line = prompt + "".join(self._buffer)
        # 先清除整行，然后写上完整内容
        sys.stdout.write(_CLEAR_LINE + _CARRIAGE_RETURN + line)
        # 把光标移回正确位置（提示符后的第 cursor 个字符）
        col = len(prompt) + self._cursor
        sys.stdout.write(_CARRIAGE_RETURN)
        if col > 0:
            sys.stdout.write(f"{_CURSOR_RIGHT * col}")
        sys.stdout.flush()

    def _write_output(self, text: str) -> None:
        """在终端上输出文本，不会干扰输入行。

        注意：原始模式下 OPOST 关闭，\n 不会自动转 \r\n。
        如果 text 含多段文字，必须把 \n 替换为 \r\n，
        否则光标不会回到行首，就会出现"每行之间好多空格"的锯齿现象。
        """
        # 统一换行符：\r\n → \n（防重复），然后每段末尾补 \r
        text = text.replace("\r\n", "\n")
        text = text.replace("\n", "\r\n")

        if self._reading:
            # 正在 readline 中：先保存当前输入行，写完输出再恢复
            current_line = self.config.prompt + "".join(self._buffer)
            sys.stdout.write(_CARRIAGE_RETURN + _CLEAR_LINE)
            sys.stdout.write(text)
            sys.stdout.write("\r\n")
            sys.stdout.write(current_line)
            col = len(self.config.prompt) + self._cursor
            sys.stdout.write(_CARRIAGE_RETURN)
            if col > 0:
                sys.stdout.write(f"\033[{col}C")
            sys.stdout.flush()
        else:
            # 未在 readline 中：直接输出
            sys.stdout.write(text)
            sys.stdout.write("\r\n")
            sys.stdout.flush()

    def _insert_char(self, ch: str) -> None:
        """在光标处插入一个字符。"""
        self._buffer.insert(self._cursor, ch)
        self._cursor += 1

    def _delete_before_cursor(self) -> bool:
        """删除光标前的一个字符（Backspace）。"""
        if self._cursor <= 0:
            return False
        self._cursor -= 1
        self._buffer.pop(self._cursor)
        return True

    def _delete_at_cursor(self) -> bool:
        """删除光标处的一个字符（Delete）。"""
        if self._cursor >= len(self._buffer):
            return False
        self._buffer.pop(self._cursor)
        return True

    def _cursor_left(self) -> None:
        """光标左移一格。"""
        if self._cursor > 0:
            self._cursor -= 1

    def _cursor_right(self) -> None:
        """光标右移一格。"""
        if self._cursor < len(self._buffer):
            self._cursor += 1

    def _cursor_home(self) -> None:
        """光标移到行首。"""
        self._cursor = 0

    def _cursor_end(self) -> None:
        """光标移到行尾。"""
        self._cursor = len(self._buffer)

    # ── 历史管理 ──

    def _load_history(self) -> None:
        """从文件加载历史记录。"""
        if not self.config.history_file:
            return
        try:
            with open(self.config.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        self._history.append(line)
            # 限制历史长度
            if len(self._history) > self.config.history_max:
                self._history = self._history[-self.config.history_max:]
            self._history_loaded = True
        except (FileNotFoundError, OSError):
            pass

    def _save_history(self) -> None:
        """将历史记录保存到文件。"""
        if not self.config.history_file:
            return
        try:
            dir_path = os.path.dirname(self.config.history_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.config.history_file, "w", encoding="utf-8") as f:
                for entry in self._history:
                    f.write(entry + "\n")
        except OSError:
            pass

    def _add_to_history(self, line: str) -> None:
        """将一行添加到历史记录。"""
        if not line or line.startswith("/exit") or line.startswith("/quit"):
            return
        # 不重复添加连续相同的行
        if self._history and self._history[-1] == line:
            return
        self._history.append(line)
        if len(self._history) > self.config.history_max:
            self._history.pop(0)

    def _history_up(self) -> bool:
        """上键：回到上一条历史。"""
        if not self._history:
            return False
        # 初次按上键时，从末尾开始
        if self._history_index >= len(self._history):
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1

        # 替换缓冲区为历史条目
        entry = self._history[self._history_index]
        self._buffer = list(entry)
        self._cursor = len(self._buffer)
        return True

    def _history_down(self) -> bool:
        """下键：回到下一条历史。"""
        if not self._history:
            return False
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            entry = self._history[self._history_index]
            self._buffer = list(entry)
            self._cursor = len(self._buffer)
            return True
        else:
            # 已经是最后一条，回到空白行
            self._history_index = len(self._history)
            self._buffer = []
            self._cursor = 0
            return True

    # ── Tab 补全 ──

    def _do_completion(self) -> bool:
        """执行 Tab 补全。"""
        if not self.config.completer or not self._buffer:
            return False

        current_text = "".join(self._buffer)
        completions = self.config.completer(current_text)
        if not completions:
            return False

        if len(completions) == 1:
            # 唯一匹配：直接替换
            self._buffer = list(completions[0])
            self._cursor = len(self._buffer)
            return True
        else:
            # 多个匹配：找出最长公共前缀
            common = os.path.commonprefix(completions)
            if common and common != current_text:
                self._buffer = list(common)
                self._cursor = len(self._buffer)
                return True
            else:
                # 显示所有匹配项
                match_list = "  ".join(completions)
                self._write_output(match_list)
                return True

    # ── 主接口：readline ──

    async def readline(self, prompt: str | None = None) -> str:
        """异步读取一行输入。

        支持：
          - 方向键上下：浏览历史
          - 方向键左右：光标移动
          - Backspace / Delete：删除字符
          - Tab：自动补全
          - Ctrl+C：KeyboardInterrupt
          - Ctrl+D：返回空字符串

        参数：
            prompt: 行提示符，None 则用配置中的默认值

        返回：
            用户输入的字符串（不含尾随换行符）
        """
        if self._interrupted:
            raise KeyboardInterrupt("上一次 Ctrl+C 未处理")

        pk = prompt if prompt is not None else self.config.prompt
        self._buffer = []
        self._cursor = 0
        # 如果终端非 TTY（pipe 输入），降级到普通 sys.stdin.readline
        if not sys.stdin.isatty():
            loop = asyncio.get_running_loop()
            line = await loop.run_in_executor(None, lambda: sys.stdin.readline())
            return line.rstrip("\n")

        # 绘制初始提示符
        self._reading = True
        self._render_line(pk)

        while True:
            key = await self._read_key()

            if key == "enter":
                # 换行并返回结果
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                line = "".join(self._buffer)
                self._reading = False
                self._add_to_history(line)
                self._history_index = len(self._history)  # 重置浏览位置
                return line

            elif key == "backspace":
                if self._delete_before_cursor():
                    self._render_line(pk)

            elif key == "delete":
                if self._delete_at_cursor():
                    self._render_line(pk)

            elif key == "left":
                self._cursor_left()
                # 只移动光标，不用重绘整行
                sys.stdout.write(_CURSOR_LEFT)
                sys.stdout.flush()

            elif key == "right":
                self._cursor_right()
                sys.stdout.write(_CURSOR_RIGHT)
                sys.stdout.flush()

            elif key == "up":
                if self._history_up():
                    self._render_line(pk)

            elif key == "down":
                if self._history_down():
                    self._render_line(pk)

            elif key == "home":
                self._cursor_home()
                self._render_line(pk)

            elif key == "end":
                self._cursor_end()
                self._render_line(pk)

            elif key == "tab":
                self._do_completion()
                self._render_line(pk)

            elif key == "eof":
                # Ctrl+D：返回空字符串
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return ""
                self._reading = False

            elif len(key) == 1:
                # 普通可打印字符
                self._insert_char(key)
                self._render_line(pk)

            # 其他按键忽略

    # ── 输出方法（供外部使用） ──

    def output(self, text: str) -> None:
        """在终端输出文本，不干扰当前输入行。

        这个方法在 await readline() 之外也能调用。
        """
        self._write_output(text)
