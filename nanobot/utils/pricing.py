"""模型定价表和费用计算。

根据模型名称模式匹配每百万 token 的单价（USD）。
"""

from __future__ import annotations
from typing import Any

from loguru import logger
from nanobot.agent.hook import AgentHook, AgentRunHookContext

# ---------------------------------------------------------------------------
# 定价表：(子串匹配, 输入价格/百万token, 输出价格/百万token)
# 价格单位为 USD。数据来源：各厂商官网定价页（截至 2026-07）。
# 顺序：精确匹配优先，通配兜底在末尾。
# ---------------------------------------------------------------------------
_MODEL_PRICES: list[tuple[str, float, float]] = [
    # -- OpenAI --------------------------------------------------------------
    ("gpt-4o-mini",           0.15,  0.60),
    ("gpt-4o",                2.50, 10.00),
    ("gpt-4-turbo",          10.00, 30.00),
    ("gpt-4",                30.00, 60.00),
    ("gpt-3.5-turbo",         0.50,  1.50),
    ("o1-mini",               1.10,  4.40),
    ("o1-preview",           15.00, 60.00),
    ("o1",                   15.00, 60.00),
    ("o3-mini",               1.10,  4.40),
    # -- Anthropic -----------------------------------------------------------
    ("claude-opus-4",        15.00, 75.00),
    ("claude-opus-4-5",      15.00, 75.00),
    ("claude-sonnet-4",       3.00, 15.00),
    ("claude-haiku-3",        0.25,  1.25),
    ("claude-3-opus",        15.00, 75.00),
    ("claude-3-sonnet",       3.00, 15.00),
    ("claude-3-haiku",        0.25,  1.25),
    ("claude-2",              8.00, 24.00),
    # -- DeepSeek ------------------------------------------------------------
    ("deepseek-reasoner",     0.55,  2.19),
    ("deepseek-chat",         0.27,  1.10),
    ("deepseek-coder",        0.14,  0.28),
    # -- Google Gemini -------------------------------------------------------
    ("gemini-2.5-pro",        1.25, 10.00),
    ("gemini-2.0-flash",      0.10,  0.40),
    ("gemini-1.5-pro",        1.25,  5.00),
    ("gemini-1.5-flash",      0.075, 0.30),
    # -- 智谱 GLM ------------------------------------------------------------
    ("glm-4-plus",            0.50,  0.50),
    ("glm-4",                 0.10,  0.10),
    ("glm-4-air",             0.05,  0.05),
    ("glm-4-flash",           0.03,  0.03),
    # -- 阿里 Qwen -----------------------------------------------------------
    ("qwen-max",              2.00,  6.00),
    ("qwen-plus",             0.80,  2.00),
    ("qwen-turbo",            0.30,  0.60),
    ("qwen2.5-72b",           4.00, 12.00),
    ("qwen2.5-32b",           3.50,  7.00),
    ("qwen2.5-14b",           2.00,  4.00),
    ("qwen2.5-7b",            0.50,  1.00),
    # -- 字节豆包 / Doubao ----------------------------------------------------
    ("doubao-pro-256k",       5.00,  9.00),
    ("doubao-pro-128k",       5.00,  9.00),
    ("doubao-pro-32k",        0.80,  2.00),
    ("doubao-lite-128k",      1.00,  2.00),
    ("doubao-lite-32k",       0.30,  0.60),
    # -- 月之暗面 Kimi --------------------------------------------------------
    ("kimi-k2.5",             2.00,  8.00),
    ("kimi-k2",               1.00,  4.00),
    ("moonshot-v1",          12.00, 12.00),
    # -- 零一万物 Yi ----------------------------------------------------------
    ("yi-lightning",          0.50,  0.50),
    ("yi-large",              2.00,  2.00),
    # -- 百川 Baichuan -------------------------------------------------------
    ("baichuan4",             0.40,  0.80),
    ("baichuan3-turbo",       0.20,  0.20),
    # -- Minimax --------------------------------------------------------------
    ("minimax-m1",            0.10,  0.35),
    # -- StepFun（阶跃星辰）---------------------------------------------------
    ("step-2",                2.00,  8.00),
    ("step-1",                1.00,  4.00),
    # -- SenseTime（商汤）----------------------------------------------------
    ("sensechat-5",           1.00,  4.00),
    # -- 本地 / 免费 / 默认兜底 ------------------------------------------------
    ("token_monad",           0.00,  0.00),
    ("local",                 0.00,  0.00),
]

# ---------------------------------------------------------------------------
# 缓存读取价格（每百万 token），缓存命中按此价格计费
# 不在该列表中的模型默认缓存价 = 输入价 × 0.5
# ---------------------------------------------------------------------------
_MODEL_CACHE_PRICES: list[tuple[str, float]] = [
    # OpenAI：缓存 = 输入价 × 0.5
    ("gpt-4o-mini",           0.075),
    ("gpt-4o",                1.25),
    ("gpt-4-turbo",           5.00),
    ("gpt-3.5-turbo",         0.25),
    ("o1-mini",               0.55),
    ("o3-mini",               0.55),
    # Anthropic：缓存 ≈ 输入价 × 0.1
    ("claude-opus-4",         1.50),
    ("claude-sonnet-4",       0.30),
    ("claude-haiku-3",        0.025),
    ("claude-3-opus",         1.50),
    ("claude-3-sonnet",       0.30),
    ("claude-3-haiku",        0.025),
    # DeepSeek：缓存 ≈ 输入价 × 0.1
    ("deepseek-reasoner",     0.055),
    ("deepseek-chat",         0.027),
    # 智谱：缓存 ≈ 输入价 × 0.5
    ("glm-4-plus",            0.25),
    ("glm-4",                 0.05),
    ("glm-4-air",             0.025),
    ("glm-4-flash",           0.015),
    # 阿里 Qwen：缓存 ≈ 输入价 × 0.5
    ("qwen-max",              1.00),
    ("qwen-plus",             0.40),
    ("qwen-turbo",            0.15),
    # Kimi：缓存 ≈ 输入价 × 0.5
    ("kimi-k2.5",             1.00),
    ("kimi-k2",               0.50),
]

# 未匹配到定价表时的默认价格
_DEFAULT_INPUT_PRICE = 1.00
_DEFAULT_OUTPUT_PRICE = 2.00
_DEFAULT_CACHE_RATIO = 0.5  # 默认缓存价为输入价的 50%

# USD → CNY 汇率
_CNY_RATE = 7.20


def _find_prices(model: str) -> tuple[float, float]:
    """根据模型名查找（输入价格, 输出价格），单位 USD/百万 token。"""
    lower = model.lower().strip()
    for pattern, inp, out in _MODEL_PRICES:
        if pattern in lower:
            return inp, out
    return _DEFAULT_INPUT_PRICE, _DEFAULT_OUTPUT_PRICE


def _find_cache_price(model: str, input_price: float) -> float:
    """查找模型的缓存读取价格，未匹配则按输入价 × 默认比例。"""
    lower = model.lower().strip()
    for pattern, cache_price in _MODEL_CACHE_PRICES:
        if pattern in lower:
            return cache_price
    return input_price * _DEFAULT_CACHE_RATIO


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """计算一次模型调用的费用。

    返回字段：
        - cost_usd: USD 总费用（float，6 位小数）
        - cost_str: 可读格式 "$0.001234"
        - prompt_cost: 非缓存的 prompt token 费用
        - cached_cost: 缓存的 prompt token 费用
        - completion_cost: completion token 费用
        - input_price_per_1M: 匹配到的输入单价
        - cache_price_per_1M: 匹配到的缓存读取单价
        - output_price_per_1M: 匹配到的输出单价
        - cost_cny: 按汇率折算的人民币费用（float，6 位小数）
        - cost_cny_str: 可读格式 "¥0.0089"
    """
    inp_price, out_price = _find_prices(model)
    cache_price = _find_cache_price(model, inp_price)

    uncached = max(0, prompt_tokens - cached_tokens)
    prompt_cost = (uncached / 1_000_000) * inp_price
    cached_cost = (cached_tokens / 1_000_000) * cache_price
    completion_cost = (completion_tokens / 1_000_000) * out_price
    total = prompt_cost + cached_cost + completion_cost
    total_cny = total * _CNY_RATE

    return {
        "cost_usd": round(total, 6),
        "cost_str": f"${total:.6f}",
        "cost_cny": round(total_cny, 6),
        "cost_cny_str": f"¥{total_cny:.4f}",
        "prompt_cost": round(prompt_cost, 6),
        "cached_cost": round(cached_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "input_price_per_1M": inp_price,
        "cache_price_per_1M": cache_price,
        "output_price_per_1M": out_price,
    }


def format_cost_line(
    usage: dict[str, int],
    model: str | None,
    *,
    prefix: str = "",
) -> str:
    """格式化一行 token 用量和费用摘要。

    输出示例：
        token 用量：输入=14242（缓存=13952） + 输出=281 = 14523  费用=$0.004154  ≈¥0.0299  (deepseek-chat)
    """
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", 0) or (prompt + completion)
    cached = usage.get("cached_tokens", 0)

    prompt_str = f"输入={prompt}"
    if cached:
        prompt_str += f"（缓存={cached}）"

    if model:
        cost = calculate_cost(prompt, completion, model, cached_tokens=cached)
        parts = [f"输入${cost['prompt_cost']:.6f}"]
        if cached:
            parts.append(f"缓存${cost['cached_cost']:.6f}")
        parts.append(f"输出${cost['completion_cost']:.6f}")
        detail = "+".join(parts)
        cost_part = f"  费用={cost['cost_str']}（{detail}）  ≈{cost['cost_cny_str']}"
        model_part = f"  ({model})"
    else:
        cost_part = ""
        model_part = ""

    return (
        f"{prefix}token 用量：{prompt_str} + 输出={completion} = {total}"
        f"{cost_part}{model_part}"
    )


class CostTrackingHook(AgentHook):
    """钩子：每轮对话后追加 token 用量和费用摘要到回复末尾。"""

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self._model = model

    async def after_run(self, ctx: AgentRunHookContext) -> None:
        """在回复完成后追加 token 用量和费用摘要。"""
        usage = ctx.usage
        if not usage:
            return
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0) or (prompt + completion)
        if total <= 0:
            return
        if ctx.final_content and ctx.stop_reason != "error":
            line = format_cost_line(usage, self._model)
            logger.info(line)
            ctx.final_content = ctx.final_content.rstrip() + "\n\n`" + line + "`"
