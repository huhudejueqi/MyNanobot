"""运行结果通知评估器：用于心跳检测等后台任务的结果评估。

心跳执行完内部检查后，此模块做一次轻量级 LLM 调用，判断结果是否值得通知用户。
避免每一条后台日志都推送给用户造成骚扰。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

# 评估工具定义：LLM 通过调用此函数来决定是否通知用户
_EVALUATE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_notification",
            "description": "判断是否应就后台任务结果通知用户。",
            "parameters": {
                "type": "object",
                "properties": {
                    "should_notify": {
                        "type": "boolean",
                        "description": "true = 结果包含重要的/可操作的信息，用户应该看到；false = 例行或空结果，安全忽略",
                    },
                },
                "required": ["should_notify"],
            },
        },
    },
]


async def evaluate_response(
    provider: LLMProvider,
    model: str,
    response_content: str,
    system_prompt_extra: str = "",
) -> bool:
    """判断 LLM 的响应是否需要通知用户。

    使用一个极简的 LLM 调用（只返回一个布尔值），判断内容是否重要到值得推送通知。

    参数：
      provider:           LLM 提供者
      model:              模型名称
      response_content:   需要评估的响应文本
      system_prompt_extra: 附加的系统提示（如频道上下文）

    返回：
      True 表示应通知用户，False 表示可安全忽略
    """
    system_prompt = render_template(
        "agent/evaluator.md",
        strip=True,
        system_prompt_extra=system_prompt_extra,
    )
    try:
        res = await provider.chat_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": response_content},
            ],
            tools=_EVALUATE_TOOL,
            tool_choice="required",
        )
    except Exception:
        logger.exception("评估 LLM 调用失败")
        return False

    if not res.tool_calls:
        logger.warning("评估结果中未包含工具调用")
        return False

    call = res.tool_calls[0]
    if call.name != "evaluate_notification":
        logger.warning("评估工具名称不匹配: {}", call.name)
        return False

    should_notify = call.arguments.get("should_notify", False)
    return bool(should_notify)
