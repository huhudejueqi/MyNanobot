"""共享的 Pydantic 基础模型，用于配置 DTO。

此模块特意放在 nanobot.config 包外部，这样运行时模块可以定义本地配置 DTO，
而无需导入完整的根配置模式。
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Base(BaseModel):
    """基础模型，同时接受 camelCase 和 snake_case 键名。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
