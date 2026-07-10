# from pydantic import BaseModel

# # 继承Pydantic BaseModel
# class AgentDefaults(BaseModel):
#     model: str = "anthropic/claude-opus-4-5"
#     max_tokens: int = 8192
#     temperature: float = 0.1
#     fallback_models: list = []
#     # 无默认值，必填字段，实例化不传就报错
#     disabled_skills: list[str]


# # 测试1：不传 disabled_skills → 直接抛校验异常
# if __name__ == "__main__":
#     try:
#         cfg = AgentDefaults(max_tokens="test")
#         print(cfg)
#     except Exception as e:
#         print("=== 捕获到必填字段报错 ===")
#         print(e)

#     print("\n=== 正确传参示例 ===")
#     # 正常传入必填字段，不会报错
#     cfg_ok = AgentDefaults(disabled_skills=["file_write", "shell"])
#     print(cfg_ok.model_dump())

from pydantic import BaseModel, Field

class AgentDefaults(BaseModel):
    model: str = "anthropic/claude-opus-4-5"
    disabled_skills: list[str] = Field(default_factory=list)

class GlobalConfig(BaseModel):
    agent: AgentDefaults = Field(default=AgentDefaults())

cfg1 = GlobalConfig()
cfg1.agent.disabled_skills.append("read_file")

cfg2 = GlobalConfig()
print(cfg2.agent.disabled_skills)
print(cfg1.agent.disabled_skills)
