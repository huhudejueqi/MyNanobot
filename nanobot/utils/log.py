import logging

# AgentLoop 日志记录器
logger = logging.getLogger("nanobot.agent.loop")

# 预先清理 httpx 的日志，避免 INFO 级别的请求日志混入终端
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
