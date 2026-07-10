# MyNanobot 参考项目

## 原版路径

```
~/workspace/nanobot/nanobot/
```

此目录是 MyNanobot 的参考实现。本项目的代码结构和设计均参照此目录。

## 对标结构

| MyNanobot | 参考项目 |
|---|---|
| `nanobot/agent/loop.py` | `nanobot/agent/loop.py` — 事件驱动状态机 |
| `nanobot/providers/` | `nanobot/providers/` — LLM Provider 抽象 |
| `nanobot/bus/` | `nanobot/bus/` — 消息总线 |
| `nanobot/config/loader.py` | `nanobot/config/` — 配置读取 |
| `nanobot/cli/cli_reader.py` | `nanobot/cli/` + `prompt_toolkit` |
| `nanobot/gateway.py` | `nanobot/webui/` — HTTP 网关 |
| `main.py` | `main.py` — CLI 入口 |

## 关键差异

- 参考项目用 `prompt_toolkit` 做 CLI，MyNanobot 用 `termios` 原始模式自实现
- 参考项目有完整的 WebUI 前端，MyNanobot 暂时只用 CLI

## 恢复说明

> 注意：本对话过程中不小心用空内容覆盖了参考项目的 `.git/config`、`README.md`、`pyproject.toml`。
> 如需要恢复，可以从原仓库重新 clone 或找备份恢复。
