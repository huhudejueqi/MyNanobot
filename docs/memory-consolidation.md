# 记忆压缩（Consolidator）

## 概述

Consolidator 是 nanobot 记忆系统的第一阶段——**在线压缩**。当对话历史膨胀到接近上下文窗口预算时，它将最旧的安全消息片段压缩为 LLM 摘要，存入 `history.jsonl`，避免对话历史无限增长。

属于第二阶段的是 **Dream**（定时运行的深层次记忆提炼），本文档不涉及。

## 触发时机

Consolidator 在每个 turn 的 `_state_save` 流程末尾以后台任务方式触发：

```
_state_save
  └─ 非临时对话
       ├─ enforce_file_cap()      → 文件上限管理
       └─ maybe_consolidate_by_tokens() → 后台触发压缩
```

## 压缩流程总览

```
                        maybe_consolidate_by_tokens
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  前置条件检查                      │
                  │  context_window_tokens <= 0 → 返回 │
                  └─────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  1. 加锁 + 刷新 session 引用      │
                  │     避免 AutoCompact 替换对象后    │
                  │     操作旧引用                     │
                  └─────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  2. 计算预算                      │
                  │     budget  = 输入 token 预算     │
                  │     target  = budget × ratio     │
                  └─────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  3. replay 溢出预压缩             │
                  │     _consolidate_replay_overflow  │
                  └─────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  4. 估算 token                   │
                  │     estimate_session_prompt_     │
                  │     tokens()                     │
                  │                                  │
                  │  estimated < budget → idle 返回   │
                  │  estimated >= budget → 进入压缩   │
                  └─────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  5. 多轮压缩循环                   │
                  │     (最多 _MAX_CONSOLIDATION     │
                  │      _ROUNDS = 5 轮)             │
                  │                                  │
                  │  每轮：                           │
                  │  a. estimated <= target → break  │
                  │  b. pick_consolidation_boundary() │
                  │  c. archive(chunk) → LLM 总结     │
                  │  d. 推进 last_consolidated       │
                  │  e. sessions.save()              │
                  │  f. archive 失败 → break         │
                  │  g. 重新估算，继续下一轮           │
                  └─────────────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │  6. 持久化摘要到 session metadata │
                  │     _persist_last_summary()      │
                  └─────────────────────────────────┘
```

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `context_window_tokens` | 模型相关（如 128K） | LLM 上下文窗口大小 |
| `max_completion_tokens` | 4096 | 为生成回复预留的 token |
| `_SAFETY_BUFFER` | 1024 | tokenizer 估算偏差的安全余量 |
| `consolidation_ratio` | 0.5 | 压缩触发阈值比例 |
| `_MAX_CONSOLIDATION_ROUNDS` | 5 | 单次触发最多压缩轮数 |

## 预算计算

```python
# _input_token_budget 属性
budget = context_window_tokens - max_completion_tokens - _SAFETY_BUFFER
target = int(budget * consolidation_ratio)
```

举例（GPT-4o，128K 窗口）：

```
窗口格局：
┌──────────────────────────────────────────────────────────────┐
│  122_880 (budget)     │  4_096    │ 1_024                    │
│  ← 输入 token 预算 →  │  生成占位  │ 安全余量                  │
├────────────────────────┴──────────┴──────────────────────────┤
│                  128_000 (context_window_tokens)              │
└──────────────────────────────────────────────────────────────┘

target = 122_880 × 0.5 = 61_440

触发条件：estimated >= 61_440 → 开始压缩
压缩目标：降到 61_440 以下
```

## 安全边界选择

`pick_consolidation_boundary()` 从 `last_consolidated` 开始遍历，累计 token，**只在 user 消息处切**：

```python
for idx in range(start, len(session.messages)):
    message = session.messages[idx]
    if idx > start and message.get("role") == "user":
        last_boundary = (idx, removed_tokens)
        if removed_tokens >= tokens_to_remove:
            return last_boundary
    removed_tokens += estimate_message_tokens(message)
```

示例（`tokens_to_remove = 400`）：

```
idx  role       token  检查时的累计    行为
───────────────────────────────────────────
 0   system     200    0              跳过，累计 += 200 → 200
 1   user        50    200            记边界 (1, 200)，200 < 400，累计 += 50 → 250
 2   assistant  100    —              不是 user，累计 += 100 → 350
 3   tool        30    —              不是 user，累计 += 30 → 380
 4   user        40    380            记边界 (4, 380)，380 < 400，累计 += 40 → 420
 5   assistant   80    —              不是 user，累计 += 80 → 500
 6   tool        20    —              不是 user，累计 += 20 → 520
 7   user        30    520            记边界 (7, 520)，520 >= 400 → **返回 (7, 520)**
```

返回 `end_idx = 7`，归档 `messages[last_consolidated:7]`（idx 0~6）。

**原则：只在 user 消息处切**，保证不截断对话回合。宁可多删一点，也不在 assistant/tool 中间截断。

## 摘要归档

### 正常路径

```python
archive(chunk):
  1. 将 chunk 格式化为文本
  2. 调用 LLM（使用 consolidator_archive.md 提示词）生成摘要
  3. append_history(summary) → 写入 history.jsonl
  4. 返回摘要文本
```

### 失败降级

```python
archive(chunk):
  LLM 调用失败:
    raw_archive(messages) → 写入带 [RAW] 前缀的原文转储到 history.jsonl
    返回 None
```

## 归档后结构

```
session.messages:
┌────── 已归档 ──────┬────── 活跃区 ─────────────────────┐
│ m0   m1  ...  m6  │ m7   m8   m9   m10  m11  ...     │
└───────────────────┴───────────────────────────────────┘
                     ↑
                last_consolidated = 7

session.metadata._last_summary:
  "用户先查询了北京的天气（25°C），
   然后又查询了上海的天气（28°C）..."

history.jsonl (session_key 隔离):
  {cursor: 1, session_key: "cli:...", content: "用户先查询了北京的天气..."}
```

归档后原始消息仍留在 `session.messages` 中，但 prompt 构建时只取 `last_consolidated` 之后的活跃区。已归档区域被一条 LLM 生成的摘要替代。

## replay 溢出预压缩

`_consolidate_replay_overflow` 处理的是**消息数量超出 replay 窗口**的情况——即使 token 预算没超，但消息条数超过了配置的 `replay_max_messages`，那些被裁掉的消息会在丢失前被归档。

```
_consolidate_replay_overflow(session):
  1. _replay_overflow_boundary()
     └─ 找到 replay 窗口的起点
  2. archive(chunk) → LLM 总结
  3. 推进 last_consolidated
  4. sessions.save()
```

## tokenizer

使用 `tiktoken`（`cl100k_base`）做本地 token 估算，优先尝试 provider 厂商的原生统计接口。

| 函数 | 位置 | 用途 |
|---|---|---|
| `estimate_prompt_tokens` | helpers.py:453 | 估算整套消息的 token 总量 |
| `estimate_message_tokens` | helpers.py:495 | 估算单条消息的 token |
| `estimate_prompt_tokens_chain` | helpers.py:556 | 优先厂商接口，失败降级 tiktoken |

## 数据存储

所有归档摘要写入 `history.jsonl`，但通过 `session_key` 字段隔离：

```json
{"cursor": 42, "ts": "...", "content": "摘要...", "session_key": "cli:chat_0_xxx"}
```

读取时按 `session_key` 过滤，确保 A session 的摘要不会被 B session 读到。
