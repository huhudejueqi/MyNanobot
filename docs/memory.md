# nanobot 的记忆系统

nanobot 的记忆系统建立在这样一个信念上：记忆应该有生命力，但不该变得混乱。

好的记忆不是笔记的堆砌，而是一套安静的关注系统。它留意哪些值得保留，哪些不需要继续占据焦点，并将经历转化为沉稳、持久且有用的东西。

这就是 nanobot 记忆系统的设计。

## 设计理念

nanobot 不把记忆当作一个大文件来处理。

它将记忆分为多层，因为不同类型的记忆需要不同的工具：

- `session.messages` — 活跃的短期对话
- `memory/history.jsonl` — 压缩后的历史会话归档
- `SOUL.md`、`USER.md` 和 `memory/MEMORY.md` — 持久化的知识文件
- `GitStore` — 记录持久化文件的变更历史

这让系统在当下保持轻量，在时间维度上保持反思能力。

## 处理流程

记忆在 nanobot 中分两个阶段流转。

### 第一阶段：Consolidator（在线压缩）

当对话增长到对上下文窗口产生压力时，nanobot 不会试图永远携带每一条旧消息。

相反，Consolidator 会将最安全的旧消息片段总结为摘要，追加到 `memory/history.jsonl`。

详细的实现说明见 [memory-consolidation.md](memory-consolidation.md)。以下是关键要点。

#### 触发时机

每个 turn 的 `_state_save` 末尾以后台任务方式触发：

```
_state_save
  └─ 非临时对话
       └─ maybe_consolidate_by_tokens() → 后台触发压缩
```

#### 流程概览

```
maybe_consolidate_by_tokens
  |
  +-- 1. 加锁 + 刷新 session 引用
  +-- 2. 计算预算（budget / target）
  +-- 3. replay 溢出预压缩
  +-- 4. 估算 token
  |    +-- estimated < budget -> idle 返回
  |    +-- estimated >= budget -> 进入压缩
  +-- 5. 多轮压缩循环（最多 5 轮）
  |    每轮：pick boundary -> archive chunk -> 推进游标 -> save
  +-- 6. 持久化摘要到 session metadata
```

#### 预算计算

```python
budget = context_window_tokens - max_completion_tokens - _SAFETY_BUFFER
target = int(budget * consolidation_ratio)   # consolidation_ratio = 0.5
```

举例（128K 窗口）：budget = 122_880，target = 61_440。

#### 归档后结构

```
session.messages:
+--------- 已归档 ---------+-------- 活跃区 -------------------+
| m0  m1  ...  m6         | m7  m8  m9  m10  m11  ...       |
+--------------------------+----------------------------------+
                           ^
                      last_consolidated = 7

session.metadata._last_summary -> 摘要文本
history.jsonl                  -> {cursor, ts, content, session_key}
```

归档后原始消息仍留在 `session.messages` 中，但 prompt 构建时只取 `last_consolidated` 之后的活跃区。

#### 安全边界

只在 user 消息处切，保证不截断对话回合：

```python
for idx in range(start, len(session.messages)):
    if idx > start and messages[idx]["role"] == "user":
        if removed_tokens >= tokens_to_remove:
            return (idx, removed_tokens)
    removed_tokens += estimate_message_tokens(messages[idx])
```

#### 摘要归档

| 路径 | 行为 |
|------|------|
| 正常 | LLM 生成摘要 -> append_history(summary) 写入 history.jsonl |
| 失败 | raw_archive(messages) -> 带 [RAW] 前缀的原文转储 |

#### 数据存储

所有归档写入 `history.jsonl`，通过 `session_key` 字段隔离：

```json
{"cursor": 42, "content": "摘要...", "session_key": "cli:chat_0_xxx"}
```

### 第二阶段：Dream（深度提炼）

`Dream` 是更慢、更深思熟虑的一层。它默认按 cron 计划运行，也可以手动触发。

Dream 读取：

- `memory/history.jsonl` 中的新条目
- 当前的 `SOUL.md`
- 当前的 `USER.md`
- 当前的 `memory/MEMORY.md`

然后它通过一次微创操作来编辑长期记忆文件——不是重写所有内容，而是做出能保持记忆连贯的最小诚实变更。

这就是为什么 nanobot 的记忆不只是归档。它是具有解释力的。

## 文件结构

```text
workspace/
├── SOUL.md              # 机器人的长期语气和交流风格
├── USER.md              # 关于用户的稳定信息
└── memory/
    ├── MEMORY.md        # 项目事实、决策和持久上下文
    ├── history.jsonl    # 只追加的历史摘要
    ├── .cursor          # Consolidator 写入游标
    ├── .dream_cursor    # Dream 消费游标
    └── .git/            # 长期记忆文件的版本历史
```

这些文件各有分工：

- `SOUL.md` 记住 nanobot 应该怎么说话
- `USER.md` 记住用户是谁以及他们的偏好
- `MEMORY.md` 记住关于工作本身的真实信息
- `history.jsonl` 记住一路上发生了什么

## 为什么用 `history.jsonl`

旧版 `HISTORY.md` 格式适合人工阅读，但作为操作基础太脆弱了。

`history.jsonl` 给 nanobot 带来了：

- 稳定的增量游标
- 更安全的机器解析
- 更容易的批量处理
- 更干净的迁移和压缩
- 原始历史和整理知识之间更清晰的边界

你仍然可以用常用工具搜索它：

```bash
# grep
grep -i "关键字" memory/history.jsonl

# jq
cat memory/history.jsonl | jq -r 'select(.content | test("关键字"; "i")) | .content' | tail -20

# Python
python -c "import json; [print(json.loads(l).get('content','')) for l in open('memory/history.jsonl','r',encoding='utf-8') if l.strip() and '关键字' in l.lower()][-20:]"
```

这既是技术上的区别，也是哲学上的区别：

- `history.jsonl` 服务于结构
- `SOUL.md`、`USER.md` 和 `MEMORY.md` 服务于意义

## 命令

记忆并非不可窥探的黑箱。用户可以检查和引导它。

| 命令 | 功能 |
|------|------|
| `/dream` | 立即运行 Dream |
| `/dream-log` | 显示最新的 Dream 记忆变更 |
| `/dream-log <sha>` | 显示指定 Dream 变更 |
| `/dream-restore` | 列出最近的 Dream 记忆版本 |
| `/dream-restore <sha>` | 恢复到指定变更前的记忆状态 |

这些命令的存在是有原因的：自动记忆很强大，但用户始终应该保留检查、理解和恢复它的权利。

## 版本化记忆

Dream 修改长期记忆文件后，nanobot 可以用 `GitStore` 记录这次变更。

这给了记忆自己的历史：

- 你可以查看变更了什么
- 你可以比较不同版本
- 你可以恢复之前的状态

这让记忆从静默的突变变成了可审计的过程。

## 配置

Dream 在 `agents.defaults.dream` 下配置：

```json
{
  "agents": {
    "defaults": {
      "dream": {
        "intervalH": 2,
        "modelOverride": null,
        "maxBatchSize": 20,
        "maxIterations": 10
      }
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `intervalH` | Dream 的运行间隔（小时） |
| `cron` | cron 表达式覆盖（优先级高于 intervalH） |
| `modelOverride` | （开发中）Dream 专用模型覆盖 |
| `maxBatchSize` | （已废弃 — 不再使用） |
| `maxIterations` | （已废弃 — 不再使用） |

实际使用：

- `intervalH` 是配置 Dream 频率的正常方式。内部以 `every` 计划运行
- `cron` 设置时会覆盖 `intervalH`，允许精确的 cron 表达式（如 `0 */4 * * *`）
- `modelOverride` 为未来版本预留。目前 Dream 使用与主 Agent 相同的模型
- `maxBatchSize` 和 `maxIterations` 保留是为了配置兼容，不再影响行为

## 实际效果

在日常使用中这意味着什么很简单：

- 对话可以保持快速，无需携带无限的上下文
- 持久的事实可以随时间变得更清晰而非更嘈杂
- 用户可以在需要时检查和恢复记忆

记忆不应该像垃圾堆。它应该像连续性。

这就是这个设计试图保护的东西。
