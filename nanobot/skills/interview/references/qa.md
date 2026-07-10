# 面试题记录

<!-- 按时间倒序排列 -->

## Q002 — AgentLoop 和 AgentRunner 的职责边界

**分类:** nanobot / 架构设计
**日期:** 2026-07-10
**掌握程度:** 需深入
**来源:** nanobot 项目面试题 #2

### 回答

AgentLoop 是 agent 的"手脚"，负责记忆、保存、读取图片等，处理好了再交给 AgentRunner 作为 ReAct 核心执行。

### 补充

这个比喻方向对，但边界不够精确。实际分工：
- **AgentLoop** 是状态机编排器：dispatch → restore → compact → command → build → run → save → respond。管的是什么时候做、做什么状态（session 恢复、记忆加载、上下文构建、工具注册）。
- **AgentRunner** 是 ReAct 执行引擎：拿到完整的 messages + tools，反复调 LLM → 执行 tools → 继续，直到 finish。
- 更准确的比喻：AgentLoop = 导演（决定剧本走到哪一幕），AgentRunner = 摄制组（拿到剧本就开机拍摄，不断喊"卡"→重来→杀青）。

---

## Q001 — MessageBus 解耦 channels 和 agent core 的好处

**分类:** nanobot / 架构设计
**日期:** 2026-07-10
**掌握程度:** 待复习
**来源:** nanobot 项目面试题 #1

### 回答

消费者生产者结构，可以让 LLM 处理不至于让队列卡。

### 补充

核心直觉（生产者-消费者 + 不阻塞）是对的，但深度不够。完善的回答应涵盖：
1. **异步解耦** — Channel publish 后不等 LLM 返回，避免 webhook 超时重发。
2. **多 channel 共享 agent** — Channel 只认识 bus，agent 只订阅 bus，零耦合。
3. **背压控制** — `asyncio.Queue(maxsize=N)` 限流，LLM 处理不过来时反压到 channel 层。
4. **事件多播** — 一条 InboundMessage 可被多个 subscriber 消费（如 agent + 审计日志 hook）。
5. **可测试性** — 可分别 mock bus 测 channel 和 agent。
