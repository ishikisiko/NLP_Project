## Context

当前仓库同时存在三层重复：入口层有 `SmartSearchOrchestrator`、`LangChainOrchestrator`、`ReactAgentOrchestrator` 三种编排方式；执行层有 legacy `rag/` 和 LangChain `langchain/` 两套 Local/Search RAG；返回层又在不同 orchestrator 中重复拼装 `control` 元数据。CLI 仍保留 legacy 入口，Web 端已默认走 LangChain，这使得同一查询语义在不同入口下可能出现不一致。

这次变更是跨模块收敛，不是单点重构。它需要同时调整 `main.py`、`server.py`、编排器、RAG 执行层和响应元数据，因此适合先以设计文档固定边界和迁移顺序。

## Goals / Non-Goals

**Goals:**
- 将 `LangChainOrchestrator` 固定为唯一默认主编排器
- 将查询路由逻辑收敛为一套共享能力，避免 legacy 与 LangChain 各自维护
- 将本地 RAG 和搜索 RAG 收敛到一套主执行链
- 将 `ReactAgentOrchestrator` 固定为 post-check 失败后的补救执行器
- 统一 CLI、API、前端可见的 `control.search_mode` 和 fallback 元数据
- 在迁移过程中保留必要兼容开关，避免一次性破坏现有调用方

**Non-Goals:**
- 不在本次变更中重新设计 search provider 路由策略
- 不在本次变更中重写 domain API 本身的取数逻辑
- 不要求一次性删除全部 legacy 文件；允许先停用主路径，再逐步退役
- 不修改前端交互模型，只保证返回结构兼容

## Decisions

### 1. 保留 LangChainOrchestrator 作为唯一默认入口

`LangChainOrchestrator` 已经是当前默认主线，并且已经具备 post-check 与 ReAct fallback 机制。继续保留 `SmartSearchOrchestrator` 作为并列主线只会维持重复成本，因此 CLI 和 Web 的默认行为都应收敛到 LangChain 主链路。

备选方案：
- 保留双主线，按环境切换。放弃，因为相同职责会继续分叉。
- 回退到 Smart 作为唯一主线。放弃，因为它已经不是 Web 默认路径，且缺少当前主线的 post-check 架构。

### 2. 将 ReactAgentOrchestrator 调整为 fallback-only executor

ReAct 对复杂恢复有效，但不适合作为所有请求的长期顶层默认入口。顶层与 fallback 两种 ReAct 并存，会带来配置和观察语义混乱。设计上应保留其独立 `answer()` 接口，以便测试和调试，但生产默认链路只在 post-check 判定可恢复失败时调用它。

备选方案：
- 保留 `orchestrator_mode=react` 作为正式并列模式。放弃，因为会继续造成两条顶层执行语义并存。

### 3. 抽取共享路由核心，而不是在 orchestrator 中复制判断

时间约束解析、small talk 判定、domain classification、search decision 和 keyword generation 应抽成共享路由核心，由默认 orchestrator 调用。legacy 兼容路径若保留，也应复用这套核心，而不是继续维护自己的判断分支。

这样做的结果是：
- 查询路由规则只定义一次
- 测试可以围绕路由核心组织
- 未来新增路径时不需要复制整套决策逻辑

### 4. 统一 RAG 执行层，优先保留 LangChain 实现

`langchain/langchain_rag.py` 已经提供 Local 和 Search 两条执行链，并与当前默认编排器自然对接。因此执行层收敛将以 LangChain RAG 为主，把 legacy `rag/` 中仍有价值的能力逐步迁入，例如时间变化类查询的颗粒化搜索与补充证据逻辑。

备选方案：
- 统一到 legacy RAG。放弃，因为需要反向适配当前默认主链路。
- 长期保留双实现。放弃，因为这正是当前维护成本的来源。

### 5. 统一响应控制语义，保留字段兼容

所有对外返回都保留现有 `answer`、`search_hits`、`control` 等主要字段，但要求：
- `control.search_mode` 的枚举语义一致
- `control.final_executor` 明确最终执行者
- `control.fallback_triggered`、`postcheck`、search source metadata 的含义一致

这允许前端和评测逻辑在不立即改协议的前提下获得更稳定的执行语义。

## Risks / Trade-offs

- [Legacy 路径仍被少量 CLI 调用依赖] → 先保留兼容开关，把默认入口收敛与彻底删除 legacy 拆成两个阶段
- [合并 RAG 时丢失 legacy 中的特殊查询优化] → 先识别并迁移时间变化、排名提取等特殊能力，再停用 legacy 执行链
- [`search_mode` 语义收紧后影响现有调试脚本] → 保留主要字段名不变，并在变更说明中列出旧值到新语义的映射
- [React 顶层模式下线可能影响实验场景] → 保留独立构造和手动调用能力，但不再作为默认正式入口暴露

## Migration Plan

1. 先抽取共享路由核心，并让 `LangChainOrchestrator` 使用它
2. 将 CLI 默认路径、Web 路径和 post-check fallback 的元数据收敛到统一 `control` 语义
3. 将 legacy RAG 的特殊能力迁移到 LangChain RAG 主执行链
4. 将 `SmartSearchOrchestrator` 降级为兼容实现，仅通过显式开关访问
5. 将 `ReactAgentOrchestrator` 固定为 fallback-only 执行器，并移除其作为长期默认顶层模式的要求
6. 在完成回归验证后，再评估是否彻底退役 legacy 文件

回滚策略：
- 保留 legacy 开关和兼容代码路径，若统一主链路出现严重回归，可临时切回兼容路径

## Open Questions

- 是否完全移除 `orchestrator_mode=react` 配置，还是仅将其标记为调试/实验用途
- 时间变化类查询中的 legacy 特化逻辑，哪些必须迁移，哪些可以删除
- 是否需要为统一后的 `search_mode` 建立显式枚举文档，供前端和评测脚本复用
