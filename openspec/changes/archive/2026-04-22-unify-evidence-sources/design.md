## Context

当前默认主链路已经通过 `LangChainOrchestrator -> SearchRAGChain` 统一了大部分 search/local 执行语义，但内部仍把三类证据分别处理：

- 网页搜索输出 `search_hits`
- 本地文档输出 `retrieved_docs`
- 领域 API 通过 `extra_context` 或旁路直出接入

这意味着系统已经共享了“回答阶段”，却没有共享“证据阶段”。结果是：

- 不同来源在抽象层级上不对称，难以做统一过滤、重排和观测
- ReAct 本地工具仍依赖 legacy `LocalRAG`
- `source_selector`、默认主链路和 fallback 工具没有共享同一套来源模型
- 后续若要退役 `rag/local_rag.py` / `rag/search_rag.py`，缺少稳定的中间抽象可供迁移

这次变更属于跨模块架构收敛：它同时影响 orchestrator、RAG 执行层、领域来源接线、ReAct 工具和响应元数据。因此需要先明确统一来源层和统一证据模型，再实现具体迁移。

## Goals / Non-Goals

**Goals:**
- 定义统一的 `EvidenceSource` 抽象，覆盖 `web`、`domain`、`local` 三类一级来源
- 定义统一的 `EvidenceItem` 结构，使不同来源的证据可以进入同一条 fusion 流程
- 让默认主链路围绕统一 evidence retrieval / normalization / rerank / answer context 工作
- 让 ReAct local/search recovery 工具复用统一来源层，而不是直接绑定 legacy RAG 类
- 保留现有 `search_hits`、`retrieved_docs`、`control` 等兼容字段，同时为后续退役 legacy 执行层创造条件

**Non-Goals:**
- 不把本地文件伪装成 `SearchClient` provider，也不修改 Brave/Google/You/Bright Data 的 provider 合并策略
- 不在本次变更中重新设计 domain classification 规则或具体领域 API 的取数协议
- 不要求一次性删除全部 legacy 文件；允许先让默认路径与 fallback 工具切到 EvidenceSource，再决定是否退役旧类
- 不在本次变更中重做前端交互，仅保证元数据可兼容扩展

## Decisions

### 1. 引入一级来源抽象 `EvidenceSource`，而不是把 local/domain 硬塞进 `SearchClient`

`SearchClient` 的语义是“网页搜索 provider”，天然输出 `SearchHit[]`，并带有 `per_source_limit`、provider timings、quota/fallback 等属性。本地文件和领域 API 并不符合这一语义，因此不应强行伪装成 `SearchClient` 的一个 provider。

设计上将新增统一来源层：

```text
EvidenceSource
  - source_type: web | domain | local
  - source_id
  - display_name
  - retrieve(query, options) -> EvidenceItem[]
```

其中：
- `web` 类型可以包装现有 `SearchClient` / `CombinedSearchClient` / `PrioritySearchClient`
- `local` 类型负责向量检索本地文档
- `domain` 类型负责把领域 API 数据转成可引用证据，而不仅是字符串上下文

备选方案：
- 让 `local` 和 `domain` 继承 `SearchClient`。放弃，因为会扭曲接口语义，并把 URL/snippet/provider 限制强加到非网页来源。
- 维持 `search_hits` / `retrieved_docs` / `extra_context` 三轨并存。放弃，因为这正是当前无法进一步收敛的根源。

### 2. 定义统一的 `EvidenceItem` 作为跨来源证据模型

为了让 fusion、重排、回答拼装和 fallback 工具共用同一份证据语义，所有来源都应被归一化为统一结构。建议的核心字段包括：

- `source_type`
- `source_id`
- `title`
- `content`
- `reference`（URL、文件路径、API 标识等）
- `snippet`
- `metadata`
- `score` / `rank`（可选）

这不是要求所有来源字段完全相同，而是要求它们先进入一个共享的最小证据模型，再由兼容层投影回 `search_hits` / `retrieved_docs`。

备选方案：
- 仅在回答前拼 prompt，不保留统一证据对象。放弃，因为无法支撑统一重排、统一观测和 ReAct 工具复用。

### 3. 默认主链路改为 “retrieve -> normalize -> fuse -> answer”

当前 `SearchRAGChain` 已经在同一条回答链里处理 search/local 两类数据。本次变更不推翻这一点，而是将其内部结构显式化：

```text
query
  -> EvidenceSource fan-out
  -> EvidenceItem normalization
  -> filter / dedupe / rerank
  -> answer context builder
  -> llm answer
```

这允许：
- search on/off 只是启用哪些来源，而不是切换不同 pipeline
- local-only 和 search+local 使用完全相同的后处理语义
- domain API 继续保留直出路径，但在“继续搜索”或“补证据”场景下进入统一 evidence fusion 流程

备选方案：
- 新建一个与 `SearchRAGChain` 平行的 `EvidenceFusionChain`。暂不选，因为会引入新的并行主执行层；更合理的做法是在现有默认主链路内部收敛。

### 4. 兼容输出保留旧字段，但内部以 `EvidenceItem` 为真源

外部调用方仍依赖：
- `search_hits`
- `retrieved_docs`
- `control`

因此内部统一来源后，不应立刻删除这些字段，而应把它们视为从统一 `EvidenceItem[]` 投影出的兼容视图。例如：

- `search_hits` 由 `source_type=web` 的 evidence items 派生
- `retrieved_docs` 由 `source_type=local` 的 evidence items 派生
- 领域 API 证据可通过 `control.evidence_sources_*`、新增 evidence summary 或兼容字段补充暴露

这样可以让实现层收敛，而不强迫前端、测试、评测脚本同一轮全部迁移。

### 5. ReAct 工具也应切到统一来源层

`ReActLocalDocTool` 当前直接绑定 legacy `LocalRAG`，这是统一默认主链路后最明显的残留分叉。设计上应让：

- `local_docs` 工具调用 local `EvidenceSource`
- 高层搜索恢复工具调用统一 fusion pipeline
- `domain_api` 工具在需要结构化证据时也能输出 EvidenceItem 兼容摘要

这样 default pipeline 和 fallback pipeline 才能真正共享证据语义，而不是只共享“最终回答”。

## Risks / Trade-offs

- [统一来源层过度设计] → 先定义最小 `EvidenceSource` / `EvidenceItem` 契约，只覆盖当前 web/domain/local 三类来源
- [兼容字段与内部模型双轨导致复杂度上升] → 明确 `EvidenceItem` 为内部真源，旧字段只做只读投影
- [domain API 证据难以统一成文档式片段] → 允许 `EvidenceItem.metadata` 保存结构化数据，`content` 保存用于回答和引用的摘要文本
- [ReAct 工具迁移时行为变化] → 保留工具名和基础输出格式，先换底层实现，再补回归测试
- [legacy 类短期仍然存在] → 分阶段推进；先让默认路径和 fallback 工具都切到 EvidenceSource 层，再决定是否删除旧类

## Migration Plan

1. 定义 `EvidenceSource` / `EvidenceItem` 契约，并为 web/local/domain 准备适配器
2. 在默认主链路中引入 evidence retrieval / normalization / fusion 流程，保留兼容输出字段
3. 让 `ReActLocalDocTool` 和高层 search recovery 工具切到统一来源层
4. 为统一来源层补充来源元数据与回归覆盖，验证 local-only、search+local、domain+search 和 fallback 路径
5. 在默认路径稳定后，评估 `LocalRAGChain`、`rag/local_rag.py`、`rag/search_rag.py` 是否还能完全退役

回滚策略：
- 在迁移期保留旧字段和必要的 legacy 类；若统一来源层导致行为回归，可临时将默认主链路切回现有 `SearchRAGChain` 的旧内部拼装方式

## Open Questions

- domain API 在“直出回答”和“产出 EvidenceItem 进入 fusion”之间是否需要显式双模式接口
- 是否需要新增统一的 `control.evidence_sources_active` / `control.evidence_sources_used` 元数据，而不是继续只暴露 `search_sources_*`
- `EvidenceItem` 是否需要统一的跨来源分数语义，还是允许各来源仅在 `metadata` 中暴露原始得分
- local 文件路径、URL、domain API 记录在引用展示上的统一格式应如何定义，才能既兼容现有 UI 又利于评测
