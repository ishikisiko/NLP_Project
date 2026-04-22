## Why

当前默认主链路虽然已经把网络搜索与本地文档放进同一条 `SearchRAGChain`，但三类证据来源仍然处在不同抽象层级：网页搜索是显式 provider，领域 API 是旁路补充上下文，本地文档则是布尔开关控制的隐藏能力。这种不对称导致检索、重排、fallback 工具复用和返回元数据都缺少统一模型，也让后续退役 legacy `LocalRAG` / `SearchRAG` 变得困难。

## What Changes

- 引入统一的 `EvidenceSource` 抽象，覆盖 `web`、`domain`、`local` 三类一级来源
- 定义统一的 `EvidenceItem` 数据模型，使网页结果、本地文档片段和领域 API 证据都能进入同一条检索后处理链
- 将默认主链路中的证据获取、过滤、重排和回答上下文拼装收敛到统一 Evidence fusion pipeline，而不是分别处理 `search_hits`、`retrieved_docs` 和 `extra_context`
- 调整 ReAct 相关工具，使其本地文档和高层搜索恢复能力复用统一 EvidenceSource 层，而不是直接绑定 legacy `LocalRAG`
- 为统一来源层补充来源元数据和可观测性语义，确保 CLI、API、fallback 和评测脚本能观察到激活来源、来源类型和证据摘要

## Capabilities

### New Capabilities
- `evidence-source-layer`: 定义统一的 `EvidenceSource` 接口、来源类型和检索调用语义
- `evidence-item-normalization`: 定义统一的 `EvidenceItem` 结构以及 web/domain/local 证据到统一结构的归一化规则
- `evidence-fusion-pipeline`: 定义默认主链路如何从多个 `EvidenceSource` 收集证据、执行统一过滤/重排并生成回答上下文

### Modified Capabilities
- `react-tool-wrapper`: 将 ReAct 本地知识库和高层恢复工具的行为改为复用统一 EvidenceSource / EvidenceItem 层，而不是直接依赖 legacy `LocalRAG`

## Impact

- **主要影响文件**: `langchain/langchain_orchestrator.py`, `langchain/langchain_rag.py`, `langchain/langchain_react_tools.py`, `search/source_selector.py`, `search/search.py`
- **潜在新模块**: `evidence/` 或等价共享检索层，用于承载 `EvidenceSource`、`EvidenceItem` 和 fusion 逻辑
- **兼容性影响**: 需要在保留 `search_hits` / `retrieved_docs` 等兼容字段的同时，引入统一证据层内部模型
- **迁移影响**: 为后续退役 `rag/local_rag.py`、`rag/search_rag.py` 和 `LocalRAGChain` 留出更明确的抽象边界
- **验证重点**: local-only、search+local、domain+search、post-check fallback 和 ReAct local tool 路径在统一来源层下保持一致行为
