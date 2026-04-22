# unified-rag-execution Specification

## Purpose
Define the single primary RAG execution layer used by the default orchestrator for local, search, and hybrid execution.

## Requirements
### Requirement: Default pipeline SHALL use a single primary RAG execution layer
系统 SHALL 为默认主编排器提供单一主 RAG 执行层，用于处理 local RAG、search RAG 以及 search+local 的混合执行，而不是长期并行维护两套主执行实现。

#### Scenario: Default search path uses unified execution layer
- **WHEN** 默认主编排器判定需要执行搜索增强回答
- **THEN** 系统 SHALL 使用统一主 RAG 执行层完成搜索结果整合、可选本地文档检索和答案生成
- **AND** 系统 SHALL NOT 在默认路径上并行依赖一套独立的 legacy SearchRAG 主实现

#### Scenario: Search disabled path uses unified local execution semantics
- **WHEN** 查询在搜索关闭的条件下执行
- **THEN** 系统 SHALL 使用统一主执行层中的 local-only 语义处理本地文档检索或 direct fallback
- **AND** CLI 与 API 的该路径行为 SHALL 保持一致

### Requirement: Unified execution layer SHALL preserve required specialized search behavior
统一主 RAG 执行层 SHALL 保留默认系统所需的特殊查询处理能力，特别是时间变化类查询和需要补充证据的搜索场景。

#### Scenario: Temporal-change query requires extra evidence gathering
- **WHEN** 查询属于时间变化、历史趋势或跨年份比较场景
- **THEN** 统一主执行层 SHALL 支持追加的历史证据收集或颗粒化搜索行为
- **AND** 该能力 SHALL 在 legacy 执行链退役后继续可用

#### Scenario: Unified execution fails to search
- **WHEN** 外部搜索不可用但本地文档或 direct answer 仍可用
- **THEN** 系统 SHALL 使用统一主执行层定义的回退语义生成结果
- **AND** 返回结构 SHALL 明确记录搜索不可用而非 silently dropping the failure
