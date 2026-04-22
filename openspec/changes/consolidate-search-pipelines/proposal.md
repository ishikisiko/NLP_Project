## Why

当前搜索流程同时维护 `SmartSearchOrchestrator`、`LangChainOrchestrator`、`ReactAgentOrchestrator` 以及两套 RAG 实现，导致相同职责在多条 pipeline 中重复出现。随着 LangChain 主链路已经成为默认路径，这种分叉开始显著增加维护成本、行为漂移风险和新能力接入复杂度。

## What Changes

- 统一搜索主编排器，保留 `LangChainOrchestrator` 作为唯一默认 orchestrator
- 将 `SmartSearchOrchestrator` 调整为兼容迁移路径，并逐步移除其重复的路由与执行逻辑
- 合并 `rag/` 与 `langchain/` 中重复的 Local RAG / Search RAG 实现，收敛为一套主执行链
- 将 `ReactAgentOrchestrator` 调整为默认搜索主链路的 fallback-only 执行器，而不是长期并列的顶层模式
- 抽取统一的搜索路由核心，负责时间约束解析、small talk 判定、领域分类、搜索决策与关键词生成
- 统一返回结构与 `control.search_mode` 语义，确保 CLI、API 与前端观察到一致的执行元数据
- 保留必要的兼容开关，确保迁移期间现有调用方可继续工作

## Capabilities

### New Capabilities
- `search-routing-core`: 统一的搜索路由核心，负责查询分类、搜索决策和关键词生成
- `unified-rag-execution`: 统一的 search/local RAG 执行层，替代重复的 legacy 与 LangChain RAG 实现
- `search-response-control`: 统一搜索响应中的 `control` 元数据、`search_mode` 语义和 fallback 标记

### Modified Capabilities
- `react-orchestrator`: 调整 ReAct 编排器定位，使其以 fallback-only 执行器为主，而不是默认长期并列入口
- `query-postcheck-fallback`: 明确 post-check 到 ReAct fallback 的衔接关系，要求其围绕统一默认主链路工作

## Impact

- **修改文件**: `main.py`, `server.py`, `langchain/langchain_orchestrator.py`, `langchain/langchain_rag.py`, `orchestrators/react_agent_orchestrator.py`, `search/source_selector.py`
- **可能收敛或退役的文件**: `orchestrators/smart_orchestrator.py`, `rag/local_rag.py`, `rag/search_rag.py`
- **兼容性影响**: 保留 legacy 开关和现有 API 返回字段作为迁移过渡，避免前端和 CLI 立即断裂
- **验证重点**: 时间变化类查询、领域 API 查询、搜索不可用回退以及 post-check 触发的 ReAct fallback
