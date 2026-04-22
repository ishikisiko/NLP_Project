## 1. Routing Core Consolidation

- [x] 1.1 Inventory the existing routing responsibilities in `SmartSearchOrchestrator` and `LangChainOrchestrator`, and define the shared routing interface for time parsing, small-talk detection, domain classification, search decision, and keyword generation
- [x] 1.2 Implement the shared routing core module and wire `LangChainOrchestrator` to use it for default CLI and API execution
- [x] 1.3 Update or isolate legacy routing code so the compatibility path reuses the shared routing core instead of maintaining divergent routing logic

## 2. Unified RAG Execution

- [x] 2.1 Identify the legacy-only behaviors in `rag/search_rag.py` and `rag/local_rag.py` that must survive consolidation, especially temporal-change and evidence-gathering behavior
- [x] 2.2 Migrate the required legacy search/local execution behaviors into the LangChain RAG execution layer in `langchain/langchain_rag.py`
- [x] 2.3 Update the default orchestrator flow so search, local-only, and search-unavailable handling all use the unified primary execution layer

## 3. Fallback and Response Semantics

- [x] 3.1 Update `ReactAgentOrchestrator` and its call sites so ReAct is treated as a fallback-only executor in the default production path
- [x] 3.2 Normalize `control.search_mode`, `control.final_executor`, fallback flags, and related metadata across direct, local, search, search-unavailable, and fallback paths
- [x] 3.3 Preserve compatibility fields in CLI and API responses while documenting or codifying the normalized control semantics

## 4. Compatibility, Cleanup, and Validation

- [x] 4.1 Keep the legacy compatibility switch in place while removing `SmartSearchOrchestrator` from the default execution path in `main.py` and `server.py`
- [x] 4.2 Add or update regression coverage for default CLI/API routing, temporal-change queries, domain API queries, search-unavailable fallback, and post-check-triggered ReAct fallback
- [x] 4.3 Evaluate whether the legacy files can be retired immediately or should remain as dormant compatibility code, and update project documentation accordingly
