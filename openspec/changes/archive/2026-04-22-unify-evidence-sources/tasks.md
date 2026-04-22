## 1. EvidenceSource Contract

- [x] 1.1 新增统一来源层模块，定义 `EvidenceSource` 接口、来源类型枚举以及 `EvidenceItem` 数据结构
- [x] 1.2 为现有网页搜索客户端提供 `web` 类型 EvidenceSource 适配器，并保留现有 provider 合并语义
- [x] 1.3 为本地向量检索提供 `local` 类型 EvidenceSource 适配器，支持从 `data_path` 检索本地证据
- [x] 1.4 为领域 API 提供 `domain` 类型 EvidenceSource 适配器，使领域证据可转换为统一来源层输出

## 2. Unified Fusion Pipeline

- [x] 2.1 在默认 LangChain 主链路中引入 evidence retrieval / normalization 流程，替代内部对 `search_hits`、`retrieved_docs` 和 `extra_context` 的分轨处理
- [x] 2.2 在默认主链路中实现统一的 `EvidenceItem` 过滤、去重、排序或等价后处理逻辑
- [x] 2.3 让 local-only、search+local、search-unavailable 和 domain+search 路径都复用统一 fusion 语义
- [x] 2.4 从统一 `EvidenceItem` 集合投影生成兼容的 `search_hits`、`retrieved_docs` 和来源引用输出

## 3. ReAct and Tooling Integration

- [x] 3.1 将 `ReActLocalDocTool` 改为复用 `local` EvidenceSource，而不是直接依赖 legacy `LocalRAG`
- [x] 3.2 将高层搜索恢复工具改为复用统一来源层和 fusion pipeline，而不是重复拼装 search/local 逻辑
- [x] 3.3 调整 fallback 相关工具输出，使其保留统一来源标识和证据摘要语义

## 4. Compatibility, Observability, and Validation

- [x] 4.1 为默认主链路和 fallback 路径补充统一来源元数据，如 active/used evidence sources 和来源类型信息
- [x] 4.2 更新或新增回归测试，覆盖 web、local、domain、多来源融合、ReAct local tool 和 fallback 路径
- [x] 4.3 验证 legacy `LocalRAGChain`、`rag/local_rag.py`、`rag/search_rag.py` 的剩余调用点，并记录后续可退役范围
