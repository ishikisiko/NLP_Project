## MODIFIED Requirements

### Requirement: LocalDocTool 封装
系统 SHALL 将本地知识库工具封装为 LangChain BaseTool，名为 `local_docs`，并通过统一 `EvidenceSource` / `EvidenceItem` 层复用默认主链路的本地证据检索能力，而不是直接依赖 legacy `LocalRAG`。

#### Scenario: 查询本地知识库
- **WHEN** Agent 调用 `local_docs` 工具并传入查询
- **THEN** 工具 SHALL 通过 `source_type=local` 的统一来源层检索本地证据
- **AND** 返回结果 SHALL 包含可读的证据摘要与文档来源信息

#### Scenario: 无本地文档
- **WHEN** `data_path` 未配置或文档不存在
- **THEN** 工具 SHALL 返回"本地知识库不可用"

### Requirement: ReAct tools SHALL expose high-level search recovery capabilities
系统 SHALL 为 ReAct fallback 提供高层搜索恢复工具，使 Agent 能复用当前默认 pipeline 的统一 EvidenceSource、归一化证据和融合能力，而不只依赖基础网页搜索或 legacy RAG 类。

#### Scenario: Agent uses high-level search tool during fallback
- **WHEN** ReAct fallback 需要补检索、补证据或重新综合搜索结果
- **THEN** Agent SHALL 可调用高层搜索工具
- **AND** 该工具 SHALL 复用统一来源层、统一证据归一化和适用的过滤/重排能力

#### Scenario: High-level search tool returns structured recovery output
- **WHEN** 高层搜索工具完成一次恢复性检索
- **THEN** 工具 SHALL 返回足以支持后续推理的结果
- **AND** 返回内容 SHALL 至少包含回答或证据摘要以及对应来源信息

### Requirement: ReAct fallback tools SHALL preserve domain and local-context recovery
系统 SHALL 在 fallback 场景中保留对领域数据和本地知识库的复用能力，并允许与统一来源层中的其他证据来源组合使用。

#### Scenario: Agent combines domain data and search recovery
- **WHEN** 查询既需要领域 API 数据也需要网络搜索补证据
- **THEN** Agent SHALL 能在同一次 fallback 中组合使用领域工具与高层搜索工具
- **AND** 最终结果 SHALL 能整合多种来源的信息

#### Scenario: Agent reuses local documents during fallback
- **WHEN** 查询涉及已上传的本地文档且 post-check 判断证据不足
- **THEN** Agent SHALL 仍可访问本地知识库工具
- **AND** 工具输出 SHALL 保留文档来源信息以便最终回答引用
