# react-tool-wrapper Specification

## Purpose
TBD - created by archiving change langchain-react-agent. Update Purpose after archive.
## Requirements
### Requirement: WebSearchTool 封装
系统 SHALL 将 `search_client.search()` 封装为 LangChain BaseTool，名为 `web_search`。

#### Scenario: 执行网络搜索
- **WHEN** Agent 调用 `web_search` 工具并传入查询字符串
- **THEN** 工具 SHALL 调用 `search_client.search(query, num_results=5)`
- **AND** 返回格式化后的搜索结果字符串

#### Scenario: 搜索结果格式化
- **WHEN** search 返回 SearchHit 列表
- **THEN** 工具 SHALL 将结果格式化为可读字符串，每条结果包含序号、标题、URL、摘要
- **AND** 如果无结果，返回 "未找到相关结果"

### Requirement: DomainApiTool 封装
系统 SHALL 将 `IntelligentSourceSelector` 封装为 LangChain BaseTool，名为 `domain_api`。

#### Scenario: 获取领域专业数据
- **WHEN** Agent 调用 `domain_api` 工具并传入领域查询
- **THEN** 工具 SHALL 调用 `source_selector.select_sources(query)`
- **AND** 如果领域为 weather/finance/transport/sports，进一步调用 `fetch_domain_data()`
- **AND** 返回领域数据或自然语言回答

#### Scenario: 通用领域查询
- **WHEN** Agent 调用 `domain_api` 但无匹配领域
- **THEN** 工具 SHALL 返回空或提示"无相关领域数据"

### Requirement: LocalDocTool 封装
系统 SHALL 将 `LocalRAG` 封装为 LangChain BaseTool，名为 `local_docs`。

#### Scenario: 查询本地知识库
- **WHEN** Agent 调用 `local_docs` 工具并传入查询
- **THEN** 工具 SHALL 调用 `LocalRAG.query(query, num_retrieved_docs=3)`
- **AND** 返回检索到的文档片段

#### Scenario: 无本地文档
- **WHEN** `data_path` 未配置或文档不存在
- **THEN** 工具 SHALL 返回"本地知识库不可用"

### Requirement: ReAct tools SHALL expose high-level search recovery capabilities
系统 SHALL 为 ReAct fallback 提供高层搜索恢复工具，使 Agent 能复用当前默认 pipeline 的成熟搜索能力，而不只依赖基础网页搜索。

#### Scenario: Agent uses high-level search tool during fallback
- **WHEN** ReAct fallback 需要补检索、补证据或重新综合搜索结果
- **THEN** Agent SHALL 可调用高层搜索工具
- **AND** 该工具 SHALL 复用现有 SearchRAG、重排、时间约束或本地文档整合能力中的适用部分

#### Scenario: High-level search tool returns structured recovery output
- **WHEN** 高层搜索工具完成一次恢复性检索
- **THEN** 工具 SHALL 返回足以支持后续推理的结果
- **AND** 返回内容 SHALL 至少包含回答或证据摘要以及对应来源信息

### Requirement: ReAct fallback tools SHALL preserve domain and local-context recovery
系统 SHALL 在 fallback 场景中保留对领域数据和本地知识库的复用能力，并允许与高层搜索能力组合使用。

#### Scenario: Agent combines domain data and search recovery
- **WHEN** 查询既需要领域 API 数据也需要网络搜索补证据
- **THEN** Agent SHALL 能在同一次 fallback 中组合使用领域工具与高层搜索工具
- **AND** 最终结果 SHALL 能整合多种来源的信息

#### Scenario: Agent reuses local documents during fallback
- **WHEN** 查询涉及已上传的本地文档且 post-check 判断证据不足
- **THEN** Agent SHALL 仍可访问本地知识库工具
- **AND** 工具输出 SHALL 保留文档来源信息以便最终回答引用

