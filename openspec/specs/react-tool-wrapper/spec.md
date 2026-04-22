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

