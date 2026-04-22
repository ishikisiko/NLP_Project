# react-orchestrator Specification

## Purpose
TBD - created by archiving change langchain-react-agent. Update Purpose after archive.
## Requirements
### Requirement: ReactAgentOrchestrator 接口
系统 SHALL 提供 `ReactAgentOrchestrator` 类，对外暴露与现有编排器一致的 `answer()` 接口。

#### Scenario: 基本回答流程
- **WHEN** 用户调用 `orchestrator.answer(query)`
- **THEN** Orchestrator SHALL 使用 ReAct Agent 处理查询
- **AND** 返回包含 `answer`、`control`、`search_hits` 字段的字典

#### Scenario: 返回结构兼容
- **WHEN** 任意编排器返回结果
- **THEN** `answer` 字段包含最终回答文本
- **AND** `control` 字段包含元数据（search_performed、decision 等）
- **AND** `search_hits` 字段包含搜索结果（如有）

### Requirement: 配置化工具列表
系统 SHALL 支持在初始化时配置哪些工具可用。

#### Scenario: 默认工具集
- **WHEN** 创建 ReactAgentOrchestrator 时不指定 tools 参数
- **THEN** 默认使用 [web_search, domain_api, local_docs] 三个工具

#### Scenario: 自定义工具集
- **WHEN** 创建 ReactAgentOrchestrator 时传入 tools 参数
- **THEN** 只使用传入的工具

### Requirement: 迭代次数控制
系统 SHALL 支持通过参数控制 ReAct Agent 最大迭代次数。

#### Scenario: 默认迭代限制
- **WHEN** 创建时不指定 max_iterations
- **THEN** 默认值为 5

#### Scenario: 自定义迭代限制
- **WHEN** 创建时指定 max_iterations=3
- **THEN** Agent 最多执行 3 次 Thought-Action-Observation 循环

### Requirement: 向后兼容切换
系统 SHALL 支持通过配置切换执行模式，不影响现有调用方。

#### Scenario: 模式切换
- **WHEN** 在 config.json 中设置 `"orchestrator_mode": "react"`
- **THEN** `create_langchain_orchestrator` 返回 ReactAgentOrchestrator
- **AND** `create_langchain_orchestrator` 返回 LangChainOrchestrator（默认）

### Requirement: ReactAgentOrchestrator SHALL support fallback execution context
系统 SHALL 支持将 ReactAgentOrchestrator 作为默认 pipeline 的补救执行器使用，并接收 fallback 上下文。

#### Scenario: Fallback invocation carries prior answer context
- **WHEN** 默认 pipeline 的 post-check 决定触发 ReAct fallback
- **THEN** ReactAgentOrchestrator SHALL 能接收原始 query、首答摘要和 post-check failure types
- **AND** Agent SHALL 使用这些上下文作为补救执行的起点

#### Scenario: Fallback invocation carries available evidence summary
- **WHEN** 默认 pipeline 已经获取了搜索结果、领域数据或本地文档
- **THEN** ReactAgentOrchestrator SHALL 支持接收这些证据的摘要或引用
- **AND** Agent SHALL 可在不完全从零开始的前提下继续补救

### Requirement: ReactAgentOrchestrator SHALL return fallback-compatible metadata
系统 SHALL 在 ReAct 作为 fallback 返回结果时输出与默认主链路兼容的元数据。

#### Scenario: Successful fallback response
- **WHEN** ReactAgentOrchestrator 产出用于替代首答的最终结果
- **THEN** 返回结构 SHALL 包含 `answer`、`control` 和 `search_hits`
- **AND** `control` SHALL 标明执行来源为 fallback ReAct

#### Scenario: Fallback execution ends without full resolution
- **WHEN** ReAct fallback 达到迭代上限或未能完全补救首答问题
- **THEN** 返回结构 SHALL 保留失败或截断信息
- **AND** `control` SHALL 包含与 fallback 执行相关的原因说明

