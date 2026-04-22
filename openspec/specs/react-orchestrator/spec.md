# react-orchestrator Specification

## Purpose
Define the standalone React agent executor and its fallback role within the unified default search pipeline.
## Requirements
### Requirement: ReactAgentOrchestrator 接口
系统 SHALL 提供 `ReactAgentOrchestrator` 类，对外暴露与现有编排器一致的 `answer()` 接口。

#### Scenario: 基本回答流程
- **WHEN** 用户调用 `orchestrator.answer(query)`
- **THEN** Orchestrator SHALL 使用 ReAct Agent 处理查询
- **AND** 返回包含 `answer`、`control`、`search_hits` 的兼容字典
- **AND** 返回结构 SHALL 允许附带统一 evidence 元数据

#### Scenario: 返回结构兼容
- **WHEN** 任意编排器返回结果
- **THEN** `answer` 字段包含最终回答文本
- **AND** `control` 字段包含元数据（search_performed、decision 等）
- **AND** `search_hits` 字段包含搜索结果（如有）
- **AND** 如存在 fallback 上下文中的统一证据元数据，返回结构 SHALL 保留 `evidence_items` 或 `evidence_sources_*` 语义

### Requirement: 配置化工具列表
系统 SHALL 支持在初始化时配置哪些工具可用。

#### Scenario: 默认工具集
- **WHEN** 创建 ReactAgentOrchestrator 时不指定 tools 参数
- **THEN** 默认使用 `web_search`、`domain_api`、`local_docs`
- **AND** 如启用 LLM 支持，还 SHALL 提供 `search_recovery`

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
系统 SHALL 保留 `ReactAgentOrchestrator` 的独立构造与调用能力，但其在默认产品链路中的主要定位 SHALL 为统一默认搜索主链路的 fallback-only 执行器，而不是长期并列的顶层默认模式。

#### Scenario: Default production path does not select top-level React mode
- **WHEN** 系统按默认产品配置处理查询
- **THEN** 默认主链路 SHALL 先由非 ReAct 的统一搜索编排器执行
- **AND** `ReactAgentOrchestrator` SHALL 仅在补救条件满足时作为 fallback 执行器运行

#### Scenario: React orchestrator remains constructible for explicit use
- **WHEN** 调用方显式构造或测试 `ReactAgentOrchestrator`
- **THEN** 系统 SHALL 继续提供与现有编排器兼容的 `answer()` 接口
- **AND** 该能力 SHALL 不要求其继续作为长期并列的默认顶层模式暴露

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

#### Scenario: Fallback invocation carries unified evidence metadata
- **WHEN** 默认 pipeline 已经生成 unified evidence metadata
- **THEN** ReactAgentOrchestrator SHALL 支持接收 `evidence_items`、`evidence_sources_active`、`evidence_sources_used` 或对应来源类型信息
- **AND** fallback 响应 SHALL 保留这些元数据语义

### Requirement: ReactAgentOrchestrator SHALL return fallback-compatible metadata
系统 SHALL 在 ReAct 作为 fallback 返回结果时输出与默认主链路兼容的元数据。

#### Scenario: Successful fallback response
- **WHEN** ReactAgentOrchestrator 产出用于替代首答的最终结果
- **THEN** 返回结构 SHALL 包含 `answer`、`control` 和 `search_hits`
- **AND** `control` SHALL 标明执行来源为 fallback ReAct
- **AND** `control` SHALL 能暴露 fallback 过程中沿用的 evidence source metadata

#### Scenario: Fallback execution ends without full resolution
- **WHEN** ReAct fallback 达到迭代上限或未能完全补救首答问题
- **THEN** 返回结构 SHALL 保留失败或截断信息
- **AND** `control` SHALL 包含与 fallback 执行相关的原因说明
