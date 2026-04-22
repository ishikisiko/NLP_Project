## ADDED Requirements

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
