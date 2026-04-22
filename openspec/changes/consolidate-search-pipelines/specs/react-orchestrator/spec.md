## MODIFIED Requirements

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
