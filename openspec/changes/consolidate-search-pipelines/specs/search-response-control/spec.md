## ADDED Requirements

### Requirement: Response control metadata SHALL use normalized search mode semantics
系统 SHALL 在默认主链路及其回退路径中统一 `control.search_mode`、`control.final_executor` 和 fallback 标记的语义。

#### Scenario: Non-search path returns normalized control metadata
- **WHEN** 查询走 direct answer、small talk、local-only 或 domain API 直出路径
- **THEN** 返回结果 SHALL 包含规范化的 `control.search_mode`
- **AND** `control.final_executor` SHALL 与实际执行路径一致

#### Scenario: Search path returns normalized control metadata
- **WHEN** 查询走搜索增强主链路并返回结果
- **THEN** 返回结果 SHALL 包含规范化的 `control.search_mode=search` 或其统一等价值
- **AND** 搜索是否执行、搜索来源元数据和 post-check 元数据 SHALL 以一致字段暴露

### Requirement: Response control metadata SHALL preserve compatibility fields during migration
系统 SHALL 在统一控制语义的同时保留现有主要响应字段，以避免 CLI、API 和前端在迁移期立即断裂。

#### Scenario: Existing callers parse response during migration
- **WHEN** 现有调用方继续读取 `answer`、`search_hits` 和 `control`
- **THEN** 系统 SHALL 继续返回这些字段
- **AND** 统一后的控制语义 SHALL 在兼容这些字段名的前提下提供更稳定的一致行为

#### Scenario: Fallback path returns compatibility metadata
- **WHEN** 查询经过 search unavailable 或 ReAct fallback 等回退路径
- **THEN** 系统 SHALL 保留与当前返回结构兼容的主要字段
- **AND** 系统 SHALL 明确暴露 fallback 是否触发、触发原因和最终执行者
