## ADDED Requirements

### Requirement: System SHALL expose a unified EvidenceSource interface for first-class evidence retrieval
系统 SHALL 为默认主链路中的一级证据来源提供统一的 `EvidenceSource` 接口，至少覆盖 `web`、`domain`、`local` 三类来源。

#### Scenario: Web search source is represented as an EvidenceSource
- **WHEN** 系统需要从网页搜索 provider 获取证据
- **THEN** 系统 SHALL 通过 `source_type=web` 的 `EvidenceSource` 执行检索
- **AND** 该来源 SHALL 暴露稳定的 `source_id` 和显示名称以供观测与调试

#### Scenario: Local document retrieval is represented as an EvidenceSource
- **WHEN** 系统需要从上传目录或本地文档目录检索证据
- **THEN** 系统 SHALL 通过 `source_type=local` 的 `EvidenceSource` 执行检索
- **AND** 系统 SHALL NOT 要求本地来源伪装为 `SearchClient` provider

#### Scenario: Domain API retrieval is represented as an EvidenceSource
- **WHEN** 查询需要 weather、finance、sports 或其他领域 API 提供的结构化证据
- **THEN** 系统 SHALL 通过 `source_type=domain` 的 `EvidenceSource` 获取证据
- **AND** 该来源 SHALL 能与 web/local 来源一起进入统一后处理流程

### Requirement: EvidenceSource retrieval SHALL be selectively enabled by the orchestrator
系统 SHALL 允许默认主编排器按查询条件启用或禁用特定 EvidenceSource，而不是通过切换完全不同的 pipeline 来改变来源集合。

#### Scenario: Search-disabled query enables local-only evidence retrieval
- **WHEN** 查询在 `allow_search=false` 条件下执行
- **THEN** 默认主编排器 SHALL 禁用 `web` 类型来源
- **AND** 默认主编排器 SHALL 继续允许 `local` 类型来源参与检索

#### Scenario: Domain-assisted query enables multiple source types
- **WHEN** 领域 API 返回需要继续搜索或继续补证据的结果
- **THEN** 默认主编排器 SHALL 能同时启用 `domain` 和其他来源类型
- **AND** 系统 SHALL 不要求领域数据只能旁路注入字符串上下文

### Requirement: Response metadata SHALL expose active evidence source semantics
系统 SHALL 在默认主链路与 fallback 路径中暴露统一的来源元数据语义，以说明哪些一级来源被启用和实际使用。

#### Scenario: Default pipeline reports active evidence source types
- **WHEN** 默认主链路完成一次回答
- **THEN** 返回结果 SHALL 能标识本次执行中启用的一级来源类型和来源标识
- **AND** 该元数据 SHALL 不局限于仅描述网页搜索 provider

#### Scenario: Fallback pipeline reports reused evidence source types
- **WHEN** ReAct fallback 或高层恢复工具复用统一来源层
- **THEN** 返回结果或工具输出 SHALL 能说明复用了哪些来源类型
- **AND** 该元数据 SHALL 与默认主链路使用兼容的来源语义
