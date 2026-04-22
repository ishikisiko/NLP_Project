## ADDED Requirements

### Requirement: Default pipeline SHALL retrieve evidence through the unified EvidenceSource layer
系统 SHALL 在默认主链路中先通过统一来源层收集证据，再执行统一的过滤、重排和回答上下文构建，而不是分别处理 `search_hits`、`retrieved_docs` 和 `extra_context`。

#### Scenario: Search-enabled query uses multi-source evidence retrieval
- **WHEN** 查询允许联网搜索且存在可用来源
- **THEN** 默认主链路 SHALL 从已启用的 EvidenceSource 收集证据并归一化为 `EvidenceItem`
- **AND** 回答阶段 SHALL 使用统一后的证据集合，而不是并列维护多个独立上下文变量

#### Scenario: Local-only query uses the same fusion pipeline with fewer source types
- **WHEN** 查询在 `allow_search=false` 条件下执行
- **THEN** 默认主链路 SHALL 继续经过统一 evidence retrieval / fusion 流程
- **AND** 唯一差异 SHALL 是启用的来源集合减少，而不是切换为另一条独立主执行链

### Requirement: Unified evidence fusion SHALL support shared filtering, deduplication, and ranking semantics
系统 SHALL 对归一化后的 `EvidenceItem` 集合执行统一的过滤、去重和排序语义，以便不同来源的证据能在同一回答上下文中协同工作。

#### Scenario: Multi-source evidence is filtered and ranked before answer generation
- **WHEN** 查询同时返回网页、本地或领域证据
- **THEN** 系统 SHALL 在回答生成前对统一证据集合执行过滤、去重、排序或等价的后处理
- **AND** 回答上下文 SHALL 基于处理后的统一证据集合生成

#### Scenario: Temporal or specialized search behavior survives the source-layer refactor
- **WHEN** 查询属于时间变化、历史趋势或其他需要特殊补证据行为的场景
- **THEN** 系统 SHALL 继续支持额外证据收集和统一后处理
- **AND** 该能力 SHALL 在迁移到统一来源层后继续可用

### Requirement: Domain evidence SHALL support both direct answer and fusion modes
系统 SHALL 允许领域 API 证据在适合直出时继续直接回答，同时在需要补检索或补证据时进入统一 fusion pipeline。

#### Scenario: Domain API can short-circuit when no additional evidence is needed
- **WHEN** 领域 API 已返回足以直接回答且无需继续搜索的结果
- **THEN** 系统 SHALL 继续允许领域路径直接返回答案
- **AND** 系统 SHALL 不要求所有领域查询都必须进入统一 fusion pipeline

#### Scenario: Domain API enters fusion when additional evidence is needed
- **WHEN** 领域 API 返回需要继续搜索、继续补证据或与本地文档联合判断的结果
- **THEN** 系统 SHALL 将领域证据归一化为 `EvidenceItem` 并送入统一 fusion pipeline
- **AND** 最终回答 SHALL 能整合领域与非领域来源的证据
