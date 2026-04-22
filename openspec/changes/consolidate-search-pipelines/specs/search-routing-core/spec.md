## ADDED Requirements

### Requirement: System SHALL use a unified routing core for default search handling
系统 SHALL 为默认搜索主链路提供统一的查询路由核心，用于处理 small talk 判定、时间约束解析、领域分类、搜索决策和关键词生成。

#### Scenario: Default CLI query uses shared routing core
- **WHEN** CLI 通过默认主编排器处理查询
- **THEN** 系统 SHALL 使用统一路由核心完成查询判定与搜索前决策
- **AND** 系统 SHALL NOT 依赖一套与 Web 端分离的 legacy 路由实现

#### Scenario: Default API query uses shared routing core
- **WHEN** Web API 通过默认主编排器处理查询
- **THEN** 系统 SHALL 使用与 CLI 默认主链路相同的统一路由核心
- **AND** small talk、direct answer、domain routing 和 search routing 的语义 SHALL 保持一致

### Requirement: Routing core SHALL produce deterministic routing outputs
统一路由核心 SHALL 输出结构化且可复用的路由结果，至少包括是否需要搜索、领域分类结果和关键词生成结果。

#### Scenario: Query does not require search
- **WHEN** 路由核心判定查询无需外部搜索
- **THEN** 系统 SHALL 返回结构化决策结果，明确 `needs_search=false`
- **AND** 该结果 SHALL 可直接驱动 direct answer 或 local-only 路径

#### Scenario: Query requires search
- **WHEN** 路由核心判定查询需要外部搜索
- **THEN** 系统 SHALL 返回结构化决策结果，明确 `needs_search=true`
- **AND** 系统 SHALL 提供可用于执行搜索的关键词或等价搜索查询信息
