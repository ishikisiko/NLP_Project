## MODIFIED Requirements

### Requirement: Default pipeline SHALL run post-check before returning search answers
系统 SHALL 在统一默认搜索主链路生成首答后执行统一的 post-check 阶段，再决定是直接返回首答还是升级到补救路径。

#### Scenario: Unified default pipeline search answer passes post-check
- **WHEN** 统一默认搜索主链路生成首答且 post-check 判定该回答满足需求
- **THEN** 系统 SHALL 直接返回首答
- **AND** 返回结果 SHALL 记录 post-check verdict 和最终执行路径为默认主链路

#### Scenario: Non-search paths can skip post-check
- **WHEN** 查询走 small talk、直接回答、纯领域 API 直出或其他未进入统一搜索主链路的路径
- **THEN** 系统 SHALL 支持跳过或短路 post-check
- **AND** 返回结果 SHALL 明确记录未执行完整 post-check 的原因

### Requirement: System SHALL trigger ReAct fallback only for recoverable failures
系统 SHALL 仅在统一默认搜索主链路的 post-check 失败且失败类型适合多步工具补救时触发 ReAct fallback。

#### Scenario: Recoverable unified-pipeline failure triggers ReAct fallback
- **WHEN** post-check 识别出统一默认搜索主链路的回答存在可恢复失败，如约束覆盖缺失、需要补证据或需要多跳综合
- **THEN** 系统 SHALL 调用 `ReactAgentOrchestrator` 作为 fallback-only 执行器
- **AND** 返回结果 SHALL 标记 fallback 已触发以及触发原因

#### Scenario: Non-recoverable failure does not trigger ReAct fallback
- **WHEN** post-check 失败原因是搜索不可用、外部 API 不可用、数据源缺失或首答已明确承认无足够信息
- **THEN** 系统 SHALL NOT 自动触发 ReAct fallback
- **AND** 系统 SHALL 返回首答或现有错误信息并保留 post-check verdict
