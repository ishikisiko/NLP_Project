## 1. Post-check Core

- [x] 1.1 在默认 LangChain 搜索路径中抽出统一的 post-check 阶段，并明确只对搜索首答执行该阶段
- [x] 1.2 实现规则筛选器，覆盖时间约束缺失、比较对象缺失、未支撑具体数字、证据不足和明显多跳未完成等硬信号
- [x] 1.3 定义结构化 post-check verdict 数据结构，并统一写入返回结果的 `control` 元数据
- [x] 1.4 接入结构化 LLM judge，仅在规则层命中高风险或需进一步审查时调用
- [x] 1.5 实现 recoverable / non-recoverable failure 分类，确保并非所有 post-check 失败都会触发 fallback

## 2. ReAct Fallback Integration

- [x] 2.1 扩展 `ReactAgentOrchestrator` 以支持接收 fallback 上下文，包括原始 query、首答摘要、failure types 和可用证据摘要
- [x] 2.2 为 ReAct 增加高层搜索恢复工具，使 fallback 可复用现有 SearchRAG、rerank、时间约束和本地文档整合能力
- [x] 2.3 保留并校准领域 API 与本地文档工具，使其能在 fallback 场景中与高层搜索工具组合使用
- [x] 2.4 在默认 pipeline 中接入 ReAct fallback 调用，只对 recoverable failure 执行升级
- [x] 2.5 统一 fallback 返回结构，补充 `fallback_triggered`、`fallback_reason` 和 `final_executor` 等元数据

## 3. Configuration And Observability

- [x] 3.1 在配置中新增 post-check 开关、judge 配置和 ReAct fallback 迭代上限
- [x] 3.2 在 `main.py` 和 `server.py` 中接入新配置，保证 CLI 和 Web API 使用一致的 post-check/fallback 行为
- [x] 3.3 为 post-check 和记分结果补充日志与 timing/metadata，便于评测和调参
- [x] 3.4 确保在跳过 post-check、judge 失败或 fallback 不触发时也能输出清晰的控制信息

## 4. Validation

- [x] 4.1 为规则筛选器与 LLM judge verdict 解析补充单元测试
- [x] 4.2 为 recoverable / non-recoverable failure 分流补充集成测试
- [x] 4.3 为 ReAct fallback 上下文传递、高层搜索工具和返回元数据兼容性补充测试
- [x] 4.4 通过 CLI 和 Flask `/api/answer` 做手工验证，覆盖简单 query、比较型 query、时间约束 query 和应跳过 fallback 的失败场景
