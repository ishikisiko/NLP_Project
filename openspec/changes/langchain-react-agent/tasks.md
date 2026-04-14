## 1. 工具封装层

- [x] 1.1 创建 `langchain/langchain_react_tools.py`
- [x] 1.2 实现 `ReActSearchTool` 封装 `search_client.search()`
- [x] 1.3 实现 `ReActDomainTool` 封装 `IntelligentSourceSelector`
- [x] 1.4 实现 `ReActLocalDocTool` 封装 `LocalRAG`
- [x] 1.5 实现工具工厂函数 `create_react_tools_from_config()`

## 2. LangChainOrchestrator 工厂扩展

- [x] 2.1 在 `langchain/langchain_orchestrator.py` 添加 `create_react_agent()` 静态方法
- [x] 2.2 实现自定义 ReAct prompt 支持（中文优化）
- [x] 2.3 添加 `max_iterations` 参数支持

## 3. ReactAgentOrchestrator 实现

- [x] 3.1 创建 `orchestrators/react_agent_orchestrator.py`
- [x] 3.2 实现 `__init__` 初始化 ReAct Agent 和工具列表
- [x] 3.3 实现 `answer()` 接口方法
- [x] 3.4 实现返回结构兼容（`answer`、`control`、`search_hits`）
- [x] 3.5 实现 `ReactAgentOrchestrator.create_from_config()` 工厂方法

## 4. 配置集成与切换

- [x] 4.1 在 `config.example.json` 添加 `orchestrator_mode` 配置项
- [x] 4.2 在 `main.py` 中根据配置创建对应编排器
- [x] 4.3 在 `server.py` 中支持运行时切换编排器模式
- [x] 4.4 添加健康检查确认 Agent 正常初始化

## 5. 测试验证

- [ ] 5.1 单元测试：`ReActSearchTool`、`ReActDomainTool`、`ReActLocalDocTool`
- [ ] 5.2 集成测试：`ReactAgentOrchestrator.answer()` 完整流程
- [ ] 5.3 模式切换测试：确认配置切换后行为正确
