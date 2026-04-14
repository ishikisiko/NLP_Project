## Why

当前系统（SmartSearchOrchestrator / LangChainOrchestrator）是线性决策管道，每步只执行一次判断（路由决策 → 关键词生成 → 搜索 → RAG → 回答）。这种架构对简单查询足够，但无法处理需要**多步推理和迭代工具调用**的复杂查询。

ReAct（Reasoning + Acting）模式让 Agent 能够：
- 在每步推理中决定使用哪个工具
- 根据工具返回结果决定下一步行动
- 支持复杂任务的迭代分解

## What Changes

- 新增 **ReAct Agent 执行模式**：基于 LangChain ReAct Agent 实现迭代推理循环
- 封装现有工具为 LangChain Tool：`WebSearchTool`、`DomainApiTool`、`LocalDocTool`
- 新增 `ReactAgentOrchestrator`：使用 ReAct 循环的编排器，替代现有线性管道
- 保留现有 `SmartSearchOrchestrator` 和 `LangChainOrchestrator` 作为兼容模式
- 新增 ReAct 专用 System Prompt，支持中文推理痕迹展示

## Capabilities

### New Capabilities

- `react-agent`: 通用 ReAct Agent 引擎，支持多工具迭代推理
- `react-tool-wrapper`: 将现有工具（搜索、领域API、本地文档）封装为 LangChain Tool
- `react-orchestrator`: 基于 ReAct Agent 的新编排器，对外提供 `answer()` 接口

### Modified Capabilities

- （无，现有能力不变，只是新增执行模式）

## Impact

- **新增文件**: `orchestrators/react_agent_orchestrator.py`、`langchain/langchain_react_tools.py`
- **修改文件**: `langchain/langchain_orchestrator.py`（添加工厂方法创建 ReAct Agent）、`langchain/langchain_tools.py`（添加 ReAct 专用工具封装）
- **依赖**: LangChain `create_react_agent`、`AgentExecutor`
- **向后兼容**: 现有编排器保持不变，通过配置切换执行模式
