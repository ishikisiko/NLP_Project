## Why

当前默认搜索 pipeline 对简单和单跳问题效率较高，但缺少统一的结果后验检查层。对于多跳推理、约束覆盖不完整、证据不足或需要跨搜索/领域 API/本地文档综合的问题，系统往往只能返回“勉强可用”的首答，无法在发现不足后自动升级到更强的多步工具调用模式。

## What Changes

- 在默认 LangChain 搜索 pipeline 末尾新增 post-check 阶段，先用规则筛选识别明显失败或高风险回答。
- 为 post-check 增加结构化 LLM judge，对“是否满足需求、是否覆盖关键约束、是否证据充分、是否适合多步补救”进行判定。
- 当规则筛选和 LLM judge 判定需要升级时，将请求自动 fallback 到 ReAct 模式，而不是直接返回首答。
- 为 fallback 引入受控的升级条件、失败类型分类和可观测元数据，避免所有低质量回答都无差别进入 ReAct。
- 让 ReAct fallback 复用当前成熟的搜索能力，而不是退化为仅调用基础 `web_search` 工具。

## Capabilities

### New Capabilities
- `query-postcheck-fallback`: 定义默认搜索 pipeline 的后验检查、升级判定和 ReAct fallback 行为。

### Modified Capabilities
- `react-tool-wrapper`: 扩展 ReAct 工具封装，使 fallback 可复用现有 SearchRAG/领域数据/本地文档的高层能力，而不只依赖低级搜索工具。
- `react-orchestrator`: 扩展 ReAct 编排器以支持作为默认 pipeline 的补救执行器接收 fallback 上下文，并返回与主链路兼容的元数据。

## Impact

- 受影响代码：`server.py`、`main.py`、`langchain/langchain_orchestrator.py`、`orchestrators/react_agent_orchestrator.py`、`langchain/langchain_react_tools.py`、`rag/search_rag.py`。
- 运行时影响：默认请求路径会增加一个 post-check 判定阶段，部分复杂或高风险 query 会触发额外的 ReAct fallback 延迟与成本。
- 观测与返回结构：需要在 `control` 中新增 post-check verdict、fallback 原因、judge 结果与执行来源等元数据。
- 配置影响：需要新增 post-check 开关、LLM judge 配置以及 ReAct fallback 迭代上限等参数。
