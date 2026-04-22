## Context

当前系统存在两个编排器：

1. **SmartSearchOrchestrator**（1300+ 行）：基于 `llm/api.py` LLMClient 的 legacy 实现
2. **LangChainOrchestrator**（1000+ 行）：基于 LangChain Chain 的 modern 实现

两者都是**线性管道**，执行模式为：

```
Query → [路由决策] → [关键词] → [搜索] → [RAG] → Answer
         (一次)      (一次)    (一次)    (一次)
```

这种架构的局限：
- 复杂查询无法迭代：例如"帮我比较苹果和微软近三年股价"需要多次搜索/获取数据
- 推理过程不可见：用户无法看到 Agent 如何逐步推理
- 工具调用不灵活：每步只能按固定顺序执行

**利益相关方**：希望用 LLM API 调用实现更智能问答的用户

## Goals / Non-Goals

**Goals:**
- 实现基于 LangChain ReAct Agent 的迭代推理能力
- 封装现有工具（WebSearch、DomainApi、LocalDoc）为 LangChain Tool
- 新增 `ReactAgentOrchestrator`，对外提供与现有编排器一致的 `answer()` 接口
- 支持多轮推理追踪（thoughts/actions/observations）

**Non-Goals:**
- 不替换现有 SmartSearchOrchestrator 和 LangChainOrchestrator（保持向后兼容）
- 不实现自定义 ReAct 循环（使用 LangChain 原生实现）
- 不新增基础工具（只封装现有工具）
- 不改变现有 API 接口

## Decisions

### Decision 1: 使用 LangChain ReAct Agent 而非自定义循环

**选择**：使用 `langchain.agents.create_react_agent`

**原因**：
- LangChain 提供完整生态：prompt 管理、tool binding、agent executor
- 支持 `max_iterations` 控制 token 消耗
- 社区广泛使用，文档完善

**替代方案**：
- 自定义 ReAct 循环：更灵活但实现成本高，需要处理边界情况

### Decision 2: 工具封装策略

**选择**：在 `langchain/langchain_tools.py` 中添加 ReAct 专用工具封装

```python
# 新增 ReActTool 封装
class ReActSearchTool(BaseTool):
    """封装 search_client.search() 为 LangChain Tool"""
    name = "web_search"
    description = "用于获取最新网络信息。输入搜索查询，返回搜索结果列表。"
    
    def _run(self, query: str) -> str:
        hits = self.search_client.search(query, num_results=5)
        return self._format_results(hits)

class ReActDomainTool(BaseTool):
    """封装 IntelligentSourceSelector 为 LangChain Tool"""
    name = "domain_api"
    description = "获取天气、金融、体育等专业领域数据"
    # ...

class ReActLocalDocTool(BaseTool):
    """封装 LocalRAG 为 LangChain Tool"""
    name = "local_docs"
    description = "查询本地知识库文档"
    # ...
```

**原因**：复用现有 `search_client`、`IntelligentSourceSelector`、`LocalRAG`，改动最小

### Decision 3: 编排器工厂模式

**选择**：在 `LangChainOrchestrator` 中添加 `create_react_agent()` 工厂方法

```python
# langchain/langchain_orchestrator.py 新增
@staticmethod
def create_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    max_iterations: int = 5,
) -> AgentExecutor:
    """创建 ReAct Agent"""
    agent = create_react_agent(llm, tools)
    return AgentExecutor.from_agent_and_tools(
        agent, tools, max_iterations=max_iterations, verbose=True
    )
```

**原因**：保持与现有 `create_langchain_orchestrator` 一致的工厂模式

### Decision 4: ReactAgentOrchestrator 位置

**选择**：`orchestrators/react_agent_orchestrator.py`

**原因**：
- 独立文件，与 `smart_orchestrator.py` 和 `langchain_orchestrator.py` 并列
- 体现这是替代执行模式而非修改现有实现

## Risks / Trade-offs

| 风险 | 影响 |  Mitigation |
|------|------|-------------|
| ReAct prompt 对中文支持不佳 | Agent 推理质量下降 | 使用中文优化的 ReAct prompt，或在 `create_react_agent` 时传入自定义 prompt |
| max_iterations 过低导致任务未完成 | 回答不完整 | 默认可配置，通过 `ReactAgentOrchestrator.__init__` 参数控制 |
| 现有工具封装后接口不匹配 | 工具调用失败 | 保持 `search_client.search()` 等现有接口不变，只做包装 |
| Token 消耗增加 | 成本上升 | 默认 `max_iterations=5`，超出强制结束 |

## Migration Plan

1. **Phase 1**: 实现 `langchain/langchain_react_tools.py`，封装现有工具为 ReAct Tool
2. **Phase 2**: 在 `LangChainOrchestrator` 中添加 `create_react_agent()` 工厂方法
3. **Phase 3**: 实现 `orchestrators/react_agent_orchestrator.py`
4. **Phase 4**: 在 `main.py` / `server.py` 中添加配置项切换执行模式
5. **回滚**：配置项切换回 `smart_orchestrator` 或 `langchain_orchestrator`

## Open Questions

1. **ReAct prompt 是否需要完全自定义？** LangChain 默认 ReAct prompt 是英文的，需要评估中文场景是否需要自定义
2. **max_iterations 默认值？** 需要在真实场景中测试后确定，建议初始值 5
3. **推理痕迹（thoughts）如何暴露给用户？** 当前设计只返回最终答案，是否需要暴露中间步骤？
