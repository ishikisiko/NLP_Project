# react-agent Specification

## Purpose
TBD - created by archiving change langchain-react-agent. Update Purpose after archive.
## Requirements
### Requirement: ReAct Agent 引擎
系统 SHALL 提供基于 LangChain ReAct Agent 的迭代推理引擎。该引擎接收用户查询，输出最终回答。

#### Scenario: 基础 ReAct 推理流程
- **WHEN** 用户提交查询且系统使用 ReAct Agent 模式
- **THEN** Agent 执行 Thought → Action → Observation 循环，最多迭代 max_iterations 次
- **AND** 每次迭代后检查是否满足终止条件（找到答案或达到上限）
- **AND** 返回最终答案

#### Scenario: 多工具迭代选择
- **WHEN** 复杂查询需要多个工具
- **THEN** Agent 在每次迭代中根据当前状态选择合适的工具
- **AND** Agent 考虑工具 description 和当前上下文

#### Scenario: 达到最大迭代次数
- **WHEN** Agent 达到 max_iterations 上限仍未生成最终答案
- **THEN** Agent SHALL 返回当前已有的答案（即使不完整）
- **AND** 在返回结果中标记 `truncated: true`

### Requirement: 自定义 ReAct Prompt
系统 SHALL 支持注入自定义 ReAct System Prompt，以支持中文推理场景。

#### Scenario: 使用默认英文 Prompt
- **WHEN** 系统未提供自定义 prompt
- **THEN** 使用 LangChain 默认 ReAct prompt

#### Scenario: 使用自定义中文 Prompt
- **WHEN** 调用 `create_react_agent` 时传入 `react_prompt` 参数
- **THEN** 使用传入的 prompt 替代默认 prompt

