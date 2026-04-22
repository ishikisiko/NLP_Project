## Context

当前默认查询路径由 `LangChainOrchestrator.answer()` 驱动，流程是时间约束处理、small talk/图片分流、领域 API 尝试、`needs_search` 决策、关键词生成、`SearchRAG` 检索与重排、可选本地文档检索，最后生成首答。该路径对简单问题效率较高，但一旦首答遗漏关键约束、证据不足或需要多步补救，流程会直接结束，没有统一的后验检查和升级机制。

仓库中已经存在 `ReactAgentOrchestrator` 和基础 ReAct 工具，但其默认能力仍偏底层：`web_search` 只暴露基础搜索，缺少对当前成熟 SearchRAG 能力的复用；同时 ReAct 也没有被集成为默认 pipeline 的补救层。因此这次变更是一个跨模块的编排调整，而不是单点功能补丁。

约束包括：
- 默认 pipeline 仍应保持主路径，不能把所有请求都切到 ReAct。
- post-check 必须可解释、可观测，避免“黑盒觉得不好就升级”。
- ReAct fallback 只应用在适合多步工具补救的失败类型上，避免徒增成本与延迟。
- 返回结构必须继续兼容现有 `answer` / `control` / `search_hits` 约定。

## Goals / Non-Goals

**Goals:**
- 为默认搜索 pipeline 增加一层受控的 post-check，统一检查首答是否满足 query 的最低完成条件。
- 用“规则筛选 + 结构化 LLM judge”两级判定决定是否需要升级，而不是单纯依赖主观质量打分。
- 让 ReAct 作为默认 pipeline 的补救执行器，仅在合适的失败类型出现时介入。
- 让 ReAct fallback 复用现有 SearchRAG/领域数据/本地文档能力，避免降级成简单搜索工具。
- 在 `control` 元数据中沉淀 post-check verdict、失败类型、fallback 触发原因和最终执行路径。

**Non-Goals:**
- 不把默认主路径整体替换为 agent-first 架构。
- 不在本次变更中构建完整的离线评测框架或 LLM-as-a-judge 训练集。
- 不要求 ReAct 覆盖图片理解、流式输出或所有实验性工具形态。
- 不把所有低质量回答都自动重试；仅处理明确适合多步补救的情况。

## Decisions

### 1. 默认 pipeline 保持主路径，ReAct 仅作为 post-check 失败后的升级通道

默认路径已有成熟的时间约束处理、领域 API、关键词生成、SearchRAG 和 rerank。直接改成 ReAct 为主会显著增加时延、成本和行为不确定性。将 ReAct 作为补救层可以在保留简单 query 性能的同时，提高复杂 query 的恢复能力。

备选方案：
- 直接切成默认 ReAct：能力更统一，但对简单 query 过重，且当前 ReAct 工具能力仍弱于主路径。
- 完全不引入 ReAct，只做 answer re-generation：能修复措辞问题，但无法补足多步检索和工具调用。

### 2. post-check 采用“两级门控”：规则筛选优先，LLM judge 只处理候选失败样本

规则层负责识别硬信号，例如时间约束未覆盖、比较对象缺失、答案包含未支撑数字、回答极短且未使用已有证据、多跳任务被单步直答等。只有规则层判定为“高风险”或“需要进一步审查”时，才进入结构化 LLM judge。

这样设计的原因：
- 规则便宜、可解释，适合筛掉明显失败。
- LLM judge 更擅长判断“是否满足需求”和“是否适合多步补救”，但成本更高，也更容易漂移。
- 两级门控比“全部交给 LLM judge”更稳。

备选方案：
- 仅规则：可解释，但难以覆盖复杂语义缺失。
- 仅 LLM judge：实现简单，但成本和误判风险都更高。

### 3. LLM judge 输出结构化 verdict，而不是自由文本评论

judge 输出应至少包含：
- `passes_postcheck`
- `failure_types`
- `missing_constraints`
- `evidence_sufficiency`
- `should_fallback_to_react`
- `reason`

结构化输出能直接驱动编排决策，并沉淀为 `control.postcheck` 元数据。自由文本结论不利于自动化分流，也不利于后续评测和调参。

### 4. fallback 只在“适合多步工具补救”的失败类型上触发

不是所有 post-check 失败都值得进 ReAct。设计上应限制为：
- 约束覆盖不完整，但已有部分线索可继续扩展。
- 证据不足，需要补检索、交叉验证或多源综合。
- query 本质上是多跳/比较/综合型任务。

以下情况不应默认升级：
- 搜索源或领域 API 明确无数据。
- 首答已经正确承认信息不足。
- 明显是配置错误、模型错误或搜索系统不可用。

这样可以避免把“不可恢复失败”也送入 ReAct，造成额外成本却无收益。

### 5. ReAct fallback 复用高层搜索能力，而不是只调用基础 `web_search`

当前 `web_search` 工具只做基础 `search_client.search()`，无法复用关键词生成、rerank、时间约束、SearchRAG prompt 组装和本地文档融合。为了让 fallback 真正比首答更强，ReAct 应新增高层工具包装，例如 `search_rag_answer` 或等价能力，使 agent 能直接利用成熟主链路能力。

备选方案：
- 仅用现有 `web_search/domain_api/local_docs`：改动小，但 fallback 容易退化。
- 让 ReAct 直接调用整个 orchestrator：边界过大，可能形成递归或职责混乱。

### 6. ReAct 编排器需要接收 fallback 上下文

作为补救执行器时，ReAct 不应从零开始。它需要至少看到：
- 原始 query
- 首答摘要
- post-check failure types
- 可用证据摘要（search hits / domain data / local docs）
- 明确的补救目标，例如“补齐缺失约束”或“重新综合并校验证据”

这样能减少 agent 无效探索，并让 fallback 更可控。

### 7. 统一观测和返回结构

无论最终由默认首答还是 ReAct fallback 返回，都需要保持兼容结构，并在 `control` 中新增：
- `postcheck`
- `fallback_triggered`
- `fallback_reason`
- `final_executor`（如 `default_pipeline` / `react_fallback`）

必要时还可增加 `agent_steps_summary`，但不要求暴露完整思维链。

## Risks / Trade-offs

- [额外延迟与成本] → 先用规则层拦截，仅对候选失败样本启用 judge；ReAct 仅对适合补救的失败类型触发。
- [judge 误判导致过度升级] → 使用结构化 failure type，并加入“不升级”的硬边界；保留配置开关和阈值。
- [fallback 后能力反而下降] → 为 ReAct 提供高层 SearchRAG 能力，而不是只暴露基础 `web_search`。
- [编排逻辑变复杂] → 将 post-check 设计为清晰的单独阶段，统一记录 verdict 和升级原因，避免把逻辑散落在多个分支里。
- [结果结构不一致] → 统一在 orchestrator 层组装 `control` 和观测元数据，避免由各工具各自返回不同形态。

## Migration Plan

1. 先在默认 pipeline 中加入可关闭的规则层 post-check，并仅写入 verdict 元数据，不触发 fallback。
2. 接入结构化 LLM judge，在观测结果稳定后启用“只记录是否建议 fallback”模式。
3. 为 ReAct 增加高层搜索能力和 fallback 上下文接入。
4. 最后开启受控的 ReAct fallback，并通过配置控制启用范围和最大迭代次数。
5. 若线上表现不稳定，可通过配置关闭 judge 或关闭 fallback，保留规则层诊断信息。

## Open Questions

- post-check 的规则命中阈值是否需要按 query 类型区分，例如 finance / ranking / general search 使用不同标准。
- ReAct fallback 是否需要单独的模型或更低温度，以控制多步执行稳定性。
- 高层 SearchRAG 工具应返回“最终答案文本”还是“结构化证据包 + answer”，这会影响 agent 如何继续迭代。
