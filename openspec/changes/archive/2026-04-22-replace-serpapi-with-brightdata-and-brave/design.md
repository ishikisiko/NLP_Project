## Context

当前通用网页搜索由 `build_search_client()` 和 `create_search_tool_from_config()` 统一装配，搜索结果最终被标准化为 `SearchHit(title, url, snippet)` 并进入 RAG 与前端展示链路。现状中仍保留了 SerpAPI 客户端和 `serp` 搜索源元数据，但项目已经同时具备多搜索源并发聚合、源级 timing、搜索源显式选择等机制，适合在现有抽象上替换具体 provider。

这次变更是跨模块的搜索 provider 迁移，涉及：
- 搜索客户端实现与装配逻辑
- 配置模型与前后端搜索源元数据
- 通用网页搜索默认路由
- Brave Search 的速率限制、配额记录、主备 key 切换

项目当前没有数据库或现成的后端审计存储，因此任何 Brave 用量记录都需要建立在现有 Flask/Python 运行形态上，且应尽量减少对当前部署方式的要求。

## Goals / Non-Goals

**Goals:**
- 用 Bright Data SERP provider 替代当前 SerpAPI provider，保持统一的 `SearchHit` 输出契约。
- 引入 Brave Search provider，支持主 key、备选 key、失败切换和 `RPS=1` 约束。
- 让 general web search 默认优先走 Brave 主 key。
- 在后端记录 Brave 调用，能够为后续人工或程序化统计月度 2000 次配额消耗提供基础数据。
- 保持现有编排器、RAG、前端搜索源展示的接口形状尽量稳定。

**Non-Goals:**
- 不在本次设计中引入数据库、消息队列或集中式限流中间件。
- 不改变 `SearchHit` 的公共结构。
- 不改变天气、金融、体育等 domain API 的行为。
- 不在本次变更中解决 MCP 搜索源接入问题。

## Decisions

### Decision 1: 以 provider replacement 的方式移除 SerpAPI，而不是保留兼容别名

**选择**：删除 `SerpAPISearchClient` 和 `SERPAPI_API_KEY` 配置入口；新增独立的 Bright Data 搜索客户端与配置段，前后端搜索源标识也同步移除 `serp`。

**原因**：
- 用户要求是“去掉所有 Serpapi 这个来源”，保留兼容别名会让显示层、配置层和实际行为继续不一致。
- Bright Data 的调用方式、鉴权方式、结果解析方式与 SerpAPI 不同，复用旧命名会掩盖新的故障模式和限流策略。

**替代方案**：
- 复用 `serp` 作为旧 token，把 Bright Data 挂在旧 token 背后。
  - 未选用，因为这会让接口参数、前端展示和后端日志继续携带误导性的 SerpAPI 命名。

### Decision 2: Bright Data SERP 搜索客户端继续对齐 `SearchClient -> SearchHit` 抽象

**选择**：新增 Bright Data provider 客户端，内部负责调用 `https://api.brightdata.com/request`，并从返回的 raw SERP HTML/文本中抽取标题、URL、摘要，输出现有 `SearchHit` 列表。

**原因**：
- 现有 RAG、重排、前端展示都依赖 `SearchHit` 标准结构，保持这一层稳定可以把变更局限在 provider 边界。
- 这样 `CombinedSearchClient`、timing 收集和现有 orchestrator 不需要重新设计。

**替代方案**：
- 让 Bright Data 直接返回 provider-specific 原始结果并修改上层消费。
  - 未选用，因为会扩大变更面，且不利于以后继续新增 provider。

### Decision 3: Brave Search 作为 general web search 默认首选，并采用顺序回退而不是并发聚合

**选择**：对 general web search 的默认路径采用有序优先级：
1. Brave primary
2. Brave secondary
3. 其他仍保留的通用网页搜索 provider（如 Bright Data、Google Custom Search、You.com，按最终配置决定）

**原因**：
- Brave 有明确月配额与 `RPS=1` 约束，需要优先控制请求流向和消耗，不适合与其他 provider 一起默认并发 fan-out。
- 用户明确要求“让目前的 General web search 先默认走第一个 Brave search”。

**替代方案**：
- 继续沿用并发多源聚合，把 Brave 仅作为其中一个源。
  - 未选用，因为这会让 Brave 的月配额消耗不可控，也无法体现默认首选策略。

### Decision 4: Brave 的 RPS 限制在应用进程内强制执行，并把每次调用记录为后端审计事件

**选择**：
- 在 Brave provider 内部实现每个 key 槽位 `RPS=1` 的节流。
- 每次 Brave 请求都写入结构化后端记录，至少包含：时间戳、key 槽位（primary/secondary）、查询摘要、结果状态、HTTP 状态、是否命中回退、返回结果数。
- 记录格式采用本地 JSONL 审计日志，路径由配置指定并提供默认值。

**原因**：
- 项目当前没有数据库，JSONL 是对现有部署侵入最小、可直接落地的持久化方式。
- 单进程 Flask 场景下，进程内节流已足够满足当前 `RPS=1` 要求。
- 审计事件比“只维护一个数字计数器”更适合后续排查额度消耗、错误激增和 key 切换。

**替代方案**：
- 只在内存里累计计数，不落盘。
  - 未选用，因为服务重启后用量信息会丢失，不满足“要做后端记录”。
- 引入 SQLite/Redis。
  - 未选用，因为对当前项目来说过重，超出本次变更目标。

### Decision 5: Brave 主备 key 以槽位配置建模，而不是把多个 key 混在一个列表里盲轮询

**选择**：配置层显式区分 `primary` 和 `secondary` 两个 Brave key 槽位，并记录当前请求实际使用的槽位。

**原因**：
- 用户需求明确指定第一个 key 为默认、第二个为备选。
- 槽位化建模能让日志、配额统计、运维告警和回退行为更清晰。

**替代方案**：
- 用数组维护多个 key 并自动轮询。
  - 未选用，因为会弱化“默认首选”和“备选”的语义，也不利于月度用量归因。

### Decision 6: 搜索源元数据与 UI 展示必须只暴露真实可用来源

**选择**：后端 `search_sources_requested / active / configured / missing` 和前端搜索源选项都同步更新为新的 provider 集合，不再显示 SerpAPI。

**原因**：
- 当前项目已经把搜索源元数据回传给前端和 timing 视图；如果只换后端实现，不改这些元数据，用户看到的来源会与真实请求流向不一致。

**替代方案**：
- 只修改 provider 实现，不调整前端/控制元数据。
  - 未选用，因为这会制造新的可观测性偏差。

## Risks / Trade-offs

- [Bright Data 返回 raw 内容，解析 HTML/文本稳定性不如 JSON 搜索 API] → 先限定抽取最小必要字段，并为解析失败保留 provider 级错误与空结果回退路径。
- [JSONL 审计日志在多进程部署下不能提供强一致计数] → 本次设计默认单进程部署有效；若后续需要多实例部署，再升级为集中式存储。
- [Brave 设为默认首选后，配额可能被快速消耗] → 强制 `RPS=1`、记录每次调用、并为主备 key 分开统计。
- [删除 SerpAPI 会让现有请求参数或前端缓存中的 `serp` 失效] → 在接口校验与前端展示中同步移除旧 token，并在变更说明中标记为 breaking。
- [顺序回退会增加异常情况下的响应延迟] → 仅在 Brave 失败、限流或配额不可用时触发后续 provider，优先保证主路径配额可控。

## Migration Plan

1. 新增 Bright Data 与 Brave provider 配置结构，并从示例配置、文档、接口校验中移除 SerpAPI。
2. 实现 Bright Data provider 与 Brave provider，接入现有 `SearchClient` 抽象。
3. 在搜索客户端装配逻辑中更新 provider 注册、默认优先级和搜索源元数据。
4. 增加 Brave 后端审计日志与 `RPS=1` 约束。
5. 更新前端搜索源选项、返回的 control/timing 元数据以及批量测试入口。
6. 验证 general web search 默认走 Brave primary，失败时按设计回退。

**Rollback**：
- 保留 Google/You 等现有 provider 代码路径不变；
- 若 Brave 或 Bright Data 集成不稳定，可临时将默认 general web search 重新指向其他现有 provider；
- 回滚时不恢复 SerpAPI 命名，避免再次引入双重语义。

## Open Questions

1. Bright Data raw 返回内容的解析策略是否优先使用 HTML 解析，还是优先尝试 Bright Data 自带结构化字段（如果运行时可获得）？
2. Brave 月配额是否需要在应用内实现“接近 2000 次时主动停用 primary key”的硬阈值控制，还是先只做记录与人工监控？
3. Brave 审计日志的默认落盘路径是否需要纳入 `.gitignore` 并提供轮转策略？
