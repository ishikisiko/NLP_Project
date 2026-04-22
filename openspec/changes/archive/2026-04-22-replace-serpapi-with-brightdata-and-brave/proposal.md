## Why

当前项目的通用网页搜索仍依赖 SerpAPI，且缺少对 Brave Search 配额与调用频率的统一治理。现在需要将通用搜索主路径切换到更可控的 Brave Search，同时用 Bright Data 的 SERP 代理替代原有 SerpAPI 来源，并为受月度配额限制的 Brave key 建立后端记录与备援机制。

## What Changes

- **BREAKING** 移除项目中的 SerpAPI 搜索来源与对应配置入口，不再将 `serp` 作为可用搜索源。
- 新增 Bright Data SERP 搜索来源，使用 Bright Data request API 发起 Google SERP 请求，并将结果标准化为现有 `SearchHit` 结构。
- 新增 Brave Search 搜索来源，支持主 key 与备选 key 配置、自动切换、以及明确的 `RPS=1` 约束。
- 为 Brave Search 增加后端调用记录，至少记录每次调用所用 key 槽位、请求时间、请求结果与累计用量，以便控制每月 2000 次配额。
- 将当前 general web search 的默认首选来源切换为 Brave 主 key；当 Brave 主 key 不可用时，按设计回退到备选 Brave key，再按配置决定是否继续回退到其他来源。
- 更新前端与接口暴露的搜索源元数据，使搜索源展示与实际可用来源保持一致。

## Capabilities

### New Capabilities
- `web-search-provider-routing`: 管理通用网页搜索来源的注册、选择、优先级和默认路由，包括 Bright Data SERP 与 Brave Search。
- `brave-search-quota-tracking`: 管理 Brave Search 的 RPS 限制、调用审计、主备 key 选择与配额可见性。

### Modified Capabilities
- （无，仓库当前没有已归档的现有 capability specs）

## Impact

- 受影响代码：`search/search.py`、`main.py`、`server.py`、`langchain/langchain_tools.py`、编排器中的搜索源元数据汇总逻辑。
- 配置变更：移除 `SERPAPI_API_KEY`，新增 Bright Data 与 Brave Search 相关配置项，以及 Brave 主备 key 与配额记录配置。
- 运行时影响：通用网页搜索默认策略、搜索源展示、搜索限流与失败回退逻辑都会变化。
- 运维影响：需要保护新的 Bright Data / Brave 凭据，并监控 Brave 月配额消耗。
