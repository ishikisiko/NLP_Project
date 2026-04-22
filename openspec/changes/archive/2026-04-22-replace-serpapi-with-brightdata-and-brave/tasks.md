## 1. Search provider migration

- [x] 1.1 Remove SerpAPI client code, SerpAPI config handling, and legacy `serp` source tokens from backend validation and search client assembly paths.
- [x] 1.2 Add a Bright Data SERP search client that calls the Bright Data request API and normalizes accepted results into `SearchHit(title, url, snippet)`.
- [x] 1.3 Register Bright Data as a first-class general web search source in `build_search_client`, LangChain search tool factories, and search source metadata reporting.

## 2. Brave Search integration

- [x] 2.1 Add Brave Search configuration support with explicit `primary` and `secondary` key slots and provider metadata for general web search.
- [x] 2.2 Implement a Brave Search client that enforces `RPS=1` per key slot and converts Brave responses into the existing `SearchHit` structure.
- [x] 2.3 Implement deterministic fallback from Brave primary to Brave secondary, then to lower-priority general web search providers when Brave cannot serve the request.

## 3. Backend quota tracking and observability

- [x] 3.1 Add backend-managed Brave usage recording that persists request timestamp, key slot, outcome, and fallback context across process restart.
- [x] 3.2 Surface updated search source metadata and Brave-related timing/control details so the active/configured sources reflect the real provider set after migration.
- [x] 3.3 Ensure the Brave usage log location is treated as runtime data and excluded from source control/documented as operational state.

## 4. Default routing, UI, and verification

- [x] 4.1 Update general web search routing so default web search attempts Brave primary first when no explicit source override is provided.
- [x] 4.2 Update frontend source selection and API validation so SerpAPI is no longer offered and Bright Data / Brave are represented consistently.
- [x] 4.3 Add or update regression coverage for provider selection, Brave fallback, quota logging, and source metadata behavior, then document the new configuration and migration notes.
