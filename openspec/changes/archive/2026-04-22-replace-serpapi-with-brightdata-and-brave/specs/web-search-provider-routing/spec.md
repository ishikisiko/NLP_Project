## ADDED Requirements

### Requirement: Search source catalog excludes SerpAPI
The system SHALL remove SerpAPI from the configurable and user-visible general web search source catalog once this change is enabled.

#### Scenario: API rejects legacy SerpAPI source token
- **WHEN** a client submits `search_sources` containing the legacy SerpAPI source token
- **THEN** the backend SHALL reject the request as an unsupported search source

#### Scenario: Search source metadata no longer advertises SerpAPI
- **WHEN** the backend returns control or timing metadata describing configured or active search sources
- **THEN** the metadata SHALL NOT list SerpAPI as a configured, active, requested, or missing provider

### Requirement: Bright Data SERP acts as a first-class general web search provider
The system SHALL support Bright Data SERP as a general web search provider and normalize its search results into the existing `SearchHit` structure.

#### Scenario: Bright Data result normalization
- **WHEN** Bright Data returns general web search results for a query
- **THEN** the backend SHALL convert each accepted result into `title`, `url`, and `snippet` fields before passing the results to RAG, reranking, or response serialization

#### Scenario: Bright Data participates in provider metadata
- **WHEN** Bright Data is configured and available
- **THEN** the backend SHALL expose Bright Data in configured and active search source metadata using its own provider identity rather than the removed SerpAPI identity

### Requirement: General web search defaults to Brave primary
The system SHALL make Brave primary the default first-choice provider for general web search.

#### Scenario: Default general web search path
- **WHEN** a query requires general web search and the caller does not explicitly override the source set
- **THEN** the backend SHALL attempt Brave primary before any fallback general web search provider

#### Scenario: Explicit source selection constrains provider usage
- **WHEN** a caller explicitly provides a supported `search_sources` subset
- **THEN** the backend SHALL limit general web search provider selection to the supported members of that subset

### Requirement: General web search falls back in deterministic order
The system SHALL use a deterministic fallback order for general web search when the preferred provider is unavailable, rate-limited, or returns a provider-level failure.

#### Scenario: Brave secondary is used after primary failure
- **WHEN** Brave primary cannot serve a general web search request
- **THEN** the backend SHALL try the configured Brave secondary key before trying lower-priority general web search providers

#### Scenario: Non-Brave fallback only occurs after Brave paths are exhausted
- **WHEN** Brave primary and Brave secondary are both unavailable for a general web search request
- **THEN** the backend SHALL only then use the next configured fallback provider in the routing order
