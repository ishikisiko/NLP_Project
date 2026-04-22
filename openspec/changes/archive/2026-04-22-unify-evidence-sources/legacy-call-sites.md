# Legacy Call Sites

Verified on 2026-04-22 with:

```bash
rg -n "LocalRAGChain|LocalRAG\\b|rag/local_rag.py|rag/search_rag.py|SearchRAG\\b" -g '!openspec/changes/archive/**' .
```

## Remaining Runtime Call Sites

- `orchestrators/smart_orchestrator.py`
  - Still imports and instantiates legacy `LocalRAG` and `SearchRAG`.
  - This file is the main remaining runtime surface preventing full retirement of `rag/local_rag.py` and `rag/search_rag.py`.
- `langchain/langchain_rag.py`
  - `LocalRAGChain` remains for backward compatibility and factory exports.
  - The default `SearchRAGChain` path now uses the unified evidence layer internally.
- `langchain/langchain_orchestrator.py`
  - Keeps a cached `_local_rag` / `_get_local_rag()` path, but the default execution path is `_get_primary_rag()` and the unified `SearchRAGChain`.
- `langchain/__init__.py`
  - Still re-exports `LocalRAGChain` for compatibility.
- `rag/__init__.py`
  - Still re-exports legacy `LocalRAG` and `SearchRAG`.

## Non-Runtime References

- `openspec/specs/*`, `openspec/changes/*`, `README.md`, reports, and architecture docs still mention the legacy classes.
- Dataset and report files still use `local_rag` / `SearchRAG` terminology.

## Retirement Scope

The next safe retirement boundary is:

1. Migrate or delete `orchestrators/smart_orchestrator.py`.
2. Remove compatibility exports that keep `LocalRAGChain`, `LocalRAG`, and `SearchRAG` publicly reachable.
3. Update repo docs/spec text that still describes the legacy runtime as active.

Until `smart_orchestrator.py` is retired, `rag/local_rag.py` and `rag/search_rag.py` should be treated as compatibility code rather than dead code.
