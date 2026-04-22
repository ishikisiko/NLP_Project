from __future__ import annotations

from typing import Any, Dict, Tuple


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def _coerce_non_negative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field}' must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"'{field}' must be greater than or equal to 0.")
    return parsed


def resolve_chunk_settings(
    config: Dict[str, Any] | None,
    *,
    chunk_size: Any = None,
    chunk_overlap: Any = None,
) -> Tuple[int, int]:
    """Resolve chunk settings from config and optional runtime overrides."""

    settings = {}
    if isinstance(config, dict):
        settings = config.get("localRag") or config.get("local_rag") or {}
        if not isinstance(settings, dict):
            settings = {}

    resolved_chunk_size = (
        chunk_size
        if chunk_size is not None
        else settings.get("chunk_size", DEFAULT_CHUNK_SIZE)
    )
    resolved_chunk_overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else settings.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    )

    size = _coerce_non_negative_int(resolved_chunk_size, "chunk_size")
    overlap = _coerce_non_negative_int(resolved_chunk_overlap, "chunk_overlap")

    if size <= 0:
        raise ValueError("'chunk_size' must be greater than 0.")
    if overlap >= size:
        raise ValueError("'chunk_overlap' must be smaller than 'chunk_size'.")

    return size, overlap
