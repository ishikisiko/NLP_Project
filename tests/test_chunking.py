from __future__ import annotations

import pytest

from tests.local_chunk_grid_search import first_gold_rank, parse_int_list, source_matches_gold
from utils.chunking import resolve_chunk_settings


def test_resolve_chunk_settings_prefers_overrides():
    size, overlap = resolve_chunk_settings(
        {"localRag": {"chunk_size": 900, "chunk_overlap": 150}},
        chunk_size=1200,
        chunk_overlap=240,
    )

    assert size == 1200
    assert overlap == 240


def test_resolve_chunk_settings_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        resolve_chunk_settings({"localRag": {"chunk_size": 300, "chunk_overlap": 400}})


def test_parse_int_list_rejects_negative_values():
    with pytest.raises(ValueError, match="negative"):
        parse_int_list("100,-20", "chunk_overlaps")


def test_source_matches_gold_accepts_absolute_source_paths():
    assert source_matches_gold(
        "/root/code/NLP_Project/uploads/System_Architecture.md",
        "System_Architecture.md",
    )


def test_first_gold_rank_returns_first_match():
    rank = first_gold_rank(
        [
            "/tmp/a.md",
            "/tmp/System_Architecture.md",
            "/tmp/b.md",
        ],
        "System_Architecture.md",
    )

    assert rank == 2
