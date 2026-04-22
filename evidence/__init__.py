"""Unified evidence retrieval and normalization primitives."""

from .source_layer import (
    DomainEvidenceSource,
    EvidenceItem,
    EvidenceSource,
    EvidenceSourceType,
    LocalEvidenceSource,
    RetrievalOptions,
    WebEvidenceSource,
    build_evidence_summary,
    describe_used_sources,
    evidence_items_to_documents,
    evidence_items_to_search_hits,
    normalize_reference_label,
    source_identity_label,
)

__all__ = [
    "DomainEvidenceSource",
    "EvidenceItem",
    "EvidenceSource",
    "EvidenceSourceType",
    "LocalEvidenceSource",
    "RetrievalOptions",
    "WebEvidenceSource",
    "build_evidence_summary",
    "describe_used_sources",
    "evidence_items_to_documents",
    "evidence_items_to_search_hits",
    "normalize_reference_label",
    "source_identity_label",
]
