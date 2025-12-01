from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class TimingRecorder:
    """Collects timing details for LLM calls and search sources."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._overall_start: Optional[float] = None
        self._total_recorded = False
        self._total_ms: Optional[float] = None
        self.llm_calls: List[Dict[str, Any]] = []
        self.search_sources: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []

    def start(self) -> None:
        if not self.enabled:
            return
        self._overall_start = time.perf_counter()
        self._total_recorded = False

    def stop(self) -> None:
        if not self.enabled or self._total_recorded:
            return
        if self._overall_start is None:
            return
        duration_ms = (time.perf_counter() - self._overall_start) * 1000
        self._total_ms = round(duration_ms, 2)
        self._overall_start = None
        self._total_recorded = True

    def record_llm_call(
        self,
        *,
        label: str,
        duration_ms: float,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        entry: Dict[str, Any] = {
            "label": label,
            "duration_ms": round(duration_ms, 2),
        }
        if provider:
            entry["provider"] = provider
        if model:
            entry["model"] = model
        if extra:
            entry.update(extra)
        self.llm_calls.append(entry)

    def record_tool_call(
        self,
        *,
        tool: str,
        duration_ms: float,
        success: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        entry: Dict[str, Any] = {
            "tool": tool,
            "duration_ms": round(duration_ms, 2),
            "success": success,
        }
        if extra:
            entry.update(extra)
        self.tool_calls.append(entry)

    def record_search_timing(
        self,
        *,
        source: Optional[str],
        label: Optional[str],
        duration_ms: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        entry: Dict[str, Any] = {
            "source": source,
            "label": label,
            "duration_ms": round(duration_ms, 2),
        }
        if extra:
            entry.update(extra)
        self.search_sources.append(entry)

    def extend_search_timings(self, timings: Optional[List[Dict[str, Any]]]) -> None:
        if not self.enabled or not timings:
            return
        for item in timings:
            if not isinstance(item, dict):
                continue
            raw_duration = item.get("duration_ms", 0.0)
            try:
                duration_value = float(raw_duration)
            except (TypeError, ValueError):
                duration_value = 0.0
            entry = {
                "source": item.get("source"),
                "label": item.get("label"),
                "duration_ms": round(duration_value, 2),
            }
            if item.get("error"):
                entry["error"] = item["error"]
            self.search_sources.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        payload: Dict[str, Any] = {}
        if self._overall_start is not None and not self._total_recorded:
            self.stop()
        if self._total_ms is not None:
            payload["total_ms"] = self._total_ms
        if self.search_sources:
            payload["search_sources"] = self.search_sources
        if self.llm_calls:
            payload["llm_calls"] = self.llm_calls
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload
