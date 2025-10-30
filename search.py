from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchClient:
    """Abstract base class for search providers."""

    def search(self, query: str, num_results: int = 5) -> List[SearchHit]:
        raise NotImplementedError


class SerpAPISearchClient(SearchClient):
    """Simple wrapper around the SerpAPI search endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://serpapi.com/search.json",
        timeout: int = 15,
        engine: str = "google",
    ) -> None:
        if not api_key:
            raise ValueError("SerpAPI API key is required.")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.engine = engine

    def search(self, query: str, num_results: int = 5) -> List[SearchHit]:
        params = {
            "engine": self.engine,
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"SerpAPI search failed: {exc}") from exc

        payload = response.json()
        organic_results = payload.get("organic_results") or []

        hits: List[SearchHit] = []
        for entry in organic_results[:num_results]:
            title = entry.get("title") or ""
            link = entry.get("link") or entry.get("url") or ""
            snippet = entry.get("snippet") or entry.get("snippet_highlighted_words") or ""
            if isinstance(snippet, list):
                snippet = " ".join(snippet)

            if title or link or snippet:
                hits.append(
                    SearchHit(
                        title=title.strip(),
                        url=link.strip(),
                        snippet=snippet.strip(),
                    )
                )
        return hits


class FallbackSearchClient(SearchClient):
    """Fallback client for environments without a search API key."""

    def __init__(self, static_results: Optional[List[SearchHit]] = None) -> None:
        self.static_results = static_results or []

    def search(self, query: str, num_results: int = 5) -> List[SearchHit]:
        if not self.static_results:
            raise RuntimeError("No search backend configured. Provide a search API key or static results.")
        return self.static_results[:num_results]
