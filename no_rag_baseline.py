from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from api import HKGAIClient
from search import SearchClient, SearchHit
from rerank import BaseReranker


DEFAULT_SYSTEM_PROMPT = (
    "You are an information assistant. "
    "Answer user questions concisely using only the provided search results. "
    "When unsure, acknowledge the uncertainty."
)


class NoRAGBaseline:
    """Minimal pipeline that sends search snippets to the LLM without local retrieval."""

    def __init__(
        self,
        llm_client: HKGAIClient,
        search_client: SearchClient,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        *,
        reranker: Optional[BaseReranker] = None,
        min_rerank_score: float = 0.0,
        max_per_domain: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client
        self.system_prompt = system_prompt
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score
        self.max_per_domain = max(1, max_per_domain)

    def _format_search_hits(self, hits: List[SearchHit]) -> str:
        if not hits:
            return "No search results were returned."

        formatted_rows = []
        for idx, hit in enumerate(hits, start=1):
            snippet = hit.snippet or "No snippet available."
            url = hit.url or "No URL available."
            title = hit.title or f"Result {idx}"
            formatted_rows.append(
                f"{idx}. {title}\n"
                f"   URL: {url}\n"
                f"   Snippet: {snippet}"
            )
        return "\n".join(formatted_rows)

    def build_prompt(self, query: str, hits: List[SearchHit]) -> str:
        context_block = self._format_search_hits(hits)
        return (
            "You are given a set of search results. "
            "Use them to answer the question at the end. "
            "When citing sources, use the format (URL 1), (URL 2), etc., "
            "where the number corresponds to the search result number.\n\n"
            f"Search Results:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def answer(
        self,
        query: str,
        *,
        search_query: Optional[str] = None,
        num_search_results: int = 5,
        per_source_limit: Optional[int] = None,
        max_tokens: int = 5000,
        temperature: float = 0.3,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> Dict[str, object]:
        # Prefer keyword-focused query generated upstream when available.
        effective_query = search_query.strip() if search_query else query

        per_source_cap = per_source_limit if per_source_limit is not None else num_search_results
        hits = self.search_client.search(
            effective_query,
            num_results=num_search_results,
            per_source_limit=per_source_cap,
            freshness=freshness,
            date_restrict=date_restrict,
        )
        search_warnings: List[str] = []
        get_last_errors = getattr(self.search_client, "get_last_errors", None)
        if callable(get_last_errors):
            errors = get_last_errors() or []
            if hits and errors:
                for item in errors:
                    source = str(item.get("source") or "搜索服务")
                    detail = str(item.get("error") or "未知错误")
                    if source.lower().startswith("mcp"):
                        search_warnings.append(f"{source} 未正常工作，已使用其他搜索结果。原因：{detail}")
                    else:
                        search_warnings.append(f"{source} 出现异常：{detail}")
        hits, rerank_meta = self._apply_rerank(query, hits, limit=num_search_results)
        user_prompt = self.build_prompt(query, hits)
        response = self.llm_client.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Build answer with URL references
        answer = response.get("content")
        if answer and hits:
            # Append reference list
            answer += "\n\n**参考链接：**\n"
            for idx, hit in enumerate(hits, start=1):
                url = hit.url or "No URL available."
                title = hit.title or f"结果 {idx}"
                answer += f"{idx}. [{title}]({url})\n"

        result: Dict[str, object] = {
            "query": query,
            "answer": answer,
            "search_hits": [asdict(hit) for hit in hits],
            "llm_raw": response.get("raw"),
            "llm_warning": response.get("warning"),
            "llm_error": response.get("error"),
            "rerank": rerank_meta or None,
            "search_query": effective_query,
        }
        if search_warnings:
            result["search_warnings"] = search_warnings
        return result

    def _apply_rerank(
        self,
        query: str,
        hits: List[SearchHit],
        *,
        limit: Optional[int] = None,
    ) -> Tuple[List[SearchHit], List[Dict[str, object]]]:
        if not self.reranker or not hits:
            return hits, []

        try:
            reranked = self.reranker.rerank(query, hits)
        except Exception as exc:  # pragma: no cover - best effort resilience
            return hits, [{"error": str(exc)}]

        filtered: List[SearchHit] = []
        metadata: List[Dict[str, object]] = []
        domain_counts: Dict[str, int] = {}
        max_results = limit or len(reranked)

        for item in reranked:
            domain = self._extract_domain(item.hit.url)
            if domain and domain_counts.get(domain, 0) >= self.max_per_domain:
                metadata.append(
                    {
                        "url": item.hit.url,
                        "score": item.score,
                        "dropped": "per_domain_limit",
                    }
                )
                continue
            if item.score is not None and item.score < self.min_rerank_score:
                metadata.append(
                    {
                        "url": item.hit.url,
                        "score": item.score,
                        "dropped": "below_min_score",
                    }
                )
                continue

            filtered.append(item.hit)
            metadata.append(
                {
                    "url": item.hit.url,
                    "score": item.score,
                    "kept": True,
                }
            )

            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            if len(filtered) >= max_results:
                break

        if not filtered:
            return hits, metadata

        return filtered, metadata

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        if not url:
            return None
        return urlparse(url).netloc or None
