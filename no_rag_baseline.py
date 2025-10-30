from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

from api import HKGAIClient
from search import SearchClient, SearchHit


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
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client
        self.system_prompt = system_prompt

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
        num_search_results: int = 5,
        max_tokens: int = 5000,
        temperature: float = 0.3,
    ) -> Dict[str, object]:
        hits = self.search_client.search(query, num_results=num_search_results)
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

        return {
            "query": query,
            "answer": answer,
            "search_hits": [asdict(hit) for hit in hits],
            "llm_raw": response.get("raw"),
            "llm_warning": response.get("warning"),
            "llm_error": response.get("error"),
        }
