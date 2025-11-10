from __future__ import annotations

import textwrap
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from api import LLMClient
from local_rag import Document, FileReader, TextSplitter, VectorStore
from search import SearchClient, SearchHit
from rerank import BaseReranker


HYBRID_SYSTEM_PROMPT = (
    "You are an information assistant. Combine insights from web search results and local documents. "
    "Only answer with information grounded in the provided sources, and cite them inline when relevant "
    "using (Web #) for search results and (Doc #) for local references. If the context is insufficient, "
    "state that explicitly."
)


class HybridRAG:
    """Pipeline that fuses web search snippets with locally indexed documents."""

    def __init__(
        self,
        llm_client: LLMClient,
        search_client: SearchClient,
        data_path: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker: Optional[BaseReranker] = None,
        min_rerank_score: float = 0.0,
        max_per_domain: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score
        self.max_per_domain = max(1, max_per_domain)

        reader = FileReader(data_path)
        documents = reader.load()

        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split(documents) if documents else []

        self.vector_store = VectorStore(model_name=embedding_model)
        if chunks:
            self.vector_store.add_documents(chunks)

        self._indexed_chunks = len(chunks)

    def _format_search_hits(self, hits: List[SearchHit]) -> str:
        if not hits:
            return "No web results available."

        formatted_rows = []
        for idx, hit in enumerate(hits, start=1):
            title = hit.title or f"Result {idx}"
            url = hit.url or "No URL available."
            snippet = hit.snippet or "No snippet provided."
            formatted_rows.append(
                f"{idx}. {title}\n"
                f"   URL: {url}\n"
                f"   Snippet: {snippet}"
            )
        return "\n".join(formatted_rows)

    def _format_local_docs(self, docs: List[Document]) -> str:
        if not docs:
            if self._indexed_chunks == 0:
                return "No local documents were indexed."
            return "No relevant local passages were retrieved."

        formatted_rows = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.source or f"Document {idx}"
            snippet = textwrap.shorten(" ".join(doc.content.split()), width=400, placeholder="…")
            formatted_rows.append(
                f"{idx}. Source: {source}\n"
                f"   Excerpt: {snippet}"
            )
        return "\n".join(formatted_rows)

    def _build_prompt(
        self,
        query: str,
        hits: List[SearchHit],
        docs: List[Document],
    ) -> str:
        search_block = self._format_search_hits(hits)
        local_block = self._format_local_docs(docs)

        return (
            "Use the following information to answer the question. "
            "Cite web sources as (Web #) and local passages as (Doc #).\n\n"
            f"Web Search Results:\n{search_block}\n\n"
            f"Local Document Passages:\n{local_block}\n\n"
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
        num_retrieved_docs: int = 5,
        max_tokens: int = 5000,
        temperature: float = 0.3,
        enable_search: bool = True,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> Dict[str, object]:
        hits: List[SearchHit] = []
        effective_query = search_query.strip() if search_query else query
        search_error: Optional[str] = None

        search_warnings: List[str] = []
        raw_hits: List[SearchHit] = []

        if enable_search:
            try:
                per_source_cap = per_source_limit if per_source_limit is not None else num_search_results
                raw_hits = self.search_client.search(
                    effective_query,
                    num_results=num_search_results,
                    per_source_limit=per_source_cap,
                    freshness=freshness,
                    date_restrict=date_restrict,
                )
                hits = list(raw_hits)
            except Exception as exc:
                # Surface search errors while still letting the LLM see local docs
                hits = []
                search_error = str(exc)

        if raw_hits:
            get_last_errors = getattr(self.search_client, "get_last_errors", None)
            if callable(get_last_errors):
                errors = get_last_errors() or []
                if errors:
                    for item in errors:
                        source = str(item.get("source") or "搜索服务")
                        detail = str(item.get("error") or "未知错误")
                        if source.lower().startswith("mcp"):
                            search_warnings.append(f"{source} 未正常工作，已使用其他搜索结果。原因：{detail}")
                        else:
                            search_warnings.append(f"{source} 出现异常：{detail}")

        hits, rerank_meta = self._apply_rerank(query, hits, limit=num_search_results)

        retrieved_docs: List[Document] = []
        if self._indexed_chunks:
            retrieved_docs = self.vector_store.search(query, k=num_retrieved_docs)

        user_prompt = self._build_prompt(query, hits, retrieved_docs)
        response = self.llm_client.chat(
            system_prompt=HYBRID_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer = response.get("content")
        if answer:
            if hits:
                answer += "\n\n**Web Sources:**\n"
                for idx, hit in enumerate(hits, start=1):
                    title = hit.title or f"Result {idx}"
                    url = hit.url or ""
                    bullet = f"{idx}. [{title}]({url})" if url else f"{idx}. {title}"
                    answer += f"{bullet}\n"
            if retrieved_docs:
                answer += "\n\n**Local Sources:**\n"
                for idx, doc in enumerate(retrieved_docs, start=1):
                    source = doc.source or f"Document {idx}"
                    answer += f"{idx}. {source}\n"

        payload: Dict[str, object] = {
            "query": query,
            "answer": answer,
            "search_hits": [asdict(hit) for hit in hits],
            "retrieved_docs": [asdict(doc) for doc in retrieved_docs],
            "llm_raw": response.get("raw"),
            "llm_warning": response.get("warning"),
            "llm_error": response.get("error"),
            "rerank": rerank_meta or None,
        }
        if search_error:
            payload["search_error"] = search_error
        if search_warnings:
            payload["search_warnings"] = search_warnings
        payload["search_query"] = effective_query if enable_search else None
        return payload

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
        except Exception as exc:  # pragma: no cover - defensive fallback
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
