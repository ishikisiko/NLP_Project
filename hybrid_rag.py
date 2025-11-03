from __future__ import annotations

import textwrap
from dataclasses import asdict
from typing import Dict, List, Optional

from api import LLMClient
from local_rag import Document, FileReader, TextSplitter, VectorStore
from search import SearchClient, SearchHit


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
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client

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
        num_search_results: int = 5,
        num_retrieved_docs: int = 5,
        max_tokens: int = 5000,
        temperature: float = 0.3,
    ) -> Dict[str, object]:
        hits: List[SearchHit] = []
        try:
            hits = self.search_client.search(query, num_results=num_search_results)
        except Exception as exc:
            # Surface search errors while still letting the LLM see local docs
            hits = []
            search_error = str(exc)
        else:
            search_error = None

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
        }
        if search_error:
            payload["search_error"] = search_error
        return payload
