from __future__ import annotations

import time
from dataclasses import asdict
from typing import List, Optional, Dict

from api import LLMClient
from langchain_support import Document, FileReader, LangChainVectorStore
from timing_utils import TimingRecorder


class LocalRAG:
    """A local RAG pipeline that uses a vector store for retrieval."""

    def __init__(
        self,
        llm_client: LLMClient,
        data_path: str,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.llm_client = llm_client
        self.vector_store = LangChainVectorStore(model_name=embedding_model)

        print("Loading and indexing documents...")
        reader = FileReader(data_path)
        documents = reader.load()
        chunk_count = self.vector_store.index(documents)
        print(f"Indexed {chunk_count} chunks.")

    def answer(
        self,
        query: str,
        *,
        num_retrieved_docs: int = 5,
        max_tokens: int = 5000,
        temperature: float = 0.3,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Dict[str, object]:
        """Answer a query using the local RAG pipeline."""
        retrieved_docs = self.vector_store.search(query, k=num_retrieved_docs)
        context = "\n".join([doc.content for doc in retrieved_docs])

        user_prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response_start = time.perf_counter()
        try:
            response = self.llm_client.chat(
                system_prompt="You are a helpful assistant. Always answer in the same language as the user's question.",
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="local_rag_answer",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm_client, "provider", None),
                    model=getattr(self.llm_client, "model_id", None),
                )

        # Build answer with source references
        answer = response.get("content")
        if answer and retrieved_docs:
            answer += "\n\n**本地文档来源：**\n"
            for idx, doc in enumerate(retrieved_docs, start=1):
                source = doc.source or f"文档 {idx}"
                answer += f"{idx}. {source}\n"

        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": [asdict(doc) for doc in retrieved_docs],
            "llm_raw": response.get("raw"),
            "llm_warning": response.get("warning"),
            "llm_error": response.get("error"),
            "search_hits": [],
        }
