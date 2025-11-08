from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import PyPDF2
from sentence_transformers import SentenceTransformer

from api import LLMClient


@dataclass
class Document:
    """Represents a single text document with optional metadata."""

    content: str
    source: Optional[str] = None


class FileReader:
    """Loads documents from a directory, supporting .txt, .md, and .pdf files."""

    def __init__(self, path: str, recursive: bool = True) -> None:
        if not os.path.isdir(path):
            raise ValueError(f"Path '{path}' is not a valid directory.")
        self.path = path
        self.recursive = recursive

    def load(self) -> List[Document]:
        """Load all supported documents from the configured path."""
        documents = []
        for root, _, files in os.walk(self.path):
            if not self.recursive and root != self.path:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(".pdf"):
                    documents.extend(self._load_pdf(file_path))
                elif file_path.endswith((".txt", ".md")):
                    documents.append(self._load_text(file_path))
        return documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF and return a Document per page."""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return [
                    Document(
                        content=page.extract_text(),
                        source=f"{file_path} (page {i + 1})",
                    )
                    for i, page in enumerate(reader.pages)
                ]
        except Exception as e:
            print(f"Skipping corrupted or invalid PDF '{file_path}': {e}")
            return []

    def _load_text(self, file_path: str) -> Document:
        """Load a single text-based document."""
        with open(file_path, "r", encoding="utf-8") as f:
            return Document(content=f.read(), source=file_path)


class TextSplitter:
    """Splits a list of documents into smaller, overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            content = doc.content
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk_content = content[i : i + self.chunk_size]
                chunks.append(Document(content=chunk_content, source=doc.source))
        return chunks


class VectorStore:
    """An in-memory vector store using SentenceTransformers for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.vectors = {}
        self.documents = []

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and create their embeddings."""
        self.documents.extend(documents)
        embeddings = self.model.encode([doc.content for doc in documents])
        for i, doc in enumerate(documents):
            self.vectors[len(self.vectors)] = embeddings[i]

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for the top k most similar documents to a query."""
        query_embedding = self.model.encode([query])
        scores = {}
        for i, vector in self.vectors.items():
            scores[i] = self.model.similarity(query_embedding, vector).item()

        sorted_indices = sorted(scores, key=scores.get, reverse=True)
        return [self.documents[i] for i in sorted_indices[:k]]


class LocalRAG:
    """A local RAG pipeline that uses a vector store for retrieval."""

    def __init__(self, llm_client: LLMClient, data_path: str) -> None:
        self.llm_client = llm_client
        self.vector_store = VectorStore()

        print("Loading and indexing documents...")
        reader = FileReader(data_path)
        documents = reader.load()
        splitter = TextSplitter()
        chunks = splitter.split(documents)
        self.vector_store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks.")

    def answer(
        self,
        query: str,
        *,
        num_retrieved_docs: int = 5,
        max_tokens: int = 5000,
        temperature: float = 0.3,
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

        response = self.llm_client.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
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
