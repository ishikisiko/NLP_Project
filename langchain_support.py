from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument


@dataclass
class Document:
    """Simple text container with an optional source label."""

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
                    Document(content=text, source=f"{file_path} (page {i + 1})")
                    for i, page in enumerate(reader.pages)
                    if (text := page.extract_text())
                ]
        except Exception as exc:
            print(f"Skipping corrupted or invalid PDF '{file_path}': {exc}")
            return []

    def _load_text(self, file_path: str) -> Document:
        """Load a single text-based document."""
        with open(file_path, "r", encoding="utf-8") as f:
            return Document(content=f.read(), source=file_path)


class LangChainVectorStore:
    """FAISS-based vector store backed by LangChain primitives."""

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._store: Optional[FAISS] = None

    def index(self, documents: List[Document]) -> int:
        """Index a list of documents and return the number of chunks stored."""
        lc_docs = [
            LCDocument(page_content=doc.content, metadata={"source": doc.source})
            for doc in documents
            if doc.content
        ]
        split_docs = self._splitter.split_documents(lc_docs)
        if not split_docs:
            self._store = None
            return 0

        self._store = FAISS.from_documents(split_docs, self._embedder)
        return len(split_docs)

    def search(self, query: str, *, k: int = 5) -> List[Document]:
        """Search for the top k most similar documents to a query."""
        if not self._store:
            return []
        hits = self._store.similarity_search(query, k=k)
        return [
            Document(content=hit.page_content, source=hit.metadata.get("source"))
            for hit in hits
        ]

    @property
    def is_ready(self) -> bool:
        return self._store is not None
