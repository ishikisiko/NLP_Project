"""LangChain-based RAG pipelines using LCEL (LangChain Expression Language).

This module provides modern, composable RAG implementations using LangChain's
LCEL syntax for building retrieval-augmented generation pipelines.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import re
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Union

from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.langchain_support import Document, FileReader, LangChainVectorStore
from langchain.langchain_tools import SearchRetriever, WebSearchTool
from search.search import SearchClient, SearchHit, GoogleSearchClient
from utils.timing_utils import TimingRecorder
from utils.query_config import (
    TEMPORAL_CHANGE_KEYWORDS,
    TIME_RANGE_CONFIG,
    QUERY_SIMPLIFICATION_PROMPT,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


# Default prompts
DEFAULT_LOCAL_RAG_SYSTEM_PROMPT = """You are a helpful assistant. 
Answer the user's question based on the provided context from local documents.
Always answer in the same language as the user's question.
If the context doesn't contain relevant information, say so clearly."""

DEFAULT_SEARCH_RAG_SYSTEM_PROMPT = """You are an information assistant.
Answer user questions concisely using ONLY the provided search results and local documents.
CRITICAL: Do NOT fabricate, invent, or guess any specific data (such as scores, numbers, statistics, dates, or names) that is not EXPLICITLY stated in the search results or local documents.
If specific information is not found, clearly state '未在搜索结果和本地文档中找到具体数据' or 'specific data not found'.
When unsure, acknowledge the uncertainty instead of guessing.
Always answer in the same language as the user's question."""


class LocalRAGChain:
    """Local RAG pipeline using LangChain LCEL.
    
    This class provides a modern LCEL-based implementation of local document
    retrieval-augmented generation.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        data_path: str,
        *,
        embedding_model: str = "all-MiniLM-L6-v2",
        system_prompt: str = DEFAULT_LOCAL_RAG_SYSTEM_PROMPT,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        
        # Initialize vector store
        self.vector_store = LangChainVectorStore(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Load and index documents
        print("Loading and indexing documents...")
        chunk_count = self.vector_store.index_from_directory(data_path)
        print(f"Indexed {chunk_count} chunks.")
        
        # Build the LCEL chain
        self._chain = self._build_chain()
    
    def _build_chain(self) -> Runnable:
        """Build the LCEL chain for local RAG."""
        
        # Create retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Format documents function
        def format_docs(docs: List[LCDocument]) -> str:
            return "\n\n".join(
                f"[Document {i+1}]\nSource: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        
        # Build the chain using LCEL
        chain = (
            RunnableParallel(
                context=retriever | RunnableLambda(format_docs),
                question=RunnablePassthrough(),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def invoke(self, query: str, **kwargs: Any) -> str:
        """Run the RAG chain and return the answer."""
        return self._chain.invoke(query, **kwargs)
    
    def stream(self, query: str, **kwargs: Any) -> Iterator[str]:
        """Stream the RAG chain response."""
        for chunk in self._chain.stream(query, **kwargs):
            yield chunk
    
    def answer(
        self,
        query: str,
        *,
        num_retrieved_docs: int = 5,
        max_tokens: int = 5000,
        temperature: float = 0.3,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Dict[str, Any]:
        """Answer a query with full response metadata (legacy interface).
        
        This method provides backward compatibility with the old LocalRAG interface.
        """
        # Retrieve documents
        retrieved_docs = self.vector_store.search(query, k=num_retrieved_docs)
        context = "\n".join([doc.content for doc in retrieved_docs])
        
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        
        # Generate response
        response_start = time.perf_counter()
        try:
            response = self.llm.invoke(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.content if hasattr(response, 'content') else str(response)
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="local_rag_answer",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm, "provider", None),
                    model=getattr(self.llm, "model_name", None),
                )
        
        # Build answer with source references
        answer = content
        if answer and retrieved_docs:
            answer += "\n\n**本地文档来源：**\n"
            for idx, doc in enumerate(retrieved_docs, start=1):
                source = doc.source or f"文档 {idx}"
                answer += f"{idx}. {source}\n"
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": [asdict(doc) for doc in retrieved_docs],
            "llm_raw": response.response_metadata if hasattr(response, "response_metadata") else None,
            "search_hits": [],
        }


class SearchRAGChain:
    """Search-augmented RAG pipeline using LangChain LCEL.
    
    Combines web search with optional local document retrieval for comprehensive
    information retrieval.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        search_client: SearchClient,
        *,
        data_path: Optional[str] = None,
        system_prompt: str = DEFAULT_SEARCH_RAG_SYSTEM_PROMPT,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker: Optional[Any] = None,
        min_rerank_score: float = 0.0,
        max_per_domain: int = 1,
    ) -> None:
        self.llm = llm
        self.search_client = search_client
        self.system_prompt = system_prompt
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score
        self.max_per_domain = max(1, max_per_domain)
        
        # Initialize local vector store if data_path provided
        self.vector_store: Optional[LangChainVectorStore] = None
        if data_path:
            print("Loading and indexing local documents...")
            try:
                self.vector_store = LangChainVectorStore(model_name=embedding_model)
                chunk_count = self.vector_store.index_from_directory(data_path)
                print(f"Indexed {chunk_count} chunks from local documents.")
            except Exception as e:
                print(f"Failed to load local documents: {e}")
                self.vector_store = None
    
    def _format_search_hits(self, hits: List[SearchHit]) -> str:
        """Format search hits for prompt context."""
        if not hits:
            return "No search results were returned."
        
        formatted = []
        for idx, hit in enumerate(hits, 1):
            formatted.append(
                f"{idx}. {hit.title or f'Result {idx}'}\n"
                f"   URL: {hit.url or 'N/A'}\n"
                f"   {hit.snippet or 'No snippet available.'}"
            )
        return "\n".join(formatted)
    
    def _format_local_docs(self, docs: List[Document]) -> str:
        """Format local documents for prompt context."""
        if not docs:
            return ""
        
        formatted = []
        for idx, doc in enumerate(docs, 1):
            source = doc.source or f"Document {idx}"
            content = doc.content[:500]
            if len(doc.content) > 500:
                content += "..."
            formatted.append(f"{idx}. {source}\n   {content}")
        return "\n".join(formatted)
    
    def _apply_rerank(
        self,
        query: str,
        hits: List[SearchHit],
        limit: Optional[int] = None,
    ) -> tuple[List[SearchHit], List[Dict[str, Any]]]:
        """Apply reranking to search results."""
        if not self.reranker or not hits:
            return hits, []
        
        try:
            from urllib.parse import urlparse
            
            reranked = self.reranker.rerank(query, hits)
            filtered: List[SearchHit] = []
            metadata: List[Dict[str, Any]] = []
            domain_counts: Dict[str, int] = {}
            max_results = limit or len(reranked)
            
            for item in reranked:
                if item.score < self.min_rerank_score:
                    metadata.append({
                        "url": item.hit.url,
                        "score": item.score,
                        "dropped": "below_min_score",
                    })
                    continue
                
                domain = urlparse(item.hit.url).netloc if item.hit.url else None
                if domain and domain_counts.get(domain, 0) >= self.max_per_domain:
                    metadata.append({
                        "url": item.hit.url,
                        "score": item.score,
                        "dropped": "per_domain_limit",
                    })
                    continue
                
                filtered.append(item.hit)
                metadata.append({
                    "url": item.hit.url,
                    "score": item.score,
                    "kept": True,
                })
                
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                if len(filtered) >= max_results:
                    break
            
            return (filtered or hits, metadata)
        except Exception as exc:
            return hits, [{"error": str(exc)}]

    def _is_temporal_change_query(self, query: str) -> bool:
        """Check if query is related to temporal changes."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in TEMPORAL_CHANGE_KEYWORDS)

    def _check_google_client_availability(self) -> Optional[Any]:
        """Check availability of Google search client."""
        # Check if search_client is CombinedSearchClient with clients
        if hasattr(self.search_client, "clients"):
            for client in self.search_client.clients:
                if hasattr(client, "source_id") and client.source_id == "google":
                    return client
        
        # Check if search_client itself is a GoogleSearchClient
        if hasattr(self.search_client, "source_id") and self.search_client.source_id == "google":
            return self.search_client
            
        return None

    def _extract_years_from_hits(self, hits: List[SearchHit]) -> Set[str]:
        """Extract years found in search hits."""
        if not hits:
            return set()
            
        # Check both snippets and titles
        combined_text = " ".join(f"{hit.title} {hit.snippet}" for hit in hits).lower()
        year_pattern = r'\b(20\d{2})\b'
        years_found = set(re.findall(year_pattern, combined_text))
        return years_found

    def _detect_missing_years(self, query: str, hits: List[SearchHit]) -> List[str]:
        """Detect which years from the specified time range are missing in the search results."""
        query_lower = query.lower()
        
        # Check if this is a time-based query
        is_time_query = False
        time_range_years = DEFAULT_CONFIG["max_granular_search_years"]
        coverage_threshold = DEFAULT_CONFIG["default_coverage_threshold"]
        
        # Check for specific time ranges
        for range_name, config in TIME_RANGE_CONFIG.items():
            if any(k in query_lower for k in config["keywords"]):
                is_time_query = True
                time_range_years = config["years"]
                coverage_threshold = config["coverage_threshold"]
                break
        
        # Also check if it's a temporal change query
        if not is_time_query and self._is_temporal_change_query(query):
            is_time_query = True
        
        if not is_time_query:
            return []
            
        import datetime
        now_year = datetime.datetime.now().year
        start_year = now_year - time_range_years + 1
        target_years = {str(y) for y in range(start_year, now_year + 1)}
        
        found_years = self._extract_years_from_hits(hits)
        
        # Check if we have sufficient coverage
        if len(found_years.intersection(target_years)) / len(target_years) < coverage_threshold:
             return sorted(list(target_years), reverse=True)

        # Calculate missing years
        missing = target_years - found_years
        
        # If we are missing more than a few years, return them
        if missing:
            return sorted(list(missing), reverse=True)
            
        return []

    def _perform_granular_search_fallback(
        self, 
        original_query: str, 
        effective_query: str, 
        num_search_results: int, 
        per_source_cap: int,
        freshness: Optional[str],
        date_restrict: Optional[str],
        timing_recorder: Optional[TimingRecorder],
        missing_years: Optional[List[str]] = None
    ) -> List[SearchHit]:
        """Perform granular search for historical data."""
        
        granular_hits = []
        google_client = self._check_google_client_availability()
        active_client = google_client if google_client else self.search_client
        
        # Determine years to search
        if missing_years:
            selected_years = missing_years
            # If too many missing years (e.g. > 8), we might want to cap it to avoid excessive API calls
            # But for "last 10 years", usually at most 10.
            # Let's cap at 8 to be safe, prioritizing recent ones.
            if len(selected_years) > 8:
                selected_years = selected_years[:8]
        else:
            # Fallback to default sampling if no missing_years provided
            import datetime
            current_year = datetime.datetime.now().year
            years = [str(year) for year in range(current_year - 9, current_year + 1)]
            if len(years) > 6:
                step = max(1, len(years) // 5)
                selected_years = [years[i] for i in range(0, len(years), step)]
                if years[-1] not in selected_years:
                    selected_years.append(years[-1])
            else:
                selected_years = years

        logger.info(f"Granular search targeting years: {selected_years}")
        
        # Prepare base query
        base_query = effective_query if effective_query and len(effective_query) < len(original_query) * 1.5 else original_query
        
        # If base_query is too long/complex (likely a full sentence), simplify it using LLM
        if len(base_query) > DEFAULT_CONFIG["max_query_length_for_simplification"]:
            try:
                prompt = QUERY_SIMPLIFICATION_PROMPT.format(query=base_query)
                # Simple invocation to get keywords
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content if hasattr(response, 'content') else str(response)
                cleaned_keywords = content.strip().replace('"', '').replace("'", "")
                if cleaned_keywords and len(cleaned_keywords) < len(base_query):
                    logger.info(f"Simplified query for granular search: '{base_query}' -> '{cleaned_keywords}'")
                    base_query = cleaned_keywords
            except Exception as e:
                logger.warning(f"Failed to simplify query for granular search: {e}")

        for year in selected_years:
            query_lower = original_query.lower()
            universities = []
            if "香港中文大學" in original_query or "香港中文大学" in original_query or "cuhk" in query_lower:
                universities.extend(["Chinese University of Hong Kong", "CUHK"])
            if "香港科技大學" in original_query or "香港科技大学" in original_query or "hkust" in query_lower:
                universities.extend(["Hong Kong University of Science and Technology", "HKUST"])
            
            year_query = f"{base_query} {year}"
            
            if "qs" in query_lower and ("排名" in original_query or "ranking" in query_lower):
                if universities:
                    for uni in universities:
                        year_query = f"QS world university rankings {year} {uni}"
                        try:
                            search_kwargs = {
                                "num_results": max(2, num_search_results // (len(selected_years) * len(universities))),
                                "freshness": None, # Ignore freshness for historical search
                                "date_restrict": None, # Ignore date_restrict for historical search
                            }
                            if google_client:
                                pass
                            
                            year_hits = active_client.search(year_query, **search_kwargs)
                            granular_hits.extend(year_hits)
                            time.sleep(1)  # Avoid rate limits
                        except Exception as e:
                            logger.warning(f"Year {year} search failed: {e}")
                    continue
                else:
                    year_query = f"QS world university rankings {year}"
            
            try:
                search_kwargs = {
                    "num_results": max(3, num_search_results // len(selected_years)),
                    "freshness": None, # Ignore freshness for historical search
                    "date_restrict": None, # Ignore date_restrict for historical search
                }
                if google_client:
                    pass
                    
                year_hits = active_client.search(year_query, **search_kwargs)
                logger.info(f"Year {year} search found {len(year_hits)} hits.")
                granular_hits.extend(year_hits)
                time.sleep(1)  # Avoid rate limits
            except Exception as e:
                logger.warning(f"Year {year} search failed: {e}")
                
        return granular_hits
    
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
        enable_local_docs: bool = True,
        reference_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
        timing_recorder: Optional[TimingRecorder] = None,
        images: Optional[List[Dict[str, str]]] = None,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a query using search + local docs RAG pipeline."""
        
        effective_query = search_query.strip() if search_query else query
        hits: List[SearchHit] = []
        search_error: Optional[str] = None
        search_warnings: List[str] = []
        
        # Execute search
        if enable_search:
            try:
                per_source_cap = per_source_limit or num_search_results
                fetch_limit = num_search_results
                if self.reranker and hasattr(self.search_client, "clients"):
                    fetch_limit = per_source_cap * len(self.search_client.clients)
                
                hits = self.search_client.search(
                    effective_query,
                    num_results=fetch_limit,
                    per_source_limit=per_source_cap,
                    freshness=freshness,
                    date_restrict=date_restrict,
                )
                
                # Check for granular search fallback
                if self._is_temporal_change_query(query):
                    missing_years = self._detect_missing_years(query, hits)
                    if missing_years:
                        logger.info(f"Insufficient historical data found (missing: {missing_years}), performing granular search fallback.")
                        granular_hits = self._perform_granular_search_fallback(
                            query, effective_query, num_search_results, per_source_cap,
                            freshness, date_restrict, timing_recorder, missing_years=missing_years
                        )
                        # Merge hits
                        hits.extend(granular_hits)
            except Exception as exc:
                hits = []
                search_error = str(exc)
            finally:
                if timing_recorder:
                    timings_getter = getattr(self.search_client, "get_last_timings", None)
                    if callable(timings_getter):
                        timing_recorder.extend_search_timings(timings_getter())
            
            # Collect errors from combined client
            get_last_errors = getattr(self.search_client, "get_last_errors", None)
            if callable(get_last_errors):
                errors = get_last_errors() or []
                for item in errors:
                    source = str(item.get("source") or "搜索服务")
                    detail = str(item.get("error") or "未知错误")
                    search_warnings.append(f"{source} 出现异常：{detail}")
        
        # Apply reranking
        hits, rerank_meta = self._apply_rerank(query, hits, limit=num_search_results)
        
        # Retrieve local documents
        retrieved_docs: List[Document] = []
        if enable_local_docs and self.vector_store:
            retrieved_docs = self.vector_store.search(query, k=num_retrieved_docs)
        
        # Build prompt context
        search_context = self._format_search_hits(hits)
        local_context = self._format_local_docs(retrieved_docs)
        
        prompt_parts = [f"Question: {query}\n\n"]
        if search_context:
            prompt_parts.append(f"Web Search Results:\n{search_context}\n\n")
        if local_context:
            prompt_parts.append(f"Local Documents:\n{local_context}\n\n")
        
        if extra_context:
            prompt_parts.append(f"Additional Context (Domain Data):\n{extra_context}\n\n")
            
        prompt_parts.append(
            "Based on the above information, please answer the question. "
            "If information is insufficient, acknowledge it."
        )
        
        user_prompt = "".join(prompt_parts)
        
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
        ]
        
        # Handle images for multimodal
        if images:
            # Check if LLM supports vision
            vision_keywords = ["grok", "gpt-4", "claude", "gemini", "glm-4v", "glm-4.5v", "claude-4.5-haiku", "vision", "minimax"]
            model_name = getattr(self.llm, 'model_name', '')
            is_vision_model = any(k in model_name.lower() for k in vision_keywords)
            
            if is_vision_model:
                content_list = [{"type": "text", "text": user_prompt}]
                for img in images:
                    b64 = img.get("base64", "")
                    if "," in b64:
                        b64 = b64.split(",")[1]
                    mime = img.get("mime_type", "image/jpeg")
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
                messages.append(HumanMessage(content=content_list))
            else:
                # For non-vision models, just send the original user prompt
                # The system prompt should contain information about images and any vision metadata
                messages.append(HumanMessage(content=user_prompt))
        else:
            messages.append(HumanMessage(content=user_prompt))
        
        # Generate response
        response_start = time.perf_counter()
        try:
            response = self.llm.invoke(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.content if hasattr(response, 'content') else str(response)
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="search_rag_answer",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm, "provider", None),
                    model=getattr(self.llm, "model_name", None),
                )
        
        # Build answer with references
        answer = content
        reference_hits = hits if reference_limit is None else hits[:reference_limit]
        
        if answer:
            if reference_hits:
                answer += "\n\n**网络来源：**\n"
                for idx, hit in enumerate(reference_hits, 1):
                    title = hit.title or f"Result {idx}"
                    url = hit.url or ""
                    bullet = f"{idx}. [{title}]({url})" if url else f"{idx}. {title}"
                    answer += f"{bullet}\n"
            
            if retrieved_docs:
                answer += "\n\n**本地文档来源：**\n"
                for idx, doc in enumerate(retrieved_docs, 1):
                    source = doc.source or f"文档 {idx}"
                    answer += f"{idx}. {source}\n"
        
        # Build response
        payload: Dict[str, Any] = {
            "query": query,
            "answer": answer,
            "search_hits": [asdict(hit) for hit in hits],
            "retrieved_docs": [asdict(doc) for doc in retrieved_docs],
            "llm_raw": response.response_metadata if hasattr(response, "response_metadata") else None,
            "rerank": rerank_meta or None,
        }
        
        if search_error:
            payload["search_error"] = search_error
        if search_warnings:
            payload["search_warnings"] = search_warnings
        
        return payload
    
    def answer_stream(
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
        enable_local_docs: bool = True,
        reference_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Iterator[str]:
        """Stream answer using search + local docs RAG pipeline."""
        import json
        
        effective_query = search_query.strip() if search_query else query
        hits: List[SearchHit] = []
        search_warnings: List[str] = []
        
        # Execute search
        if enable_search:
            per_source_cap = per_source_limit or num_search_results
            hits = self.search_client.search(
                effective_query,
                num_results=num_search_results,
                per_source_limit=per_source_cap,
                freshness=freshness,
                date_restrict=date_restrict,
            )
            
            if timing_recorder:
                timings_getter = getattr(self.search_client, "get_last_timings", None)
                if callable(timings_getter):
                    timing_recorder.extend_search_timings(timings_getter())
        
        # Apply reranking
        hits, rerank_meta = self._apply_rerank(query, hits, limit=num_search_results)
        
        # Retrieve local documents
        retrieved_docs: List[Document] = []
        if enable_local_docs and self.vector_store:
            retrieved_docs = self.vector_store.search(query, k=num_retrieved_docs)
        
        # Yield preliminary data
        preliminary = {
            "query": query,
            "search_hits": [asdict(hit) for hit in hits],
            "retrieved_docs": [asdict(doc) for doc in retrieved_docs],
            "rerank": rerank_meta or None,
            "search_query": effective_query,
        }
        yield json.dumps({"type": "preliminary", "data": preliminary})
        
        # Build prompt
        search_context = self._format_search_hits(hits)
        local_context = self._format_local_docs(retrieved_docs)
        
        prompt_parts = [f"Question: {query}\n\n"]
        if search_context:
            prompt_parts.append(f"Web Search Results:\n{search_context}\n\n")
        if local_context:
            prompt_parts.append(f"Local Documents:\n{local_context}\n\n")
        prompt_parts.append("Answer the question based on the above information.")
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="".join(prompt_parts)),
        ]
        
        # Stream response
        response_start = time.perf_counter()
        full_answer = ""
        
        try:
            for chunk in self.llm.stream(messages, max_tokens=max_tokens, temperature=temperature):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    full_answer += content
                    yield json.dumps({"type": "content", "data": content})
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="search_rag_answer_stream",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm, "provider", None),
                    model=getattr(self.llm, "model_name", None),
                )
        
        # Yield references
        reference_hits = hits if reference_limit is None else hits[:reference_limit]
        if reference_hits:
            ref_text = "\n\n**网络来源：**\n"
            for idx, hit in enumerate(reference_hits, 1):
                title = hit.title or f"Result {idx}"
                url = hit.url or ""
                bullet = f"{idx}. [{title}]({url})" if url else f"{idx}. {title}"
                ref_text += f"{bullet}\n"
            yield json.dumps({"type": "references", "data": ref_text})
        
        if retrieved_docs:
            ref_text = "\n\n**本地文档来源：**\n"
            for idx, doc in enumerate(retrieved_docs, 1):
                source = doc.source or f"文档 {idx}"
                ref_text += f"{idx}. {source}\n"
            yield json.dumps({"type": "local_references", "data": ref_text})


# Factory functions for creating RAG chains
def create_local_rag_chain(
    data_path: str,
    llm: Optional[BaseChatModel] = None,
    **kwargs: Any,
) -> LocalRAGChain:
    """Create a local RAG chain.
    
    Args:
        data_path: Path to directory containing documents
        llm: LangChain chat model (created from config if not provided)
        **kwargs: Additional arguments passed to LocalRAGChain
    
    Returns:
        Configured LocalRAGChain instance
    """
    if llm is None:
        from langchain_llm import create_chat_model
        llm = create_chat_model()
    
    return LocalRAGChain(llm=llm, data_path=data_path, **kwargs)


def create_search_rag_chain(
    search_client: SearchClient,
    llm: Optional[BaseChatModel] = None,
    data_path: Optional[str] = None,
    **kwargs: Any,
) -> SearchRAGChain:
    """Create a search RAG chain.
    
    Args:
        search_client: Search client for web search
        llm: LangChain chat model (created from config if not provided)
        data_path: Optional path to local documents
        **kwargs: Additional arguments passed to SearchRAGChain
    
    Returns:
        Configured SearchRAGChain instance
    """
    if llm is None:
        from langchain_llm import create_chat_model
        llm = create_chat_model()
    
    return SearchRAGChain(
        llm=llm,
        search_client=search_client,
        data_path=data_path,
        **kwargs,
    )
