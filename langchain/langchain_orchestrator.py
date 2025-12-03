"""LangChain-based intelligent orchestrator with agent-style routing.

This module provides a modern implementation of the smart orchestrator using
LangChain's agent and router patterns for intelligent query handling.
"""

from __future__ import annotations

import json
import os
import sys
import time
import requests
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.langchain_rag import LocalRAGChain, SearchRAGChain
from langchain.langchain_support import Document, LangChainVectorStore
from search.search import SearchClient, SearchHit
from search.source_selector import IntelligentSourceSelector
from utils.time_parser import TimeConstraint, parse_time_constraint
from utils.timing_utils import TimingRecorder
from utils.current_time import get_current_date_str


class QueryIntent(str, Enum):
    """Enum for query intent classification."""
    SEARCH = "search"
    LOCAL_RAG = "local_rag"
    DIRECT_ANSWER = "direct_answer"
    DOMAIN_API = "domain_api"
    SMALL_TALK = "small_talk"


class RouterDecision(BaseModel):
    """Schema for router decision output."""
    needs_search: bool = Field(description="Whether web search is needed")
    reason: str = Field(description="Reasoning for the decision")
    answer: Optional[str] = Field(default=None, description="Direct answer if no search needed")


class KeywordGeneration(BaseModel):
    """Schema for keyword generation output."""
    keywords: List[str] = Field(description="Generated search keywords")


class LangChainOrchestrator:
    """Intelligent orchestrator using LangChain for routing and RAG.
    
    This orchestrator decides the best approach for answering queries:
    - Direct LLM response for simple questions
    - Local RAG for document-based queries
    - Web search RAG for current information
    - Domain-specific APIs for specialized queries
    """
    
    # System prompts
    DECISION_SYSTEM_PROMPT = """You are a routing assistant that decides whether a user's question needs fresh web or document search.

Respond strictly in JSON format with the following structure:
{
    "needs_search": true/false,
    "reason": "brief explanation",
    "answer": "direct answer if needs_search is false, otherwise empty string"
}

Guidelines:
- Set needs_search=true for: current events, real-time data, recent news, prices, scores, weather
- Set needs_search=false for: general knowledge, definitions, concepts, historical facts, greetings
- If needs_search=false, provide a complete answer in the "answer" field"""

    KEYWORD_SYSTEM_PROMPT = """You help generate high quality web search keywords.

Generate up to 4 bilingual (Chinese/English) search keywords or phrases for the given query.

Respond in JSON format:
{
    "keywords": ["关键词1", "keyword 1", "关键词2", "keyword 2"]
}

Rules:
1. Keywords should cover the core information of the query
2. Include English keywords to improve search effectiveness
3. For sports queries, add keywords like '战报 highlights', '得分统计 box score'
4. For news queries, add keywords like '最新 latest', '新闻 news'"""

    DIRECT_ANSWER_SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer clearly based on your existing knowledge.
Always answer in the same language as the user's question."""

    SMALL_TALK_PATTERNS = {
        "hi", "hello", "hey", "thanks", "thank you", "good morning", "good night",
        "bye", "goodbye", "see you", "你好", "您好", "嗨", "谢谢", "感谢", "早上好",
        "晚上好", "晚安", "再见", "拜拜", "哈囉", "謝謝", "感謝", "早安", "再見", "掰掰",
    }

    def __init__(
        self,
        llm: BaseChatModel,
        search_client: Optional[SearchClient] = None,
        *,
        classifier_llm: Optional[BaseChatModel] = None,
        routing_llm: Optional[BaseChatModel] = None,
        data_path: Optional[str] = None,
        reranker: Optional[Any] = None,
        min_rerank_score: float = 0.0,
        max_per_domain: int = 1,
        source_selector: Optional[IntelligentSourceSelector] = None,
        show_timings: bool = False,
        google_api_key: Optional[str] = None,
        finnhub_api_key: Optional[str] = None,
        sportsdb_api_key: Optional[str] = None,
        apisports_api_key: Optional[str] = None,
        # Search source metadata
        requested_search_sources: Optional[List[str]] = None,
        active_search_sources: Optional[List[str]] = None,
        active_search_source_labels: Optional[List[str]] = None,
        missing_search_sources: Optional[List[str]] = None,
        configured_search_sources: Optional[List[str]] = None,
    ) -> None:
        self.llm = llm
        self.classifier_llm = classifier_llm or llm
        self.routing_llm = routing_llm or llm
        self.search_client = search_client
        self.data_path = data_path
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score
        self.max_per_domain = max(1, max_per_domain)
        self.show_timings = show_timings
        self.google_api_key = google_api_key
        
        # Initialize source selector
        if source_selector:
            self.source_selector = source_selector
        else:
            # Create legacy wrapper for classifier LLM
            from langchain.langchain_llm import LangChainLLMWrapper
            legacy_client = LangChainLLMWrapper(self.classifier_llm)
            self.source_selector = IntelligentSourceSelector(
                llm_client=legacy_client,
                use_llm=True,
                google_api_key=google_api_key,
                finnhub_api_key=finnhub_api_key,
                sportsdb_api_key=sportsdb_api_key,
                apisports_api_key=apisports_api_key,
            )
        
        # Search source metadata
        self.requested_search_sources = self._normalize_sources(requested_search_sources)
        self.active_search_sources = self._normalize_sources(
            active_search_sources or getattr(search_client, "active_sources", [])
        )
        self.active_search_source_labels = [
            str(label).strip() for label in (active_search_source_labels or []) if str(label).strip()
        ]
        self.missing_search_sources = self._normalize_sources(missing_search_sources)
        self.configured_search_sources = self._normalize_sources(configured_search_sources)
        
        # Lazy-initialized pipelines
        self._local_rag: Optional[LocalRAGChain] = None
        self._search_rag: Optional[SearchRAGChain] = None
        self._local_signature: Optional[tuple] = None
        self._search_signature: Optional[tuple] = None
        
        # Build decision chain
        self._decision_chain = self._build_decision_chain()
        self._keyword_chain = self._build_keyword_chain()

    def _build_decision_chain(self):
        """Build the routing decision chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.DECISION_SYSTEM_PROMPT),
            ("human", "{query}"),
        ])
        
        return prompt | self.routing_llm

    def _build_keyword_chain(self):
        """Build the keyword generation chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.KEYWORD_SYSTEM_PROMPT),
            ("human", "{query}"),
        ])
        
        return prompt | self.routing_llm

    @staticmethod
    def _normalize_sources(sources: Optional[List[str]]) -> List[str]:
        """Normalize source list."""
        if not sources:
            return []
        normalized = []
        for item in sources:
            if item is None:
                continue
            token = str(item).strip().lower()
            if token and token not in normalized:
                normalized.append(token)
        return normalized

    def _is_small_talk(self, query: str) -> bool:
        """Check if query is small talk."""
        stripped = (query or "").strip()
        if not stripped:
            return True
        
        lowered = stripped.lower()
        if lowered in self.SMALL_TALK_PATTERNS or stripped in self.SMALL_TALK_PATTERNS:
            return True
        
        # Check for substring matches
        substring_triggers = ["你好", "您好", "嗨", "哈喽", "拜拜", "谢谢", "感谢", "哈囉", "掰掰", "謝謝", "感謝"]
        return any(token in stripped for token in substring_triggers)

    def _make_routing_decision(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Dict[str, Any]:
        """Make routing decision using LLM."""
        start = time.perf_counter()
        try:
            response = self._decision_chain.invoke({"query": query})
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        parsed = json.loads(content[start_idx:end_idx])
                    except json.JSONDecodeError:
                        parsed = {"needs_search": True, "reason": "parse_error"}
                else:
                    parsed = {"needs_search": True, "reason": "parse_error"}
            
            return {
                "needs_search": parsed.get("needs_search", True),
                "reason": parsed.get("reason", ""),
                "direct_answer": parsed.get("answer", ""),
                "raw_text": content[:100] if content else None,
            }
        except Exception as exc:
            return {
                "needs_search": True,
                "reason": f"decision_error: {exc}",
                "direct_answer": None,
                "raw_text": None,
            }
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_llm_call(
                    label="search_decision",
                    duration_ms=duration_ms,
                    provider=getattr(self.routing_llm, "provider", None),
                    model=getattr(self.routing_llm, "model_name", None),
                )

    def _generate_keywords(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Dict[str, Any]:
        """Generate search keywords using LLM."""
        start = time.perf_counter()
        try:
            response = self._keyword_chain.invoke({"query": query})
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        parsed = json.loads(content[start_idx:end_idx])
                    except json.JSONDecodeError:
                        parsed = {"keywords": []}
                else:
                    parsed = {"keywords": []}
            
            keywords = parsed.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(";") if k.strip()]
            
            return {
                "keywords": keywords[:10],
                "raw_text": content[:200] if content else None,
            }
        except Exception as exc:
            return {
                "keywords": [],
                "raw_text": None,
                "error": str(exc),
            }
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_llm_call(
                    label="keyword_generation",
                    duration_ms=duration_ms,
                    provider=getattr(self.routing_llm, "provider", None),
                    model=getattr(self.routing_llm, "model_name", None),
                )

    def _get_local_rag(self, snapshot: Optional[tuple]) -> Optional[LocalRAGChain]:
        """Get or create local RAG pipeline."""
        if not self.data_path or snapshot is None:
            return None
        
        if self._local_rag is None or self._local_signature != snapshot:
            try:
                self._local_rag = LocalRAGChain(
                    llm=self.llm,
                    data_path=self.data_path,
                )
                self._local_signature = snapshot
            except Exception as exc:
                print(f"Failed to initialize local RAG: {exc}")
                self._local_rag = None
                self._local_signature = None
        
        return self._local_rag

    def _get_search_rag(self, snapshot: Optional[tuple]) -> Optional[SearchRAGChain]:
        """Get or create search RAG pipeline."""
        if not self.search_client:
            return None
        
        search_signature = (id(self.search_client), snapshot)
        
        if self._search_rag is None or self._search_signature != search_signature:
            self._search_rag = SearchRAGChain(
                llm=self.llm,
                search_client=self.search_client,
                data_path=self.data_path,
                reranker=self.reranker,
                min_rerank_score=self.min_rerank_score,
                max_per_domain=self.max_per_domain,
            )
            self._search_signature = search_signature
        
        return self._search_rag

    def _snapshot_local_docs(self) -> Optional[tuple]:
        """Create a snapshot of local documents for cache invalidation."""
        if not self.data_path or not os.path.isdir(self.data_path):
            return None
        
        records = []
        for root, _, files in os.walk(self.data_path):
            for name in files:
                if name.lower().endswith((".txt", ".md", ".pdf")):
                    full_path = os.path.join(root, name)
                    try:
                        records.append((full_path, os.path.getmtime(full_path)))
                    except OSError:
                        continue
        return tuple(sorted(records))

    def _direct_answer(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder] = None,
        images: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate direct answer without search."""
        messages = [
            SystemMessage(content=system_prompt or self.DIRECT_ANSWER_SYSTEM_PROMPT),
        ]
        
        if images:
            content_list = [{"type": "text", "text": query}]
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
            messages.append(HumanMessage(content=query))
        
        start = time.perf_counter()
        try:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            return {
                "content": content,
                "raw": response.response_metadata if hasattr(response, "response_metadata") else None,
            }
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_llm_call(
                    label="direct_answer",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm, "provider", None),
                    model=getattr(self.llm, "model_name", None),
                )

    def answer(
        self,
        query: str,
        *,
        num_search_results: int = 10,
        per_source_search_results: Optional[int] = None,
        num_retrieved_docs: int = 5,
        max_tokens: int = 8000,
        temperature: float = 0.3,
        allow_search: bool = True,
        reference_limit: Optional[int] = None,
        force_search: bool = False,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Answer a query using intelligent routing.
        
        This method determines the best approach for answering:
        1. Small talk detection
        2. Domain-specific API routing
        3. Direct LLM answer for simple questions
        4. Web search RAG for current information
        5. Local RAG for document-based queries
        """
        timing_recorder = TimingRecorder(enabled=self.show_timings)
        timing_recorder.start()
        
        total_limit = max(1, int(num_search_results))
        per_source_limit = max(1, int(per_source_search_results or total_limit))
        force_search = bool(force_search and allow_search)
        
        # Parse time constraints
        time_constraint = parse_time_constraint(query)
        effective_query = time_constraint.cleaned_query if time_constraint.days else query
        
        if time_constraint.days:
            current_date = get_current_date_str()
            effective_query = f"{effective_query} (Current Date: {current_date})"
        
        snapshot = self._snapshot_local_docs()
        has_docs = bool(snapshot)
        
        # Handle images (visual retrieval)
        if images:
            visual_response = self._handle_visual_query(
                query, images, max_tokens, temperature, 
                has_docs, allow_search, timing_recorder
            )
            return self._finalize_response(visual_response, timing_recorder)
        
        # Small talk detection
        if self._is_small_talk(effective_query):
            response = self._handle_small_talk(
                query, max_tokens, temperature, 
                has_docs, allow_search, timing_recorder
            )
            return self._finalize_response(response, timing_recorder)
        
        # Search disabled
        if not allow_search:
            response = self._handle_local_only(
                query, snapshot, has_docs, num_retrieved_docs,
                max_tokens, temperature, timing_recorder
            )
            return self._finalize_response(response, timing_recorder)
        
        # Domain classification and API handling
        domain, sources = self.source_selector.select_sources(
            effective_query, timing_recorder=timing_recorder
        )
        enhanced_query = self.source_selector.generate_domain_specific_query(
            effective_query, domain
        )
        
        # For queries with finance keywords, use original query to preserve time expressions
        # like "前三年" which yfinance needs for proper date range parsing
        finance_keywords = ["股价", "stock", "股票", "市值", "market cap", "收益", "revenue",
                           "英伟达", "nvidia", "nvda", "英特尔", "intel", "intc", "amd",
                           "苹果", "apple", "aapl", "微软", "microsoft", "msft"]
        has_finance_keywords = any(kw in query.lower() for kw in finance_keywords)
        finance_query = query if has_finance_keywords else effective_query
        
        domain_api_result = self.source_selector.fetch_domain_data(
            finance_query, domain, timing_recorder=timing_recorder
        )
        
        should_continue = domain_api_result.get("continue_search", False) if domain_api_result else False
        
        if domain_api_result and domain_api_result.get("handled") and domain_api_result.get("answer") and not should_continue:
            response = self._handle_domain_api(
                query, domain, sources, enhanced_query,
                domain_api_result, has_docs, allow_search, force_search, timing_recorder
            )
            return self._finalize_response(response, timing_recorder)
        
        # Routing decision
        if not force_search:
            decision = self._make_routing_decision(effective_query, timing_recorder)
            
            if not decision["needs_search"] and decision.get("direct_answer"):
                response = self._build_direct_response(
                    query, decision["direct_answer"], decision,
                    has_docs, allow_search
                )
                return self._finalize_response(response, timing_recorder)
            
            if not decision["needs_search"]:
                direct = self._direct_answer(query, timing_recorder)
                response = self._build_direct_response(
                    query, direct.get("content", ""), decision,
                    has_docs, allow_search,
                    llm_raw=direct.get("raw"),
                    llm_error=direct.get("error"),
                )
                return self._finalize_response(response, timing_recorder)
        
        # Search is needed
        if not self.search_client:
            response = self._handle_search_unavailable(
                query, snapshot, has_docs, num_retrieved_docs,
                max_tokens, temperature, timing_recorder
            )
            if force_search:
                response.setdefault("control", {})["force_search_enabled"] = True
            return self._finalize_response(response, timing_recorder)
        
        # Generate keywords
        keyword_info = self._generate_keywords(effective_query, timing_recorder)
        keywords = keyword_info.get("keywords") or [effective_query]
        search_query = " ".join(keywords).strip() or effective_query
        
        # Execute search RAG
        pipeline = self._get_search_rag(snapshot)
        if not pipeline:
            response = self._handle_search_unavailable(
                query, snapshot, has_docs, num_retrieved_docs,
                max_tokens, temperature, timing_recorder
            )
            return self._finalize_response(response, timing_recorder)
        
        result = pipeline.answer(
            query,
            search_query=search_query,
            num_search_results=total_limit,
            per_source_limit=per_source_limit,
            num_retrieved_docs=num_retrieved_docs,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_search=True,
            freshness=time_constraint.you_freshness if time_constraint.days else None,
            date_restrict=time_constraint.google_date_restrict if time_constraint.days else None,
            reference_limit=reference_limit,
            timing_recorder=timing_recorder,
            extra_context=domain_api_result.get("answer") if domain_api_result and should_continue else None,
        )
        
        # Add control metadata
        control = {
            "search_performed": True,
            "decision": {"needs_search": True, "reason": "search_required"},
            "search_mode": "search",
            "keywords": keywords,
            "keyword_generation": keyword_info,
            "hybrid_mode": True,
            "local_docs_present": has_docs,
            "search_allowed": True,
            "domain": domain,
            "selected_sources": sources,
            "enhanced_query": enhanced_query,
            "search_total_limit": total_limit,
            "search_per_source_limit": per_source_limit,
            "force_search_enabled": force_search,
        }
        
        if time_constraint.days:
            control["time_constraint"] = {
                "original_query": time_constraint.original_query,
                "cleaned_query": time_constraint.cleaned_query,
                "time_expression": time_constraint.time_expression,
                "days": time_constraint.days,
            }
        
        result["control"] = control
        result["search_query"] = search_query
        
        return self._finalize_response(result, timing_recorder)

    def _handle_visual_query(
        self,
        query: str,
        images: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        has_docs: bool,
        allow_search: bool,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        """Handle queries with images."""
        # Check if LLM supports vision
        vision_keywords = ["grok", "gpt-4", "claude", "gemini", "glm-4v", "glm-4.5v", "claude-4.5-haiku", "vision", "minimax"]
        is_vision_model = any(k in self.llm.model_name.lower() if hasattr(self.llm, 'model_name') else '' for k in vision_keywords)
        
        # Try to get visual metadata from Google Vision API if available
        visual_context = ""
        if self.google_api_key:
            try:
                visual_info = self._perform_visual_retrieval(images, timing_recorder)
                if visual_info:
                    labels = [label.get("label") for label in visual_info.get("bestGuessLabels", [])]
                    entities = [entity.get("description") for entity in visual_info.get("webEntities", []) if entity.get("description")]
                    
                    visual_context = (
                        "我通过搜索引擎找到了关于这张图片的以下线索（元数据）：\n\n"
                        f"最佳猜测标签：[{', '.join(labels)}]\n\n"
                        f"关联实体：[{', '.join(entities)}]\n\n"
                        "请结合图片内容和上述线索回答用户的问题。如果线索不足，请诚实告知。请始终使用与用户提问相同的语言回答。"
                    )
            except Exception as e:
                print(f"Visual retrieval failed: {e}")
        
        if is_vision_model:
            system_prompt = "你是一个智能视觉助手。用户上传了一张图片。"
            if visual_context:
                system_prompt += "\n\n" + visual_context
            else:
                system_prompt += "\n请结合图片内容回答用户的问题。请始终使用与用户提问相同的语言回答。"
        else:
            system_prompt = "你是一个智能助手。用户上传了图片，但你无法查看图片内容。"
            if visual_context:
                system_prompt += "\n\n" + visual_context
                system_prompt += "\n\n虽然你无法直接查看图片，但可以根据上述元数据信息尝试回答用户的问题。请始终使用与用户提问相同的语言回答。"
            else:
                system_prompt += "\n请明确告知用户你无法查看图片，并询问他们是否可以描述图片内容或提供其他相关信息。请始终使用与用户提问相同的语言回答。"
        
        direct = self._direct_answer(
            query, timing_recorder, images=images, system_prompt=system_prompt
        )
        
        return {
            "query": query,
            "answer": direct.get("content", ""),
            "search_hits": [],
            "llm_raw": direct.get("raw"),
            "llm_error": direct.get("error"),
            "control": {
                "search_performed": False,
                "decision": {"needs_search": False, "reason": "image_content_present"},
                "search_mode": "image_content_present",
                "local_docs_present": has_docs,
                "search_allowed": allow_search,
            },
        }

    def _perform_visual_retrieval(self, images: List[Dict[str, str]], timing_recorder: Optional[TimingRecorder]) -> Optional[Dict[str, Any]]:
        if not self.google_api_key or not images:
            return None
        
        start = time.perf_counter()
        try:
            # Use the first image for visual retrieval
            img = images[0]
            b64_content = img.get("base64", "")
            if "," in b64_content:
                b64_content = b64_content.split(",")[1]
            
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_api_key}"
            payload = {
                "requests": [
                    {
                        "image": {
                            "content": b64_content
                        },
                        "features": [
                            {
                                "type": "WEB_DETECTION"
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            responses = result.get("responses", [])
            if responses:
                web_detection = responses[0].get("webDetection", {})
                return web_detection
            return None
            
        except Exception as e:
            print(f"Google Cloud Vision API error: {e}")
            return None
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_tool_call(
                    tool="google_vision",
                    duration_ms=duration_ms,
                    success=True # Simplified
                )

    def _handle_small_talk(
        self,
        query: str,
        max_tokens: int,
        temperature: float,
        has_docs: bool,
        allow_search: bool,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        """Handle small talk queries."""
        direct = self._direct_answer(query, timing_recorder)
        
        return {
            "query": query,
            "answer": direct.get("content", ""),
            "search_hits": [],
            "llm_raw": direct.get("raw"),
            "llm_error": direct.get("error"),
            "control": {
                "search_performed": False,
                "decision": {"needs_search": False, "reason": "small_talk_heuristic"},
                "search_mode": "small_talk",
                "local_docs_present": has_docs,
                "search_allowed": allow_search,
            },
        }

    def _handle_local_only(
        self,
        query: str,
        snapshot: Optional[tuple],
        has_docs: bool,
        num_retrieved_docs: int,
        max_tokens: int,
        temperature: float,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        """Handle local-only queries (search disabled)."""
        if not has_docs:
            direct = self._direct_answer(query, timing_recorder)
            return {
                "query": query,
                "answer": direct.get("content", ""),
                "search_hits": [],
                "llm_raw": direct.get("raw"),
                "llm_error": direct.get("error"),
                "control": {
                    "search_performed": False,
                    "decision": {"needs_search": False, "reason": "search_disabled"},
                    "search_mode": "direct_llm",
                    "local_docs_present": False,
                    "search_allowed": False,
                },
            }
        
        pipeline = self._get_local_rag(snapshot)
        if not pipeline:
            direct = self._direct_answer(query, timing_recorder)
            return {
                "query": query,
                "answer": direct.get("content", ""),
                "search_hits": [],
                "llm_raw": direct.get("raw"),
                "llm_error": direct.get("error"),
                "control": {
                    "search_performed": False,
                    "decision": {"needs_search": False, "reason": "search_disabled"},
                    "search_mode": "direct_llm",
                    "local_docs_present": True,
                    "search_allowed": False,
                },
            }
        
        result = pipeline.answer(
            query,
            num_retrieved_docs=num_retrieved_docs,
            max_tokens=max_tokens,
            temperature=temperature,
            timing_recorder=timing_recorder,
        )
        
        result["control"] = {
            "search_performed": False,
            "decision": {"needs_search": False, "reason": "search_disabled"},
            "search_mode": "local_rag",
            "local_docs_present": True,
            "search_allowed": False,
        }
        
        return result

    def _handle_domain_api(
        self,
        query: str,
        domain: str,
        sources: List[Dict[str, Any]],
        enhanced_query: str,
        domain_api_result: Dict[str, Any],
        has_docs: bool,
        allow_search: bool,
        force_search: bool,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        """Handle domain-specific API responses."""
        answer = domain_api_result.get("answer", "")
        
        # Enhance with LLM if data available
        domain_data = domain_api_result.get("data")
        if domain_data:
            enhanced = self._enhance_domain_answer(
                query, domain, domain_data, timing_recorder
            )
            if enhanced.get("content"):
                answer = enhanced["content"]
        
        return {
            "query": query,
            "answer": answer,
            "search_hits": [],
            "domain_data": domain_data,
            "llm_raw": None,
            "control": {
                "search_performed": False,
                "decision": {"needs_search": False, "reason": f"domain_api_{domain}"},
                "search_mode": "domain_api",
                "domain": domain,
                "selected_sources": sources,
                "enhanced_query": enhanced_query,
                "local_docs_present": has_docs,
                "search_allowed": allow_search,
                "force_search_enabled": force_search,
            },
        }

    def _enhance_domain_answer(
        self,
        query: str,
        domain: str,
        domain_data: Any,  # Can be Dict or List for finance
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        """Enhance domain API answer with LLM."""
        prompts = {}
        
        # Handle weather domain - only if domain_data is a dict
        if domain == "weather" and isinstance(domain_data, dict):
            prompts["weather"] = (
                "你是天气助手。根据实时数据，给出自然、丰富的回复，包括当前状况、穿衣/出行建议。",
                f"位置：{domain_data.get('location', {}).get('formatted_address', '未知')}\n"
                f"概况：{domain_data.get('weatherCondition', {}).get('description', {}).get('text', '未知')}\n"
                f"温度：{domain_data.get('temperature', {}).get('degrees', '未知')}°C"
            )
        elif domain == "weather":
            # Fallback for weather when data is not a dict
            prompts["weather"] = (
                "你是天气助手。根据提供的天气数据，给出自然、丰富的回复。",
                f"天气数据：{json.dumps(domain_data, ensure_ascii=False, indent=2)}"
            )
        
        # Handle finance domain - works with both dict and list
        if domain == "finance":
            prompts["finance"] = (
                "你是金融助手。根据提供的股票数据，分析价格走势和表现。如果包含多只股票，请进行对比。",
                json.dumps(domain_data, ensure_ascii=False, indent=2)
            )
        
        if domain not in prompts:
            return {"content": ""}
        
        system_prompt, data_summary = prompts[domain]
        user_prompt = f"查询：{query}\n数据：\n{data_summary}\n生成回复："
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        start = time.perf_counter()
        try:
            response = self.llm.invoke(messages, max_tokens=300, temperature=0.3)
            return {"content": response.content if hasattr(response, 'content') else str(response)}
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_llm_call(
                    label=f"domain_enhance_{domain}",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm, "provider", None),
                    model=getattr(self.llm, "model_name", None),
                )

    def _handle_search_unavailable(
        self,
        query: str,
        snapshot: Optional[tuple],
        has_docs: bool,
        num_retrieved_docs: int,
        max_tokens: int,
        temperature: float,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        """Handle case where search is requested but unavailable."""
        result = self._handle_local_only(
            query, snapshot, has_docs, num_retrieved_docs,
            max_tokens, temperature, timing_recorder
        )
        result["control"]["search_mode"] = "search_unavailable"
        result["control"]["search_allowed"] = True
        return result

    def _build_direct_response(
        self,
        query: str,
        answer: str,
        decision: Dict[str, Any],
        has_docs: bool,
        allow_search: bool,
        llm_raw: Any = None,
        llm_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build response for direct answer."""
        return {
            "query": query,
            "answer": answer,
            "search_hits": [],
            "llm_raw": llm_raw,
            "llm_error": llm_error,
            "control": {
                "search_performed": False,
                "decision": decision,
                "search_mode": "direct_llm",
                "local_docs_present": has_docs,
                "search_allowed": allow_search,
            },
        }

    def _finalize_response(
        self,
        result: Dict[str, Any],
        timing_recorder: TimingRecorder,
    ) -> Dict[str, Any]:
        """Finalize response with timing and metadata."""
        control = result.get("control", {})
        
        # Add search source metadata
        control.setdefault("search_sources_requested", self.requested_search_sources)
        control.setdefault("search_sources_active", self.active_search_sources)
        control.setdefault("search_sources_configured", self.configured_search_sources)
        if self.missing_search_sources:
            control.setdefault("search_sources_missing", self.missing_search_sources)
        
        result["control"] = control
        
        # Add timing information
        if timing_recorder.enabled:
            timing_recorder.stop()
            timing_payload = timing_recorder.to_dict()
            if timing_payload:
                domain = control.get("domain", "")
                timing_payload["领域智能类型"] = domain if domain and domain.lower() != "general" else "无"
                result["response_times"] = timing_payload
        
        return result


# Factory function
def create_langchain_orchestrator(
    config: Optional[Dict[str, Any]] = None,
    llm: Optional[BaseChatModel] = None,
    search_client: Optional[SearchClient] = None,
    **kwargs: Any,
) -> LangChainOrchestrator:
    """Create a LangChain orchestrator from configuration.
    
    Args:
        config: Configuration dictionary (loaded from config.json if not provided)
        llm: LangChain chat model (created from config if not provided)
        search_client: Search client for web search
        **kwargs: Additional arguments passed to LangChainOrchestrator
    
    Returns:
        Configured LangChainOrchestrator instance
    """
    if config is None:
        import json
        import os
        config_path = os.getenv("NLP_CONFIG_PATH", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
    
    if llm is None:
        from langchain.langchain_llm import create_chat_model
        llm = create_chat_model(config=config)
    
    # Create classifier LLM if configured
    classifier_llm = None
    classifier_cfg = config.get("domainClassifier", {})
    if classifier_cfg.get("enabled", True):
        provider = classifier_cfg.get("provider") or classifier_cfg.get("model")
        if provider:
            from langchain.langchain_llm import create_chat_model
            classifier_llm = create_chat_model(provider=provider, config=config)
    
    # Create routing LLM if configured
    routing_llm = None
    routing_cfg = config.get("routingAndKeywords", {})
    if routing_cfg.get("enabled", True):
        provider = routing_cfg.get("provider") or routing_cfg.get("model")
        if provider:
            from langchain.langchain_llm import create_chat_model
            routing_llm = create_chat_model(provider=provider, config=config)
    
    return LangChainOrchestrator(
        llm=llm,
        search_client=search_client,
        classifier_llm=classifier_llm,
        routing_llm=routing_llm,
        google_api_key=config.get("googleSearch", {}).get("api_key") or config.get("GOOGLE_API_KEY"),
        sportsdb_api_key=config.get("SPORTSDB_API_KEY"),
        apisports_api_key=config.get("APISPORTS_KEY"),
        show_timings=kwargs.pop("show_timings", False),
        **kwargs,
    )
