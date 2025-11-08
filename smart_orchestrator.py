from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from api import LLMClient
from hybrid_rag import HybridRAG
from local_rag import LocalRAG
from no_rag_baseline import NoRAGBaseline
from rerank import BaseReranker
from search import SearchClient
from source_selector import IntelligentSourceSelector


class SmartSearchOrchestrator:
    """Coordinator that decides whether search is necessary before answering."""

    DECISION_SYSTEM_PROMPT = (
        "You are a routing assistant that decides whether a user's question needs fresh "
        "web or document search. Respond strictly in minified JSON."
    )
    KEYWORD_SYSTEM_PROMPT = (
        "You help generate high quality web search keywords. Respond strictly in JSON."
    )
    DIRECT_ANSWER_SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Answer clearly based on your existing knowledge."
    )
    SEARCH_SOURCE_LABELS = {
        "serp": "SerpAPI",
        "you": "You.com",
        "mcp": "MCP",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        search_client: Optional[SearchClient],
        *,
        data_path: Optional[str] = None,
        reranker: Optional[BaseReranker] = None,
        min_rerank_score: float = 0.0,
        max_per_domain: int = 1,
        requested_search_sources: Optional[List[str]] = None,
        active_search_sources: Optional[List[str]] = None,
        active_search_source_labels: Optional[List[str]] = None,
        missing_search_sources: Optional[List[str]] = None,
        configured_search_sources: Optional[List[str]] = None,
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client
        self.data_path = data_path
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score
        self.max_per_domain = max(1, max_per_domain)
        self._local_pipeline: Optional[LocalRAG] = None
        self._hybrid_pipeline: Optional[HybridRAG] = None
        self._search_pipeline: Optional[NoRAGBaseline] = None
        self._local_signature: Optional[tuple] = None
        self._hybrid_signature: Optional[tuple] = None
        self.source_selector = IntelligentSourceSelector()
        self.requested_search_sources = self._normalize_sources(requested_search_sources)
        raw_active_sources = active_search_sources or getattr(search_client, "active_sources", [])
        self.active_search_sources = self._normalize_sources(raw_active_sources)
        raw_labels = active_search_source_labels or getattr(search_client, "active_source_labels", [])
        self.active_search_source_labels = [str(label).strip() for label in raw_labels if str(label).strip()]
        missing_sources = missing_search_sources or getattr(search_client, "missing_requested_sources", [])
        self.missing_search_sources = self._normalize_sources(missing_sources)
        configured_sources = configured_search_sources or getattr(search_client, "configured_sources", [])
        self.configured_search_sources = self._normalize_sources(configured_sources)

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
    ) -> Dict[str, Any]:
        try:
            total_limit = max(1, int(num_search_results))
        except (TypeError, ValueError):
            total_limit = 10
        try:
            per_source_limit = (
                max(1, int(per_source_search_results))
                if per_source_search_results is not None
                else total_limit
            )
        except (TypeError, ValueError):
            per_source_limit = total_limit

        snapshot = self._snapshot_local_docs()
        has_docs = bool(snapshot)

        if self._looks_like_small_talk(query):
            return self._finalize(
                self._respond_small_talk(
                    query=query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    allow_search=allow_search,
                    has_docs=has_docs,
                )
            )

        if not allow_search:
            return self._finalize(
                self._answer_local_mode(
                    query=query,
                    snapshot=snapshot,
                    has_docs=has_docs,
                    num_retrieved_docs=num_retrieved_docs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    search_allowed=False,
                    decision_reason="search_disabled",
                )
            )
        
        # 在决策之前先进行领域分类
        domain, sources = self.source_selector.select_sources(query)
        enhanced_query = self.source_selector.generate_domain_specific_query(query, domain)

        decision = self._decide(query)
        decision_meta = {
            "needs_search": decision.get("needs_search", True),
            "reason": decision.get("reason"),
            "raw_text": decision.get("raw_text"),
            "llm_warning": decision.get("llm_warning"),
            "llm_error": decision.get("llm_error"),
        }
        decision_raw = decision.get("llm_raw")

        if not decision_meta["needs_search"] and decision.get("direct_answer"):
            return self._finalize(
                self._direct_answer_from_decision(
                    query=query,
                    answer=decision["direct_answer"],
                    decision_meta=decision_meta,
                    decision_raw=decision_raw,
                    has_docs=has_docs,
                    allow_search=True,
                )
            )

        if not decision_meta["needs_search"]:
            return self._finalize(
                self._direct_answer_via_llm(
                    query=query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    has_docs=has_docs,
                    allow_search=True,
                    reason="direct_llm_fallback",
                    decision_meta=decision_meta,
                )
            )

        if not self.search_client:
            return self._finalize(
                self._search_unavailable_response(
                    query=query,
                    snapshot=snapshot,
                    has_docs=has_docs,
                    decision_meta=decision_meta,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_retrieved_docs=num_retrieved_docs,
                )
            )

        keyword_info = self._generate_keywords(query)
        keywords = keyword_info.get("keywords") or [query]
        search_query = " ".join(keywords).strip() or query

        pipeline = self._build_pipeline(
            allow_search=True,
            has_docs=has_docs,
            snapshot=snapshot,
        )
        if pipeline is None:
            return self._finalize(
                self._search_unavailable_response(
                    query=query,
                    snapshot=snapshot,
                    has_docs=has_docs,
                    decision_meta=decision_meta,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_retrieved_docs=num_retrieved_docs,
                )
            )

        pipeline_kwargs: Dict[str, Any] = {
            "search_query": search_query,
            "num_search_results": total_limit,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if isinstance(pipeline, HybridRAG):
            pipeline_kwargs.update(
                {
                    "num_retrieved_docs": num_retrieved_docs,
                    "enable_search": True,
                }
            )
        if isinstance(pipeline, (HybridRAG, NoRAGBaseline)):
            pipeline_kwargs["per_source_limit"] = per_source_limit

        result = pipeline.answer(query, **pipeline_kwargs)
        control_payload = {
            "search_performed": True,
            "decision": decision_meta,
            "keywords": keywords,
            "keyword_generation": {
                "raw_text": keyword_info.get("raw_text"),
                "llm_warning": keyword_info.get("llm_warning"),
                "llm_error": keyword_info.get("llm_error"),
            },
            "hybrid_mode": isinstance(pipeline, HybridRAG),
            "local_docs_present": has_docs,
            "search_allowed": True,
            "domain": domain,
            "selected_sources": sources,
            "enhanced_query": enhanced_query,
            "search_total_limit": total_limit,
            "search_per_source_limit": per_source_limit,
        }

        if result.get("search_warnings"):
            control_payload["search_warnings"] = result["search_warnings"]

        self._merge_control(result, control_payload)
        result.setdefault("search_query", search_query)
        return self._finalize(result)

    def _respond_small_talk(
        self,
        *,
        query: str,
        max_tokens: int,
        temperature: float,
        allow_search: bool,
        has_docs: bool,
    ) -> Dict[str, Any]:
        direct = self.llm_client.chat(
            system_prompt=self.DIRECT_ANSWER_SYSTEM_PROMPT,
            user_prompt=query,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        control = {
            "search_performed": False,
            "decision": {
                "needs_search": False,
                "reason": "small_talk_heuristic",
                "raw_text": None,
                "llm_warning": None,
                "llm_error": None,
            },
            "search_mode": "small_talk",
            "keywords": [],
            "hybrid_mode": False,
            "local_docs_present": has_docs,
            "search_allowed": allow_search,
        }
        return {
            "query": query,
            "answer": direct.get("content") or "",
            "search_hits": [],
            "llm_raw": direct.get("raw"),
            "llm_warning": direct.get("warning"),
            "llm_error": direct.get("error"),
            "control": control,
            "search_query": None,
        }

    def _answer_local_mode(
        self,
        *,
        query: str,
        snapshot: Optional[tuple],
        has_docs: bool,
        num_retrieved_docs: int,
        max_tokens: int,
        temperature: float,
        search_allowed: bool,
        decision_reason: str,
    ) -> Dict[str, Any]:
        decision_meta = {
            "needs_search": False,
            "reason": decision_reason,
            "raw_text": None,
            "llm_warning": None,
            "llm_error": None,
        }

        if not has_docs:
            return self._direct_answer_via_llm(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                has_docs=False,
                allow_search=search_allowed,
                reason=decision_reason,
                decision_meta=decision_meta,
            )

        pipeline = self._ensure_local_pipeline(snapshot)
        if pipeline is None:
            return self._direct_answer_via_llm(
                query=query,
                max_tokens=max_tokens,
                temperature=temperature,
                has_docs=False,
                allow_search=search_allowed,
                reason=decision_reason,
                decision_meta=decision_meta,
            )

        result = pipeline.answer(
            query,
            num_retrieved_docs=num_retrieved_docs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result.setdefault("search_hits", [])
        result["search_query"] = None

        control_payload = {
            "search_performed": False,
            "decision": decision_meta,
            "search_mode": "local_rag",
            "keywords": [],
            "hybrid_mode": False,
            "local_docs_present": True,
            "search_allowed": search_allowed,
        }
        self._merge_control(result, control_payload)
        return result

    def _direct_answer_from_decision(
        self,
        *,
        query: str,
        answer: str,
        decision_meta: Dict[str, Any],
        decision_raw: Optional[Dict[str, Any]],
        has_docs: bool,
        allow_search: bool,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "answer": answer,
            "search_hits": [],
            "llm_raw": decision_raw,
            "llm_warning": decision_meta.get("llm_warning"),
            "llm_error": decision_meta.get("llm_error"),
            "search_query": None,
        }
        control_payload = {
            "search_performed": False,
            "decision": decision_meta,
            "search_mode": "direct_llm",
            "keywords": [],
            "hybrid_mode": False,
            "local_docs_present": has_docs,
            "search_allowed": allow_search,
        }
        self._merge_control(payload, control_payload)
        return payload

    def _direct_answer_via_llm(
        self,
        *,
        query: str,
        max_tokens: int,
        temperature: float,
        has_docs: bool,
        allow_search: bool,
        reason: str,
        decision_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        fallback = self.llm_client.chat(
            system_prompt=self.DIRECT_ANSWER_SYSTEM_PROMPT,
            user_prompt=query,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        meta = (
            dict(decision_meta)
            if decision_meta is not None
            else {
                "needs_search": False,
                "reason": reason,
                "raw_text": None,
                "llm_warning": None,
                "llm_error": None,
            }
        )
        if fallback.get("warning") is not None:
            meta["llm_warning"] = fallback.get("warning")
        if fallback.get("error") is not None:
            meta["llm_error"] = fallback.get("error")

        payload = {
            "query": query,
            "answer": fallback.get("content") or "",
            "search_hits": [],
            "llm_raw": fallback.get("raw"),
            "llm_warning": fallback.get("warning"),
            "llm_error": fallback.get("error"),
            "search_query": None,
        }
        control_payload = {
            "search_performed": False,
            "decision": meta,
            "search_mode": reason,
            "keywords": [],
            "hybrid_mode": False,
            "local_docs_present": has_docs,
            "search_allowed": allow_search,
        }
        self._merge_control(payload, control_payload)
        return payload

    def _search_unavailable_response(
        self,
        *,
        query: str,
        snapshot: Optional[tuple],
        has_docs: bool,
        decision_meta: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        num_retrieved_docs: int,
    ) -> Dict[str, Any]:
        result = self._answer_local_mode(
            query=query,
            snapshot=snapshot,
            has_docs=has_docs,
            num_retrieved_docs=num_retrieved_docs,
            max_tokens=max_tokens,
            temperature=temperature,
            search_allowed=True,
            decision_reason="search_unavailable",
        )
        self._merge_control(
            result,
            {
                "decision": decision_meta,
                "search_mode": "search_unavailable",
                "search_allowed": True,
            },
        )
        return result

    def _build_pipeline(
        self,
        *,
        allow_search: bool,
        has_docs: bool,
        snapshot: Optional[tuple],
    ) -> Optional[NoRAGBaseline | HybridRAG | LocalRAG]:
        if not allow_search:
            return self._ensure_local_pipeline(snapshot)

        if has_docs:
            pipeline = self._ensure_hybrid_pipeline(snapshot)
            if pipeline is not None:
                return pipeline

        return self._ensure_search_pipeline()

    def _ensure_local_pipeline(self, snapshot: Optional[tuple]) -> Optional[LocalRAG]:
        if not self.data_path or snapshot is None:
            self._local_pipeline = None
            self._local_signature = None
            return None

        if self._local_pipeline is None or self._local_signature != snapshot:
            try:
                self._local_pipeline = LocalRAG(
                    llm_client=self.llm_client,
                    data_path=self.data_path,
                )
                self._local_signature = snapshot
            except ValueError:
                self._local_pipeline = None
                self._local_signature = None
        return self._local_pipeline

    def _ensure_hybrid_pipeline(self, snapshot: Optional[tuple]) -> Optional[HybridRAG]:
        if not self.data_path or snapshot is None or not self.search_client:
            return None

        if self._hybrid_pipeline is None or self._hybrid_signature != snapshot:
            self._hybrid_pipeline = HybridRAG(
                llm_client=self.llm_client,
                search_client=self.search_client,
                data_path=self.data_path,
                reranker=self.reranker,
                min_rerank_score=self.min_rerank_score,
                max_per_domain=self.max_per_domain,
            )
            self._hybrid_signature = snapshot
        return self._hybrid_pipeline

    def _ensure_search_pipeline(self) -> Optional[NoRAGBaseline]:
        if not self.search_client:
            return None

        if self._search_pipeline is None:
            self._search_pipeline = NoRAGBaseline(
                llm_client=self.llm_client,
                search_client=self.search_client,
                reranker=self.reranker,
                min_rerank_score=self.min_rerank_score,
                max_per_domain=self.max_per_domain,
            )
        return self._search_pipeline

    def _snapshot_local_docs(self) -> Optional[tuple]:
        if not self.data_path or not os.path.isdir(self.data_path):
            return None

        records = []
        for root, _, files in os.walk(self.data_path):
            for name in files:
                lowered = name.lower()
                if lowered.endswith((".txt", ".md", ".pdf")):
                    full_path = os.path.join(root, name)
                    try:
                        records.append((full_path, os.path.getmtime(full_path)))
                    except OSError:
                        continue
        return tuple(sorted(records))

    @staticmethod
    def _merge_control(target: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if "control" in target and isinstance(target["control"], dict):
            target["control"].update(payload)
        else:
            target["control"] = payload

    @staticmethod
    def _normalize_sources(sources: Optional[List[str]]) -> List[str]:
        normalized: List[str] = []
        if not sources:
            return normalized
        for item in sources:
            if item is None:
                continue
            token = str(item).strip().lower()
            if not token or token in normalized:
                continue
            normalized.append(token)
        return normalized

    def _attach_search_source_metadata(self, control: Dict[str, Any]) -> None:
        if "search_sources_requested" not in control:
            control["search_sources_requested"] = list(self.requested_search_sources)
        if "search_sources_active" not in control:
            control["search_sources_active"] = list(self.active_search_sources)
        if self.active_search_source_labels and "search_sources_active_labels" not in control:
            control["search_sources_active_labels"] = list(self.active_search_source_labels)
        if self.configured_search_sources and "search_sources_configured" not in control:
            control["search_sources_configured"] = list(self.configured_search_sources)
        if self.missing_search_sources and "search_sources_missing" not in control:
            control["search_sources_missing"] = list(self.missing_search_sources)

    def _apply_search_source_warnings(self, result: Dict[str, Any]) -> None:
        if not self.missing_search_sources:
            return
        labels = [self.SEARCH_SOURCE_LABELS.get(src, src) for src in self.missing_search_sources]
        warning = "部分搜索源不可用: " + ", ".join(labels)
        existing = result.get("search_warnings")
        if existing is None:
            result["search_warnings"] = [warning]
            return
        if isinstance(existing, list):
            if warning not in existing:
                existing.append(warning)
            return
        if existing != warning:
            result["search_warnings"] = [existing, warning]

    def _finalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        control = result.get("control")
        if not isinstance(control, dict):
            control = {}
            result["control"] = control
        self._attach_search_source_metadata(control)
        self._apply_search_source_warnings(result)
        return result

    def _decide(self, query: str) -> Dict[str, Any]:
        try:
            response = self.llm_client.chat(
                system_prompt=self.DECISION_SYSTEM_PROMPT,
                user_prompt=self._decision_prompt(query),
                max_tokens=400,
                temperature=0.0,
            )

            content = response.get("content") or ""
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            
            payload = {
                "needs_search": True,
                "reason": None,
                "direct_answer": None,
                "raw_text": content[:100] if content else None,
                "llm_raw": response.get("raw"),
                "llm_warning": str(response.get("warning")) if response.get("warning") else None,
                "llm_error": str(response.get("error")) if response.get("error") else None,
            }

            if response.get("error"):
                payload["reason"] = "decision_llm_error"
                return payload

            parsed = self._extract_json_object(content)
            if not parsed:
                payload["reason"] = "decision_parse_error"
                return payload

            needs_search = self._coerce_bool(parsed.get("needs_search"))
            payload["needs_search"] = needs_search
            reason = parsed.get("reason")
            payload["reason"] = str(reason)[:100] if reason else None
            direct_answer = parsed.get("answer") if not needs_search else None
            if direct_answer:
                payload["direct_answer"] = str(direct_answer).strip()[:1000]
            return payload
            
        except Exception as e:
            return {
                "needs_search": True,
                "reason": f"Error in _decide: {str(e)[:100]}",
                "direct_answer": None,
                "raw_text": None,
                "llm_raw": None,
                "llm_warning": None,
                "llm_error": str(e)[:100],
            }

    @staticmethod
    def _looks_like_small_talk(query: str) -> bool:
        stripped = (query or "").strip()
        if not stripped:
            return True

        lowered = stripped.lower()
        english_small_talk = {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "good morning",
            "good night",
            "bye",
            "goodbye",
            "see you",
        }
        chinese_small_talk = {
            "你好",
            "您好",
            "嗨",
            "谢谢",
            "感谢",
            "早上好",
            "晚上好",
            "晚安",
            "再见",
            "拜拜",
            "哈囉",
            "謝謝",
            "感謝",
            "早安",
            "再見",
            "掰掰",
        }

        if lowered in english_small_talk or stripped in chinese_small_talk:
            return True

        substring_triggers = [
            "你好",
            "您好",
            "嗨",
            "哈喽",
            "拜拜",
            "谢谢",
            "感谢",
            "哈囉",
            "掰掰",
            "謝謝",
            "感謝",
            "hello",
            "hi",
            "bye",
        ]
        lowered_full = stripped.lower()
        if any(token in lowered_full for token in substring_triggers):
            return True

        return False

    def _generate_keywords(self, query: str) -> Dict[str, Any]:
        try:
            response = self.llm_client.chat(
                system_prompt=self.KEYWORD_SYSTEM_PROMPT,
                user_prompt=self._keyword_prompt(query),
                max_tokens=300,
                temperature=0.2,
            )

            content = response.get("content") or ""
            # Ensure content is a string
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            
            payload: Dict[str, Any] = {
                "keywords": [],
                "raw_text": content[:200] if content else None,
                "llm_warning": str(response.get("warning")) if response.get("warning") else None,
                "llm_error": str(response.get("error")) if response.get("error") else None,
            }

            if response.get("error"):
                return payload

            parsed = self._extract_json_object(content)
            if not parsed:
                return payload

            keywords: List[str] = []
            raw_keywords = parsed.get("keywords")

            if isinstance(raw_keywords, list):
                for item in raw_keywords:
                    if isinstance(item, str):
                        cleaned = item.strip()
                        if cleaned:
                            keywords.append(str(cleaned)[:50])
            elif isinstance(raw_keywords, str):
                cleaned = raw_keywords.strip()
                if cleaned:
                    keywords.extend([str(part.strip())[:50] for part in cleaned.split(";") if part.strip()])

            payload["keywords"] = [str(k)[:50] for k in keywords[:10]]
            return payload
            
        except Exception as e:
            return {
                "keywords": [],
                "raw_text": None,
                "llm_warning": None,
                "llm_error": f"Error in _generate_keywords: {str(e)[:100]}",
            }

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"true", "yes", "需要", "y", "1"}
        return False

    @classmethod
    def _decision_prompt(cls, query: str) -> str:
        return (
            "请判断下述用户问题是否需要实时搜索。"
            "如果无需搜索，请直接在answer字段内给出最终答案。"
            "如果需要搜索，将answer设为空字符串。\n\n"
            "输出严格的JSON，形如{\"needs_search\": bool, \"reason\": string, \"answer\": string}.\n"
            "用户问题:\n" + query
        )

    @classmethod
    def _keyword_prompt(cls, query: str) -> str:
        return (
            "请为以下问题生成不超过4个高质量的搜索关键词或短语，"
            "以数组形式返回JSON，例如{\"keywords\": [\"关键词1\", \"关键词2\"]}。"
            "关键词应覆盖查询核心信息。\n\n"
            "用户问题:\n" + query
        )
