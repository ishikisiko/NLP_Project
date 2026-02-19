from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import requests


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchClient:
    """Abstract base class for search providers."""

    source_id: str = "generic"
    display_name: str = "Search"

    def __init__(self) -> None:
        self._last_timings: List[Dict[str, Any]] = []

    def _reset_timings(self) -> None:
        self._last_timings = []

    def _append_timing(self, payload: Dict[str, Any]) -> None:
        self._last_timings.append(payload)

    def get_last_timings(self) -> List[Dict[str, Any]]:
        return list(self._last_timings)

    def _is_valid_search_result(self, title: str, url: str, snippet: str) -> bool:
        """Filter out invalid or irrelevant search results."""
        # Skip if all fields are empty
        if not (title or url or snippet):
            return False
        
        # Common patterns for invalid search result pages
        invalid_patterns = [
            r"^search results?(\s+for:?|\s*\|)",  # "Search results for:", "Search results |"
            r"^untitled\s*$",  # "Untitled"
            r"^search\s*\|\s*",  # "Search | xxx"
            r"^sorry,?\s+(no|we)",  # "Sorry, no results", "Sorry, we"
            r"^no\s+results?\s+(found|available)",  # "No results found"
            r"^\d+\s+results?$",  # "10 results"
            r"^page\s+not\s+found",  # "Page not found"
            r"^error\s+\d+",  # "Error 404"
            r"^buletin$", # Specific garbage pattern observed
        ]
        
        title_lower = title.lower()
        for pattern in invalid_patterns:
            if re.match(pattern, title_lower, re.IGNORECASE):
                return False
        
        # Skip URLs that are clearly search engine pages
        invalid_url_patterns = [
            r"/search\?",  # Search query pages
            r"/results\?",  # Results pages
            r"google\.com/search",
            r"bing\.com/search",
            r"yahoo\.com/search",
        ]
        
        url_lower = url.lower()
        for pattern in invalid_url_patterns:
            if re.search(pattern, url_lower):
                return False
        
        return True

    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> List[SearchHit]:
        raise NotImplementedError


class SerpAPISearchClient(SearchClient):
    """Simple wrapper around the SerpAPI search endpoint."""

    source_id = "serp"
    display_name = "SerpAPI"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://serpapi.com/search.json",
        timeout: int = 15,
        engine: str = "google",
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("SerpAPI API key is required.")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.engine = engine

    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> List[SearchHit]:
        self._reset_timings()
        start = time.perf_counter()
        error_message: Optional[str] = None
        try:
            limit = max(1, int(per_source_limit or num_results))
            params = {
                "engine": self.engine,
                "q": query,
                "num": limit,
                "api_key": self.api_key,
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                error_message = str(exc)
                raise RuntimeError(f"SerpAPI search failed: {exc}") from exc

            payload = response.json()
            organic_results = payload.get("organic_results") or []

            hits: List[SearchHit] = []
            for entry in organic_results[:limit]:
                title = entry.get("title") or ""
                link = entry.get("link") or entry.get("url") or ""
                snippet = entry.get("snippet") or entry.get("snippet_highlighted_words") or ""
                if isinstance(snippet, list):
                    snippet = " ".join(snippet)

                if title or link or snippet:
                    if self._is_valid_search_result(title.strip(), link.strip(), snippet.strip()):
                        hits.append(
                            SearchHit(
                                title=title.strip(),
                                url=link.strip(),
                                snippet=snippet.strip(),
                            )
                        )
            return hits
        except Exception as exc:
            if error_message is None:
                error_message = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            timing_payload: Dict[str, Any] = {
                "source": self.source_id,
                "label": self.display_name,
                "duration_ms": round(duration_ms, 2),
            }
            if error_message:
                timing_payload["error"] = error_message
            self._append_timing(timing_payload)


class FallbackSearchClient(SearchClient):
    """Fallback client for environments without a search API key."""

    def __init__(self, static_results: Optional[List[SearchHit]] = None) -> None:
        super().__init__()
        self.static_results = static_results or []

    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> List[SearchHit]:
        self._reset_timings()
        start = time.perf_counter()
        error_message: Optional[str] = None
        try:
            if not self.static_results:
                raise RuntimeError("No search backend configured. Provide a search API key or static results.")
            limit = max(1, int(per_source_limit or num_results))
            return self.static_results[:limit]
        except Exception as exc:
            error_message = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            payload = {
                "source": self.source_id,
                "label": self.display_name,
                "duration_ms": round(duration_ms, 2),
            }
            if error_message:
                payload["error"] = error_message
            self._append_timing(payload)





class GoogleSearchClient(SearchClient):
    """Client for Google Custom Search JSON API."""

    source_id = "google"
    display_name = "Google Search"

    def __init__(
        self,
        api_key: str,
        cx: str,
        *,
        base_url: str = "https://www.googleapis.com/customsearch/v1",
        timeout: int = 15,
        gl: Optional[str] = None,
        lr: Optional[str] = None,
        safe: Optional[str] = "medium",
        date_restrict: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("Google API key is required.")
        if not cx:
            raise ValueError("Google Custom Search Engine ID (cx) is required.")
        self.api_key = api_key
        self.cx = cx
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.gl = (gl or "").strip() or None  # Geolocation
        self.lr = (lr or "").strip() or None  # Language restrict
        self.safe = (safe or "").strip().lower() or None  # Safe search
        self.date_restrict = (date_restrict or "").strip() or None  # Time range filter (e.g., d3, w1, m2, y1)



    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> List[SearchHit]:
        self._reset_timings()
        start = time.perf_counter()
        error_message: Optional[str] = None
        try:
            limit = max(1, min(int(per_source_limit or num_results), 10))  # Google API max is 10 per request
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": limit,
            }
            if self.gl:
                params["gl"] = self.gl
            if self.lr:
                params["lr"] = self.lr
            if self.safe:
                params["safe"] = self.safe
            # Allow callers to specify freshness (e.g., "d1", "w1"); map to dateRestrict if unset
            effective_freshness = (freshness or "").strip() or None
            if effective_freshness:
                params.setdefault("dateRestrict", effective_freshness)
            # Use runtime date_restrict if provided, otherwise use instance default
            effective_date_restrict = date_restrict or self.date_restrict
            if effective_date_restrict:
                params["dateRestrict"] = effective_date_restrict

            try:
                response = requests.get(self.base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                error_message = str(exc)
                raise RuntimeError(f"Google Search failed: {exc}") from exc

            try:
                payload = response.json()
            except ValueError as exc:
                error_message = str(exc)
                raise RuntimeError("Google Search returned non-JSON payload") from exc

            items = payload.get("items") or []
            hits: List[SearchHit] = []
            for entry in items[:limit]:
                if not isinstance(entry, dict):
                    continue
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                snippet = (entry.get("snippet") or "").strip()

                # Filter out invalid search results
                if self._is_valid_search_result(title, link, snippet):
                    hits.append(
                        SearchHit(
                            title=title,
                            url=link,
                            snippet=snippet,
                        )
                    )
            return hits
        except Exception as exc:
            if error_message is None:
                error_message = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            payload = {
                "source": self.source_id,
                "label": self.display_name,
                "duration_ms": round(duration_ms, 2),
            }
            if error_message:
                payload["error"] = error_message
            self._append_timing(payload)


class CombinedSearchClient(SearchClient):
    """Fan out to multiple search clients concurrently and merge unique hits."""

    def __init__(self, clients: List[SearchClient]) -> None:
        super().__init__()
        if not clients:
            raise ValueError("At least one search client is required.")
        self.clients = clients
        self._last_errors: List[Dict[str, str]] = []
        self.active_sources: List[str] = [getattr(client, "source_id", type(client).__name__.lower()) for client in clients]
        self.active_source_labels: List[str] = [getattr(client, "display_name", type(client).__name__) for client in clients]
        self.requested_sources: List[str] = []
        self.missing_requested_sources: List[str] = []
        self.configured_sources: List[str] = sorted(set(self.active_sources))

    def _label_for_client(self, client: SearchClient) -> str:
        name = getattr(client, "display_name", None)
        return name or type(client).__name__

    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> List[SearchHit]:
        self._reset_timings()
        hits: List[SearchHit] = []
        self._last_errors = []

        total_limit = max(1, int(num_results))
        per_source = max(1, int(per_source_limit or total_limit))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as pool:
            future_map: Dict[concurrent.futures.Future, SearchClient] = {}
            start_times: Dict[concurrent.futures.Future, float] = {}
            for client in self.clients:
                future = pool.submit(
                    client.search,
                    query,
                    num_results=per_source,
                    per_source_limit=per_source,
                    freshness=freshness,
                    date_restrict=date_restrict,
                )
                future_map[future] = client
                start_times[future] = time.perf_counter()

            for future in concurrent.futures.as_completed(future_map):
                client = future_map[future]
                label = self._label_for_client(client)
                duration_ms = round((time.perf_counter() - start_times.get(future, 0.0)) * 1000, 2)
                timing_payload = {
                    "source": getattr(client, "source_id", type(client).__name__.lower()),
                    "label": label,
                    "duration_ms": duration_ms,
                }
                try:
                    chunk = future.result() or []
                except Exception as exc:
                    self._last_errors.append({"source": label, "error": str(exc)})
                    timing_payload["error"] = str(exc)
                    self._append_timing(timing_payload)
                    continue

                self._append_timing(timing_payload)
                hits.extend(chunk)

        deduped: List[SearchHit] = []
        seen: Set[str] = set()
        for hit in hits:
            key = (hit.url or f"{hit.title}|{hit.snippet}").strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= total_limit:
                break

        # 如果查询包含时间变化相关关键词，应用时间变化领域的特殊过滤和排序
        if self._is_temporal_change_query(query):
            deduped = self._filter_and_rank_temporal_change_results(deduped, query)

        return deduped

    def get_last_errors(self) -> List[Dict[str, str]]:
        return list(self._last_errors)
    
    def _is_temporal_change_query(self, query: str) -> bool:
        """检测查询是否与时间变化领域相关"""
        temporal_change_keywords = [
            # 教育排名相关
            "大学", "高校", "学院", "学校", "排名", "QS", "THE", "ARWU", "US News",
            "university", "college", "ranking", "rankings", "education", "higher education",
            "香港中文大學", "香港科技大學", "香港大學", "CUHK", "HKUST", "HKU",
            "香港中文大学", "香港科技大学", "香港大学",
            # 时间变化相关
            "最近10年", "过去10年", "10年", "十年", "历年", "历史", "变化", "趋势", "发展",
            "10 years", "decade", "historical", "trend", "development", "evolution",
            "对比", "比较", "变化趋势", "时间序列", "年度", "逐年",
            "comparison", "compare", "trend over time", "time series", "yearly", "year by year",
            # 其他可能的时间变化查询
            "增长", "下降", "波动", "变化率", "增长率", "涨跌",
            "growth", "decline", "fluctuation", "rate of change", "growth rate", "rise and fall"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in temporal_change_keywords)
    
    def _filter_and_rank_temporal_change_results(self, results: List[SearchHit], query: str) -> List[SearchHit]:
        """针对时间变化领域的搜索结果进行过滤和排序"""
        if not results:
            return results
        
        # 定义权威教育网站域名
        authoritative_domains = [
            "topuniversities.com",  # QS排名官网
            "timeshighereducation.com",  # THE排名官网
            "shanghairanking.com",  # ARWU排名官网
            "usnews.com",  # US News排名官网
            "cuHK.edu.hk",  # 香港中文大学官网
            "hkust.edu.hk",  # 香港科技大学官网
            "hku.hk",  # 香港大学官网
            "edu.hk",  # 香港教育机构域名
            "edu.cn",  # 中国教育机构域名
            "gov.hk",  # 香港政府网站
            "gov.cn",  # 中国政府网站
            "wikipedia.org",  # 维基百科
            "britannica.com"  # 大英百科全书
        ]
        
        # 定义高质量排名相关关键词
        ranking_keywords = [
            "qs world university rankings",
            "the world university rankings",
            "arwu academic ranking",
            "world university rankings",
            "global university rankings",
            "qs排名",
            "the排名",
            "arwu排名",
            "世界大学排名",
            "大学排名对比",
            "university ranking comparison"
        ]
        
        # 计算每个搜索结果的分数
        scored_results = []
        for result in results:
            score = 0
            title_lower = result.title.lower()
            snippet_lower = result.snippet.lower()
            url_lower = result.url.lower()
            
            # 1. 权威域名加分
            for domain in authoritative_domains:
                if domain in url_lower:
                    score += 10
                    break
            
            # 2. 排名关键词匹配加分
            for keyword in ranking_keywords:
                if keyword in title_lower:
                    score += 5
                if keyword in snippet_lower:
                    score += 3
            
            # 3. 查询中的特定大学名称匹配加分
            query_lower = query.lower()
            university_names = [
                "香港中文大學", "香港科技大學", "香港大學", "CUHK", "HKUST", "HKU",
                "香港中文大学", "香港科技大学", "香港大学"
            ]
            for uni in university_names:
                if uni in query_lower and uni in title_lower:
                    score += 8
                if uni in query_lower and uni in snippet_lower:
                    score += 5
            
            # 4. 时间相关性加分（最近的内容优先）
            recent_keywords = ["2024", "2023", "2025", "最新", "latest", "recent"]
            for keyword in recent_keywords:
                if keyword in title_lower or keyword in snippet_lower:
                    score += 2
                    break
            
            # 5. 数据丰富度加分（包含表格、图表等）
            data_keywords = ["table", "chart", "graph", "表格", "图表", "数据", "data"]
            for keyword in data_keywords:
                if keyword in snippet_lower:
                    score += 1
                    break
            
            scored_results.append((result, score))
        
        # 按分数降序排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的结果
        return [result for result, _ in scored_results]


class YouSearchClient(SearchClient):
    """Client for the You.com Search API."""

    source_id = "you"
    display_name = "You.com Search"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.ydc-index.io/v1/search",
        timeout: int = 15,
        country: Optional[str] = None,
        safesearch: Optional[str] = "moderate",
        freshness: Optional[str] = None,
        include_news: bool = True,
        default_count: int = 10,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("You.com API key is required.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.country = (country or "").strip().upper() or None
        self.safesearch = (safesearch or "").strip().lower() or None
        self.freshness = (freshness or "").strip().lower() or None
        self.include_news = include_news
        self.default_count = max(1, min(default_count, 100))
        self.extra_params = extra_params or {}

    def _build_params(self, query: str, num_results: int) -> Dict[str, Any]:
        count_value = max(1, min(max(num_results, self.default_count), 100))
        params: Dict[str, Any] = {"query": query, "count": count_value}
        if self.country:
            params["country"] = self.country
        if self.safesearch:
            params["safesearch"] = self.safesearch
        if self.freshness:
            params["freshness"] = self.freshness
        params.update(self.extra_params)
        return params

    def _extract_section(self, records: Any, limit: int) -> List[SearchHit]:
        hits: List[SearchHit] = []
        if not isinstance(records, list):
            return hits

        for entry in records:
            if not isinstance(entry, dict):
                continue
            title = (entry.get("title") or entry.get("name") or "").strip()
            url = (entry.get("url") or "").strip()
            snippets = entry.get("snippets") or []
            snippet_text = ""
            if isinstance(snippets, list) and snippets:
                snippet_parts = [str(part).strip() for part in snippets if part]
                snippet_text = " ".join(part for part in snippet_parts if part)
            if not snippet_text:
                snippet_text = str(entry.get("description") or "").strip()
            if not snippet_text and entry.get("content"):
                snippet_text = str(entry.get("content")).strip()

            if not (title or url or snippet_text):
                continue

            if self._is_valid_search_result(title, url, snippet_text):
                hits.append(SearchHit(title=title, url=url, snippet=snippet_text))
            if len(hits) >= limit:
                break

        return hits

    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: Optional[int] = None,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
    ) -> List[SearchHit]:
        self._reset_timings()
        start = time.perf_counter()
        error_message: Optional[str] = None
        try:
            effective_limit = max(1, int(per_source_limit or num_results))
            params = self._build_params(query, effective_limit)

            # Use runtime freshness if provided, otherwise use instance default
            effective_freshness = freshness or self.freshness
            if effective_freshness:
                params["freshness"] = effective_freshness
            # You.com API currently does not expose a direct date filter; accept the kwarg for interface parity.
            _ = date_restrict

            headers = {"X-API-Key": self.api_key}

            try:
                response = requests.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                error_message = str(exc)
                raise RuntimeError(f"You.com search failed: {exc}") from exc

            try:
                payload = response.json()
            except ValueError as exc:
                error_message = str(exc)
                raise RuntimeError("You.com search returned non-JSON payload") from exc

            results = payload.get("results") if isinstance(payload, dict) else None
            web_hits = self._extract_section((results or {}).get("web"), effective_limit)
            if len(web_hits) >= effective_limit or not self.include_news:
                return web_hits[:effective_limit]

            remaining = effective_limit - len(web_hits)
            news_hits = self._extract_section((results or {}).get("news"), remaining)
            combined = web_hits + news_hits
            return combined[:effective_limit]
        except Exception as exc:
            if error_message is None:
                error_message = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            payload = {
                "source": self.source_id,
                "label": self.display_name,
                "duration_ms": round(duration_ms, 2),
            }
            if error_message:
                payload["error"] = error_message
            self._append_timing(payload)
