import json
import os
import re
import time
from typing import Dict, List, Tuple, Any, Optional

import requests

import yfinance as yf
from yahoo_fin import stock_info

from api import LLMClient
from timing_utils import TimingRecorder

class IntelligentSourceSelector:
    """智能源选择器 - 带具体API配置的版本"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        use_llm: Optional[bool] = None,
        google_api_key: Optional[str] = None,
        google_weather_base_url: str = "https://weather.googleapis.com/v1",
        google_routes_base_url: str = "https://routes.googleapis.com",
        google_geocode_url: str = "https://maps.googleapis.com/maps/api/geocode/json",
        request_timeout: int = 12,
        finnhub_api_key: Optional[str] = None,
        sportsdb_api_key: Optional[str] = None,
    ):
        # 领域关键词映射
        self.domain_keywords = {
            "weather": [
                "天气", "气温", "温度", "下雨", "下雪", "台风", "暴雨",
                "天氣", "氣溫", "溫度", "颱風",
                "weather", "temperature", "rain", "snow", "typhoon"
            ],
            "transportation": [
                "交通", "公交", "地铁", "拥堵", "路况", "航班", "火车", "高铁",
                "公車", "地鐵", "擁堵", "路況", "航班", "火車", "高鐵",
                "traffic", "bus", "subway", "congestion", "flight", "train"
            ],
            "finance": [
                "股票", "股价", "金融", "汇率", "投资", "基金", "黄金", "原油",
                "股價", "匯率", "投資", "基金", "黃金", "原油",
                "stock", "finance", "exchange rate", "investment", "fund"
            ],
            "sports": [
                "体育", "足球", "篮球", "网球", "比赛", "比分", "NBA", "奥运", "世界杯", "英超",
                "sports", "football", "basketball", "tennis", "match", "score", "NBA", "Olympics", "Premier League"
            ],
            "general": []  # 通用领域，无特定关键词
        }
        
        # 具体的数据源API配置
        self.domain_sources = {
            "weather": [
                {
                    "name": "Google Weather API",
                    "url": "https://weather.googleapis.com/v1/currentConditions:lookup",
                    "type": "rest_api",
                    "description": "Google Cloud 提供的实时天气数据"
                },
                {
                    "name": "Google Geocoding API",
                    "url": "https://maps.googleapis.com/maps/api/geocode/json",
                    "type": "rest_api",
                    "description": "用于将地点名称解析为坐标以便获取天气"
                }
            ],
            "transportation": [
                {
                    "name": "Google Routes Preferred API",
                    "url": "https://routes.googleapis.com/directions/v2:computeRoutes",
                    "type": "rest_api",
                    "description": "支持交通拥堵的路线规划（含实时路况）"
                },
                {
                    "name": "Google Geocoding API",
                    "url": "https://maps.googleapis.com/maps/api/geocode/json",
                    "type": "rest_api",
                    "description": "起点/终点地名解析"
                }
            ],
            "finance": [
                {
                    "name": "yfinance",
                    "type": "python_lib", 
                    "description": "Yahoo Finance Python库 (yfinance)"
                },
                {
                    "name": "yahoo-fin",
                    "type": "python_lib", 
                    "description": "Yahoo Finance Python库 (yahoo-fin)"
                },
                {
                    "name": "Finnhub",
                    "url": "https://finnhub.io/api/v1/quote",
                    "type": "rest_api",
                    "description": "实时股票报价和金融市场数据"
                }
            ],
            "sports": [
                {
                    "name": "TheSportsDB",
                    "url": "https://www.thesportsdb.com/api/v1/json/1/search_all_events.php",
                    "type": "rest_api",
                    "description": "体育赛事、球队和比分数据"
                }
            ],
            "general": [
                {
                    "name": "Google Search API",
                    "url": "https://www.googleapis.com/customsearch/v1",
                    "type": "search_api",
                    "description": "通用网页搜索"
                },
                {
                    "name": "Wikipedia API",
                    "url": "https://en.wikipedia.org/api/rest_v1/page/summary/",
                    "type": "knowledge_api",
                    "description": "知识库数据源"
                }
            ]
        }
        self.llm_client = llm_client
        self.use_llm = use_llm if use_llm is not None else llm_client is not None
        self.google_api_key = (google_api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
        self.google_weather_base_url = google_weather_base_url.rstrip("/")
        self.google_routes_base_url = google_routes_base_url.rstrip("/")
        self.google_geocode_url = google_geocode_url
        self.request_timeout = max(3, int(request_timeout))
        self.finnhub_api_key = (finnhub_api_key or os.getenv("FINNHUB_API_KEY") or "").strip()
        
        self.sportsdb_api_key = (sportsdb_api_key or os.getenv("SPORTSDB_API_KEY") or "123").strip()
    
    def classify_domain(self, query: str, timing_recorder: Optional[TimingRecorder] = None) -> str:
        """分类查询的领域"""
        if self.use_llm and self.llm_client:
            domain = self._classify_with_llm(query, timing_recorder=timing_recorder)
            if domain:
                return domain
        return self._classify_with_keywords(query)

    def _classify_with_keywords(self, query: str) -> str:
        query_lower = query.lower()
        
        # 统计各领域关键词命中数
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            if domain == "general":
                continue
                
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            domain_scores[domain] = score
        
        # 找到最高分的领域
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:  # 至少命中一个关键词
                return best_domain[0]
        
        return "general"

    def _classify_with_llm(self, query: str, timing_recorder: Optional[TimingRecorder] = None) -> Optional[str]:
        allowed = sorted(self.domain_keywords.keys())
        prompt = (
            "你是NLU分类器，请将用户问题归类到固定领域中。"
            "只允许以下标签: weather, transportation, finance, sports, general."
            "输出严格的JSON，例如 {\"domain\": \"sports\"}.\n\n"
            f"用户问题: {query}"
        )
        try:
            response_start = time.perf_counter()
            response = self.llm_client.chat(
                system_prompt="You classify intents into fixed domains.",
                user_prompt=prompt,
                max_tokens=200,
                temperature=0.0,
            )
        except Exception:
            return None
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="domain_classification",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm_client, "provider", None),
                    model=getattr(self.llm_client, "model_id", None),
                    extra={"stage": "source_selector"},
                )

        content = response.get("content")
        if not isinstance(content, str) or not content.strip():
            return None

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed, dict):
            return None

        domain_raw = parsed.get("domain")
        if not isinstance(domain_raw, str):
            return None
        domain = domain_raw.strip().lower()
        return domain if domain in allowed else None
    
    def select_sources(self, query: str, timing_recorder: Optional[TimingRecorder] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """选择数据源 - 返回具体API信息"""
        domain = self.classify_domain(query, timing_recorder=timing_recorder)
        sources = self.domain_sources.get(domain, [
            {
                "name": "Default Search",
                "url": "https://serpapi.com/search",
                "type": "search_api",
                "description": "默认搜索引擎"
            }
        ])
        
        try:
            print(f"query: '{query}'")
            print(f"detected domain: {domain}")
            print("selected sources:")
            for source in sources:
                print(f"   - {source['name']}: {source['url']}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 在不支持UTF-8的环境中静默跳过打印
            pass
        
        return domain, sources

    def generate_domain_specific_query(self, query: str, domain: str) -> str:
        """根据识别出的领域为查询补充上下文关键词"""
        cleaned_query = query.strip()
        domain = (domain or "general").lower()

        if not cleaned_query or domain == "general":
            return cleaned_query

        domain_context = {
            "weather": "current weather forecast humidity wind speed",
            "transportation": "live traffic status transit delays road conditions",
            "finance": "latest market data stock price trend analysis",
            "sports": "latest match scores results standings fixtures sports news",
        }

        supplemental_keywords = " ".join(self.domain_keywords.get(domain, [])[:3])
        enhanced_query = " ".join(
            part for part in [cleaned_query, domain_context.get(domain, ""), supplemental_keywords] if part
        )

        try:
            print(f"🧠 领域增强查询: {enhanced_query}")
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 在不支持UTF-8的环境中静默跳过打印
            pass
        return enhanced_query
    
    def get_source_details(self, domain: str) -> List[Dict[str, Any]]:
        """获取指定领域的详细数据源信息"""
        return self.domain_sources.get(domain, [])

    # === Google Cloud 专用调用 ===
    def fetch_domain_data(
        self,
        query: str,
        domain: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """调用特定领域的Google Cloud API并返回结构化结果"""
        domain = (domain or "").lower().strip()
        if domain not in {"weather", "transportation", "finance", "sports"}:
            return None

        if not self.google_api_key:
            return {"handled": True, "error": "missing_google_api_key"}

        if domain == "weather":
            return self._handle_weather(query, timing_recorder=timing_recorder)
        if domain == "transportation":
            return self._handle_transportation(query, timing_recorder=timing_recorder)
        if domain == "finance":
            return self._handle_finance(query, timing_recorder=timing_recorder)
        if domain == "sports":
            return self._handle_sports(query, timing_recorder=timing_recorder)

    def _handle_weather(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        # 检测预报查询，fallback 搜索
        forecast_keywords = ["明天", "后天", "预报", "forecast", "tomorrow"]
        if any(kw in query for kw in forecast_keywords):
            return {"handled": False, "reason": "forecast_requested_fallback_search"}

        location_hint = self._extract_weather_location(query)
        if not location_hint:
            return {"handled": True, "error": "cannot_parse_location"}

        geocode = self._geocode_text(location_hint, timing_recorder=timing_recorder)
        if not geocode or geocode.get("error"):
            return {
                "handled": True,
                "error": geocode.get("error") if geocode else "geocode_failed",
                "location": location_hint,
            }

        if "中国" in geocode.get("formatted_address", "") or "China" in geocode.get("formatted_address", ""):
            return {"handled": True, "skipped": True, "reason": "china_location_not_supported_by_google_weather", "location": geocode}

        weather_payload = self._call_google_weather(
            geocode["lat"],
            geocode["lng"],
            timing_recorder=timing_recorder,
        )
        if not weather_payload or weather_payload.get("error"):
            return {
                "handled": True,
                "error": weather_payload.get("error") if weather_payload else "weather_request_failed",
                "location": geocode,
            }

        answer = self._format_weather_answer(location_hint, geocode, weather_payload)
        return {
            "handled": True,
            "provider": "google",
            "endpoint": f"{self.google_weather_base_url}/currentConditions:lookup",
            "location": geocode,
            "data": weather_payload,
            "answer": answer,
        }

    def _handle_transportation(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        parsed = self._extract_route(query)
        if not parsed:
            return {"handled": True, "error": "cannot_parse_route"}

        origin_geo = self._geocode_text(parsed["origin"], timing_recorder=timing_recorder)
        dest_geo = self._geocode_text(parsed["destination"], timing_recorder=timing_recorder)
        if (not origin_geo or origin_geo.get("error")) or (not dest_geo or dest_geo.get("error")):
            return {
                "handled": True,
                "error": "geocode_failed",
                "origin": origin_geo or parsed.get("origin"),
                "destination": dest_geo or parsed.get("destination"),
            }

        origin_label = origin_geo.get("formatted_address") or parsed["origin"]
        dest_label = dest_geo.get("formatted_address") or parsed["destination"]

        modes = [
            {"internal": "DRIVING", "api": "DRIVE", "display": "驾车"},
            {"internal": "TRANSIT", "api": "TRANSIT", "display": "公共交通"},
        ]

        routes = []
        answers = []
        for m in modes:
            route_payload = self._call_google_routes(
                origin_label,
                dest_label,
                mode=m["api"],
                timing_recorder=timing_recorder,
            )
            if route_payload and not route_payload.get("error"):
                answer = self._format_route_answer(
                    {"mode": m["internal"]}, origin_geo, dest_geo, route_payload
                )
                routes.append({
                    "mode": m["display"],
                    "data": route_payload,
                    "answer": answer
                })
                answers.append(answer)
            else:
                answers.append(f"{m['display']}：获取失败 ({route_payload.get('error') if route_payload else '未知错误'})")

        combined_answer = f"{origin_label} -> {dest_label}\n" + "\n".join(answers)

        return {
            "handled": True,
            "provider": "google",
            "endpoint": f"{self.google_routes_base_url}/directions/v2:computeRoutes",
            "origin": origin_geo,
            "destination": dest_geo,
            "routes": routes,
            "data": {"routes": [r["data"] for r in routes]},
            "answer": combined_answer,
        }

    def _handle_finance(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        symbol = self._extract_finance_symbol(query)
        if not symbol:
            return {"handled": True, "error": "cannot_parse_symbol"}

        if not self.finnhub_api_key:
            return {"handled": True, "error": "missing_finnhub_api_key"}

        quote = self._query_stock_price(symbol, timing_recorder=timing_recorder)
        if not quote or quote.get("error"):
            return {
                "handled": True,
                "error": quote.get("error") if quote else "finnhub_request_failed",
                "symbol": symbol,
            }

        answer = self._format_finance_answer(symbol, quote)
        return {
            "handled": True,
            "provider": "finnhub",
            "endpoint": "https://finnhub.io/api/v1/quote",
            "symbol": symbol,
            "data": quote,
            "answer": answer,
        }

    def _handle_sports(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        entity = self._extract_sports_entity(query)
        if not entity:
            return {"handled": True, "error": "cannot_parse_sports_entity"}

        if not self.sportsdb_api_key:
            return {"handled": True, "error": "missing_sportsdb_api_key"}

        data = self._call_sportsdb_events(entity, timing_recorder=timing_recorder)
        if not data or data.get("error") or not data.get("events"):
            return {
                "handled": True,
                "error": data.get("error", "no_events_found") if data else "api_failed",
                "entity": entity,
            }

        answer = self._format_sports_answer(entity, data["events"][0])
        return {
            "handled": True,
            "provider": "sportsdb",
            "endpoint": f"https://www.thesportsdb.com/api/v1/json/{self.sportsdb_api_key}/search_all_events.php",
            "entity": entity,
            "data": data,
            "answer": answer,
        }

    def _extract_weather_location(self, query: str) -> str:
        cleaned = query.strip()
        # 移除常见天气关键词，保留地点提示
        for kw in self.domain_keywords.get("weather", []):
            cleaned = cleaned.replace(kw, " ")
        cleaned = re.sub(r"[?？。,.!！]", " ", cleaned)
        cleaned = " ".join(token for token in cleaned.split() if token)
        if cleaned:
            return cleaned

        if self.use_llm and self.llm_client:
            prompt = (
                "从用户问题中提取地理位置，输出JSON格式，例如 {\"location\": \"北京\"}。"
                "如果无法提取，返回空字符串。\n\n用户问题：" + query
            )
            response = self.llm_client.chat(
                system_prompt="You extract a location name or city.",
                user_prompt=prompt,
                max_tokens=150,
                temperature=0.0,
            )
            try:
                payload = json.loads(response.get("content") or "{}")
                location = payload.get("location") or ""
                if isinstance(location, str) and location.strip():
                    return location.strip()
            except Exception:
                pass
        return query

    def _extract_route(self, query: str) -> Optional[Dict[str, str]]:
        # 基础正则：从A到B / from A to B
        match_cn = re.search(r"从(.+?)到(.+)", query)
        if match_cn:
            origin = match_cn.group(1).strip()
            destination = match_cn.group(2).strip()
            if origin and destination:
                return {"origin": origin, "destination": destination, "mode": "DRIVING"}

        match_en = re.search(r"from\s+(.+?)\s+to\s+(.+)", query, flags=re.IGNORECASE)
        if match_en:
            origin = match_en.group(1).strip()
            destination = match_en.group(2).strip()
            if origin and destination:
                return {"origin": origin, "destination": destination, "mode": "DRIVING"}

        if self.use_llm and self.llm_client:
            prompt = (
                "从用户问题里提取出行起点、终点与方式，输出JSON，如："
                "{\"origin\": \"上海\", \"destination\": \"苏州\", \"mode\": \"DRIVING\"}。"
                "mode 取 DRIVING/TRANSIT/WALKING/BICYCLING。提取不到返回空字符串。\n\n"
                f"用户问题：{query}"
            )
            response = self.llm_client.chat(
                system_prompt="You extract travel origin/destination/mode.",
                user_prompt=prompt,
                max_tokens=150,
                temperature=0.0,
            )
            try:
                payload = json.loads(response.get("content") or "{}")
                origin = (payload.get("origin") or "").strip()
                destination = (payload.get("destination") or "").strip()
                mode = (payload.get("mode") or "DRIVING").upper()
                if origin and destination:
                    if mode not in {"DRIVING", "WALKING", "BICYCLING", "TRANSIT"}:
                        mode = "DRIVING"
                    return {"origin": origin, "destination": destination, "mode": mode}
            except Exception:
                return None
        return None

    def _extract_finance_symbol(self, query: str) -> str:
        # 简单正则匹配常见股票代码，如 AAPL, TSLA, 600000 等
        match_us = re.search(r'\b([A-Z]{1,5})\b(?=\s*(?:股价|股票|price|stock))', query, re.IGNORECASE)
        if match_us:
            return match_us.group(1).upper()
        
        match_cn = re.search(r'\b([0-9]{6})\b(?=\s*(?:股价|股票))', query)
        if match_cn:
            return match_cn.group(1)
        
        # LLM fallback
        if self.use_llm and self.llm_client:
            prompt = (
                "从用户问题中提取股票代码（美股如AAPL，A股如600000），输出JSON {\"symbol\": \"AAPL\"}。"
                "无法提取返回空字符串。\n\n用户问题：" + query
            )
            response = self.llm_client.chat(
                system_prompt="Extract stock symbol.",
                user_prompt=prompt,
                max_tokens=100,
                temperature=0.0,
            )
            try:
                payload = json.loads(response.get("content") or "{}")
                symbol = payload.get("symbol") or ""
                if isinstance(symbol, str) and re.match(r'^[A-Z]{1,5}$|^[0-9]{6}$', symbol):
                    return symbol
            except Exception:
                pass
        return ""

    def _extract_sports_entity(self, query: str) -> str:
        # 正则匹配常见体育实体
        patterns = [
            r'(?:球队|队|比赛|赛事|vs|对阵)\s*[:：]?\s*([^\s,。？?]+(?:\s+[^\s,。？?]+)*)',
            r'([a-zA-Z]{2,}(?:\s+[a-zA-Z]{2,})?)(?:\s+(?:vs|对|战)\s+[a-zA-Z]{2,})?',
            r'([^\s,。？?]{2,})(?:\s*(?:比赛|score|结果))?'
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 1:
                    return candidate

        # LLM fallback
        if self.use_llm and self.llm_client:
            prompt = (
                "从体育问题中提取核心实体（球队名、赛事名），输出JSON {\"entity\": \"曼联\"}。"
                "无法提取返回空字符串。\n\n用户问题：" + query
            )
            try:
                response = self.llm_client.chat(
                    system_prompt="Extract sports entity like team or event.",
                    user_prompt=prompt,
                    max_tokens=100,
                    temperature=0.0,
                )
                parsed = json.loads(response.get("content") or "{}")
                entity = parsed.get("entity") or ""
                if isinstance(entity, str) and entity.strip():
                    return entity.strip()
            except Exception:
                pass

        # Fallback to first meaningful word
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', query)
        return words[0] if words else ""

    def _call_finnhub_quote(
        self,
        symbol: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        url = "https://finnhub.io/api/v1/quote"
        params = {
            "symbol": symbol,
            "token": self.finnhub_api_key,
        }
        start = time.perf_counter()
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="finnhub_quote",
                    label="Finnhub Quote",
                    duration_ms=duration_ms,
                )

    def _call_yfinance_quote(
        self,
        symbol: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            quote = {
                "c": info.get("currentPrice") or info.get("regularMarketPrice"),
                "h": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "l": info.get("dayLow") or info.get("regularMarketDayLow"),
                "o": info.get("regularMarketOpen"),
                "pc": info.get("regularMarketPreviousClose"),
            }
            quote = {k: v for k, v in quote.items() if v is not None}
            if not quote:
                return {"error": "no_data"}
            return quote
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="yfinance",
                    label="yfinance Quote",
                    duration_ms=duration_ms,
                )

    def _call_yahoo_fin_quote(
        self,
        symbol: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        try:
            c = stock_info.get_live_price(symbol)
            if c is None:
                raise ValueError("No price data")
            return {"c": c}
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="yahoo_fin",
                    label="yahoo_fin Quote",
                    duration_ms=duration_ms,
                )

    def _call_sportsdb_events(
        self,
        entity: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        url = f"https://www.thesportsdb.com/api/v1/json/{self.sportsdb_api_key}/search_all_events.php?e={requests.utils.quote(entity)}"
        start = time.perf_counter()
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="sportsdb_events",
                    label="TheSportsDB Events",
                    duration_ms=duration_ms,
                )

    def _format_sports_answer(self, entity: str, event: Dict) -> str:
        date_event = event.get('dateEvent', '未知日期')
        str_time = event.get('strTime', '未知时间')
        str_home_team = event.get('strHomeTeam', '未知主队')
        str_away_team = event.get('strAwayTeam', '未知客队')
        str_league = event.get('strLeague', '未知联赛')
        str_status = event.get('intHomeScore', 'N/A') + '-' + event.get('intAwayScore', 'N/A') if event.get('intHomeScore') is not None else '未开始'
        return (
            f"{entity} 最新相关赛事：\n"
            f"对阵：{str_home_team} vs {str_away_team}\n"
            f"联赛：{str_league}\n"
            f"比分：{str_status}\n"
            f"时间：{date_event} {str_time}"
        )

def test_basic_functionality():
    """基础功能测试"""
    selector = IntelligentSourceSelector(use_llm=False)
    
    test_cases = [
        "今天天气怎么样？",
        "北京交通状况",
        "腾讯股票价格",
        "曼联最近比赛",
        "什么是人工智能",
        "最新NBA比赛结果",
        "明天的天气预报",
        "从上海到北京的高铁",
        "苹果公司的股价",
        "切尔西对阵曼联的比赛"
    ]
    
    print("✅ 基础功能验证测试")
    print("=" * 40)
    
    try:
        for query in test_cases:
            domain, sources = selector.select_sources(query)
            print(f"query '{query}' -> domain: {domain}, sources: {len(sources)}")
    except (UnicodeEncodeError, UnicodeDecodeError):
        # 在不支持UTF-8的环境中静默跳过打印
        pass
    
    print("\n🎉 基础测试完成！")

if __name__ == "__main__":
    test_basic_functionality()
