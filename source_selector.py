import json
import os
import re
import time
from typing import Dict, List, Tuple, Any, Optional

import requests

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
                    "name": "Alpha Vantage",
                    "url": "https://www.alphavantage.co/query",
                    "type": "rest_api",
                    "description": "免费股票和金融市场数据"
                },
                {
                    "name": "Yahoo Finance",
                    "url": "https://yfapi.net/v6/finance/quote",
                    "type": "rest_api", 
                    "description": "实时股票行情和数据"  # 修复这里，添加了缺失的引号
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
            "只允许以下标签: weather, transportation, finance, general."
            "输出严格的JSON，例如 {\"domain\": \"weather\"}.\n\n"
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
        if domain not in {"weather", "transportation"}:
            return None

        if not self.google_api_key:
            return {"handled": True, "error": "missing_google_api_key"}

        if domain == "weather":
            return self._handle_weather(query, timing_recorder=timing_recorder)
        return self._handle_transportation(query, timing_recorder=timing_recorder)

    def _handle_weather(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
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

        route_payload = self._call_google_routes(
            origin_geo.get("formatted_address") or parsed["origin"],
            dest_geo.get("formatted_address") or parsed["destination"],
            mode=parsed.get("mode") or "DRIVING",
            timing_recorder=timing_recorder,
        )
        if not route_payload or route_payload.get("error"):
            return {
                "handled": True,
                "error": route_payload.get("error") if route_payload else "routes_request_failed",
                "origin": origin_geo,
                "destination": dest_geo,
            }

        answer = self._format_route_answer(parsed, origin_geo, dest_geo, route_payload)
        return {
            "handled": True,
            "provider": "google",
            "endpoint": f"{self.google_routes_base_url}/directions/v2:computeRoutes",
            "origin": origin_geo,
            "destination": dest_geo,
            "mode": parsed.get("mode") or "DRIVING",
            "data": route_payload,
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

    def _geocode_text(
        self,
        text: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()
        try:
            response = requests.get(
                self.google_geocode_url,
                params={"address": text, "key": self.google_api_key, "language": "zh-CN"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                return None
            geometry = results[0].get("geometry") or {}
            location = geometry.get("location") or {}
            lat = location.get("lat")
            lng = location.get("lng")
            if lat is None or lng is None:
                return None
            return {
                "lat": float(lat),
                "lng": float(lng),
                "formatted_address": results[0].get("formatted_address") or text,
            }
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="google_geocoding",
                    label="Google Geocoding",
                    duration_ms=duration_ms,
                )

    def _call_google_weather(
        self,
        lat: float,
        lng: float,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.google_weather_base_url}/currentConditions:lookup"
        params = {
            "location": f"{lat},{lng}",
            "key": self.google_api_key,
            "languageCode": "zh-CN",
            "unitSystem": "METRIC",
        }
        start = time.perf_counter()
        try:
            response = requests.get(
                url,
                params=params,
                headers={"X-Goog-Api-Key": self.google_api_key},
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
                    source="google_weather",
                    label="Google Weather",
                    duration_ms=duration_ms,
                )

    def _call_google_routes(
        self,
        origin: str,
        destination: str,
        *,
        mode: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.google_routes_base_url}/directions/v2:computeRoutes"
        # Map internal modes to Google Routes API v2 modes
        mode_mapping = {
            "DRIVING": "DRIVE",
            "WALKING": "WALK",
            "BICYCLING": "BICYCLE",
            "TRANSIT": "TRANSIT"
        }
        api_mode = mode_mapping.get(mode.upper(), "DRIVE")

        payload = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": api_mode,
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
            "computeAlternativeRoutes": False,
        }
        start = time.perf_counter()
        try:
            response = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": self.google_api_key,
                    "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.legs,routes.travelAdvisory",
                },
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
                    source="google_routes",
                    label="Google Routes",
                    duration_ms=duration_ms,
                )

    @staticmethod
    def _format_weather_answer(location: str, geo: Dict[str, Any], payload: Dict[str, Any]) -> str:
        current = payload.get("currentConditions") or payload.get("current") or {}
        summary = current.get("summary") or current.get("conditions") or current.get("weatherText")
        temp = current.get("temperature") or current.get("temperatureCelsius") or current.get("tempCelsius")
        humidity = current.get("humidity") or current.get("relativeHumidity")
        wind = current.get("windSpeed") or current.get("windSpeedKph") or current.get("windSpeedMps")

        parts = [f"{geo.get('formatted_address') or location} 当前天气"]
        if summary:
            parts.append(f"概况：{summary}")
        if temp is not None:
            try:
                temp_val = float(temp)
                parts.append(f"气温：{temp_val:.1f}°C")
            except (TypeError, ValueError):
                parts.append(f"气温：{temp}")
        if humidity is not None:
            parts.append(f"湿度：{humidity}%")
        if wind is not None:
            parts.append(f"风速：{wind}")

        if len(parts) == 1:
            parts.append("（未能解析详细字段，请检查 Google Weather API 权限或响应格式）")
        return "；".join(parts)

    @staticmethod
    def _format_route_answer(
        parsed: Dict[str, str],
        origin_geo: Dict[str, Any],
        dest_geo: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        routes = payload.get("routes") or []
        first = routes[0] if routes else {}
        distance_m = first.get("distanceMeters")
        duration = first.get("duration")
        advisory = first.get("travelAdvisory") or {}
        delay = advisory.get("trafficInfo", {}).get("delay")

        origin_label = origin_geo.get("formatted_address") or parsed.get("origin")
        dest_label = dest_geo.get("formatted_address") or parsed.get("destination")

        parts = [f"{origin_label} -> {dest_label}（{parsed.get('mode', 'DRIVING')}）"]
        if distance_m is not None:
            try:
                km = float(distance_m) / 1000.0
                parts.append(f"距离约 {km:.1f} 公里")
            except (TypeError, ValueError):
                parts.append(f"距离：{distance_m}")
        if duration:
            # Google 返回形如 "3600s"
            minutes = ""
            try:
                if isinstance(duration, str) and duration.endswith("s"):
                    seconds = float(duration.rstrip("s"))
                    minutes_val = seconds / 60
                    minutes = f"{minutes_val:.0f} 分钟"
            except (TypeError, ValueError):
                minutes = duration
            parts.append(f"预估耗时 {minutes or duration}")
        if delay:
            try:
                if isinstance(delay, str) and delay.endswith("s"):
                    delay_min = float(delay.rstrip("s")) / 60
                    parts.append(f"拥堵额外耗时约 {delay_min:.0f} 分钟")
                else:
                    parts.append(f"拥堵延迟：{delay}")
            except (TypeError, ValueError):
                parts.append(f"拥堵延迟：{delay}")

        if len(parts) == 1:
            parts.append("未能获取路线详情，请检查 Google Routes 配额或请求参数。")
        return "；".join(parts)

def test_basic_functionality():
    """基础功能测试"""
    selector = IntelligentSourceSelector(use_llm=False)
    
    test_cases = [
        "今天天气怎么样？",
        "北京交通状况",
        "腾讯股票价格",
        "什么是人工智能"
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
