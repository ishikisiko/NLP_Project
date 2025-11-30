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
        apisports_api_key: Optional[str] = None,  # 添加缺失参数
    ):
        # 领域关键词映射
        self.domain_keywords = {
            "weather": [
                "天气", "气温", "温度", "下雨", "下雪", "台风", "暴雨",
                "天氣", "氣溫", "溫度", "颱風",
                "weather", "temperature", "rain", "snow", "typhoon",
                "空气质量", "空气污染", "AQI", "PM2.5", "PM10", "雾霾", "空气指数",
                "air quality", "air pollution", "AQI", "PM2.5", "PM10", "smog", "haze",
                "指数", "污染", "pm25", "pm10", "指数", "质量", "aqi", "pm2.5"
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
            "temporal_change": [
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
            ],
            "location": [
                "最近", "附近", "距离", "哪家", "哪里", "在哪", "周边", "旁边",
                "最近的", "附近的", "离", "靠近",
                "距離", "哪裡", "週邊",
                "nearest", "nearby", "closest", "near", "around", "where is",
                "find", "locate", "location", "place", "places"
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
                    "name": "Google Air Quality API",
                    "url": "https://airquality.googleapis.com/v1/currentConditions:lookup",
                    "type": "rest_api",
                    "description": "Google Cloud 提供的实时空气质量数据，包括AQI、PM2.5等污染物信息"
                },
                {
                    "name": "Google Geocoding API",
                    "url": "https://maps.googleapis.com/maps/api/geocode/json",
                    "type": "rest_api",
                    "description": "用于将地点名称解析为坐标以便获取天气和空气质量"
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
            "temporal_change": [
                {
                    "name": "Google Search API",
                    "url": "https://www.googleapis.com/customsearch/v1",
                    "type": "search_api",
                    "description": "时间变化相关数据搜索，包括历史排名、趋势分析等"
                },
                {
                    "name": "Wikipedia API",
                    "url": "https://en.wikipedia.org/api/rest_v1/page/summary/",
                    "type": "knowledge_api",
                    "description": "历史数据和知识库信息"
                },
                {
                    "name": "Google Trends API",
                    "url": "https://trends.googleapis.com/trends/v1/",
                    "type": "rest_api",
                    "description": "获取趋势数据和变化模式"
                }
            ],
            "location": [
                {
                    "name": "Google Places API (Nearby Search)",
                    "url": "https://places.googleapis.com/v1/places:searchNearby",
                    "type": "rest_api",
                    "description": "搜索附近的地点/兴趣点(POI)"
                },
                {
                    "name": "Google Geocoding API",
                    "url": "https://maps.googleapis.com/maps/api/geocode/json",
                    "type": "rest_api",
                    "description": "将地点名称解析为坐标"
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
        self.apisports_api_key = (apisports_api_key or os.getenv("APISPORTS_KEY") or "").strip()
    
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
            "只允许以下标签: weather, transportation, finance, sports, temporal_change, location, general.\n"
            "- location: 用于查找附近地点、最近的商店/餐厅/设施等（如'最近的KFC'、'附近的医院'）\n"
            "- temporal_change: 用于涉及时间变化的查询，如历史排名、趋势分析、年度对比等（如'最近10年排名变化'、'历年数据对比'）\n"
            "输出严格的JSON，例如 {\"domain\": \"location\"}.\n\n"
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
                url = source.get('url', 'N/A')
                print(f"   - {source['name']}: {url}")
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

        # 各领域使用更具体的关键词以获取详细数据
        domain_context = {
            "weather": "current weather forecast humidity wind speed",
            "transportation": "live traffic status transit delays road conditions",
            "finance": "latest market data stock price trend analysis",
            "sports": "box score player stats 球员得分统计 比赛数据",
            "temporal_change": "historical data trend analysis year by year comparison time series",
        }

        # 特殊领域处理：检测是否是求具体数据的查询
        if domain == "sports":
            enhanced_query = self._enhance_sports_query(cleaned_query)
        elif domain == "temporal_change":
            enhanced_query = self._enhance_temporal_change_query(cleaned_query)
        else:
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
    
    def _enhance_sports_query(self, query: str) -> str:
        """增强体育查询以获取详细的比赛和球员数据
        
        策略：添加新闻导向和数据导向的精准关键词，获取最新比赛战报
        """
        query_lower = query.lower()
        
        # NBA队伍关键词映射（中英文）
        nba_teams = {
            "湖人": "Lakers", "勇士": "Warriors", "独行侠": "Mavericks",
            "篮网": "Nets", "凯尔特人": "Celtics", "热火": "Heat",
            "雷霆": "Thunder", "掘金": "Nuggets", "太阳": "Suns",
            "快船": "Clippers", "国王": "Kings", "开拓者": "Blazers",
            "火箭": "Rockets", "马刺": "Spurs", "灰熊": "Grizzlies",
            "鹈鹕": "Pelicans", "森林狼": "Timberwolves", "爵士": "Jazz",
            "雄鹿": "Bucks", "公牛": "Bulls", "骑士": "Cavaliers",
            "活塞": "Pistons", "步行者": "Pacers", "老鹰": "Hawks",
            "黄蜂": "Hornets", "魔术": "Magic", "尼克斯": "Knicks",
            "76人": "76ers", "猛龙": "Raptors", "奇才": "Wizards"
        }
        
        # 检测查询中的NBA球队
        detected_team = None
        detected_team_en = None
        for cn_name, en_name in nba_teams.items():
            if cn_name in query_lower or en_name.lower() in query_lower:
                detected_team = cn_name
                detected_team_en = en_name
                break
        
        # 检测是否是最近比赛查询
        recent_keywords = ["上一场", "最近", "最新", "昨天", "今天", "概况",
                          "last", "latest", "recent", "yesterday", "today"]
        is_recent = any(kw in query_lower for kw in recent_keywords)
        
        # 构建精准增强 - 使用新闻导向关键词获取战报而非赛程
        enhancements = []
        
        if detected_team and is_recent:
            # 最近比赛查询：添加新闻/战报导向关键词
            # 核心策略：
            # 1. "战报" "highlights" 获取比赛新闻而非赛程
            # 2. "得分 统计" "box score" 获取球员数据
            # 3. 添加英文关键词提升搜索质量
            enhancements = [
                f"{detected_team_en} game highlights",  # 英文新闻关键词
                "战报",  # 中文新闻关键词
                "得分统计",  # 球员数据关键词
                "box score"  # 英文数据关键词
            ]
        elif detected_team:
            # 一般球队查询
            enhancements = [detected_team_en, "比赛战报", "score highlights"]
        elif is_recent:
            # 最近比赛但未指定球队
            enhancements = ["战报", "得分统计", "highlights box score"]
        else:
            # 通用体育查询
            enhancements = ["最新战报", "results highlights"]
        
        # 组合：原始查询 + 精准关键词
        enhanced = query + " " + " ".join(enhancements)
        return enhanced
    
    def _enhance_temporal_change_query(self, query: str) -> str:
        """增强时间变化查询以获取详细的历史数据和趋势分析
        
        策略：添加时间变化相关的关键词，获取历史数据和趋势分析
        """
        query_lower = query.lower()
        
        # 香港大学关键词映射（中英文）
        hk_universities = {
            "香港中文大學": "CUHK", "香港科技大學": "HKUST", "香港大學": "HKU",
            "香港中文大学": "CUHK", "香港科技大学": "HKUST", "香港大学": "HKU",
            "中文大学": "CUHK", "科技大学": "HKUST", "香港大学": "HKU"
        }
        
        # 检测查询中的香港大学
        detected_universities = []
        for cn_name, en_name in hk_universities.items():
            if cn_name in query_lower or en_name.lower() in query_lower:
                detected_universities.append((cn_name, en_name))
        
        # 检测是否是排名查询
        ranking_keywords = ["排名", "rankings", "对比", "comparison", "比较", "compare"]
        is_ranking_query = any(kw in query_lower for kw in ranking_keywords)
        
        # 检测是否是时间范围查询
        time_keywords = ["最近10年", "过去10年", "10年", "十年", "10 years", "decade", "历年", "历史", "变化", "趋势"]
        is_time_range_query = any(kw in query_lower for kw in time_keywords)
        
        # 检测是否是增长/变化查询
        growth_keywords = ["增长", "下降", "波动", "变化率", "增长率", "涨跌", "growth", "decline", "fluctuation", "rate of change"]
        is_growth_query = any(kw in query_lower for kw in growth_keywords)
        
        # 构建精准增强 - 使用时间变化导向关键词获取历史数据
        enhancements = []
        
        if detected_universities and is_ranking_query:
            # 特定大学排名查询
            enhancements = [
                "QS World University Rankings",  # QS排名关键词
                "THE World University Rankings",  # THE排名关键词
                "ARWU Academic Ranking",  # ARWU排名关键词
                "历年排名对比",  # 中文排名对比关键词
                "historical rankings comparison",  # 英文排名对比关键词
                "ranking trends",  # 排名趋势
                "year by year ranking"  # 年度排名
            ]
            if is_time_range_query:
                enhancements.extend(["历年数据", "historical data", "trend analysis", "time series"])
        elif detected_universities and is_growth_query:
            # 特定大学增长/变化查询
            enhancements = [
                "historical performance",
                "development trends",
                "growth analysis",
                "变化趋势分析",
                "历史表现"
            ]
        elif detected_universities:
            # 一般大学查询
            enhancements = [
                "university profile",
                "学术排名",
                "academic reputation",
                "教育质量",
                "historical development"
            ]
        elif is_ranking_query:
            # 排名查询但未指定大学
            enhancements = [
                "QS World University Rankings",
                "THE World University Rankings",
                "世界大学排名",
                "global university rankings"
            ]
            if is_time_range_query:
                enhancements.extend(["历年变化", "ranking trends", "historical data", "time series analysis"])
        elif is_growth_query:
            # 一般增长/变化查询
            enhancements = [
                "historical trends",
                "trend analysis",
                "time series data",
                "historical comparison",
                "变化趋势",
                "历史对比"
            ]
        else:
            # 通用时间变化查询
            enhancements = ["历史数据", "historical data", "趋势分析", "trend analysis"]
        
        # 组合：原始查询 + 精准关键词
        enhanced = query + " " + " ".join(enhancements)
        return enhanced
    
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
        if domain not in {"weather", "transportation", "finance", "sports", "location"}:
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
        if domain == "location":
            return self._handle_location(query, timing_recorder=timing_recorder)

    def _handle_weather(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder],
    ) -> Dict[str, Any]:
        # 检测空气质量查询
        air_quality_keywords = ["空气质量", "空气污染", "AQI", "PM2.5", "PM10", "雾霾", "空气指数",
                               "air quality", "air pollution", "AQI", "PM2.5", "PM10", "smog", "haze"]
        is_air_quality_query = any(kw in query for kw in air_quality_keywords)
        
        # 检测预报查询，fallback 搜索
        forecast_keywords = ["明天", "后天", "预报", "forecast", "tomorrow"]
        if any(kw in query for kw in forecast_keywords):
            return {"handled": False, "reason": "forecast_requested_fallback_search"}

        location_hint = self._extract_weather_location(query)
        if not location_hint:
            return {"handled": False, "reason": "cannot_parse_location", "skipped": True}

        geocode = self._geocode_text(location_hint, timing_recorder=timing_recorder)
        if not geocode or geocode.get("error"):
            return {
                "handled": True,
                "error": geocode.get("error") if geocode else "geocode_failed",
                "location": location_hint,
            }

        # 如果是空气质量查询，使用空气质量API
        if is_air_quality_query:
            # 检查是否是中国地区（Google Air Quality API支持中国）
            air_quality_payload = self._call_google_air_quality(
                geocode["lat"],
                geocode["lng"],
                timing_recorder=timing_recorder,
            )
            if not air_quality_payload or air_quality_payload.get("error"):
                return {
                    "handled": True,
                    "error": air_quality_payload.get("error") if air_quality_payload else "air_quality_request_failed",
                    "location": geocode,
                }

            answer = self._format_air_quality_answer(location_hint, geocode, air_quality_payload)
            return {
                "handled": True,
                "provider": "google",
                "endpoint": "https://airquality.googleapis.com/v1/currentConditions:lookup",
                "location": geocode,
                "data": air_quality_payload,
                "answer": answer,
            }
        
        # 普通天气查询
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
            return {"handled": False, "skipped": True, "reason": "cannot_parse_route"}

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
        symbols = self._extract_finance_symbols(query)
        if not symbols:
            # 无法提取代码时跳过而不报错，交由通用搜索处理
            return {"handled": False, "reason": "cannot_parse_symbol", "skipped": True}

        # Check for historical/reasoning intent
        query_lower = query.lower()
        history_keywords = [
            "过去",
            "過去",
            "past",
            "days",
            "history",
            "trend",
            "历史",
            "歷史",
            "走势",
            "走勢",
            "表现",
            "表現",
            "近",
            "最近",
        ]
        reasoning_keywords = ["为什么", "why", "reason", "cause", "news", "analysis", "分析", "原因", "影响"]
        
        is_history = any(kw in query_lower for kw in history_keywords)
        is_reasoning = any(kw in query_lower for kw in reasoning_keywords)

        # Regex to extract time period if present (e.g. "5 days", "2年", "十年")
        period = "1d"
        
        # Check for years first (e.g. "2年", "3 years", "十年")
        match_years = re.search(r'(十|(\d+))\s*(?:年|years?)', query_lower)
        if match_years:
            is_history = True
            if match_years.group(1) == "十":
                years = 10
            else:
                years = int(match_years.group(2))
            
            # Map years to yfinance valid periods
            if years <= 1:
                period = "1y"
            elif years <= 2:
                period = "2y"
            elif years <= 5:
                period = "5y"
            elif years <= 10:
                period = "10y"
            else:
                period = "max"
        else:
            # Check for days (e.g. "5天", "10 days")
            match_days = re.search(r'(\d+)\s*(?:天|days)', query_lower)
            if match_days:
                is_history = True
                days = int(match_days.group(1))
                # Map to yfinance valid periods roughly
                if days <= 5:
                    period = "5d"
                elif days <= 30:
                    period = "1mo"
                elif days <= 90:
                    period = "3mo"
                elif days <= 180:
                    period = "6mo"
                elif days <= 365:
                    period = "1y"
                else:
                    period = "max"
            elif is_history:
                period = "1mo"  # Default history if not specified

        results = []
        for symbol in symbols:
            if is_history:
                quote = self._query_stock_history(symbol, period, timing_recorder=timing_recorder)
            else:
                quote = self._query_stock_price(symbol, timing_recorder=timing_recorder)
            
            if quote:
                quote["symbol"] = symbol
                results.append(quote)

        if not results:
             return {
                "handled": True,
                "error": "data_fetch_failed_for_all_symbols",
                "symbols": symbols,
            }

        # Identify key events for historical data
        key_events = []
        if is_history:
            for result in results:
                if not result.get("error") and "yearly_returns" in result:
                    events = self._identify_key_events(result, query)
                    if events:
                        key_events.extend(events)
        
        answer = self._format_finance_answer_multi(results, is_history)
        
        # Add key events to answer if any were identified
        if key_events:
            answer += "\n\n🔍 **关键事件分析**:\n"
            for event in key_events:
                answer += f"   • {event}\n"
        
        return {
            "handled": True,
            "provider": "yfinance" if is_history else "finnhub/yfinance",
            "endpoint": "yfinance.history" if is_history else "quote",
            "symbols": symbols,
            "data": results,
            "answer": answer,
            "key_events": key_events,
            # If the user asks for reasons/analysis, we must continue to the main search pipeline
            # to retrieve news/web content, while providing the data we found as context.
            "continue_search": is_reasoning
        }

    def _extract_finance_symbols(self, query: str) -> List[str]:
        """Extract multiple finance symbols from query.
        
        Uses a multi-step approach:
        1. Check predefined mappings (indices, crypto, company names)
        2. Use regex patterns to find potential symbols
        3. Use LLM to identify company names and get stock symbols (if available)
        4. Use Google Search as fallback to find stock symbols for unknown companies
        """
        symbols = set()
        query_upper = query.upper()
        
        # 1. Check specific maps first
        index_map = {
            "HANG SENG": "^HSI", "恒生": "^HSI",
            "NASDAQ": "^IXIC", "纳斯达克": "^IXIC",
            "DOW JONES": "^DJI", "道琼斯": "^DJI",
            "S&P 500": "^GSPC", "标普500": "^GSPC",
        }
        for name, sym in index_map.items():
            if name in query_upper:
                symbols.add(sym)

        # 2. Crypto Map (including Chinese names)
        crypto_map = {
            "BTC": "BTC-USD",
            "BITCOIN": "BTC-USD",
            "比特币": "BTC-USD",
            "比特幣": "BTC-USD",
            "ETH": "ETH-USD",
            "ETHEREUM": "ETH-USD",
            "以太坊": "ETH-USD",
            "以太幣": "ETH-USD",
            "DOGE": "DOGE-USD",
            "狗狗币": "DOGE-USD",
            "狗狗幣": "DOGE-USD",
            "SOL": "SOL-USD",
            "SOLANA": "SOL-USD",
            "索拉纳": "SOL-USD",
            "XRP": "XRP-USD",
            "瑞波币": "XRP-USD",
            "瑞波幣": "XRP-USD",
            "ADA": "ADA-USD",
            "艾达币": "ADA-USD",
            "DOT": "DOT-USD",
            "波卡": "DOT-USD",
            "MATIC": "MATIC-USD",
            "AVAX": "AVAX-USD",
            "LINK": "LINK-USD",
            "UNI": "UNI-USD",
        }
        # Check both uppercase and original query for crypto
        for name, sym in crypto_map.items():
            if name in query_upper or name in query:
                symbols.add(sym)

        # 2.5. Chinese Company Name Map
        chinese_company_map = {
            "苹果": "AAPL",
            "苹果公司": "AAPL",
            "微软": "MSFT",
            "微软公司": "MSFT",
            "谷歌": "GOOGL",
            "谷歌公司": "GOOGL",
            "亚马逊": "AMZN",
            "亚马逊公司": "AMZN",
            "Amazon": "AMZN",
            "特斯拉": "TSLA",
            "特斯拉公司": "TSLA",
            "脸书": "META",
            "脸书公司": "META",
            "Meta": "META",
            "英伟达": "NVDA",
            "英伟达公司": "NVDA",
            # Intel 和 AMD
            "英特尔": "INTC",
            "英特尔公司": "INTC",
            "Intel": "INTC",
            "INTEL": "INTC",
            "超威半导体": "AMD",
            "超微半导体": "AMD",
            "AMD公司": "AMD",
            # 其他公司
            "阿里巴巴": "BABA",
            "阿里巴巴集团": "BABA",
            "腾讯": "0700.HK",
            "腾讯控股": "0700.HK",
            "台积电": "TSM",
            "台积电公司": "TSM",
            "比亚迪": "BYD",
            "比亚迪公司": "BYD",
            "茅台": "600519.SS",
            "贵州茅台": "600519.SS",
            "中国平安": "601318.SS",
            "平安": "601318.SS",
            "中国移动": "0941.HK",
            "中国联通": "600050.SS",
            "中国电信": "601728.SS",
            "京东": "JD",
            "京东集团": "JD",
            "百度": "BIDU",
            "百度公司": "BIDU",
            "网易": "NTES",
            "网易公司": "NTES",
            "小米": "1810.HK",
            "小米集团": "1810.HK",
            "美团": "3690.HK",
            "美团点评": "3690.HK",
            "拼多多": "PDD",
            "拼多多公司": "PDD",
            "蔚来": "NIO",
            "蔚来汽车": "NIO",
            "理想汽车": "LI",
            "小鹏汽车": "XPEV",
            "高通": "QCOM",
            "高通公司": "QCOM",
            "博通": "AVGO",
            "博通公司": "AVGO",
            "标普500": "^GSPC",
            "标普": "^GSPC",
            "道琼斯": "^DJI",
            "道指": "^DJI",
            "纳斯达克": "^IXIC",
            "纳指": "^IXIC",
            "恒生指数": "^HSI",
            "恒指": "^HSI",
        }
        
        # Check for Chinese company names
        for name, sym in chinese_company_map.items():
            if name in query:
                symbols.add(sym)

        # 3. Regex for Stock Symbols
        
        # Matches: (NVDA), (AMD)
        matches_paren = re.findall(r'\(([A-Z]{1,5})\)', query_upper)
        symbols.update(matches_paren)
        
        # Matches: 6 digits (CN/HK codes)
        matches_digits = re.findall(r'(?<!\d)\d{6}(?!\d)', query)
        symbols.update(matches_digits)

        # Matches: NVDA, AMD, AMAZON (2-6 letters)
        # Improved regex to handle mixed language boundaries and case insensitivity
        # Look for 2-6 letter words not surrounded by other letters
        candidates = re.findall(r'(?<![a-zA-Z])[a-zA-Z]{2,6}(?![a-zA-Z])', query)
        
        stopwords = {
            "AND", "THE", "FOR", "WHO", "WHY", "USD", "HKD", "RMB",
            "STOCK", "PRICE", "DAYS", "PAST", "COMPARE", "WITH", "FROM", "TO",
            "WHAT", "WHEN", "WHERE", "HOW", "IS", "ARE", "WAS", "WERE",
            "YEAR", "MONTH", "WEEK", "DAY", "TODAY", "NOW", "NEWS",
            "ANALYSIS", "TREND", "HISTORY", "PERFORMANCE", "VS", "OR",
            "TOP", "BEST", "NEW", "OLD", "BIG", "SMALL", "BUY", "SELL",
            "OF", "IN", "ON", "AT", "BY", "IT", "AS", "IF", "DO", "GO",
            "MY", "ME", "WE", "UP", "SO", "NO", "KEY", "MODEL", "DATA",
            "INFO", "SHOW", "TELL", "GIVE", "GET", "SET", "RUN", "USE",
            "TRY", "ASK", "SAY", "SEE", "SAW", "LOT", "BIT", "PUT", "LET",
            "ANY", "ALL", "ONE", "TWO", "SIX", "TEN", "HAS", "HAD", "NOT",
            "BUT", "CAN", "MAY", "OUT", "OFF", "TAX", "LAW", "ACT", "ART",
            "MAP", "APP", "WEB", "NET", "COM", "ORG", "EDU", "GOV", "MIL",
            "INT", "DESCRIBE", "EXPLAIN", "ABOUT",
            # Date/time related words that should not be treated as stock symbols
            "DATE", "TIME", "CURRENT", "UTC", "ISO",
            # Company name words that are not stock symbols themselves
            "INTEL", "NVIDIA", "GOOGLE", "APPLE", "AMAZON", "TESLA",
        }
        
        for m in candidates:
            if m.upper() not in stopwords:
                symbols.add(m.upper())

        # 4. If no symbols found, try LLM extraction
        if not symbols and self.use_llm and self.llm_client:
            llm_symbols = self._extract_symbols_with_llm(query)
            if llm_symbols:
                symbols.update(llm_symbols)
        
        # 5. If still no symbols, try Google Search fallback
        if not symbols and self.google_api_key:
            search_symbols = self._extract_symbols_with_search(query)
            if search_symbols:
                symbols.update(search_symbols)

        return list(symbols)
    
    def _extract_symbols_with_llm(self, query: str) -> List[str]:
        """Use LLM to extract company names and convert to stock symbols."""
        if not self.llm_client:
            return []
        
        prompt = (
            "从用户的金融查询中提取所有公司名称，并返回它们对应的股票代码。\n"
            "输出JSON格式，例如：{\"symbols\": [\"AAPL\", \"MSFT\"]}\n"
            "规则：\n"
            "- 美股使用标准代码（如AAPL, MSFT, GOOGL）\n"
            "- 港股使用.HK后缀（如0700.HK）\n"
            "- A股使用.SS（上海）或.SZ（深圳）后缀（如600519.SS）\n"
            "- 如果无法确定股票代码，返回空数组\n"
            "- 只返回确定的股票代码，不要猜测\n\n"
            f"用户查询：{query}"
        )
        
        try:
            response = self.llm_client.chat(
                system_prompt="You are a financial assistant that extracts stock symbols from queries.",
                user_prompt=prompt,
                max_tokens=200,
                temperature=0.0,
            )
            content = response.get("content", "{}")
            
            # Try to parse JSON
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(content[start:end+1])
                    except json.JSONDecodeError:
                        return []
                else:
                    return []
            
            symbols = parsed.get("symbols", [])
            if isinstance(symbols, list):
                # Validate symbols format
                valid_symbols = []
                for sym in symbols:
                    if isinstance(sym, str) and sym.strip():
                        sym = sym.strip().upper()
                        # Basic validation: alphanumeric with optional suffix
                        if re.match(r'^[A-Z0-9\^]{1,6}(\.[A-Z]{1,2})?$', sym):
                            valid_symbols.append(sym)
                return valid_symbols
        except Exception as exc:
            try:
                print(f"[LLM Symbol Extraction] Error: {exc}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        
        return []
    
    def _extract_symbols_with_search(self, query: str) -> List[str]:
        """Use Google Search to find stock symbols for companies mentioned in query."""
        if not self.google_api_key:
            return []
        
        # Extract potential company names from query
        # Look for capitalized words or Chinese company patterns
        company_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized words like "Intel Corporation"
            r'([\u4e00-\u9fff]{2,}(?:公司|集团|控股|科技|电子|半导体)?)',  # Chinese company names
        ]
        
        potential_companies = []
        for pattern in company_patterns:
            matches = re.findall(pattern, query)
            potential_companies.extend(matches)
        
        if not potential_companies:
            return []
        
        symbols = []
        for company in potential_companies[:3]:  # Limit to 3 companies to avoid too many API calls
            symbol = self._search_stock_symbol(company)
            if symbol:
                symbols.append(symbol)
        
        return symbols
    
    def _search_stock_symbol(self, company_name: str) -> Optional[str]:
        """Search for a company's stock symbol using Google Search."""
        if not self.google_api_key:
            return None
        
        # Construct search query
        search_query = f"{company_name} stock symbol ticker"
        
        try:
            # Use Google Custom Search API
            params = {
                "key": self.google_api_key,
                "cx": os.getenv("GOOGLE_CX", ""),  # Custom Search Engine ID
                "q": search_query,
                "num": 3,
            }
            
            # Skip if no CX configured
            if not params["cx"]:
                return self._search_stock_symbol_simple(company_name)
            
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                combined = f"{title} {snippet}"
                
                # Look for stock symbol patterns in results
                # Pattern: (SYMBOL) or SYMBOL: or ticker: SYMBOL
                symbol_patterns = [
                    r'\(([A-Z]{1,5})\)',  # (AAPL)
                    r'(?:ticker|symbol|stock)[\s:]+([A-Z]{1,5})',  # ticker: AAPL
                    r'([A-Z]{1,5})(?:\s+stock|\s+shares)',  # AAPL stock
                ]
                
                for pattern in symbol_patterns:
                    match = re.search(pattern, combined, re.IGNORECASE)
                    if match:
                        symbol = match.group(1).upper()
                        # Validate it's a real symbol by checking with yfinance
                        if self._validate_symbol(symbol):
                            return symbol
            
        except Exception as exc:
            try:
                print(f"[Google Search Symbol] Error: {exc}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        
        return None
    
    def _search_stock_symbol_simple(self, company_name: str) -> Optional[str]:
        """Simple fallback to search stock symbol using yfinance search."""
        try:
            import yfinance as yf
            
            # Try common variations
            variations = [
                company_name,
                company_name.replace(" ", ""),
                company_name.split()[0] if " " in company_name else company_name,
            ]
            
            for variation in variations:
                try:
                    # Try to get ticker info directly
                    ticker = yf.Ticker(variation)
                    info = ticker.info
                    if info and info.get("symbol"):
                        return info["symbol"]
                except Exception:
                    continue
            
            # Try yfinance search (if available)
            try:
                # yfinance doesn't have a direct search API, but we can try common patterns
                # For well-known companies, the symbol is often the first few letters
                if len(company_name) >= 2:
                    potential_symbol = company_name[:4].upper()
                    ticker = yf.Ticker(potential_symbol)
                    info = ticker.info
                    if info and info.get("symbol") and info.get("shortName"):
                        # Verify the company name matches
                        if company_name.lower() in info.get("shortName", "").lower():
                            return info["symbol"]
            except Exception:
                pass
                
        except Exception as exc:
            try:
                print(f"[yfinance Symbol Search] Error: {exc}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        
        return None
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is a real stock symbol using yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Check if we got valid data
            return bool(info and (info.get("symbol") or info.get("shortName")))
        except Exception:
            return False

    def _identify_key_events(self, stock_data: Dict[str, Any], query: str) -> List[str]:
        """Identify key events based on stock performance data."""
        events = []
        symbol = stock_data.get("symbol", "Unknown")
        
        # Analyze yearly returns for significant events
        yearly_returns = stock_data.get("yearly_returns", [])
        if yearly_returns:
            # Find best and worst years
            best_year = max(yearly_returns, key=lambda x: x["return"])
            worst_year = min(yearly_returns, key=lambda x: x["return"])
            
            if best_year["return"] > 30:
                events.append(f"{best_year['year']}年表现强劲，上涨{best_year['return']:.2f}%")
            
            if worst_year["return"] < -20:
                events.append(f"{worst_year['year']}年表现疲软，下跌{abs(worst_year['return']):.2f}%")
            
            # Check for consecutive up/down years
            consecutive_up = 0
            consecutive_down = 0
            max_consecutive_up = 0
            max_consecutive_down = 0
            
            for i, year_data in enumerate(yearly_returns):
                if year_data["return"] > 0:
                    consecutive_up += 1
                    consecutive_down = 0
                    max_consecutive_up = max(max_consecutive_up, consecutive_up)
                else:
                    consecutive_down += 1
                    consecutive_up = 0
                    max_consecutive_down = max(max_consecutive_down, consecutive_down)
            
            if max_consecutive_up >= 3:
                events.append(f"曾连续{max_consecutive_up}年上涨，表现持续向好")
            
            if max_consecutive_down >= 2:
                events.append(f"曾连续{max_consecutive_down}年下跌，面临调整压力")
        
        # Analyze volatility
        volatility = stock_data.get("volatility")
        if volatility is not None:
            if volatility > 40:
                events.append("价格波动极大，市场情绪不稳定")
            elif volatility > 30:
                events.append("价格波动较大，投资风险较高")
        
        # Analyze drawdown
        max_drawdown = stock_data.get("max_drawdown")
        if max_drawdown is not None and max_drawdown < -40:
            events.append(f"曾经历大幅回撤({max_drawdown:.2f}%)，需注意风险控制")
        
        # Analyze trend and momentum
        trend = stock_data.get("trend_direction", "")
        momentum = stock_data.get("momentum")
        momentum_desc = stock_data.get("momentum_desc", "")
        
        if "强劲上涨" in trend and momentum and momentum > 10:
            events.append("当前呈现强劲上涨趋势，短期动量充足")
        elif "大幅下跌" in trend and momentum and momentum < -10:
            events.append("近期表现疲软，短期动量不足")
        
        # Check for moving average signals
        ma_20 = stock_data.get("ma_20")
        ma_50 = stock_data.get("ma_50")
        ma_200 = stock_data.get("ma_200")
        current_price = stock_data.get("end_price")
        
        if all(x is not None for x in [ma_20, ma_50, ma_200, current_price]):
            if current_price > ma_20 > ma_50 > ma_200:
                events.append("技术面呈现多头排列，长期趋势向好")
            elif current_price < ma_20 < ma_50 < ma_200:
                events.append("技术面呈现空头排列，长期趋势向淡")
        
        # Add symbol-specific insights if available
        symbol_insights = {
            "^GSPC": "标普500指数",
            "^DJI": "道琼斯指数",
            "^IXIC": "纳斯达克指数",
            "^HSI": "恒生指数",
            "AAPL": "苹果公司",
            "MSFT": "微软公司",
            "GOOGL": "谷歌公司",
            "TSLA": "特斯拉公司",
            "AMZN": "亚马逊公司",
        }
        
        symbol_name = symbol_insights.get(symbol, symbol)
        if symbol_name != symbol:
            events.insert(0, f"分析对象：{symbol_name}({symbol})")
        
        return events

    def _query_stock_history(self, symbol: str, period: str, timing_recorder: Optional[TimingRecorder] = None) -> Dict[str, Any]:
        """Fetch historical data using yfinance with technical indicators."""
        start = time.perf_counter()
        try:
            # Ensure yfinance is available
            import yfinance as yf
            import numpy as np
            
            ticker = yf.Ticker(symbol)
            # Fetch history
            df = ticker.history(period=period)
            if df.empty:
                return {"error": "no_history_data"}
            
            # Calculate basic stats
            # Start/End close
            if len(df) >= 2:
                start_price = df.iloc[0]['Close']
                end_price = df.iloc[-1]['Close']
                change = end_price - start_price
                pct_change = (change / start_price) * 100
                
                # Calculate technical indicators
                closes = df['Close'].values
                
                # 1. Moving averages
                ma_20 = None
                ma_50 = None
                ma_200 = None
                
                if len(closes) >= 20:
                    ma_20 = np.mean(closes[-20:])
                if len(closes) >= 50:
                    ma_50 = np.mean(closes[-50:])
                if len(closes) >= 200:
                    ma_200 = np.mean(closes[-200:])
                
                # 2. Volatility (standard deviation of daily returns)
                daily_returns = np.diff(closes) / closes[:-1]
                volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized volatility
                
                # 3. Maximum drawdown
                cumulative_returns = np.cumprod(1 + daily_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns) * 100
                
                # 4. Yearly returns (if data spans multiple years)
                yearly_returns = []
                if len(df) > 365:
                    df_years = df.groupby(df.index.year)
                    for year, group in df_years:
                        if len(group) > 1:  # Need at least 2 days to calculate return
                            year_start = group.iloc[0]['Close']
                            year_end = group.iloc[-1]['Close']
                            year_return = ((year_end - year_start) / year_start) * 100
                            yearly_returns.append({"year": year, "return": year_return})
                
                # 5. Trend analysis
                trend_direction = "横盘"
                if pct_change > 20:
                    trend_direction = "强劲上涨"
                elif pct_change > 10:
                    trend_direction = "温和上涨"
                elif pct_change < -20:
                    trend_direction = "大幅下跌"
                elif pct_change < -10:
                    trend_direction = "温和下跌"
                
                # 6. Price momentum (recent 30 days vs previous 30 days)
                momentum = None
                momentum_desc = "无法计算"
                if len(closes) >= 60:
                    recent_30 = np.mean(closes[-30:])
                    prev_30 = np.mean(closes[-60:-30])
                    momentum = ((recent_30 - prev_30) / prev_30) * 100
                    if momentum > 5:
                        momentum_desc = "强劲上升"
                    elif momentum > 2:
                        momentum_desc = "温和上升"
                    elif momentum < -5:
                        momentum_desc = "快速下降"
                    elif momentum < -2:
                        momentum_desc = "温和下降"
                    else:
                        momentum_desc = "基本稳定"
                
                # Serialize a summary with technical indicators
                history_summary = {
                    "start_date": str(df.index[0].date()),
                    "end_date": str(df.index[-1].date()),
                    "start_price": start_price,
                    "end_price": end_price,
                    "change": change,
                    "pct_change": pct_change,
                    "high": df['High'].max(),
                    "low": df['Low'].min(),
                    "volatility": volatility,
                    "max_drawdown": max_drawdown,
                    "trend_direction": trend_direction,
                    "ma_20": ma_20,
                    "ma_50": ma_50,
                    "ma_200": ma_200,
                    "momentum": momentum,
                    "momentum_desc": momentum_desc,
                    "yearly_returns": yearly_returns,
                    "daily_data": [
                        {"date": str(idx.date()), "close": row['Close'], "volume": row['Volume']}
                        for idx, row in df.iterrows()
                    ]
                }
                return history_summary
            else:
                return {"error": "insufficient_history_data"}

        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="yfinance_history",
                    label="yfinance History",
                    duration_ms=duration_ms,
                )

    def _format_finance_answer_multi(self, results: List[Dict[str, Any]], is_history: bool) -> str:
        parts = []
        for res in results:
            sym = res.get("symbol", "Unknown")
            if res.get("error"):
                parts.append(f"{sym}: 数据获取失败 ({res['error']})")
                continue
            
            if is_history:
                # Format historical summary with enhanced analysis
                pct = res.get("pct_change", 0)
                sign = "+" if pct >= 0 else ""
                
                s_p = res.get('start_price')
                e_p = res.get('end_price')
                l_p = res.get('low')
                h_p = res.get('high')
                
                s_str = f"{s_p:.2f}" if isinstance(s_p, (int, float)) else "N/A"
                e_str = f"{e_p:.2f}" if isinstance(e_p, (int, float)) else "N/A"
                l_str = f"{l_p:.2f}" if isinstance(l_p, (int, float)) else "N/A"
                h_str = f"{h_p:.2f}" if isinstance(h_p, (int, float)) else "N/A"

                # Enhanced analysis with technical indicators
                analysis_parts = []
                
                # Basic performance
                analysis_parts.append(
                    f"📊 **{sym}** ({res.get('start_date', '?')} 至 {res.get('end_date', '?')}):\n"
                    f"   - 涨跌幅: {sign}{pct:.2f}%\n"
                    f"   - 收盘价: {s_str} -> {e_str}\n"
                    f"   - 期间波动: {l_str} - {h_str}"
                )
                
                # Trend analysis
                trend = res.get("trend_direction", "未知")
                analysis_parts.append(f"\n   - 趋势分析: {trend}")
                
                # Volatility and risk
                volatility = res.get("volatility")
                if volatility is not None:
                    vol_level = "低" if volatility < 15 else "中" if volatility < 30 else "高"
                    analysis_parts.append(f"\n   - 年化波动率: {volatility:.2f}% ({vol_level}风险)")
                
                # Maximum drawdown
                max_dd = res.get("max_drawdown")
                if max_dd is not None:
                    analysis_parts.append(f"\n   - 最大回撤: {max_dd:.2f}%")
                
                # Moving averages
                ma_20 = res.get("ma_20")
                ma_50 = res.get("ma_50")
                ma_200 = res.get("ma_200")
                
                if ma_20 is not None:
                    analysis_parts.append(f"\n   - 20日均线: {ma_20:.2f}")
                    if e_p is not None:
                        ma20_signal = "上方" if e_p > ma_20 else "下方"
                        analysis_parts.append(f"     (当前价格在20日均线{ma20_signal})")
                
                if ma_50 is not None:
                    analysis_parts.append(f"\n   - 50日均线: {ma_50:.2f}")
                    if e_p is not None:
                        ma50_signal = "上方" if e_p > ma_50 else "下方"
                        analysis_parts.append(f"     (当前价格在50日均线{ma50_signal})")
                
                if ma_200 is not None:
                    analysis_parts.append(f"\n   - 200日均线: {ma_200:.2f}")
                    if e_p is not None:
                        ma200_signal = "上方" if e_p > ma_200 else "下方"
                        analysis_parts.append(f"     (当前价格在200日均线{ma200_signal})")
                
                # Momentum
                momentum = res.get("momentum")
                momentum_desc = res.get("momentum_desc")
                if momentum is not None and momentum_desc is not None:
                    analysis_parts.append(f"\n   - 近期动量: {momentum_desc} ({momentum:.2f}%)")
                
                # Yearly returns
                yearly_returns = res.get("yearly_returns", [])
                if yearly_returns:
                    analysis_parts.append("\n   - 年度收益率:")
                    for yr in yearly_returns:
                        yr_sign = "+" if yr["return"] >= 0 else ""
                        analysis_parts.append(f"     {yr['year']}年: {yr_sign}{yr['return']:.2f}%")
                
                # Key insights summary
                insights = []
                if pct > 20:
                    insights.append("整体表现强劲，显著上涨")
                elif pct < -20:
                    insights.append("整体表现疲软，显著下跌")
                
                if volatility is not None:
                    if volatility > 30:
                        insights.append("价格波动较大，投资风险较高")
                    elif volatility < 15:
                        insights.append("价格相对稳定，投资风险较低")
                
                if max_dd is not None and max_dd < -30:
                    insights.append("曾经历较大回撤，需注意风险控制")
                
                if ma_20 is not None and ma_50 is not None and ma_200 is not None:
                    if e_p > ma_20 > ma_50 > ma_200:
                        insights.append("技术面呈现多头排列，趋势向好")
                    elif e_p < ma_20 < ma_50 < ma_200:
                        insights.append("技术面呈现空头排列，趋势向淡")
                
                if insights:
                    analysis_parts.append("\n   - 关键洞察:")
                    for insight in insights:
                        analysis_parts.append(f"     • {insight}")
                
                parts.append("".join(analysis_parts))
            else:
                # Format current quote (reuse logic or simple)
                c = res.get("c") or res.get("currentPrice")
                parts.append(f"📈 **{sym}** 现价: {c}")

        return "\n\n".join(parts)


    def _handle_location(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Dict[str, Any]:
        """处理地点/POI搜索查询，如"距离HKUST最近的KFC是哪家"""
        # 提取参考地点和目标类型
        parsed = self._extract_location_query(query)
        if not parsed:
            return {"handled": False, "reason": "cannot_parse_location_query", "skipped": True}
        
        reference_location = parsed.get("reference_location")
        target_type = parsed.get("target_type")
        
        if not reference_location or not target_type:
            return {"handled": False, "reason": "missing_reference_or_target", "skipped": True}
        
        # 获取参考地点的坐标
        geocode = self._geocode_text(reference_location, timing_recorder=timing_recorder)
        if not geocode or geocode.get("error"):
            return {
                "handled": True,
                "error": geocode.get("error") if geocode else "geocode_failed",
                "reference_location": reference_location,
            }
        
        lat = geocode.get("lat")
        lng = geocode.get("lng")
        
        if lat is None or lng is None:
            return {
                "handled": True,
                "error": "invalid_coordinates",
                "reference_location": reference_location,
            }
        
        # 使用 Google Places API 搜索附近地点
        places_result = self._call_google_places_nearby(
            lat=lat,
            lng=lng,
            keyword=target_type,
            timing_recorder=timing_recorder,
        )
        
        if not places_result or places_result.get("error"):
            return {
                "handled": True,
                "error": places_result.get("error") if places_result else "places_search_failed",
                "reference_location": geocode,
                "target_type": target_type,
            }
        
        places = places_result.get("places", [])
        if not places:
            return {
                "handled": True,
                "error": "no_places_found",
                "reference_location": geocode,
                "target_type": target_type,
                "answer": f"在 {geocode.get('formatted_address', reference_location)} 附近未找到 {target_type}。",
            }
        
        # 格式化答案
        answer = self._format_location_answer(
            reference_location=reference_location,
            geocode=geocode,
            target_type=target_type,
            places=places,
        )
        
        return {
            "handled": True,
            "provider": "google_places",
            "endpoint": "https://places.googleapis.com/v1/places:searchNearby",
            "reference_location": geocode,
            "target_type": target_type,
            "data": places_result,
            "answer": answer,
        }

    def _extract_location_query(self, query: str) -> Optional[Dict[str, str]]:
        """从查询中提取参考地点和目标类型"""
        # 常见模式：
        # "距离X最近的Y是哪家" / "X附近的Y" / "离X最近的Y"
        # "nearest Y to X" / "Y near X"
        
        patterns_cn = [
            r"距离(.+?)最近的(.+?)(?:是哪|在哪|有哪)",
            r"离(.+?)最近的(.+?)(?:是哪|在哪|有哪)",
            r"(.+?)附近的(.+?)(?:是哪|在哪|有哪)",
            r"(.+?)附近有(?:什么|哪些)?(.+)",
            r"(.+?)周边的(.+)",
            r"距离(.+?)最近的(.+)",
            r"离(.+?)最近的(.+)",
            r"(.+?)附近的(.+)",
        ]
        
        patterns_en = [
            r"nearest\s+(.+?)\s+(?:to|from|near)\s+(.+)",
            r"closest\s+(.+?)\s+(?:to|from|near)\s+(.+)",
            r"(.+?)\s+near(?:est)?\s+(.+)",
            r"find\s+(.+?)\s+near\s+(.+)",
        ]
        
        # 尝试中文模式
        for pattern in patterns_cn:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                reference = match.group(1).strip()
                target = match.group(2).strip()
                # 清理目标类型中的常见后缀
                target = re.sub(r"(?:是哪家|在哪里|有哪些|是什么)$", "", target).strip()
                if reference and target:
                    return {"reference_location": reference, "target_type": target}
        
        # 尝试英文模式
        for pattern in patterns_en:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # 英文模式中目标和参考位置的顺序可能不同
                g1 = match.group(1).strip()
                g2 = match.group(2).strip()
                # "nearest KFC to HKUST" -> target=KFC, reference=HKUST
                # "KFC near HKUST" -> target=KFC, reference=HKUST
                return {"reference_location": g2, "target_type": g1}
        
        # LLM fallback
        if self.use_llm and self.llm_client:
            prompt = (
                "从用户问题中提取：\n"
                "1. reference_location: 参考地点（用户想要从哪里出发/以哪里为中心）\n"
                "2. target_type: 目标类型（用户想要找什么类型的地点，如餐厅、商店名称等）\n\n"
                "输出JSON格式，例如：\n"
                "{\"reference_location\": \"香港科技大学\", \"target_type\": \"KFC\"}\n\n"
                "如果无法提取，返回空对象 {}\n\n"
                f"用户问题：{query}"
            )
            try:
                response = self.llm_client.chat(
                    system_prompt="You extract location search parameters from queries.",
                    user_prompt=prompt,
                    max_tokens=200,
                    temperature=0.0,
                )
                content = response.get("content", "{}")
                parsed = json.loads(content)
                ref = (parsed.get("reference_location") or "").strip()
                target = (parsed.get("target_type") or "").strip()
                if ref and target:
                    return {"reference_location": ref, "target_type": target}
            except Exception:
                pass
        
        return None

    def _call_google_places_nearby(
        self,
        lat: float,
        lng: float,
        keyword: str,
        radius: int = 5000,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """调用 Google Places API (New) 搜索附近地点"""
        url = "https://places.googleapis.com/v1/places:searchNearby"
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.google_api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.googleMapsUri,places.primaryType,places.shortFormattedAddress"
        }
        
        payload = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": radius
                }
            },
            "maxResultCount": 10,
            "languageCode": "zh-CN"
        }
        
        # 添加关键词/文本查询
        if keyword:
            # 使用 includedTypes 或 textQuery 取决于关键词类型
            # 对于品牌名称（如KFC），使用 textQuery 更合适
            payload["textQuery"] = keyword
        
        start = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.HTTPError as exc:
            # 尝试使用旧版 Places API 作为备选
            return self._call_google_places_nearby_legacy(
                lat=lat,
                lng=lng,
                keyword=keyword,
                radius=radius,
                timing_recorder=timing_recorder,
            )
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="google_places_nearby",
                    label="Google Places Nearby",
                    duration_ms=duration_ms,
                )

    def _call_google_places_nearby_legacy(
        self,
        lat: float,
        lng: float,
        keyword: str,
        radius: int = 5000,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """使用旧版 Google Places API (nearbysearch) 作为备选"""
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "keyword": keyword,
            "key": self.google_api_key,
            "language": "zh-CN",
        }
        
        start = time.perf_counter()
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            # 转换为统一格式
            if data.get("status") == "OK" and data.get("results"):
                places = []
                for result in data["results"][:10]:
                    place = {
                        "displayName": {"text": result.get("name", "")},
                        "formattedAddress": result.get("vicinity", ""),
                        "location": {
                            "latitude": result.get("geometry", {}).get("location", {}).get("lat"),
                            "longitude": result.get("geometry", {}).get("location", {}).get("lng"),
                        },
                        "rating": result.get("rating"),
                        "userRatingCount": result.get("user_ratings_total"),
                    }
                    places.append(place)
                return {"places": places}
            elif data.get("status") == "ZERO_RESULTS":
                return {"places": []}
            else:
                return {"error": data.get("status", "unknown_error")}
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="google_places_nearby_legacy",
                    label="Google Places Nearby (Legacy)",
                    duration_ms=duration_ms,
                )

    def _format_location_answer(
        self,
        reference_location: str,
        geocode: Dict[str, Any],
        target_type: str,
        places: List[Dict[str, Any]],
    ) -> str:
        """格式化地点搜索结果"""
        ref_address = geocode.get("formatted_address", reference_location)
        ref_lat = geocode.get("lat")
        ref_lng = geocode.get("lng")
        
        if not places:
            return f"在 {ref_address} 附近未找到 {target_type}。"
        
        # 计算距离并排序
        places_with_distance = []
        for place in places:
            place_lat = place.get("location", {}).get("latitude")
            place_lng = place.get("location", {}).get("longitude")
            
            distance = None
            if ref_lat and ref_lng and place_lat and place_lng:
                # 使用 Haversine 公式计算距离
                distance = self._haversine_distance(ref_lat, ref_lng, place_lat, place_lng)
            
            places_with_distance.append({
                "place": place,
                "distance": distance,
            })
        
        # 按距离排序
        places_with_distance.sort(key=lambda x: x["distance"] if x["distance"] is not None else float("inf"))
        
        # 格式化输出
        lines = [f"📍 在 **{ref_address}** 附近找到以下 **{target_type}**：\n"]
        
        for i, item in enumerate(places_with_distance[:5], 1):
            place = item["place"]
            distance = item["distance"]
            
            name = place.get("displayName", {}).get("text", "未知名称")
            address = place.get("formattedAddress") or place.get("shortFormattedAddress", "")
            rating = place.get("rating")
            rating_count = place.get("userRatingCount")
            
            line = f"{i}. **{name}**"
            if distance is not None:
                if distance < 1:
                    line += f" - 约 {int(distance * 1000)} 米"
                else:
                    line += f" - 约 {distance:.1f} 公里"
            if address:
                line += f"\n   📫 {address}"
            if rating:
                stars = "⭐" * int(rating)
                line += f"\n   {stars} {rating}"
                if rating_count:
                    line += f" ({rating_count} 条评价)"
            
            lines.append(line)
        
        # 添加最近的地点总结
        if places_with_distance:
            nearest = places_with_distance[0]
            nearest_name = nearest["place"].get("displayName", {}).get("text", "未知")
            nearest_dist = nearest["distance"]
            
            summary = f"\n\n✅ **最近的 {target_type}** 是 **{nearest_name}**"
            if nearest_dist is not None:
                if nearest_dist < 1:
                    summary += f"，距离约 {int(nearest_dist * 1000)} 米。"
                else:
                    summary += f"，距离约 {nearest_dist:.1f} 公里。"
            lines.append(summary)
        
        return "\n".join(lines)

    @staticmethod
    def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """使用 Haversine 公式计算两点之间的距离（公里）"""
        import math
        
        R = 6371  # 地球半径（公里）
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def _handle_sports(
        self,
        query: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Dict[str, Any]:
        # 不启用 SportsDB API（保留代码但跳过调用）
        if not self.sportsdb_api_key or self.sportsdb_api_key == "123":
            return {"handled": False, "skipped": True, "reason": "sportsdb_disabled"}

        entity = self._extract_sports_entity(query)
        if not entity:
            return {"handled": False, "reason": "cannot_parse_sports_entity", "skipped": True}

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
        # 常见指数名称到符号的映射
        index_map = {
            "hang seng index": "^HSI",
            "恒生指数": "^HSI",
            "nasdaq": "^IXIC",
            "纳斯达克": "^IXIC",
            "dow jones": "^DJI",
            "道琼斯": "^DJI",
            "s&p 500": "^GSPC",
            "标普500": "^GSPC",
        }
        query_lower = query.lower()
        for iname, symbol in index_map.items():
            if iname in query_lower:
                return symbol

        # 加密货币常见名称到符号映射
        crypto_map = {
            "比特币": "BTC",
            "比特幣": "BTC",
            "以太坊": "ETH",
            "狗狗币": "DOGE",
            "莱特币": "LTC",
            "索拉纳": "SOL",
        }
        for cname, symbol in crypto_map.items():
            if cname.lower() in query_lower:
                return symbol
        
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
                "从用户问题中提取股票代码、指数代码或加密货币符号（美股如AAPL，A股如600000，港股如0700.HK, 指数如^HSI, 加密货币如BTC），"
                "输出JSON {\"symbol\": \"^HSI\"}。"
                "无法提取返回空字符串。\n\n用户问题：" + query
            )
            response = self.llm_client.chat(
                system_prompt="Extract stock, index, or crypto symbol.",
                user_prompt=prompt,
                max_tokens=100,
                temperature=0.0,
            )
            try:
                payload = json.loads(response.get("content") or "{}")
                symbol = payload.get("symbol") or ""
                if isinstance(symbol, str) and re.match(r'^[A-Z\^]{1,6}(\.HK)?$|^[0-9]{6}$', symbol, re.IGNORECASE):
                    return symbol.upper()
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

    def _query_stock_price(
        self,
        symbol: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """尝试按优先级获取股票/加密货币报价：Finnhub -> yfinance -> yahoo_fin。

        返回第一个成功的、非 error 的结果字典；若所有后端都失败，返回最后一个错误结构。
        """
        last_error: Optional[Dict[str, Any]] = None

        # 1) Finnhub（如果配置了 api key）
        if self.finnhub_api_key:
            try:
                quote = self._call_finnhub_quote(symbol, timing_recorder=timing_recorder)
                if quote and not quote.get("error") and quote.get("c", 0) != 0:
                    quote["source_name"] = "Finnhub"
                    return quote
                last_error = quote or {"error": "finnhub_no_response"}
            except Exception as exc:
                last_error = {"error": str(exc)}

        # 2) yfinance
        try:
            quote = self._call_yfinance_quote(symbol, timing_recorder=timing_recorder)
            if quote and not quote.get("error"):
                quote["source_name"] = "Yahoo Finance (yfinance)"
                return quote
            last_error = quote or {"error": "yfinance_no_data"}
        except Exception as exc:
            last_error = {"error": str(exc)}

        # 3) yahoo_fin
        try:
            quote = self._call_yahoo_fin_quote(symbol, timing_recorder=timing_recorder)
            if quote and not quote.get("error"):
                quote["source_name"] = "Yahoo Finance (yahoo-fin)"
                return quote
            last_error = quote or {"error": "yahoo_fin_no_data"}
        except Exception as exc:
            last_error = {"error": str(exc)}

        return last_error or {"error": "no_price_providers_available"}

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

    def _geocode_text(
        self,
        location: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """使用 Google Geocoding API 将文本地点转换为坐标"""
        if not self.google_api_key:
            return {"error": "missing_google_api_key"}

        params = {
            "address": location,
            "key": self.google_api_key,
        }
        start = time.perf_counter()
        try:
            response = requests.get(
                self.google_geocode_url,
                params=params,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return {"error": "no_geocode_results"}

            result = results[0]
            geometry = result.get("geometry", {}).get("location", {})
            formatted = result.get("formatted_address", "")

            return {
                "lat": geometry.get("lat"),
                "lng": geometry.get("lng"),
                "formatted_address": formatted,
            }
        except Exception as exc:
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - start) * 1000
                timing_recorder.record_search_timing(
                    source="google_geocode",
                    label="Google Geocode",
                    duration_ms=duration_ms,
                )

    def _call_google_weather(
        self,
        lat: float,
        lng: float,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """调用 Google Weather API 获取当前天气"""
        params = {
            "key": self.google_api_key,
            "lat": lat,
            "lon": lng,
        }
        start = time.perf_counter()
        try:
            response = requests.get(
                f"{self.google_weather_base_url}/currentConditions:lookup",
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
                    source="google_weather",
                    label="Google Weather",
                    duration_ms=duration_ms,
                )

    def _call_google_air_quality(
        self,
        lat: float,
        lng: float,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """调用 Google Air Quality API 获取当前空气质量"""
        url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "location": {
                "latitude": lat,
                "longitude": lng
            },
            "extraComputations": [
                "HEALTH_RECOMMENDATIONS",
                "DOMINANT_POLLUTANT_CONCENTRATION",
                "POLLUTANT_CONCENTRATION",
                "LOCAL_AQI"
            ],
            "uaqiColorPalette": "RED_GREEN",
            "universalAqi": True
        }
        
        start = time.perf_counter()
        try:
            response = requests.post(
                url,
                headers=headers,
                params={"key": self.google_api_key},
                json=payload,
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
                    source="google_air_quality",
                    label="Google Air Quality",
                    duration_ms=duration_ms,
                )

    def _format_weather_answer(
        self,
        location_hint: str,
        geocode: Dict[str, Any],
        weather_data: Dict[str, Any],
    ) -> str:
        """格式化天气回复"""
        try:
            condition = (
                weather_data.get("weatherCondition", {})
                .get("description", {})
                .get("text", "未知")
            )
            temp = weather_data.get("temperature", {}).get("degrees", "未知")
            humidity = weather_data.get("relativeHumidity", {}).get("value", "未知")
            wind = (
                weather_data.get("wind", {})
                .get("speed", {})
                .get("value", "未知")
            )
            location = geocode.get("formatted_address", location_hint)
            return (
                f"{location} 当前天气：{condition}，"
                f"{temp}°C，湿度{int(humidity)}%，风速{wind}km/h。"
            )
        except Exception:
            return f"{location_hint} 天气数据获取成功，但解析失败。"

    def _format_air_quality_answer(
        self,
        location_hint: str,
        geocode: Dict[str, Any],
        air_quality_data: Dict[str, Any],
    ) -> str:
        """格式化空气质量回复"""
        try:
            location = geocode.get("formatted_address", location_hint)
            
            # 获取AQI信息
            indexes = air_quality_data.get("indexes", [])
            if not indexes:
                return f"{location} 空气质量数据获取成功，但缺少AQI信息。"
            
            # 优先使用通用AQI (UAQI)
            uaqi = next((idx for idx in indexes if idx.get("code") == "uaqi"), indexes[0])
            aqi_value = uaqi.get("aqi", "未知")
            aqi_category = uaqi.get("category", "未知")
            dominant_pollutant = uaqi.get("dominantPollutant", "未知")
            
            # 获取污染物信息
            pollutants = air_quality_data.get("pollutants", [])
            pollutant_info = []
            
            # 查找主要污染物的详细信息
            if pollutants and dominant_pollutant:
                main_pollutant = next(
                    (p for p in pollutants if p.get("code") == dominant_pollutant), 
                    None
                )
                if main_pollutant:
                    concentration = main_pollutant.get("concentration", {})
                    value = concentration.get("value", "未知")
                    units = concentration.get("units", "未知")
                    display_name = main_pollutant.get("displayName", dominant_pollutant)
                    pollutant_info.append(f"{display_name}: {value} {units}")
            
            # 获取健康建议
            health_recommendations = air_quality_data.get("healthRecommendations", {})
            general_recommendation = health_recommendations.get("generalPopulation", "")
            
            # 构建回复
            answer = f"{location} 当前空气质量：\n"
            answer += f"• AQI指数: {aqi_value} ({aqi_category})\n"
            
            if pollutant_info:
                answer += f"• 主要污染物: {', '.join(pollutant_info)}\n"
            
            # 添加其他常见污染物信息
            common_pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]
            other_pollutants = []
            for pollutant_code in common_pollutants:
                if pollutant_code == dominant_pollutant:
                    continue  # 已作为主要污染物显示
                pollutant = next(
                    (p for p in pollutants if p.get("code") == pollutant_code), 
                    None
                )
                if pollutant:
                    concentration = pollutant.get("concentration", {})
                    value = concentration.get("value")
                    if value is not None:
                        display_name = pollutant.get("displayName", pollutant_code)
                        units = concentration.get("units", "")
                        other_pollutants.append(f"{display_name}: {value} {units}")
            
            if other_pollutants:
                answer += f"• 其他污染物: {', '.join(other_pollutants[:3])}\n"  # 限制显示前3个
            
            # 添加健康建议
            if general_recommendation:
                answer += f"• 健康建议: {general_recommendation}\n"
            
            # 添加运动建议（针对跑步等户外活动）
            if aqi_value != "未知" and isinstance(aqi_value, (int, float)):
                if aqi_value <= 50:
                    exercise_advice = "空气质量优秀，非常适合户外跑步等运动。"
                elif aqi_value <= 100:
                    exercise_advice = "空气质量良好，适合户外运动。"
                elif aqi_value <= 150:
                    exercise_advice = "空气质量一般，敏感人群应减少户外运动。"
                elif aqi_value <= 200:
                    exercise_advice = "空气质量较差，不建议户外跑步等运动。"
                else:
                    exercise_advice = "空气质量很差，避免户外运动。"
                answer += f"• 运动建议: {exercise_advice}"
            
            return answer
        except Exception as e:
            return f"{location_hint} 空气质量数据获取成功，但解析失败: {str(e)}"

    def _format_finance_answer(self, symbol: str, quote: Dict[str, Any]) -> str:
        """格式化股票/加密货币报价为可读的中文回答。

        支持多种后端返回结构（finnhub、yfinance、yahoo_fin），尽量提取常见字段。
        """
        if not isinstance(quote, dict):
            return f"{symbol} 报价获取失败：无效返回格式。"

        if quote.get("error"):
            return f"{symbol} 报价获取失败：{quote.get('error')}"

        # 常见字段映射
        def _num(val: Any) -> Optional[float]:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        # Finnhub: c (current), h (high), l (low), o (open), pc (prev close)
        c = _num(quote.get("c") or quote.get("currentPrice") or quote.get("price"))
        h = _num(quote.get("h") or quote.get("dayHigh") or quote.get("high"))
        l = _num(quote.get("l") or quote.get("dayLow") or quote.get("low"))
        o = _num(quote.get("o") or quote.get("open") or quote.get("regularMarketOpen"))
        pc = _num(quote.get("pc") or quote.get("previousClose") or quote.get("regularMarketPreviousClose"))

        change = None
        change_percent = None
        if c is not None and pc is not None and pc != 0:
            change = c - pc
            change_percent = (change / pc) * 100

        parts: List[str] = []
        if c is not None:
            parts.append(f"当前价 {c:g}")
        
        if change is not None and change_percent is not None:
            sign = "+" if change > 0 else ""
            parts.append(f"涨跌 {sign}{change:.2f} ({sign}{change_percent:.2f}%)")

        if o is not None:
            parts.append(f"开盘 {o:g}")
        if h is not None and l is not None:
            parts.append(f"区间 {l:g} - {h:g}")
        else:
            if h is not None:
                parts.append(f"最高 {h:g}")
            if l is not None:
                parts.append(f"最低 {l:g}")
        if pc is not None:
            parts.append(f"昨收 {pc:g}")

        if not parts:
            # 最后尝试把整个 quote 转为简短字符串
            try:
                summary = json.dumps({k: v for k, v in quote.items() if v is not None}, ensure_ascii=False)
                return f"{symbol} 报价：{summary}"
            except Exception:
                return f"{symbol} 报价获取成功，但无法解析具体字段。"

        source = quote.get("source_name", "unknown")
        return f"{symbol}：" + "，".join(parts) + f"（数据源: {source}）"
    

    # 1. Google Routes 路线计算方法 (修复当前的 AttributeError)
    def _call_google_routes(
        self,
        origin: str,
        destination: str,
        mode: str = "DRIVE",
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """调用 Google Routes API (v2) 计算路线"""
        url = f"{self.google_routes_base_url}/directions/v2:computeRoutes"
        
        payload = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": mode,
            "routingPreference": "TRAFFIC_AWARE" if mode == "DRIVE" else None,
            "languageCode": "zh-CN",
            "units": "METRIC"
        }
        
        # 必须添加 FieldMask 才能获取数据
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.google_api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.description,routes.legs"
        }

        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"[Routes] API Request Failed: {exc}")
            return {"error": str(exc)}
        finally:
            if timing_recorder:
                timing_recorder.record_search_timing(
                    source="google_routes",
                    label="Google Routes",
                    duration_ms=(time.perf_counter() - start) * 1000
                )

    # 2. 路线结果格式化方法
    def _format_route_answer(
        self,
        mode_info: Dict[str, str],
        origin_geo: Dict[str, Any],
        dest_geo: Dict[str, Any],
        route_payload: Dict[str, Any]
    ) -> str:
        """将路线数据格式化为可读文本"""
        if not route_payload or "routes" not in route_payload or not route_payload["routes"]:
            return f"{mode_info.get('display')}：未找到有效路线或 API 报错。"
            
        route = route_payload["routes"][0]
        
        # 解析距离 (米 -> 公里)
        dist_meters = route.get("distanceMeters", 0)
        dist_km = dist_meters / 1000
        
        # 解析时间 (格式如 "1800s")
        dur_str = route.get("duration", "0s")
        seconds = 0
        if isinstance(dur_str, str) and dur_str.endswith("s"):
            try:
                seconds = int(dur_str[:-1])
            except ValueError:
                pass
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        time_str = ""
        if hours > 0: time_str += f"{hours}小时"
        if minutes > 0 or hours == 0: time_str += f"{minutes}分钟"
        
        mode_label = mode_info.get("display", "行程")
        return f"🚗 **{mode_label}**：预计耗时 **{time_str}**，距离 **{dist_km:.1f}公里**"

    # 3. 股票查询分发方法 (预防金融查询报错)
    def _query_stock_price(
        self,
        symbol: str,
        timing_recorder: Optional[TimingRecorder] = None,
    ) -> Optional[Dict[str, Any]]:
        """尝试多种渠道查询股价"""
        # 优先尝试 Finnhub (API)
        if self.finnhub_api_key:
            res = self._call_finnhub_quote(symbol, timing_recorder)
            if res and not res.get("error") and res.get("c"):
                res["source_name"] = "Finnhub"
                return res
        
        # 其次尝试 yfinance (库)
        try:
            res = self._call_yfinance_quote(symbol, timing_recorder)
            if res and not res.get("error"):
                res["source_name"] = "Yahoo Finance (yfinance)"
                return res
        except Exception:
            pass

        return {"error": "所有金融数据源均不可用"}

    # 4. 金融结果格式化方法
    def _format_finance_answer(self, symbol: str, quote: Dict[str, Any]) -> str:
        """格式化股价信息"""
        price = quote.get("c", "N/A")
        high = quote.get("h", "N/A")
        low = quote.get("l", "N/A")
        src = quote.get("source_name", "Unknown")
        
        return (
            f"📈 **{symbol}** 实时行情 (来源: {src}):\n"
            f"💰 当前价格: **{price}**\n"
            f"⬆️ 今日最高: {high}\n"
            f"⬇️ 今日最低: {low}"
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
