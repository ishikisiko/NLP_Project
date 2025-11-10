import json
from typing import Dict, List, Tuple, Any, Optional

from api import LLMClient

class IntelligentSourceSelector:
    """智能源选择器 - 带具体API配置的版本"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, *, use_llm: Optional[bool] = None):
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
                    "name": "OpenWeatherMap",
                    "url": "https://api.openweathermap.org/data/2.5/weather",
                    "type": "rest_api",
                    "description": "全球天气数据API"
                },
                {
                    "name": "和风天气", 
                    "url": "https://devapi.qweather.com/v7/weather/now",
                    "type": "rest_api",
                    "description": "中国地区天气服务"
                }
            ],
            "transportation": [
                {
                    "name": "高德地图API",
                    "url": "https://restapi.amap.com/v3/traffic/status/rectangle",
                    "type": "rest_api", 
                    "description": "实时交通路况数据"
                },
                {
                    "name": "百度交通API",
                    "url": "https://api.map.baidu.com/traffic/v1/traffic",
                    "type": "rest_api",
                    "description": "公共交通和路况信息"
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
    
    def classify_domain(self, query: str) -> str:
        """分类查询的领域"""
        if self.use_llm and self.llm_client:
            domain = self._classify_with_llm(query)
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

    def _classify_with_llm(self, query: str) -> Optional[str]:
        allowed = sorted(self.domain_keywords.keys())
        prompt = (
            "你是NLU分类器，请将用户问题归类到固定领域中。"
            "只允许以下标签: weather, transportation, finance, general."
            "输出严格的JSON，例如 {\"domain\": \"weather\"}.\n\n"
            f"用户问题: {query}"
        )
        try:
            response = self.llm_client.chat(
                system_prompt="You classify intents into fixed domains.",
                user_prompt=prompt,
                max_tokens=200,
                temperature=0.0,
            )
        except Exception:
            return None

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
    
    def select_sources(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """选择数据源 - 返回具体API信息"""
        domain = self.classify_domain(query)
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
