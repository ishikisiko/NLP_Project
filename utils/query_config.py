"""
查询关键词和映射表配置文件
用于提高代码的鲁棒性和可维护性
"""

# 时间变化查询关键词列表
TEMPORAL_CHANGE_KEYWORDS = [
    # 大学排名相关
    "大学", "高校", "学院", "学校", "排名", "QS", "THE", "ARWU", "US News",
    "university", "college", "ranking", "rankings", "education", "higher education",
    "香港中文大學", "香港科技大學", "香港大學", "CUHK", "HKUST", "HKU",
    "香港中文大学", "香港科技大学", "香港大学",
    "北京大学", "清华大学", "复旦大学", "上海交通大学", "浙江大学", "南京大学",
    "Peking University", "Tsinghua University", "Fudan University", "Shanghai Jiao Tong University",
    "Zhejiang University", "Nanjing University",
    
    # 时间范围相关
    "最近10年", "过去10年", "10年", "十年", "历年", "历史", "变化", "趋势", "发展",
    "10 years", "decade", "historical", "trend", "development", "evolution",
    "对比", "比较", "变化趋势", "时间序列", "年度", "逐年",
    "comparison", "compare", "trend over time", "time series", "yearly", "year by year",
    
    # 变化类型相关
    "增长", "下降", "波动", "变化率", "增长率", "涨跌", "提升", "改善", "恶化",
    "growth", "decline", "fluctuation", "rate of change", "growth rate", "rise and fall",
    "increase", "decrease", "improvement", "deterioration", "enhancement",
    
    # 特定时间范围
    "前三年", "近三年", "过去三年", "3 years", "three years", "前3年", "近3年",
    "前五年", "近五年", "过去五年", "5 years", "five years", "前5年", "近5年",
    "前一年", "近一年", "过去一年", "1 year", "one year", "前1年", "近1年",
    "前两年", "近两年", "过去两年", "2 years", "two years", "前2年", "近2年",
    
    # 股价和财务相关
    "股价", "stock price", "market cap", "市值", "revenue", "营收", "利润", "profit",
    "收入", "income", "资产", "assets", "负债", "liabilities", "现金流", "cash flow",
    "市盈率", "P/E ratio", "市净率", "P/B ratio", "股息", "dividend", "收益率", "return rate",
    
    # 科技公司相关
    "苹果", "Apple", "AAPL", "微软", "Microsoft", "MSFT", "谷歌", "Google", "GOOGL",
    "亚马逊", "Amazon", "AMZN", "Meta", "Facebook", "FB", "特斯拉", "Tesla", "TSLA",
    "阿里巴巴", "Alibaba", "BABA", "腾讯", "Tencent", "0700.HK", "百度", "Baidu", "BIDU",
    "字节跳动", "ByteDance", "抖音", "TikTok", "美团", "Meituan", "3690.HK",
    
    # 经济指标相关
    "GDP", "国内生产总值", "CPI", "消费者价格指数", "通胀", "inflation", "失业率", "unemployment rate",
    "利率", "interest rate", "汇率", "exchange rate", "贸易", "trade", "出口", "export", "进口", "import",
    
    # 环境和气候相关
    "气候变化", "climate change", "全球变暖", "global warming", "碳排放", "carbon emissions",
    "可再生能源", "renewable energy", "太阳能", "solar", "风能", "wind energy", "电动汽车", "electric vehicles",
    
    # 医疗健康相关
    "COVID-19", "新冠病毒", "疫情", "pandemic", "疫苗", "vaccine", "医疗", "healthcare",
    "医院", "hospital", "医生", "doctor", "护士", "nurse", "药物", "medicine", "制药", "pharmaceutical",
    
    # 教育相关
    "学生", "student", "教师", "teacher", "课程", "curriculum", "学费", "tuition", "研究", "research",
    "论文", "paper", "学术", "academic", "期刊", "journal", "引用", "citation",
    
    # 社交媒体相关
    "微博", "Weibo", "微信", "WeChat", "Twitter", "Instagram", "YouTube", "TikTok", "抖音",
    "用户", "user", "粉丝", "followers", "点赞", "likes", "分享", "shares", "评论", "comments"
]

# 时间范围检测配置
TIME_RANGE_CONFIG = {
    "1_year": {
        "keywords": ["1 year", "one year", "前1年", "近1年", "前一年", "近一年", "过去一年", "去年", "上年"],
        "years": 1,
        "coverage_threshold": 0.8
    },
    "2_years": {
        "keywords": ["2 years", "two years", "前2年", "近2年", "前两年", "近两年", "过去两年", "近两年"],
        "years": 2,
        "coverage_threshold": 0.75
    },
    "3_years": {
        "keywords": ["3 years", "three years", "前3年", "近3年", "前三年", "近三年", "过去三年"],
        "years": 3,
        "coverage_threshold": 0.7
    },
    "5_years": {
        "keywords": ["5 years", "five years", "前5年", "近5年", "前五年", "近五年", "过去五年"],
        "years": 5,
        "coverage_threshold": 0.6
    },
    "10_years": {
        "keywords": ["10 years", "ten years", "前10年", "近10年", "前十年", "近十年", "过去十年", "十年"],
        "years": 10,
        "coverage_threshold": 0.5
    },
    "15_years": {
        "keywords": ["15 years", "fifteen years", "前15年", "近15年", "前十五年", "近十五年", "过去十五年"],
        "years": 15,
        "coverage_threshold": 0.45
    },
    "20_years": {
        "keywords": ["20 years", "twenty years", "前20年", "近20年", "前二十年", "近二十年", "过去二十年", "二十年来"],
        "years": 20,
        "coverage_threshold": 0.4
    },
    "all_time": {
        "keywords": ["all time", "all-time", "历史", "历年", "有史以来", "建校以来", "成立以来", "since inception", "since founding"],
        "years": 100,  # 设置一个较大的默认值
        "coverage_threshold": 0.3
    }
}

# 查询类型分类配置
QUERY_TYPE_CONFIG = {
    "temporal_trend": {
        "keywords": ["趋势", "变化", "发展", "演变", "trend", "change", "development", "evolution", "增长", "下降", "growth", "decline"],
        "description": "时间趋势查询，关注数据随时间的变化",
        "requires_time_range": True,
        "default_time_range": "5_years"
    },
    "comparison": {
        "keywords": ["对比", "比较", "差异", "区别", "comparison", "compare", "difference", "versus", "vs", "与", "和"],
        "description": "比较查询，对比不同实体或时间点的数据",
        "requires_time_range": False,
        "default_time_range": None
    },
    "ranking": {
        "keywords": ["排名", "排行", "最佳", "最差", "top", "ranking", "rank", "best", "worst", "第一", "最后"],
        "description": "排名查询，获取实体的排名信息",
        "requires_time_range": False,
        "default_time_range": None
    },
    "statistical": {
        "keywords": ["统计", "数据", "平均值", "中位数", "标准差", "statistics", "data", "average", "median", "standard deviation"],
        "description": "统计查询，获取数据的统计信息",
        "requires_time_range": True,
        "default_time_range": "3_years"
    },
    "forecast": {
        "keywords": ["预测", "预期", "未来", "forecast", "prediction", "expectation", "future", "projection"],
        "description": "预测查询，基于历史数据预测未来趋势",
        "requires_time_range": True,
        "default_time_range": "5_years"
    },
    "causal": {
        "keywords": ["原因", "影响", "导致", "因素", "cause", "effect", "impact", "factor", "reason", "由于"],
        "description": "因果查询，探索事件之间的因果关系",
        "requires_time_range": True,
        "default_time_range": "5_years"
    },
    "descriptive": {
        "keywords": ["描述", "介绍", "什么是", "what is", "describe", "introduction", "about", "信息"],
        "description": "描述性查询，获取实体的基本信息",
        "requires_time_range": False,
        "default_time_range": None
    }
}

# 数据源映射配置
DATA_SOURCE_MAPPING = {
    "university_ranking": {
        "keywords": ["大学排名", "高校排名", "university ranking", "college ranking", "QS排名", "THE排名", "ARWU排名"],
        "primary_sources": ["qs_world_university_rankings", "times_higher_education", "arwu_shanghai_ranking"],
        "secondary_sources": ["us_news_rankings", "cwts_leiden_ranking"],
        "data_format": "structured",
        "update_frequency": "annual",
        "reliability_score": 0.9
    },
    "stock_market": {
        "keywords": ["股价", "股票", "stock price", "market data", "股市", "证券", "shares"],
        "primary_sources": ["yahoo_finance", "alpha_vantage", "quandl"],
        "secondary_sources": ["google_finance", "bloomberg"],
        "data_format": "time_series",
        "update_frequency": "daily",
        "reliability_score": 0.95
    },
    "economic_indicators": {
        "keywords": ["GDP", "CPI", "经济指标", "economic indicators", "inflation", "unemployment", "利率"],
        "primary_sources": ["world_bank", "imf", "oecd"],
        "secondary_sources": ["national_statistics", "federal_reserve"],
        "data_format": "time_series",
        "update_frequency": "monthly_quarterly",
        "reliability_score": 0.95
    },
    "company_financials": {
        "keywords": ["财报", "财务报表", "financial statements", "revenue", "profit", "assets", "liabilities"],
        "primary_sources": ["sec_edgar", "company_reports", "annual_reports"],
        "secondary_sources": ["financial_news", "market_research"],
        "data_format": "structured",
        "update_frequency": "quarterly_annual",
        "reliability_score": 0.9
    },
    "climate_data": {
        "keywords": ["气候", "温度", "降雨", "climate", "temperature", "rainfall", "weather"],
        "primary_sources": ["noaa", "nasa_climate", "met_office"],
        "secondary_sources": ["weather_stations", "satellite_data"],
        "data_format": "time_series",
        "update_frequency": "daily_monthly",
        "reliability_score": 0.85
    },
    "social_media": {
        "keywords": ["社交媒体", "微博", "微信", "Twitter", "Facebook", "Instagram", "social media"],
        "primary_sources": ["platform_apis", "social_listening_tools"],
        "secondary_sources": ["third_party_analytics", "research_studies"],
        "data_format": "unstructured",
        "update_frequency": "real_time",
        "reliability_score": 0.7
    },
    "academic_research": {
        "keywords": ["研究", "论文", "学术", "research", "paper", "academic", "journal"],
        "primary_sources": ["pubmed", "google_scholar", "arxiv", "web_of_science"],
        "secondary_sources": ["university_repositories", "research_gate"],
        "data_format": "mixed",
        "update_frequency": "continuous",
        "reliability_score": 0.85
    }
}

# 查询简化提示模板
QUERY_SIMPLIFICATION_PROMPT = (
    "Extract the core entities and topic from this query for finding historical data. "
    "Remove any time references like 'past 3 years' or 'history'. "
    "Output ONLY the keywords (e.g. 'Entity1 Entity2 Topic').\nQuery: {query}"
)

# 查询处理规则配置
QUERY_PROCESSING_RULES = {
    "preprocessing": {
        "remove_stopwords": True,
        "normalize_case": True,
        "remove_punctuation": False,
        "expand_contractions": True,
        "lemmatize": True,
        "min_token_length": 2,
        "max_token_length": 20
    },
    "entity_extraction": {
        "min_entity_confidence": 0.7,
        "max_entities_per_query": 5,
        "entity_types": ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "DATE"],
        "custom_patterns": {
            "university": r"(大学|学院|高校|University|College)",
            "stock_symbol": r"\b[A-Z]{1,5}\b",
            "time_range": r"(\d+\s*(年|years?|months?|days?))"
        }
    },
    "query_expansion": {
        "enabled": True,
        "max_synonyms_per_term": 3,
        "synonym_sources": ["wordnet", "domain_specific"],
        "expansion_methods": ["synonym", "hypernym", "hyponym"],
        "expansion_threshold": 0.6
    },
    "query_rewriting": {
        "enabled": True,
        "temporal_focus": True,
        "entity_normalization": True,
        "abbreviation_expansion": True,
        "spelling_correction": True,
        "context_aware_rewriting": True
    },
    "result_filtering": {
        "relevance_threshold": 0.5,
        "duplicate_threshold": 0.8,
        "temporal_consistency_check": True,
        "source_reliability_weight": 0.3,
        "recency_weight": 0.2
    },
    "response_formatting": {
        "include_confidence_scores": True,
        "include_source_attribution": True,
        "max_results_per_query": 10,
        "sort_by": "relevance",
        "include_summaries": True,
        "summary_length": 100
    }
}

# 性能优化相关配置
PERFORMANCE_CONFIG = {
    "caching": {
        "enabled": True,
        "cache_type": "redis",  # 可选: redis, memory, file
        "default_ttl": 3600,  # 默认缓存时间(秒)
        "max_cache_size": 1000,  # 最大缓存条目数
        "cache_key_prefix": "query:",
        "cache_invalidations": {
            "time_based": True,
            "size_based": True,
            "manual": True
        }
    },
    "parallel_processing": {
        "enabled": True,
        "max_workers": 4,  # 最大并行工作进程数
        "chunk_size": 100,  # 批处理大小
        "timeout_per_task": 10,  # 每个任务的超时时间(秒)
        "use_thread_pool": True  # 使用线程池而非进程池
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60,  # 每分钟最大请求数
        "burst_size": 10,  # 突发请求大小
        "backoff_strategy": "exponential",  # 退避策略: linear, exponential
        "max_backoff": 60  # 最大退避时间(秒)
    },
    "resource_management": {
        "max_memory_usage": "1GB",  # 最大内存使用量
        "max_cpu_usage": 80,  # 最大CPU使用率(%)
        "gc_threshold": 0.8,  # 垃圾回收阈值
        "monitoring_interval": 30  # 监控间隔(秒)
    },
    "query_optimization": {
        "enable_query_plan_cache": True,
        "enable_result_streaming": True,
        "enable_lazy_loading": True,
        "enable_compression": True,
        "compression_level": 6,  # 压缩级别(1-9)
        "batch_similar_queries": True
    },
    "connection_pooling": {
        "enabled": True,
        "max_connections": 20,  # 最大连接数
        "min_connections": 5,  # 最小连接数
        "connection_timeout": 10,  # 连接超时(秒)
        "idle_timeout": 300,  # 空闲超时(秒)
        "max_lifetime": 3600  # 连接最大生命周期(秒)
    }
}

# 默认配置
DEFAULT_CONFIG = {
    "max_query_length_for_simplification": 20,
    "default_coverage_threshold": 0.5,
    "max_granular_search_years": 10,
    "query_timeout": 30,
    "max_retry_attempts": 3,
    "cache_expiry": 3600,
    "enable_query_logging": True,
    "log_level": "INFO"
}