from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from api import HKGAIClient
from search import SearchClient, SearchHit
from rerank import BaseReranker
from timing_utils import TimingRecorder


DEFAULT_SYSTEM_PROMPT = (
    "You are an information assistant. "
    "Answer user questions concisely using ONLY the provided search results. "
    "CRITICAL: Do NOT fabricate, invent, or guess any specific data (such as scores, numbers, statistics, dates, or names) "
    "that is not EXPLICITLY stated in the search results. "
    "If specific information is not found in the search results, clearly state '未在搜索结果中找到具体数据' or 'specific data not found in search results'. "
    "When unsure, acknowledge the uncertainty instead of guessing. "
    "Always answer in the same language as the user's question."
)


class NoRAGBaseline:
    """Minimal pipeline that sends search snippets to the LLM without local retrieval."""

    def __init__(
        self,
        llm_client: HKGAIClient,
        search_client: SearchClient,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        *,
        reranker: Optional[BaseReranker] = None,
        min_rerank_score: float = 0.0,
        max_per_domain: int = 1,
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client
        self.system_prompt = system_prompt
        self.reranker = reranker
        self.min_rerank_score = min_rerank_score
        self.max_per_domain = max(1, max_per_domain)

    def _format_search_hits(self, hits: List[SearchHit]) -> str:
        if not hits:
            return "No search results were returned."

        formatted_rows = []
        for idx, hit in enumerate(hits, start=1):
            snippet = hit.snippet or "No snippet available."
            url = hit.url or "No URL available."
            title = hit.title or f"Result {idx}"
            formatted_rows.append(
                f"{idx}. {title}\n"
                f"   URL: {url}\n"
                f"   Snippet: {snippet}"
            )
        return "\n".join(formatted_rows)
    
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
    
    def _should_fallback_to_granular_search(self, query: str, hits: List[SearchHit]) -> bool:
        """判断是否应该进行颗粒化搜索fallback"""
        if not hits:
            return True
            
        # 使用LLM判断搜索结果是否满足查询需求
        query_lower = query.lower()
        
        # 检查是否包含时间变化相关的关键词
        time_keywords = ["最近10年", "过去10年", "10年", "十年", "10 years", "decade", "历年", "历史", "变化", "趋势"]
        is_time_query = any(kw in query_lower for kw in time_keywords)
        
        if not is_time_query:
            return False
            
        # 检查搜索结果中是否包含足够的时间变化数据
        combined_snippets = " ".join(hit.snippet.lower() for hit in hits if hit.snippet)
        
        # 检查是否包含年份+排名的模式
        import re
        year_rank_pattern = r'\b(20\d{2})\b.*?(?:rank|排名|position|#\d+|top\s*\d+)'
        has_year_rank_data = bool(re.search(year_rank_pattern, combined_snippets))
        
        # 检查是否包含多个年份的数据
        year_pattern = r'\b(20\d{2})\b'
        years_found = re.findall(year_pattern, combined_snippets)
        has_multiple_years = len(set(years_found)) >= 3  # 至少3个不同年份的数据
        
        # 如果没有足够的时间变化数据，则需要fallback
        return not (has_year_rank_data or has_multiple_years)
    
    def _perform_granular_search_fallback(
        self, 
        original_query: str, 
        effective_query: str, 
        num_search_results: int, 
        per_source_cap: int,
        freshness: Optional[str],
        date_restrict: Optional[str],
        timing_recorder: Optional[TimingRecorder]
    ) -> List[SearchHit]:
        """执行颗粒化搜索fallback"""
        import json
        from api import HKGAIClient
        
        # 第一步：使用LLM生成更宽泛的搜索查询
        broad_search_prompt = (
            f"原始查询：{original_query}\n\n"
            "这是一个关于时间变化的查询，但初始搜索结果没有提供足够的历史数据。"
            "请生成一个更宽泛的搜索查询，用于获取相关的历史数据。\n\n"
            "生成规则：\n"
            "1. 如果查询涉及大学排名，生成包含'历史排名'、'历年排名'或'历年变化'的查询\n"
            "2. 如果查询涉及10年变化，生成包含2016-2025年份范围的查询\n"
            "3. 如果查询涉及其他时间变化，生成包含'历史趋势'、'历年数据'或'时间序列'的查询\n"
            "4. 宽泛查询应该更通用，但仍然保持与原始查询的相关性\n\n"
            "只返回一个JSON对象，格式如下：\n"
            '{\n'
            '  "broad_query": "更宽泛的搜索查询",\n'
            '  "years": ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]\n'
            '}'
        )
        
        try:
            response = self.llm_client.chat(
                system_prompt="你是搜索查询优化专家，擅长生成更宽泛但相关的搜索查询。",
                user_prompt=broad_search_prompt,
                max_tokens=200,
                temperature=0.3,
            )
            
            content = response.get("content", "")
            # 尝试解析JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                llm_result = json.loads(json_str)
                broad_query = llm_result.get("broad_query", effective_query)
                years = llm_result.get("years", [])
            else:
                broad_query = effective_query
                years = []
        except Exception as e:
            print(f"LLM生成宽泛查询失败: {e}")
            broad_query = effective_query
            years = []
        
        # 第二步：执行宽泛搜索
        print(f"🔍 执行宽泛搜索: {broad_query}")
        broad_hits = self.search_client.search(
            broad_query,
            num_results=num_search_results * 2,  # 获取更多结果以便筛选
            per_source_limit=per_source_cap * 2,
            freshness=freshness,
            date_restrict=date_restrict,
        )
        
        # 分析宽泛搜索结果中的时间变化数据
        if broad_hits:
            combined_snippets = " ".join(hit.snippet.lower() for hit in broad_hits if hit.snippet)
            import re
            
            # 统计找到的年份数量
            year_pattern = r'\b(20\d{2})\b'
            years_found = re.findall(year_pattern, combined_snippets)
            unique_years = set(years_found)
            
            # 检查是否包含排名/位置信息
            rank_indicators = ['rank', '排名', 'position', '#', 'top', 'ranking']
            has_rank_data = any(indicator in combined_snippets for indicator in rank_indicators)
            
            print(f"📊 宽泛搜索分析：找到 {len(unique_years)} 个不同年份的数据，包含排名信息: {has_rank_data}")
            
            # 对于时间变化查询，总是执行颗粒化搜索以获取更完整的数据
            # 注释掉早期返回，确保总是执行颗粒化搜索
            # if len(unique_years) >= 3 and has_rank_data:
            #     print("✅ 宽泛搜索已找到足够的时间变化数据，无需进一步颗粒化搜索")
            #     return broad_hits[:num_search_results * 2]
            print(f"📊 宽泛搜索分析：找到 {len(unique_years)} 个不同年份的数据，包含排名信息: {has_rank_data}")
            print("🔄 继续执行颗粒化搜索以获取更完整的历史数据...")
        
        # 第三步：从宽泛搜索结果中提取年份信息
        if not years:
            # 如果LLM没有提供年份，尝试从查询中提取
            import re
            year_range_match = re.search(r'(20\d{2})\s*[-至到]\s*(20\d{2})', original_query)
            if year_range_match:
                start_year = int(year_range_match.group(1))
                end_year = int(year_range_match.group(2))
                years = [str(year) for year in range(start_year, end_year + 1)]
            else:
                # 默认使用最近10年
                current_year = 2025
                years = [str(year) for year in range(current_year - 9, current_year + 1)]
        
        # 第四步：执行颗粒化搜索（对于时间变化查询总是执行）
        if years:  # 移除_should_fallback_to_granular_search检查，确保总是执行颗粒化搜索
            print("🔍 开始颗粒化搜索...")
            granular_hits = []
            
            # 只使用Google搜索进行颗粒化查询
            google_client = None
            print(f"🔍 查找Google搜索客户端...")
            print(f"   search_client类型: {type(self.search_client)}")
            print(f"   search_client属性: {dir(self.search_client)}")
            
            # 检查search_client是否是CombinedSearchClient
            if hasattr(self.search_client, "clients"):
                print(f"   找到clients属性，客户端数量: {len(self.search_client.clients)}")
                for i, client in enumerate(self.search_client.clients):
                    print(f"   客户端 {i}: {type(client)}")
                    if hasattr(client, "source_id"):
                        print(f"      source_id: {client.source_id}")
                    if hasattr(client, "source_id") and client.source_id == "google":
                        google_client = client
                        print(f"   ✅ 找到Google搜索客户端!")
            else:
                print(f"   ❌ search_client没有clients属性")
            
            if google_client:
                # 优化颗粒化查询：智能选择关键年份
                selected_years = years
                if len(years) > 6:
                    # 智能选择策略：选择开始年份、结束年份和中间的几个关键年份
                    # 对于10年查询，选择第1年、第3年、第5年、第7年、第10年
                    if len(years) == 10:
                        selected_years = [years[0], years[2], years[4], years[6], years[-1]]
                    else:
                        # 对于其他长度的年份列表，均匀分布选择
                        step = max(1, len(years) // 5)
                        selected_years = [years[i] for i in range(0, len(years), step)]
                        if years[-1] not in selected_years:
                            selected_years.append(years[-1])
                    
                    print(f"📅 优化颗粒化搜索，选择关键年份: {selected_years}")
                else:
                    print(f"📅 执行颗粒化搜索，年份: {selected_years}")
                
                # 为每个选定的年份生成查询
                for year in selected_years:
                    # 智能生成更精确的年份查询
                    query_lower = original_query.lower()
                    
                    # 提取查询中的关键实体（大学名称等）
                    import re
                    # 提取大学名称
                    universities = []
                    if "香港中文大學" in original_query or "香港中文大学" in original_query:
                        universities.append("Chinese University of Hong Kong")
                        universities.append("CUHK")
                    if "香港科技大學" in original_query or "香港科技大学" in original_query:
                        universities.append("Hong Kong University of Science and Technology")
                        universities.append("HKUST")
                    
                    # 根据查询类型生成不同的年份查询
                    if "qs" in query_lower and ("排名" in original_query or "ranking" in query_lower):
                        # QS排名查询 - 生成更简洁的查询
                        if universities:
                            # 如果有具体的大学，查询这些大学的QS排名
                            for uni in universities:
                                year_query = f"QS world university rankings {year} {uni}"
                                print(f"🔍 搜索年份 {year}: {year_query}")
                                try:
                                    year_hits = google_client.search(
                                        year_query,
                                        num_results=max(2, num_search_results // (len(selected_years) * len(universities))),
                                        freshness=freshness,
                                        date_restrict=f"{year}-01-01..{year}-12-31",  # 限制在特定年份内
                                    )
                                    granular_hits.extend(year_hits)
                                except Exception as e:
                                    print(f"年份 {year} 搜索失败: {e}")
                            
                            # 额外查询：香港大学QS排名（作为参考）
                            hk_query = f"QS world university rankings {year} Hong Kong universities ranking"
                            print(f"🔍 搜索年份 {year} (香港大学排名): {hk_query}")
                            try:
                                hk_hits = google_client.search(
                                    hk_query,
                                    num_results=2,
                                    freshness=freshness,
                                    date_restrict=f"{year}-01-01..{year}-12-31",  # 限制在特定年份内
                                )
                                granular_hits.extend(hk_hits)
                            except Exception as e:
                                print(f"年份 {year} 香港大学排名搜索失败: {e}")
                        else:
                            # 如果没有具体大学，查询QS排名总体情况
                            year_query = f"QS world university rankings {year}"
                            print(f"🔍 搜索年份 {year}: {year_query}")
                            try:
                                year_hits = google_client.search(
                                    year_query,
                                    num_results=max(3, num_search_results // len(selected_years)),
                                    freshness=freshness,
                                    date_restrict=f"{year}-01-01..{year}-12-31",  # 限制在特定年份内
                                )
                                granular_hits.extend(year_hits)
                            except Exception as e:
                                print(f"年份 {year} 搜索失败: {e}")
                        continue  # 跳过后续的通用查询逻辑
                    elif "the" in query_lower and ("排名" in original_query or "ranking" in query_lower):
                        # THE排名查询
                        year_query = f"THE world university rankings {year}"
                    elif "arwu" in query_lower or "软科" in original_query:
                        # ARWU排名查询
                        year_query = f"ARWU academic ranking of world universities {year}"
                    elif "排名" in original_query or "ranking" in query_lower:
                        # 通用排名查询
                        year_query = f"university rankings {year}"
                    elif "大学" in original_query or "university" in query_lower:
                        # 大学相关查询
                        year_query = f"university {year}"
                    else:
                        # 通用查询
                        year_query = f"{original_query} {year}"
                    
                    print(f"🔍 搜索年份 {year}: {year_query}")
                    
                    try:
                        year_hits = google_client.search(
                            year_query,
                            num_results=max(3, num_search_results // len(selected_years)),
                            freshness=freshness,
                            date_restrict=f"{year}-01-01..{year}-12-31",  # 限制在特定年份内
                        )
                        granular_hits.extend(year_hits)
                    except Exception as e:
                        print(f"年份 {year} 搜索失败: {e}")
                
                # 合并宽泛搜索和颗粒化搜索结果，优先保留颗粒化搜索结果
                all_hits = granular_hits + broad_hits
                
                # 智能去重：保留更相关的结果
                seen_urls = set()
                deduped_hits = []
                for hit in all_hits:
                    url_key = hit.url or ""
                    if url_key not in seen_urls:
                        seen_urls.add(url_key)
                        deduped_hits.append(hit)
                
                # 按相关性排序：优先包含年份和排名信息的结果
                def hit_relevance_score(hit):
                    score = 0
                    if hit.snippet:
                        snippet = hit.snippet.lower()
                        # 检查是否包含年份
                        import re
                        years_in_snippet = re.findall(r'\b(20\d{2})\b', snippet)
                        score += len(years_in_snippet) * 2
                        # 检查是否包含排名信息
                        rank_keywords = ['rank', '排名', 'position', '#', 'top']
                        score += sum(1 for kw in rank_keywords if kw in snippet)
                    return score
                
                deduped_hits.sort(key=hit_relevance_score, reverse=True)
                
                print(f"✅ 颗粒化搜索完成，共获得 {len(deduped_hits)} 条结果")
                return deduped_hits[:num_search_results * 2]  # 返回更多结果以便筛选
            else:
                print("⚠️ 未找到Google搜索客户端，无法执行颗粒化搜索")
                return broad_hits
        else:
            return broad_hits

    def build_prompt(self, query: str, hits: List[SearchHit], ranking_info: str = "") -> str:
        context_block = self._format_search_hits(hits)
        
        # 检查是否是排名查询
        is_ranking_query = any(keyword in query.lower() for keyword in ['排名', 'ranking', 'rank'])
        
        if is_ranking_query:
            special_instructions = (
                "SPECIAL INSTRUCTIONS FOR RANKING QUERIES:\n"
                "1. Carefully extract ALL ranking data from the search results, even if it's not in a standard format.\n"
                "2. Look for patterns like 'Year #Rank', 'Year: Rank', 'Year ranked', 'Year position', 'Rank #Year', etc.\n"
                "3. Pay special attention to university official websites which often contain ranking tables.\n"
                "4. Extract ranking information from titles, snippets, and any visible text.\n"
                "5. For university rankings, look for both global rankings and regional rankings.\n"
                "6. If ranking data is scattered across multiple results, compile it into a coherent comparison table.\n"
                "7. If specific years are missing from the results, explicitly mention which years are not covered.\n"
                "8. For Chinese universities, look for both English and Chinese names (CUHK/香港中文大學, HKUST/香港科技大學).\n"
                "9. If ranking data is provided below, use it to create a comprehensive comparison table.\n\n"
            )
            # For ranking queries, modify the important rules to allow using pre-extracted ranking data
            important_rules = (
                "IMPORTANT RULES FOR RANKING QUERIES:\n"
                "1. You may use specific ranking data that is EXPLICITLY mentioned in the search results below.\n"
                "2. You may ALSO use the pre-extracted ranking data provided below the search results.\n"
                "3. If ranking data is missing for certain years, explicitly mention which years are not covered.\n"
                "4. DO NOT guess or invent ranking numbers that are not found in either the search results or the pre-extracted data.\n"
                "5. Create a comprehensive comparison table using all available ranking data.\n\n"
            )
        else:
            special_instructions = ""
            important_rules = (
                "IMPORTANT RULES:\n"
                "1. ONLY include specific data (scores, statistics, numbers, names) that are EXPLICITLY mentioned in the search results below.\n"
                "2. If specific data (like individual player scores, detailed statistics) is NOT found in the search results, "
                "say '搜索结果中未提及具体数据' or 'not mentioned in search results' - DO NOT guess or invent numbers.\n"
                "3. For sports queries: only report scores and statistics that appear verbatim in the snippets.\n\n"
            )
        
        return (
            "You are given a set of search results. "
            "Use them to answer the question at the end. "
            "When citing sources, use the format (URL 1), (URL 2), etc., "
            "where the number corresponds to the search result number.\n\n"
            f"{important_rules}"
            f"{special_instructions}"
            f"Search Results:\n{context_block}\n"
            f"{ranking_info}\n\n"
            f"Question: {query}\n\n"
            "Answer (remember: NO fabricated data):"
        )

    def _extract_ranking_data(self, hits: List[SearchHit]) -> Dict[str, object]:
        """Extract ranking data from search hits for university ranking queries."""
        import re
        
        cuhk_rankings = {}
        hkust_rankings = {}
        other_rankings = {}
        
        # 信任度评分：官方大学网站 > QS官方网站 > 新闻媒体 > 其他
        def get_source_trust_score(url: str, title: str) -> int:
            """根据URL和标题评估来源的信任度"""
            if not url:
                return 1
            
            url_lower = url.lower()
            title_lower = title.lower() if title else ""
            
            # 官方大学网站
            if 'cuhk.edu.hk' in url_lower or 'cuhk.edu.cn' in url_lower:
                return 10
            if 'hkust.edu.hk' in url_lower:
                return 10
            
            # QS官方网站
            if 'topuniversities.com' in url_lower and 'university-rankings' in url_lower:
                return 9
            
            # 知名教育媒体
            if 'timeshighereducation.com' in url_lower:
                return 8
            if 'scmp.com' in url_lower:
                return 7
            
            # 一般新闻媒体
            if any(domain in url_lower for domain in ['news', 'reuters', 'bbc', 'cnn']):
                return 5
            
            # 其他来源
            return 3
        
        for i, hit in enumerate(hits, 1):
            title = hit.title if hit.title else ''
            snippet = hit.snippet if hit.snippet else ''
            url = hit.url if hit.url else ''
            text = f"{title} {snippet}"
            
            # 获取来源信任度
            trust_score = get_source_trust_score(url, title)
            
            # 检查是否与CUHK相关
            is_cuhk = ('cuhk' in text.lower() or 'chinese university of hong kong' in text.lower() or 
                       '香港中文' in text or '香港中文大學' in text)
            
            # 检查是否与HKUST相关
            is_hkust = ('hkust' in text.lower() or 'hong kong university of science and technology' in text.lower() or 
                        '香港科技' in text or '香港科技大學' in text)
            
            # 提取排名信息的多种模式，优先使用更精确的模式
            rank_patterns = [
                # 高可信度模式：明确的年份和排名组合
                (r'(20\d{2})[^0-9]*#?(\d{1,3})', 9),  # 2020 #42
                (r'(20\d{2})[^0-9]*ranked?[^0-9]*(\d{1,3})', 8),  # 2020 ranked 42
                (r'(20\d{2})[^0-9]*排名[^0-9]*(\d{1,3})', 8),  # 2020 排名 42
                (r'#?(\d{1,3})[^0-9]*(20\d{2})', 7),  # #42 2020
                (r'ranked?[^0-9]*(\d{1,3})[^0-9]*(20\d{2})', 7),  # ranked 42 2020
                (r'排名[^0-9]*(\d{1,3})[^0-9]*(20\d{2})', 7),  # 排名 42 2020
                
                # 中等可信度模式：QS相关
                (r'QS World University Rankings[^0-9]*(\d{1,3})', 6),  # QS World University Rankings 42
                (r'QS.*?(\d{1,3})', 5),  # QS #42
                
                # 低可信度模式：单独的排名信息
                (r'(\d{1,3})', 3),  # 单独的数字
            ]
            
            # 对每个模式进行匹配
            for pattern, pattern_score in rank_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # 处理匹配结果
                    if isinstance(match, tuple) and len(match) == 2:
                        # 包含年份和排名的情况
                        year_str, rank_str = match
                        if year_str.isdigit() and rank_str.isdigit():
                            year = int(year_str)
                            rank = int(rank_str)
                            
                            # 验证年份和排名的合理性
                            if 2000 <= year <= 2030 and 1 <= rank <= 500:
                                # 额外验证规则：过滤明显不合理的排名
                                # 对于CUHK和HKUST这样的顶尖大学，世界排名通常在1-100之间
                                # 排名在200+的可能是特定领域排名或地区排名，需要更严格的验证
                                is_reasonable_rank = True
                                if rank > 150:
                                    # 对于高排名数字，检查是否包含特定关键词
                                    if not any(keyword in text.lower() for keyword in 
                                              ['asia', '亚洲', 'subject', '学科', 'faculty', '学院', 'engineering', '工程']):
                                        # 如果没有明确说明是地区排名或学科排名，降低信任度
                                        combined_score = trust_score + pattern_score - 5
                                        print(f"警告: {year}年排名#{rank}可能不是全球排名，降低信任度")
                                
                                # 计算综合信任度
                                combined_score = trust_score + pattern_score
                                
                                # 只接受高信任度的数据
                                if combined_score >= 10:  # 只接受高信任度的数据
                                    if is_cuhk:
                                        # 如果该年份已有数据，只保留更高信任度的数据
                                        if year not in cuhk_rankings or combined_score > cuhk_rankings[year][1]:
                                            cuhk_rankings[year] = (rank, combined_score)
                                            print(f"提取到CUHK排名: {year}年 #{rank} (信任度: {combined_score}, 来源: URL {i})")
                                    elif is_hkust:
                                        if year not in hkust_rankings or combined_score > hkust_rankings[year][1]:
                                            hkust_rankings[year] = (rank, combined_score)
                                            print(f"提取到HKUST排名: {year}年 #{rank} (信任度: {combined_score}, 来源: URL {i})")
                    elif isinstance(match, str) and match.isdigit():
                        # 只有排名的情况，尝试从文本中提取年份
                        rank = int(match)
                        if 1 <= rank <= 500:
                            # 尝试从文本中提取年份
                            year_matches = re.findall(r'\b(20\d{2})\b', text)
                            for year_str in year_matches:
                                year = int(year_str)
                                if 2000 <= year <= 2030:
                                    # 额外验证规则：过滤明显不合理的排名
                                    if rank > 150:
                                        # 对于高排名数字，检查是否包含特定关键词
                                        if not any(keyword in text.lower() for keyword in 
                                                  ['asia', '亚洲', 'subject', '学科', 'faculty', '学院', 'engineering', '工程']):
                                            # 如果没有明确说明是地区排名或学科排名，降低信任度
                                            pattern_score_adjusted = pattern_score - 5
                                            print(f"警告: {year}年排名#{rank}可能不是全球排名，降低信任度")
                                        else:
                                            pattern_score_adjusted = pattern_score
                                    else:
                                        pattern_score_adjusted = pattern_score
                                    
                                    # 计算综合信任度
                                    combined_score = trust_score + pattern_score_adjusted
                                    
                                    # 只接受高信任度的数据
                                    if combined_score >= 10:  # 只接受高信任度的数据
                                        if is_cuhk:
                                            if year not in cuhk_rankings or combined_score > cuhk_rankings[year][1]:
                                                cuhk_rankings[year] = (rank, combined_score)
                                                print(f"提取到CUHK排名: {year}年 #{rank} (信任度: {combined_score}, 来源: URL {i})")
                                        elif is_hkust:
                                            if year not in hkust_rankings or combined_score > hkust_rankings[year][1]:
                                                hkust_rankings[year] = (rank, combined_score)
                                                print(f"提取到HKUST排名: {year}年 #{rank} (信任度: {combined_score}, 来源: URL {i})")
        
        # 查找包含"top"的排名信息
        for i, hit in enumerate(hits, 1):
            title = hit.title if hit.title else ''
            snippet = hit.snippet if hit.snippet else ''
            url = hit.url if hit.url else ''
            text = f"{title} {snippet}"
            
            # 获取来源信任度
            trust_score = get_source_trust_score(url, title)
            
            # 查找包含"top"的排名信息
            top_pattern = r'(20\d{2})[^0-9]*top\s*(\d{1,3})'
            matches = re.findall(top_pattern, text.lower())
            for year_str, rank_str in matches:
                if year_str.isdigit() and rank_str.isdigit():
                    year = int(year_str)
                    rank = int(rank_str)
                    
                    # 验证年份和排名的合理性
                    if 2000 <= year <= 2030 and 1 <= rank <= 500:
                        # 额外验证规则：过滤明显不合理的排名
                        top_pattern_score = 6  # "top"模式的信任度
                        if rank > 150:
                            # 对于高排名数字，检查是否包含特定关键词
                            if not any(keyword in text.lower() for keyword in 
                                      ['asia', '亚洲', 'subject', '学科', 'faculty', '学院', 'engineering', '工程']):
                                # 如果没有明确说明是地区排名或学科排名，降低信任度
                                top_pattern_score -= 5
                                print(f"警告: {year}年Top {rank}可能不是全球排名，降低信任度")
                        
                        # 计算综合信任度
                        combined_score = trust_score + top_pattern_score
                        
                        # 只接受高信任度的数据
                        if combined_score >= 10:
                            is_cuhk = ('cuhk' in text.lower() or 'chinese university of hong kong' in text.lower() or 
                                       '香港中文' in text or '香港中文大學' in text)
                            is_hkust = ('hkust' in text.lower() or 'hong kong university of science and technology' in text.lower() or 
                                         '香港科技' in text or '香港科技大學' in text)
                            
                            if is_cuhk:
                                if year not in cuhk_rankings or combined_score > cuhk_rankings[year][1]:
                                    cuhk_rankings[year] = (rank, combined_score)
                                    print(f"提取到CUHK排名(Top): {year}年 Top {rank} (信任度: {combined_score}, 来源: URL {i})")
                            elif is_hkust:
                                if year not in hkust_rankings or combined_score > hkust_rankings[year][1]:
                                    hkust_rankings[year] = (rank, combined_score)
                                    print(f"提取到HKUST排名(Top): {year}年 Top {rank} (信任度: {combined_score}, 来源: URL {i})")
        
        # 提取排名数据，只保留排名值（去掉信任度分数）
        cuhk_final = {year: data[0] for year, data in cuhk_rankings.items()}
        hkust_final = {year: data[0] for year, data in hkust_rankings.items()}
        
        return {
            'cuhk_rankings': cuhk_final,
            'hkust_rankings': hkust_final,
            'other_rankings': other_rankings
        }

    def answer(
        self,
        query: str,
        *,
        search_query: Optional[str] = None,
        num_search_results: int = 5,
        per_source_limit: Optional[int] = None,
        max_tokens: int = 5000,
        temperature: float = 0.3,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
        timing_recorder: Optional[TimingRecorder] = None,
        reference_limit: Optional[int] = None,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        # Prefer keyword-focused query generated upstream when available.
        effective_query = search_query.strip() if search_query else query

        per_source_cap = per_source_limit if per_source_limit is not None else num_search_results
        hits = self.search_client.search(
            effective_query,
            num_results=num_search_results,
            per_source_limit=per_source_cap,
            freshness=freshness,
            date_restrict=date_restrict,
        )
        
        # 检查是否需要LLM fallback（针对时间变化类查询）
        if self._is_temporal_change_query(query):
            # 对于时间变化查询，总是尝试执行颗粒化搜索以获取更全面的历史数据
            print("🔄 检测到时间变化查询，启动LLM fallback机制以获取历史数据...")
            fallback_hits = self._perform_granular_search_fallback(query, effective_query, num_search_results, per_source_cap, freshness, date_restrict, timing_recorder)
            if fallback_hits:
                hits = fallback_hits
                print(f"✅ Fallback搜索完成，获得{len(fallback_hits)}条结果")
        
        # 检查是否是排名查询，如果是则提取排名数据
        is_ranking_query = any(keyword in query.lower() for keyword in ['排名', 'ranking', 'rank'])
        ranking_data = None
        if is_ranking_query:
            print("🔍 检测到排名查询，提取排名数据...")
            ranking_data = self._extract_ranking_data(hits)
            print(f"✅ 提取到CUHK排名数据: {ranking_data['cuhk_rankings']}")
            print(f"✅ 提取到HKUST排名数据: {ranking_data['hkust_rankings']}")
        if timing_recorder:
            timings_getter = getattr(self.search_client, "get_last_timings", None)
            if callable(timings_getter):
                timing_recorder.extend_search_timings(timings_getter())
        search_warnings: List[str] = []
        get_last_errors = getattr(self.search_client, "get_last_errors", None)
        if callable(get_last_errors):
            errors = get_last_errors() or []
            if hits and errors:
                for item in errors:
                    source = str(item.get("source") or "搜索服务")
                    detail = str(item.get("error") or "未知错误")
                    if source.lower().startswith("mcp"):
                        search_warnings.append(f"{source} 未正常工作，已使用其他搜索结果。原因：{detail}")
                    else:
                        search_warnings.append(f"{source} 出现异常：{detail}")
        hits, rerank_meta = self._apply_rerank(query, hits, limit=num_search_results)
        # 如果是排名查询且有提取的排名数据，则将其添加到提示中
        if is_ranking_query and ranking_data:
            ranking_info = "\n\n提取的排名数据:\n"
            if ranking_data['cuhk_rankings']:
                ranking_info += "CUHK排名:\n"
                for year, rank in sorted(ranking_data['cuhk_rankings'].items()):
                    ranking_info += f"- {year}年: #{rank}\n"
            
            if ranking_data['hkust_rankings']:
                ranking_info += "HKUST排名:\n"
                for year, rank in sorted(ranking_data['hkust_rankings'].items()):
                    ranking_info += f"- {year}年: #{rank}\n"
            
            # 修改提示以包含排名数据
            context_block = self._format_search_hits(hits)
            user_prompt = self.build_prompt(query, hits, ranking_info)
        else:
            user_prompt = self.build_prompt(query, hits)
        response_start = time.perf_counter()
        try:
            response = self.llm_client.chat(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                images=images,
            )
        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="search_answer",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm_client, "provider", None),
                    model=getattr(self.llm_client, "model_id", None),
                )

        # Build answer with URL references
        answer = response.get("content")
        reference_hits = hits if reference_limit is None else hits[:reference_limit]
        if answer and reference_hits:
            # Append reference list
            answer += "\n\n**参考链接：**\n"
            for idx, hit in enumerate(reference_hits, start=1):
                url = hit.url or "No URL available."
                title = hit.title or f"结果 {idx}"
                answer += f"{idx}. [{title}]({url})\n"

        result: Dict[str, object] = {
            "query": query,
            "answer": answer,
            "search_hits": [asdict(hit) for hit in hits],
            "llm_raw": response.get("raw"),
            "llm_warning": response.get("warning"),
            "llm_error": response.get("error"),
            "rerank": rerank_meta or None,
            "search_query": effective_query,
        }
        if search_warnings:
            result["search_warnings"] = search_warnings
        return result

    def answer_stream(
        self,
        query: str,
        *,
        search_query: Optional[str] = None,
        num_search_results: int = 5,
        per_source_limit: Optional[int] = None,
        max_tokens: int = 5000,
        temperature: float = 0.3,
        freshness: Optional[str] = None,
        date_restrict: Optional[str] = None,
        timing_recorder: Optional[TimingRecorder] = None,
        reference_limit: Optional[int] = None,
    ):
        # Prefer keyword-focused query generated upstream when available.
        effective_query = search_query.strip() if search_query else query

        per_source_cap = per_source_limit if per_source_limit is not None else num_search_results
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
        search_warnings: List[str] = []
        get_last_errors = getattr(self.search_client, "get_last_errors", None)
        if callable(get_last_errors):
            errors = get_last_errors() or []
            if hits and errors:
                for item in errors:
                    source = str(item.get("source") or "搜索服务")
                    detail = str(item.get("error") or "未知错误")
                    if source.lower().startswith("mcp"):
                        search_warnings.append(f"{source} 未正常工作，已使用其他搜索结果。原因：{detail}")
                    else:
                        search_warnings.append(f"{source} 出现异常：{detail}")
        
        hits, rerank_meta = self._apply_rerank(query, hits, limit=num_search_results)

        # First, yield preliminary data
        preliminary_data = {
            "query": query,
            "search_hits": [asdict(hit) for hit in hits],
            "rerank": rerank_meta or None,
            "search_query": effective_query,
        }
        if search_warnings:
            preliminary_data["search_warnings"] = search_warnings
        
        yield json.dumps({"type": "preliminary", "data": preliminary_data})


        user_prompt = self.build_prompt(query, hits)
        response_start = time.perf_counter()
        
        # Stream the response
        full_answer = ""
        try:
            stream = self.llm_client.chat_stream(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for chunk in stream:
                if chunk.startswith("Error:"):
                    yield json.dumps({"type": "error", "data": chunk})
                    return
                full_answer += chunk
                yield json.dumps({"type": "content", "data": chunk})

        finally:
            if timing_recorder:
                duration_ms = (time.perf_counter() - response_start) * 1000
                timing_recorder.record_llm_call(
                    label="search_answer_stream",
                    duration_ms=duration_ms,
                    provider=getattr(self.llm_client, "provider", None),
                    model=getattr(self.llm_client, "model_id", None),
                )

        # Finally, yield the references
        reference_hits = hits if reference_limit is None else hits[:reference_limit]
        if full_answer and reference_hits:
            reference_text = "\n\n**参考链接：**\n"
            for idx, hit in enumerate(reference_hits, start=1):
                url = hit.url or "No URL available."
                title = hit.title or f"结果 {idx}"
                reference_text += f"{idx}. [{title}]({url})\n"
            yield json.dumps({"type": "references", "data": reference_text})

    def _apply_rerank(
        self,
        query: str,
        hits: List[SearchHit],
        *,
        limit: Optional[int] = None,
    ) -> Tuple[List[SearchHit], List[Dict[str, object]]]:
        if not self.reranker or not hits:
            return hits, []

        try:
            reranked = self.reranker.rerank(query, hits)
        except Exception as exc:  # pragma: no cover - best effort resilience
            return hits, [{"error": str(exc)}]

        filtered: List[SearchHit] = []
        metadata: List[Dict[str, object]] = []
        domain_counts: Dict[str, int] = {}
        max_results = limit or len(reranked)

        for item in reranked:
            domain = self._extract_domain(item.hit.url)
            if domain and domain_counts.get(domain, 0) >= self.max_per_domain:
                metadata.append(
                    {
                        "url": item.hit.url,
                        "score": item.score,
                        "dropped": "per_domain_limit",
                    }
                )
                continue
            if item.score is not None and item.score < self.min_rerank_score:
                metadata.append(
                    {
                        "url": item.hit.url,
                        "score": item.score,
                        "dropped": "below_min_score",
                    }
                )
                continue

            filtered.append(item.hit)
            metadata.append(
                {
                    "url": item.hit.url,
                    "score": item.score,
                    "kept": True,
                }
            )

            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            if len(filtered) >= max_results:
                break

        if not filtered:
            return hits, metadata

        return filtered, metadata

    @staticmethod
    def _extract_domain(url: str) -> Optional[str]:
        if not url:
            return None
        return urlparse(url).netloc or None
