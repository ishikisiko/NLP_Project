from source_selector import IntelligentSourceSelector

def test_enhanced_sources():
    """测试增强版数据源选择"""
    selector = IntelligentSourceSelector()
    
    print("🧪 测试增强版数据源选择器")
    print("=" * 50)
    
    test_cases = [
        "今天北京天气怎么样？",
        "上海交通拥堵情况",
        "腾讯股票实时价格",
        "机器学习基础知识"
    ]
    
    for query in test_cases:
        print(f"\n📝 查询: '{query}'")
        domain, sources = selector.select_sources(query)
        
        print(f"📊 数据源详情:")
        for i, source in enumerate(sources, 1):
            print(f"   {i}. {source['name']}")
            print(f"      类型: {source['type']}")
            print(f"      网址: {source['url']}")
            print(f"      描述: {source['description']}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_enhanced_sources()