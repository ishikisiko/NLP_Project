"""
测试地点搜索功能修复
验证 "距离HKUST最近的KFC是哪家" 查询能否正确处理
"""

import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_location_classification():
    """测试地点查询的领域分类"""
    from source_selector import IntelligentSourceSelector
    
    selector = IntelligentSourceSelector(use_llm=False)
    
    test_queries = [
        "距离HKUST最近的KFC是哪家",
        "香港科技大学附近的餐厅",
        "离北京大学最近的星巴克",
        "nearest McDonald's to Central",
        "find pharmacy near me",
        "附近有什么超市",
    ]
    
    print("=" * 60)
    print("测试地点查询的领域分类")
    print("=" * 60)
    
    for query in test_queries:
        domain = selector._classify_with_keywords(query)
        print(f"\n查询: {query}")
        print(f"分类结果: {domain}")
        
        if domain == "location":
            # 测试提取参考地点和目标类型
            parsed = selector._extract_location_query(query)
            if parsed:
                print(f"  参考地点: {parsed.get('reference_location')}")
                print(f"  目标类型: {parsed.get('target_type')}")
            else:
                print("  无法提取参考地点和目标类型")
    
    print("\n" + "=" * 60)

def test_location_query_extraction():
    """测试地点查询参数提取"""
    from source_selector import IntelligentSourceSelector
    
    selector = IntelligentSourceSelector(use_llm=False)
    
    test_cases = [
        ("距离HKUST最近的KFC是哪家", "HKUST", "KFC"),
        ("香港科技大学附近的麦当劳", "香港科技大学", "麦当劳"),
        ("离北京大学最近的星巴克在哪", "北京大学", "星巴克"),
        ("nearest KFC to HKUST", "HKUST", "KFC"),
        ("find Starbucks near Central", "Central", "Starbucks"),
    ]
    
    print("\n" + "=" * 60)
    print("测试地点查询参数提取")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for query, expected_ref, expected_target in test_cases:
        parsed = selector._extract_location_query(query)
        
        print(f"\n查询: {query}")
        
        if parsed:
            ref = parsed.get("reference_location", "")
            target = parsed.get("target_type", "")
            print(f"  提取结果: 参考地点='{ref}', 目标类型='{target}'")
            
            # 检查是否包含预期值（不要求完全匹配）
            ref_match = expected_ref.lower() in ref.lower() or ref.lower() in expected_ref.lower()
            target_match = expected_target.lower() in target.lower() or target.lower() in expected_target.lower()
            
            if ref_match and target_match:
                print("  ✅ 通过")
                passed += 1
            else:
                print(f"  ❌ 失败 (期望: 参考地点包含'{expected_ref}', 目标类型包含'{expected_target}')")
                failed += 1
        else:
            print("  ❌ 失败 (无法提取)")
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0

def test_haversine_distance():
    """测试距离计算"""
    from source_selector import IntelligentSourceSelector
    
    # HKUST 坐标: 22.3363, 114.2654
    # 将军澳 KFC 大约坐标: 22.3078, 114.2599
    
    hkust_lat, hkust_lng = 22.3363, 114.2654
    tko_lat, tko_lng = 22.3078, 114.2599
    
    distance = IntelligentSourceSelector._haversine_distance(
        hkust_lat, hkust_lng, tko_lat, tko_lng
    )
    
    print("\n" + "=" * 60)
    print("测试距离计算 (Haversine)")
    print("=" * 60)
    print(f"HKUST ({hkust_lat}, {hkust_lng}) 到 将军澳 ({tko_lat}, {tko_lng})")
    print(f"计算距离: {distance:.2f} 公里")
    
    # 预期距离约 3-4 公里
    if 2 < distance < 5:
        print("✅ 距离计算合理")
        return True
    else:
        print("❌ 距离计算可能有误")
        return False

def main():
    print("\n🔍 地点搜索功能修复测试\n")
    
    try:
        test_location_classification()
        extraction_ok = test_location_query_extraction()
        distance_ok = test_haversine_distance()
        
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        if extraction_ok and distance_ok:
            print("✅ 所有测试通过！地点搜索功能已正确实现。")
            print("\n下一步：使用 conda activate env1 后运行:")
            print('  python main.py "距离HKUST最近的KFC是哪家" --pretty')
        else:
            print("❌ 部分测试失败，请检查实现。")
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在正确的环境中运行此测试。")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()