#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from source_selector import IntelligentSourceSelector

def test_finance_queries():
    """测试金融查询功能"""
    selector = IntelligentSourceSelector()
    
    # 测试用例
    test_cases = [
        "近十年苹果股价走势分析",
        "AAPL过去十年表现",
        "微软近5年股价趋势",
        "特斯拉过去十年股价分析",
        "Amazon最近十年股价走势"
    ]
    
    for query in test_cases:
        print(f"\n{'='*50}")
        print(f"测试查询: {query}")
        print(f"{'='*50}")
        
        try:
            result = selector._handle_finance(query, None)
            
            if result.get("handled"):
                print("✅ 查询已处理")
                print(f"提供商: {result.get('provider')}")
                print(f"股票代码: {result.get('symbols')}")
                
                # 检查是否包含关键事件
                key_events = result.get("key_events", [])
                if key_events:
                    print("\n🔍 关键事件:")
                    for event in key_events:
                        print(f"  • {event}")
                
                # 打印部分答案
                answer = result.get("answer", "")
                if answer:
                    print("\n📊 分析结果:")
                    # 只打印前500个字符，避免输出过长
                    print(answer[:500] + "..." if len(answer) > 500 else answer)
            else:
                print("❌ 查询未处理")
                print(f"原因: {result.get('reason')}")
                
        except Exception as e:
            print(f"❌ 处理查询时出错: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_finance_queries()