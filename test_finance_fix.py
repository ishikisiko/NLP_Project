#!/usr/bin/env python
"""Test script to verify the finance symbol extraction fix for Intel and AMD."""

import sys
sys.path.insert(0, '.')

from source_selector import IntelligentSourceSelector

def test_extract_finance_symbols():
    """Test that Intel and AMD are correctly extracted as INTC and AMD."""
    selector = IntelligentSourceSelector(use_llm=False)
    
    test_cases = [
        # Test case: query, expected symbols
        ("对比Intel和AMD近三股价变化以及原因", {"INTC", "AMD"}),
        ("Intel和AMD的股价对比", {"INTC", "AMD"}),
        ("英特尔和AMD股票", {"INTC", "AMD"}),
        ("INTC和AMD的表现", {"INTC", "AMD"}),
        ("苹果和微软股价", {"AAPL", "MSFT"}),
        ("英伟达股票价格", {"NVDA"}),
        ("Current Date: 2025-11-30", set()),  # DATE should not be extracted
    ]
    
    print("=" * 60)
    print("Testing _extract_finance_symbols method")
    print("=" * 60)
    
    all_passed = True
    for query, expected in test_cases:
        result = set(selector._extract_finance_symbols(query))
        passed = result == expected
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        print(f"Status: {status}")
        
        if not passed:
            all_passed = False
            missing = expected - result
            extra = result - expected
            if missing:
                print(f"  Missing: {missing}")
            if extra:
                print(f"  Extra: {extra}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED! ✅")
    else:
        print("Some tests FAILED! ❌")
    print("=" * 60)
    
    return all_passed

def test_domain_classification():
    """Test that finance queries are correctly classified."""
    selector = IntelligentSourceSelector(use_llm=False)
    
    test_cases = [
        ("对比Intel和AMD近三股价变化以及原因", "finance"),
        ("Intel和AMD的股价对比", "finance"),
        ("苹果股票价格", "finance"),
    ]
    
    print("\n" + "=" * 60)
    print("Testing domain classification")
    print("=" * 60)
    
    all_passed = True
    for query, expected_domain in test_cases:
        domain = selector.classify_domain(query)
        passed = domain == expected_domain
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\nQuery: {query}")
        print(f"Expected domain: {expected_domain}")
        print(f"Got domain: {domain}")
        print(f"Status: {status}")
        
        if not passed:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("Finance Symbol Extraction Fix Test")
    print("=" * 60)
    
    test1_passed = test_extract_finance_symbols()
    test2_passed = test_domain_classification()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if test1_passed and test2_passed:
        print("All tests PASSED! ✅")
        sys.exit(0)
    else:
        print("Some tests FAILED! ❌")
        sys.exit(1)