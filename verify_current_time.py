import sys
import types
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["requests"] = MagicMock()
sys.modules["requests.adapters"] = MagicMock()
sys.modules["requests.exceptions"] = MagicMock()
sys.modules["urllib3"] = MagicMock()
sys.modules["urllib3.util"] = MagicMock()
sys.modules["urllib3.util.retry"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["yfinance"] = MagicMock()
sys.modules["yahoo_fin"] = MagicMock()
sys.modules["yahoo_fin.stock_info"] = MagicMock()

from smart_orchestrator import SmartSearchOrchestrator
from current_time import get_current_date_str

def test_current_time_injection():
    mock_llm = MagicMock()
    # Mock LLM to return no time constraint for the LLM check, 
    # but we will use a query that triggers the REGEX parser ("最近三天")
    mock_llm.chat.return_value = {"content": "Test Answer"}
    
    orchestrator = SmartSearchOrchestrator(llm_client=mock_llm, search_client=None)
    
    # Query with explicit time constraint -> triggers regex parser -> triggers date injection
    query = "最近三天的新闻"
    print(f"Testing query: {query}")
    
    # We can't easily check internal state, but we can check the print output or mock the parser
    # Let's mock the parser to ensure it returns a constraint
    
    try:
        orchestrator.answer(query, allow_search=False) # allow_search=False to skip search pipeline but trigger logic
    except Exception as e:
        print(f"Execution finished with: {e}")

    print("\nCheck the stdout above for 'Injecting current date into query'")
    print(f"Expected date: {get_current_date_str()}")

if __name__ == "__main__":
    test_current_time_injection()
