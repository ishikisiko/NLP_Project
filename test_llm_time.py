import sys
import unittest
import types
from unittest.mock import MagicMock

# Mock requests package structure
requests_mock = types.ModuleType("requests")
sys.modules["requests"] = requests_mock

requests_adapters_mock = types.ModuleType("requests.adapters")
sys.modules["requests.adapters"] = requests_adapters_mock
requests_mock.adapters = requests_adapters_mock

requests_exceptions_mock = types.ModuleType("requests.exceptions")
sys.modules["requests.exceptions"] = requests_exceptions_mock
requests_mock.exceptions = requests_exceptions_mock

# Add mock classes/functions
requests_adapters_mock.HTTPAdapter = MagicMock()
requests_exceptions_mock.ConnectionError = Exception
requests_exceptions_mock.Timeout = Exception
requests_exceptions_mock.RequestException = Exception

# Mock urllib3
urllib3_mock = types.ModuleType("urllib3")
sys.modules["urllib3"] = urllib3_mock

urllib3_util_mock = types.ModuleType("urllib3.util")
sys.modules["urllib3.util"] = urllib3_util_mock
urllib3_mock.util = urllib3_util_mock

urllib3_util_retry_mock = types.ModuleType("urllib3.util.retry")
sys.modules["urllib3.util.retry"] = urllib3_util_retry_mock
urllib3_util_mock.retry = urllib3_util_retry_mock

urllib3_util_retry_mock.Retry = MagicMock()

# Mock PyPDF2
pypdf2_mock = types.ModuleType("PyPDF2")
sys.modules["PyPDF2"] = pypdf2_mock

# Mock sentence_transformers
sentence_transformers_mock = types.ModuleType("sentence_transformers")
sys.modules["sentence_transformers"] = sentence_transformers_mock
sentence_transformers_mock.SentenceTransformer = MagicMock()

# Mock yfinance
yfinance_mock = types.ModuleType("yfinance")
sys.modules["yfinance"] = yfinance_mock

# Mock yahoo_fin
yahoo_fin_mock = types.ModuleType("yahoo_fin")
sys.modules["yahoo_fin"] = yahoo_fin_mock
yahoo_fin_stock_info_mock = types.ModuleType("yahoo_fin.stock_info")
sys.modules["yahoo_fin.stock_info"] = yahoo_fin_stock_info_mock
yahoo_fin_mock.stock_info = yahoo_fin_stock_info_mock

from smart_orchestrator import SmartSearchOrchestrator
from time_parser import TimeConstraint

class TestLLMTimeDetection(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.orchestrator = SmartSearchOrchestrator(
            llm_client=self.mock_llm,
            search_client=None
        )

    def test_detect_time_constraint_positive(self):
        # Mock LLM response for "Trump age"
        self.mock_llm.chat.return_value = {
            "content": '{"has_time_constraint": true, "time_range": "month", "reason": "Implies current age"}'
        }
        
        result = self.orchestrator._detect_time_constraint_with_llm("Trump多少岁", None)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.days, 30)
        self.assertEqual(result.you_freshness, "month")
        self.assertEqual(result.time_expression, "LLM_Inferred")

    def test_detect_time_constraint_negative(self):
        # Mock LLM response for "Python list methods"
        self.mock_llm.chat.return_value = {
            "content": '{"has_time_constraint": false, "time_range": null, "reason": "General knowledge"}'
        }
        
        result = self.orchestrator._detect_time_constraint_with_llm("Python list methods", None)
        
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
