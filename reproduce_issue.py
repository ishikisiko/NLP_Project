import os
import json
from source_selector import IntelligentSourceSelector

def test_transportation():
    print("Testing Transportation API...")
    
    # Load config.json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found.")
        return

    google_api_key = config.get("GOOGLE_API_KEY") or config.get("googleSearch", {}).get("api_key")
    
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in config.json")
        return

    print(f"Loaded Google API Key: {google_api_key[:5]}...")

    selector = IntelligentSourceSelector(use_llm=False, google_api_key=google_api_key)
    
    # Test case for transportation
    query = "从北京到上海"
    print(f"Query: {query}")
    
    try:
        # Force domain to transportation to test the specific handler
        result = selector.fetch_domain_data(query, domain="transportation")
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_transportation()
