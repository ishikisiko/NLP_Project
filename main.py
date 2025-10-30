import argparse
import json
import os
from typing import Optional

from api import HKGAIClient
from no_rag_baseline import NoRAGBaseline
from search import SerpAPISearchClient


def build_search_client(api_key: Optional[str]) -> SerpAPISearchClient:
    if not api_key:
        raise ValueError(
            "SERPAPI_API_KEY environment variable is not set. "
            "Provide a SerpAPI key to enable the search baseline."
        )
    return SerpAPISearchClient(api_key=api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the No-RAG baseline pipeline.")
    parser.add_argument("query", help="User question to answer using search + LLM.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens for the LLM response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of search results to include in the prompt.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print the JSON response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hkgai_api_key = os.getenv("HKGAI_API_KEY")
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    llm_client = HKGAIClient(api_key=hkgai_api_key)
    search_client = build_search_client(serpapi_api_key)

    pipeline = NoRAGBaseline(llm_client=llm_client, search_client=search_client)

    result = pipeline.answer(
        args.query,
        num_search_results=args.num_results,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
