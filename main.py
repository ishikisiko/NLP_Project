import argparse
import json
import os
from typing import Optional

from api import LLMClient, HKGAIClient
from no_rag_baseline import NoRAGBaseline
from search import SerpAPISearchClient
from local_rag import LocalRAG
from hybrid_rag import HybridRAG


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
        "--mode",
        type=str,
        default="search",
        choices=["search", "local", "hybrid"],
        help="RAG mode: 'search' for web search, 'local' for local files, 'hybrid' for both.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to the directory containing local files for local RAG mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
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
    parser.add_argument(
        "--provider",
        type=str,
        help="Override the LLM provider (openai, anthropic, google, glm, hkgai).",
    )
    return parser.parse_args()


def build_llm_client(config: dict) -> LLMClient:
    """Build LLM client based on provider configuration."""
    provider = config.get("LLM_PROVIDER", "glm")
    
    # Validate provider
    supported_providers = ["openai", "anthropic", "google", "glm", "hkgai"]
    if provider not in supported_providers:
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {', '.join(supported_providers)}")
    
    if provider == "hkgai":
        # Use legacy HKGAI client for backward compatibility
        return HKGAIClient(api_key=config.get("providers", {}).get("hkgai", {}).get("api_key"))
    
    provider_config = config.get("providers", {}).get(provider, {})
    if not provider_config:
        raise ValueError(f"Provider '{provider}' not found in configuration")
    
    return LLMClient(
        api_key=provider_config.get("api_key"),
        model_id=provider_config.get("model"),
        base_url=provider_config.get("base_url"),
        provider=provider
    )


def main() -> None:
    args = parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    # Override provider if specified via command line
    if args.provider:
        config["LLM_PROVIDER"] = args.provider

    llm_client = build_llm_client(config)

    if args.mode == "search":
        serpapi_api_key = config.get("SERPAPI_API_KEY")
        search_client = build_search_client(serpapi_api_key)
        pipeline = NoRAGBaseline(llm_client=llm_client, search_client=search_client)
        result = pipeline.answer(
            args.query,
            num_search_results=args.num_results,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.mode == "local":
        pipeline = LocalRAG(llm_client=llm_client, data_path=args.data_path)
        result = pipeline.answer(
            args.query,
            num_retrieved_docs=args.num_results,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.mode == "hybrid":
        serpapi_api_key = config.get("SERPAPI_API_KEY")
        search_client = build_search_client(serpapi_api_key)
        pipeline = HybridRAG(
            llm_client=llm_client,
            search_client=search_client,
            data_path=args.data_path,
        )
        result = pipeline.answer(
            args.query,
            num_search_results=args.num_results,
            num_retrieved_docs=args.num_results,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    # Check if there are any errors or warnings
    has_error = result.get("llm_error") is not None
    has_warning = result.get("llm_warning") is not None
    no_answer = result.get("answer") is None

    # If there are errors/warnings or no answer, output full JSON
    if has_error or has_warning or no_answer:
        if args.pretty:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(result))
    else:
        # Normal case: only output the answer
        print(result["answer"])


if __name__ == "__main__":
    main()
