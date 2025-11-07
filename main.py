import argparse
import json
import os
from typing import Optional, Tuple, Dict, Any

from api import LLMClient, HKGAIClient
from search import SerpAPISearchClient
from rerank import BaseReranker, Qwen3Reranker
from smart_orchestrator import SmartSearchOrchestrator


def build_search_client(api_key: Optional[str]) -> SerpAPISearchClient:
    if not api_key:
        raise ValueError(
            "SERPAPI_API_KEY environment variable is not set. "
            "Provide a SerpAPI key to enable the search baseline."
        )
    return SerpAPISearchClient(api_key=api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer questions with optional web search and local RAG.")
    parser.add_argument("query", help="User question to answer.")
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
        help="Override the LLM provider or model (openai, anthropic, google, glm, hkgai, openrouter, minimax, or specific model like minimax/minimax-m2:free).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the LLM model (e.g., minimax/minimax-m2:free, deepseek/deepseek-r1-0528:free).",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Skip search result reranking even if configured.",
    )
    parser.add_argument(
        "--search",
        choices=["on", "off"],
        default="on",
        help="Enable ('on') or disable ('off') live web search. Uploads remain available in both modes.",
    )
    return parser.parse_args()


def build_llm_client(config: dict) -> LLMClient:
    """Build LLM client based on provider or model configuration."""
    provider_or_model = config.get("LLM_PROVIDER", "glm")
    
    # Check if it's a specific model (contains '/') and map to provider
    if "/" in provider_or_model:
        # Map models to providers
        model_to_provider = {
            "minimax/minimax-m2:free": "openrouter",
            "deepseek/deepseek-r1-0528:free": "openrouter",
        }
        
        if provider_or_model in model_to_provider:
            provider = model_to_provider[provider_or_model]
        else:
            # For models like "openai/gpt-3.5-turbo", extract the provider
            provider = provider_or_model.split("/")[0]
            if provider not in ["openai", "anthropic", "google", "glm", "hkgai", "openrouter", "minimax"]:
                # Check if it's a direct provider name
                supported_providers = ["openai", "anthropic", "google", "glm", "hkgai", "openrouter", "minimax"]
                if provider not in supported_providers:
                    # If not a known provider, treat as provider name
                    provider = provider_or_model
        model_id = provider_or_model
    else:
        # It's a provider name
        provider = provider_or_model
        model_id = None
    
    # Validate provider
    supported_providers = ["openai", "anthropic", "google", "glm", "hkgai", "openrouter", "minimax"]
    if provider not in supported_providers:
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {', '.join(supported_providers)}")
    
    if provider == "hkgai":
        # Use legacy HKGAI client for backward compatibility
        return HKGAIClient(api_key=config.get("providers", {}).get("hkgai", {}).get("api_key"))
    
    provider_config = config.get("providers", {}).get(provider, {})
    if not provider_config:
        raise ValueError(f"Provider '{provider}' not found in configuration")
    
    # Get the model ID (use specified model or fall back to provider config)
    final_model_id = model_id or provider_config.get("model")
    if not final_model_id:
        raise ValueError(f"No model specified for provider '{provider}'")
    
    # For openrouter, check if model is in available models
    if provider == "openrouter" and "/" in provider_or_model:
        # Update the provider config with the specific model
        provider_config = provider_config.copy()
        provider_config["model"] = final_model_id
    elif "/" not in provider_or_model and model_id:
        # For other providers, use the specified model if provided
        provider_config = provider_config.copy()
        provider_config["model"] = model_id
    
    # Get global LLM settings
    llm_settings = config.get("llm_settings", {})
    default_timeout = llm_settings.get("default_timeout", 60)
    max_retries = llm_settings.get("max_retries", 3)
    backoff_factor = llm_settings.get("backoff_factor", 2.0)
    
    # Provider-specific timeout (fallback to global)
    provider_timeout = provider_config.get("request_timeout", default_timeout)
    provider_max_retries = provider_config.get("max_retries", max_retries)
    provider_backoff_factor = provider_config.get("backoff_factor", backoff_factor)
    
    return LLMClient(
        api_key=provider_config.get("api_key"),
        model_id=final_model_id,
        base_url=provider_config.get("base_url"),
        request_timeout=provider_timeout,
        provider=provider,
        max_retries=provider_max_retries,
        backoff_factor=provider_backoff_factor
    )


def build_reranker(config: dict) -> Tuple[Optional[BaseReranker], Dict[str, Any]]:
    """Instantiate a reranker based on configuration."""

    rerank_config = config.get("rerank") or {}
    provider = config.get("RERANK_PROVIDER") or rerank_config.get("provider")
    if not provider:
        return None, rerank_config

    provider_key = provider.lower()

    if provider_key in {"qwen", "qwen3", "qwen3-rerank"}:
        provider_settings = (
            (rerank_config.get("providers") or {}).get("qwen")
            or rerank_config.get("qwen")
            or {}
        )
        api_key = provider_settings.get("api_key")
        if not api_key:
            raise ValueError("DashScope API key is required for Qwen3 reranking.")

        reranker = Qwen3Reranker(
            api_key=api_key,
            model=provider_settings.get("model", "qwen3-rerank"),
            base_url=provider_settings.get("base_url"),
            request_timeout=provider_settings.get("timeout", 15),
        )
        return reranker, rerank_config

    raise ValueError(f"Unsupported rerank provider '{provider}'.")


def main() -> None:
    args = parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    # Override provider or model if specified via command line
    if args.model:
        config["LLM_PROVIDER"] = args.model
    elif args.provider:
        config["LLM_PROVIDER"] = args.provider

    llm_client = build_llm_client(config)

    allow_search = args.search == "on"

    reranker: Optional[BaseReranker] = None
    rerank_config: Dict[str, Any] = config.get("rerank") or {}
    if allow_search and not args.disable_rerank:
        reranker, rerank_config = build_reranker(config)

    min_rerank_score = float(rerank_config.get("min_score", 0.0))
    max_per_domain = max(1, int(rerank_config.get("max_per_domain", 1)))

    search_client: Optional[SerpAPISearchClient] = None
    if allow_search:
        serpapi_api_key = config.get("SERPAPI_API_KEY")
        if serpapi_api_key:
            search_client = build_search_client(serpapi_api_key)

    orchestrator = SmartSearchOrchestrator(
        llm_client=llm_client,
        search_client=search_client,
        data_path=args.data_path,
        reranker=reranker,
        min_rerank_score=min_rerank_score,
        max_per_domain=max_per_domain,
    )
    
    result = orchestrator.answer(
        args.query,
        num_search_results=args.num_results,
        num_retrieved_docs=args.num_results,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        allow_search=allow_search,
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
