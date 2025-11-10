import argparse
import json
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

from main import build_llm_client, build_search_client, build_reranker, build_domain_classifier_client
from smart_orchestrator import SmartSearchOrchestrator


def resolve_model_to_provider(model_or_provider: str, config: Dict[str, Any]) -> str:
    """Resolve a model path or provider name to the actual provider."""
    if "/" in model_or_provider:
        # Map specific models to providers
        model_to_provider = {
            "minimax/minimax-m2:free": "openrouter",
            "deepseek/deepseek-r1-0528:free": "openrouter",
        }
        
        if model_or_provider in model_to_provider:
            return model_to_provider[model_or_provider]
        else:
            # For models like "openai/gpt-3.5-turbo", extract provider
            return model_or_provider.split("/")[0]
    else:
        # It's a provider name
        return model_or_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple queries through the smart orchestrator for quick regression testing."
    )
    parser.add_argument(
        "--queries-file",
        required=True,
        help="Path to a UTF-8 text file; one query per line. Blank lines and lines starting with # are ignored.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.json. Defaults to NLP_CONFIG_PATH env or ./config.json.",
    )
    parser.add_argument(
        "--data-path",
        default="./data",
        help="Directory containing local documents for the local RAG pipeline.",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Maximum number of documents/search hits per query.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum LLM generation tokens per query.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--search",
        choices=["on", "off"],
        default="on",
        help="Enable ('on') or disable ('off') live web search.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Skip configured reranking even if rerank provider exists.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Override the default LLM provider or model for this batch run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the LLM model (e.g., minimax/minimax-m2:free, deepseek/deepseek-r1-0528:free).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print full JSON responses instead of condensed text output.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output directory for batch results. If not provided, results are printed to console.",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Save all results to a single JSON file instead of separate files per query.",
    )
    return parser.parse_args()


def read_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        queries = []
        for line in handle:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            queries.append(cleaned)
    return queries


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config_path = path or os.environ.get("NLP_CONFIG_PATH") or "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filename"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces and dots
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'\.+$', '', filename)
    return filename[:50]  # Limit length


def save_query_result(query_id: int, query: str, result: Dict[str, Any], 
                     output_dir: str, timestamp: str) -> str:
    """Save a single query result to a file"""
    # Create output data structure
    output_data = {
        "query_id": query_id,
        "query": query,
        "timestamp": timestamp,
        "result": result
    }
    
    # Generate filename
    query_filename = f"query_{query_id:03d}_{sanitize_filename(query[:50])}.json"
    output_path = os.path.join(output_dir, query_filename)
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_path


def save_all_results(all_results: List[Dict[str, Any]], output_file: str):
    """Save all results to a single JSON file"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    queries = read_queries(args.queries_file)
    if not queries:
        raise SystemExit("No queries found in the provided file.")

    config = load_config(args.config)
    if args.model:
        config["LLM_PROVIDER"] = args.model
    elif args.provider:
        config["LLM_PROVIDER"] = args.provider

    llm_client = build_llm_client(config)
    classifier_client = build_domain_classifier_client(config)
    allow_search = args.search == "on"

    reranker = None
    rerank_config: Dict[str, Any] = config.get("rerank") or {}
    if allow_search and not args.disable_rerank:
        reranker, rerank_config = build_reranker(config)

    min_rerank_score = float(rerank_config.get("min_score", 0.0))
    max_per_domain = max(1, int(rerank_config.get("max_per_domain", 1)))

    search_client = None
    requested_sources: List[str] = []
    active_sources: List[str] = []
    active_labels: List[str] = []
    missing_sources: List[str] = []
    configured_sources: List[str] = []
    if (config.get("SERPAPI_API_KEY") or "").strip():
        configured_sources.append("serp")
    you_cfg_batch = config.get("youSearch") or {}
    if (you_cfg_batch.get("api_key") or config.get("YOU_API_KEY") or "").strip():
        configured_sources.append("you")
    mcp_cfg_batch = (config.get("mcpServers") or {}).get("web-search-prime") or {}
    if (mcp_cfg_batch.get("url") or "").strip() and any(
        (mcp_cfg_batch.get("headers") or {}).get(token) for token in ("Authorization", "authorization")
    ):
        configured_sources.append("mcp")
    if allow_search:
        search_client = build_search_client(config)
        if search_client:
            requested_sources = list(getattr(search_client, "requested_sources", []))
            active_sources = list(getattr(search_client, "active_sources", []))
            active_labels = list(getattr(search_client, "active_source_labels", []))
            missing_sources = list(getattr(search_client, "missing_requested_sources", []))
            configured_sources = list(getattr(search_client, "configured_sources", []))

    orchestrator = SmartSearchOrchestrator(
        llm_client=llm_client,
        classifier_llm_client=classifier_client,
        search_client=search_client,
        data_path=args.data_path,
        reranker=reranker,
        min_rerank_score=min_rerank_score,
        max_per_domain=max_per_domain,
        requested_search_sources=requested_sources,
        active_search_sources=active_sources,
        active_search_source_labels=active_labels,
        missing_search_sources=missing_sources,
        configured_search_sources=configured_sources,
    )

    print(f"Loaded {len(queries)} queries from {args.queries_file}")
    
    # Initialize output
    output_to_file = args.output_file is not None
    all_results = []
    
    if output_to_file:
        if args.single_file:
            # Single file mode - prepare for collecting all results
            os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
            print(f"All results will be saved to: {args.output_file}")
        else:
            # Directory mode - one file per query
            os.makedirs(args.output_file, exist_ok=True)
            print(f"Results will be saved to directory: {args.output_file}")
            print("Each query will have its own JSON file.")

    for idx, query in enumerate(queries, start=1):
        print(f"\n=== Query {idx}: {query}")
        timestamp = datetime.now().isoformat()
        
        result = orchestrator.answer(
            query,
            num_search_results=args.num_results,
            num_retrieved_docs=args.num_results,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            allow_search=allow_search,
        )

        if output_to_file:
            if args.single_file:
                # Collect for single file output
                output_data = {
                    "query_id": idx,
                    "query": query,
                    "timestamp": timestamp,
                    "result": result
                }
                all_results.append(output_data)
                print(f"Query {idx} result collected for file output")
            else:
                # Save to individual file
                output_path = save_query_result(idx, query, result, args.output_file, timestamp)
                print(f"Answer saved to: {output_path}")
        else:
            # Original console output
            if args.pretty:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                continue

            answer = result.get("answer") or ""
            decision = ((result.get("control") or {}).get("decision") or {}).get("reason")
            mode = (result.get("control") or {}).get("search_mode")
            print(f"Answer: {answer}")
            if decision:
                print(f"Decision reason: {decision}")
            if mode:
                print(f"Mode: {mode}")
            warnings = result.get("search_warnings")
            if warnings:
                if isinstance(warnings, list):
                    for warning in warnings:
                        print(f"Search warning: {warning}")
                else:
                    print(f"Search warning: {warnings}")

    # Save all results to single file if requested
    if output_to_file and args.single_file:
        save_all_results(all_results, args.output_file)
        print(f"\nAll {len(all_results)} query results saved to: {args.output_file}")

    print(f"\nBatch processing completed. Processed {len(queries)} queries.")


if __name__ == "__main__":
    main()