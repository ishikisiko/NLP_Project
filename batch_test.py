import argparse
import json
import os
from typing import List, Optional, Dict, Any

from main import build_llm_client, build_search_client, build_reranker
from smart_orchestrator import SmartSearchOrchestrator


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
        help="Override the default LLM provider for this batch run.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print full JSON responses instead of condensed text output.",
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


def main() -> None:
    args = parse_args()
    queries = read_queries(args.queries_file)
    if not queries:
        raise SystemExit("No queries found in the provided file.")

    config = load_config(args.config)
    if args.provider:
        config["LLM_PROVIDER"] = args.provider

    llm_client = build_llm_client(config)
    allow_search = args.search == "on"

    reranker = None
    rerank_config: Dict[str, Any] = config.get("rerank") or {}
    if allow_search and not args.disable_rerank:
        reranker, rerank_config = build_reranker(config)

    min_rerank_score = float(rerank_config.get("min_score", 0.0))
    max_per_domain = max(1, int(rerank_config.get("max_per_domain", 1)))

    search_client = None
    if allow_search:
        serp_key = config.get("SERPAPI_API_KEY")
        if serp_key:
            search_client = build_search_client(serp_key)

    orchestrator = SmartSearchOrchestrator(
        llm_client=llm_client,
        search_client=search_client,
        data_path=args.data_path,
        reranker=reranker,
        min_rerank_score=min_rerank_score,
        max_per_domain=max_per_domain,
    )

    print(f"Loaded {len(queries)} queries from {args.queries_file}")
    for idx, query in enumerate(queries, start=1):
        print(f"\n=== Query {idx}: {query}")
        result = orchestrator.answer(
            query,
            num_search_results=args.num_results,
            num_retrieved_docs=args.num_results,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            allow_search=allow_search,
        )

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


if __name__ == "__main__":
    main()
