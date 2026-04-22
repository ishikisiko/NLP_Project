from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.langchain_support import LangChainVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local document chunk-size/chunk-overlap retrieval experiment."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory containing the local documents to index.",
    )
    parser.add_argument(
        "--dataset-file",
        default="tests/search_quality_minimal_dataset.csv",
        help="CSV dataset file containing local_rag questions and gold_doc_id.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.json. Defaults to NLP_CONFIG_PATH env or ./config.json.",
    )
    parser.add_argument(
        "--category",
        default="local_rag",
        help="Dataset category to evaluate. Defaults to local_rag.",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="300,500,800,1000,1500",
        help="Comma-separated chunk sizes to test.",
    )
    parser.add_argument(
        "--chunk-overlaps",
        default="0,50,100,150,200,300",
        help="Comma-separated chunk overlaps to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieved chunks to inspect per query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of evaluated queries.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model override passed to LangChainVectorStore.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-query ranks for each parameter pair.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to save the experiment report as JSON.",
    )
    return parser.parse_args()


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config_path = path or os.environ.get("NLP_CONFIG_PATH") or "config.json"
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_int_list(raw_value: str, field: str) -> List[int]:
    values: List[int] = []
    for token in str(raw_value).split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            parsed = int(cleaned)
        except ValueError as exc:
            raise ValueError(f"'{field}' contains a non-integer value: {cleaned}") from exc
        if parsed < 0:
            raise ValueError(f"'{field}' cannot contain negative values.")
        values.append(parsed)
    if not values:
        raise ValueError(f"'{field}' must contain at least one integer.")
    return values


def load_queries(
    dataset_file: str,
    *,
    category: str,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []
    with open(dataset_file, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        for row in reader:
            if str(row.get("category") or "").strip().lower() != category.strip().lower():
                continue
            gold_doc_id = str(row.get("gold_doc_id") or "").strip()
            query = str(row.get("query") or "").strip()
            if not gold_doc_id or not query:
                continue
            queries.append(
                {
                    "id": str(row.get("id") or "").strip(),
                    "query": query,
                    "gold_doc_id": gold_doc_id,
                }
            )
            if limit is not None and len(queries) >= limit:
                break
    return queries


def source_matches_gold(source: str, gold_doc_id: str) -> bool:
    source_text = str(source or "").strip()
    gold_text = str(gold_doc_id or "").strip()
    if not source_text or not gold_text:
        return False

    source_lower = source_text.lower()
    gold_lower = gold_text.lower()
    source_base = os.path.basename(source_lower)
    gold_base = os.path.basename(gold_lower)
    return (
        gold_lower in source_lower
        or gold_base in source_base
        or gold_base in source_lower
    )


def first_gold_rank(sources: Iterable[str], gold_doc_id: str) -> Optional[int]:
    for index, source in enumerate(sources, start=1):
        if source_matches_gold(source, gold_doc_id):
            return index
    return None


def evaluate_setting(
    *,
    data_path: str,
    queries: List[Dict[str, str]],
    config: Optional[Dict[str, Any]],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> Dict[str, Any]:
    index_start = time.perf_counter()
    store = LangChainVectorStore(
        model_name=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        config=config,
    )
    chunk_count = store.index_from_directory(data_path)
    index_ms = (time.perf_counter() - index_start) * 1000

    hits = 0
    reciprocal_rank_sum = 0.0
    total_query_ms = 0.0
    query_results: List[Dict[str, Any]] = []

    for row in queries:
        search_start = time.perf_counter()
        docs = store.search(row["query"], k=top_k)
        query_ms = (time.perf_counter() - search_start) * 1000
        total_query_ms += query_ms

        sources = [doc.source or "" for doc in docs]
        rank = first_gold_rank(sources, row["gold_doc_id"])
        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / rank

        query_results.append(
            {
                "id": row["id"],
                "query": row["query"],
                "gold_doc_id": row["gold_doc_id"],
                "gold_rank": rank,
                "hit": rank is not None,
                "query_ms": round(query_ms, 3),
                "retrieved_sources": sources,
            }
        )

    query_count = len(queries)
    hit_rate = (hits / query_count) if query_count else 0.0
    mrr = (reciprocal_rank_sum / query_count) if query_count else 0.0
    avg_query_ms = (total_query_ms / query_count) if query_count else 0.0

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": chunk_count,
        "query_count": query_count,
        "doc_hit_at_k": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "index_ms": round(index_ms, 3),
        "avg_query_ms": round(avg_query_ms, 3),
        "queries": query_results,
    }


def main() -> int:
    args = parse_args()

    if args.top_k <= 0:
        raise SystemExit("'--top-k' must be greater than 0.")

    chunk_sizes = parse_int_list(args.chunk_sizes, "chunk_sizes")
    chunk_overlaps = parse_int_list(args.chunk_overlaps, "chunk_overlaps")
    config = load_config(args.config)
    queries = load_queries(args.dataset_file, category=args.category, limit=args.limit)

    if not queries:
        raise SystemExit("No matching local queries found in the dataset.")

    results: List[Dict[str, Any]] = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            if chunk_overlap >= chunk_size:
                continue
            result = evaluate_setting(
                data_path=args.data_path,
                queries=queries,
                config=config,
                embedding_model=args.embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=args.top_k,
            )
            results.append(result)

    if not results:
        raise SystemExit("No valid (chunk_size, chunk_overlap) combinations to evaluate.")

    results.sort(
        key=lambda item: (
            -float(item["doc_hit_at_k"]),
            -float(item["mrr"]),
            float(item["avg_query_ms"]),
            float(item["index_ms"]),
        )
    )

    report = {
        "data_path": os.path.abspath(args.data_path),
        "dataset_file": os.path.abspath(args.dataset_file),
        "category": args.category,
        "top_k": args.top_k,
        "best": results[0],
        "results": results,
    }

    print("chunk_size\tchunk_overlap\tdoc_hit@k\tmrr\tchunks\tindex_ms\tavg_query_ms")
    for item in results:
        print(
            f"{item['chunk_size']}\t"
            f"{item['chunk_overlap']}\t"
            f"{item['doc_hit_at_k']:.4f}\t"
            f"{item['mrr']:.4f}\t"
            f"{item['chunk_count']}\t"
            f"{item['index_ms']:.3f}\t"
            f"{item['avg_query_ms']:.3f}"
        )
        if args.details:
            for query in item["queries"]:
                print(
                    f"  - {query['id'] or '?'} rank={query['gold_rank']} "
                    f"hit={query['hit']} query_ms={query['query_ms']:.3f} "
                    f"gold={query['gold_doc_id']}"
                )

    if args.output_file:
        parent = os.path.dirname(args.output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
