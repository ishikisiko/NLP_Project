"""Minimal web server that exposes the No-RAG baseline through a REST API."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from main import build_llm_client, build_search_client
from no_rag_baseline import NoRAGBaseline

app = Flask(__name__, static_folder="frontend", static_url_path="")


class ConfigurationError(RuntimeError):
    """Raised when the application configuration is invalid."""


@lru_cache(maxsize=1)
def load_base_config() -> Dict[str, Any]:
    """Load the project configuration file.

    The path can be overridden via the NLP_CONFIG_PATH environment variable.
    """

    config_path = os.environ.get("NLP_CONFIG_PATH", "config.json")
    if not os.path.exists(config_path):
        raise ConfigurationError(
            f"Configuration file '{config_path}' not found. "
            "Create it based on config.example.json."
        )

    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_pipeline(provider_override: Optional[str] = None) -> NoRAGBaseline:
    """Create a NoRAGBaseline pipeline configured for the current request."""

    config = load_base_config().copy()
    if provider_override:
        config["LLM_PROVIDER"] = provider_override

    llm_client = build_llm_client(config)
    search_client = build_search_client(config.get("SERPAPI_API_KEY"))
    return NoRAGBaseline(llm_client=llm_client, search_client=search_client)


@app.route("/")
def index() -> Any:
    return app.send_static_file("index.html")


@app.post("/api/answer")
def answer() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' in request body."}), 400

    try:
        pipeline = build_pipeline(payload.get("provider") or None)
    except ConfigurationError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Failed to build pipeline: {exc}"}), 500

    try:
        result = pipeline.answer(
            query,
            num_search_results=int(payload.get("num_results")) if payload.get("num_results") else 5,
            max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
            temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
        )
    except Exception as exc:  # pragma: no cover - propagate runtime issues
        return jsonify({"error": f"Pipeline execution failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
