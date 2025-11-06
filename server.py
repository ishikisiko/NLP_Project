"""Minimal web server that exposes the No-RAG baseline through a REST API."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from main import build_llm_client, build_search_client
from no_rag_baseline import NoRAGBaseline
from local_rag import LocalRAG
from hybrid_rag import HybridRAG

app = Flask(__name__, static_folder="frontend", static_url_path="")
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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


def build_pipeline(provider_override: Optional[str] = None, mode: str = "search") -> NoRAGBaseline | LocalRAG | HybridRAG:
    """Create a pipeline configured for the current request."""

    config = load_base_config().copy()
    if provider_override:
        config["LLM_PROVIDER"] = provider_override

    # Build LLM client with enhanced error handling
    try:
        llm_client = build_llm_client(config)
    except Exception as exc:
        raise ConfigurationError(f"Failed to build LLM client: {exc}")

    if mode == "search":
        search_client = build_search_client(config.get("SERPAPI_API_KEY"))
        return NoRAGBaseline(llm_client=llm_client, search_client=search_client)
    elif mode == "local":
        return LocalRAG(llm_client=llm_client, data_path=app.config['UPLOAD_FOLDER'])
    elif mode == "hybrid":
        search_client = build_search_client(config.get("SERPAPI_API_KEY"))
        return HybridRAG(
            llm_client=llm_client,
            search_client=search_client,
            data_path=app.config['UPLOAD_FOLDER'],
        )


@app.route("/")
def index() -> Any:
    return app.send_static_file("index.html")


@app.route('/api/files', methods=['GET'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify(files)


@app.route('/api/files', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": "File uploaded successfully"})


@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": "File deleted successfully"})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


@app.post("/api/answer")
def answer() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    mode = (payload.get("mode") or "search").strip()

    if not query:
        return jsonify({"error": "Missing 'query' in request body."}), 400

    try:
        pipeline = build_pipeline(payload.get("provider") or None, mode)
    except ConfigurationError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Failed to build pipeline: {exc}"}), 500

    try:
        if mode == "search":
            result = pipeline.answer(
                query,
                num_search_results=int(payload.get("num_results")) if payload.get("num_results") else 5,
                max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
                temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
            )
        elif mode == "local":
            result = pipeline.answer(
                query,
                num_retrieved_docs=int(payload.get("num_results")) if payload.get("num_results") else 5,
                max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
                temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
            )
        elif mode == "hybrid":
            num_value = int(payload.get("num_results")) if payload.get("num_results") else 5
            result = pipeline.answer(
                query,
                num_search_results=num_value,
                num_retrieved_docs=num_value,
                max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
                temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
            )
        else:
            result = pipeline.answer(
                query,
                num_retrieved_docs=int(payload.get("num_results")) if payload.get("num_results") else 5,
                max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
                temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
            )
    except Exception as exc:  # pragma: no cover - propagate runtime issues
        return jsonify({"error": f"Pipeline execution failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
