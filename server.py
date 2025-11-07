"""Minimal web server that exposes the No-RAG baseline through a REST API."""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from main import build_llm_client, build_search_client, build_reranker
from smart_orchestrator import SmartSearchOrchestrator

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


def build_pipeline(model_override: Optional[str] = None) -> SmartSearchOrchestrator:
    """Create a pipeline configured for the current request."""

    # Deep copy to avoid mutating cached configuration between requests
    config = copy.deepcopy(load_base_config())
    providers_cfg = config.get("providers", {})

    def provider_has_valid_key(name: str) -> bool:
        cfg = providers_cfg.get(name) or {}
        key = (cfg.get("api_key") or "").strip()
        if not key:
            return False
        upper_key = key.upper()
        return not any(token in upper_key for token in ("YOUR_", "REPLACE", "TODO"))

    def match_provider_by_model(model_id: str) -> Optional[str]:
        return next(
            (name for name, cfg in providers_cfg.items() if cfg.get("model") == model_id),
            None,
        )

    def resolve_default_provider() -> str:
        configured = config.get("LLM_PROVIDER")
        if configured:
            if configured in providers_cfg and provider_has_valid_key(configured):
                return configured
            matched = match_provider_by_model(configured)
            if matched and provider_has_valid_key(matched):
                if matched in providers_cfg and not providers_cfg[matched].get("model"):
                    providers_cfg[matched]["model"] = configured
                return matched

        preferred_order = [
            "glm",
            "openai",
            "anthropic",
            "google",
            "minimax",
            "hkgai",
            "openrouter",
        ]
        for candidate in preferred_order:
            if candidate in providers_cfg and provider_has_valid_key(candidate):
                return candidate

        if configured and configured in providers_cfg:
            return configured

        return next(iter(providers_cfg.keys()), "glm")

    config["LLM_PROVIDER"] = resolve_default_provider()
    if model_override:
        # Check if it's a model path (contains '/') and convert to provider
        if "/" in model_override:
            # Map specific models to providers
            model_to_provider = {
                "minimax/minimax-m2:free": "openrouter",
                "deepseek/deepseek-r1-0528:free": "openrouter",
            }
            
            if model_override in model_to_provider:
                config["LLM_PROVIDER"] = model_to_provider[model_override]
            else:
                # For models like "openai/gpt-3.5-turbo", extract provider
                config["LLM_PROVIDER"] = model_override.split("/")[0]
        else:
            if model_override in providers_cfg:
                # Direct provider selection (e.g., "glm")
                config["LLM_PROVIDER"] = model_override
            else:
                matched_provider = match_provider_by_model(model_override)
                if matched_provider:
                    config["LLM_PROVIDER"] = matched_provider
                    if matched_provider in providers_cfg:
                        providers_cfg[matched_provider]["model"] = model_override
                else:
                    # Fall back to treating the override as provider name
                    config["LLM_PROVIDER"] = model_override

    # Build LLM client with enhanced error handling
    try:
        llm_client = build_llm_client(config)
    except Exception as exc:
        raise ConfigurationError(f"Failed to build LLM client: {exc}")

    serpapi_key = config.get("SERPAPI_API_KEY")
    search_client = None
    if serpapi_key:
        try:
            search_client = build_search_client(serpapi_key)
        except Exception as exc:
            raise ConfigurationError(f"Failed to build search client: {exc}")

    rerank_config = config.get("rerank") or {}
    reranker: Optional[Any] = None
    try:
        reranker, rerank_config = build_reranker(config)
    except ValueError as exc:
        print(f"[server] Reranker disabled: {exc}")
        reranker = None
        rerank_config = config.get("rerank") or {}
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ConfigurationError(f"Unexpected reranker error: {exc}")

    min_rerank_score = float(rerank_config.get("min_score", 0.0))
    max_per_domain = max(1, int(rerank_config.get("max_per_domain", 1)))

    return SmartSearchOrchestrator(
        llm_client=llm_client,
        search_client=search_client,
        data_path=app.config['UPLOAD_FOLDER'],
        reranker=reranker,
        min_rerank_score=min_rerank_score,
        max_per_domain=max_per_domain,
    )


@app.route("/")
def index() -> Any:
    return app.send_static_file("index.html")


@app.route("/api/models")
def get_available_models():
    """Get list of available models from configuration."""
    try:
        config = load_base_config()
        models = []
        
        # Get all models from providers
        for provider_name, provider_config in config.get("providers", {}).items():
            model = provider_config.get("model")
            if model:
                models.append({
                    "id": model,
                    "provider": provider_name,
                    "display_name": f"{provider_name.upper()} - {model}"
                })
        
        # Add available models from openrouter if configured
        openrouter_config = config.get("providers", {}).get("openrouter", {})
        if openrouter_config and openrouter_config.get("available_models"):
            for model in openrouter_config.get("available_models", []):
                if not any(m["id"] == model for m in models):
                    models.append({
                        "id": model,
                        "provider": "openrouter",
                        "display_name": f"OpenRouter - {model}"
                    })
        
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

    if not query:
        return jsonify({"error": "Missing 'query' in request body."}), 400

    try:
        # Support both provider and model parameters for backward compatibility
        model = payload.get("model") or payload.get("provider")
        pipeline = build_pipeline(model_override=model)
    except ConfigurationError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Failed to build pipeline: {exc}"}), 500

    search_pref = (payload.get("search") or "").strip().lower()
    if search_pref in {"on", "off"}:
        allow_search = search_pref == "on"
    else:
        legacy_mode = (payload.get("mode") or "search").strip().lower()
        allow_search = legacy_mode != "local"

    num_value = int(payload.get("num_results")) if payload.get("num_results") else 5

    try:
        result = pipeline.answer(
            query,
            num_search_results=num_value,
            num_retrieved_docs=num_value,
            max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
            temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
            allow_search=allow_search,
        )
    except Exception as exc:  # pragma: no cover - propagate runtime issues
        error_msg = str(exc).encode('utf-8', errors='replace').decode('utf-8')
        return jsonify({"error": f"Pipeline execution failed: {error_msg}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False  # 允许UTF-8字符
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
