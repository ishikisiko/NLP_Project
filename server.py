from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional, List

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from main import build_llm_client, build_search_client, build_reranker, build_domain_classifier_client, build_routing_keywords_client
from smart_orchestrator import SmartSearchOrchestrator

app = Flask(__name__, static_folder="frontend", static_url_path="")
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class ConfigurationError(RuntimeError):
    """Raised when the application configuration is invalid."""


def ensure_json_serializable(obj: Any) -> Any:
    """Recursively ensure all values in a dict/list are JSON serializable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    # Convert any other type to string
    try:
        return str(obj)
    except Exception:
        return None


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


def _normalize_search_sources(raw_sources: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    if not raw_sources:
        return normalized
    for item in raw_sources:
        if not isinstance(item, str):
            continue
        token = item.strip().lower()
        if not token or token in normalized:
            continue
        normalized.append(token)
    return normalized


def build_pipeline(
    model_override: Optional[str] = None,
    *,
    search_sources: Optional[List[str]] = None,
) -> SmartSearchOrchestrator:
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
            "zai",
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

        return next(iter(providers_cfg.keys()), "zai")

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

    try:
        classifier_client = build_domain_classifier_client(config)
    except Exception as exc:
        raise ConfigurationError(f"Failed to build domain classifier client: {exc}")

    try:
        routing_client = build_routing_keywords_client(config)
    except Exception as exc:
        raise ConfigurationError(f"Failed to build routing/keywords client: {exc}")

    normalized_sources = _normalize_search_sources(search_sources)
    configured_sources: List[str] = []
    if (config.get("SERPAPI_API_KEY") or "").strip():
        configured_sources.append("serp")
    you_cfg = config.get("youSearch") or {}
    you_key = (you_cfg.get("api_key") or config.get("YOU_API_KEY") or "").strip()
    if you_key:
        configured_sources.append("you")
    mcp_cfg = (config.get("mcpServers") or {}).get("web-search-prime") or {}
    if (mcp_cfg.get("url") or "").strip() and any((mcp_cfg.get("headers") or {}).get(token) for token in ("Authorization", "authorization")):
        configured_sources.append("mcp")
    google_cfg = config.get("googleSearch") or {}
    google_key = (google_cfg.get("api_key") or config.get("GOOGLE_API_KEY") or "").strip()
    google_cx = (google_cfg.get("cx") or config.get("GOOGLE_CX") or "").strip()
    if google_key and google_cx:
        configured_sources.append("google")

    search_client = None
    active_sources: List[str] = []
    active_labels: List[str] = []
    missing_sources: List[str] = []
    try:
        search_client = build_search_client(config, sources=normalized_sources if normalized_sources else None)
        if search_client is not None:
            active_sources = list(getattr(search_client, "active_sources", []))
            active_labels = list(getattr(search_client, "active_source_labels", []))
            missing_sources = list(getattr(search_client, "missing_requested_sources", []))
            if not configured_sources:
                configured_sources = list(getattr(search_client, "configured_sources", []))
    except Exception as exc:
        raise ConfigurationError(f"Failed to build search client: {exc}")

    if not missing_sources and normalized_sources:
        reference = active_sources if active_sources else configured_sources
        missing_sources = [src for src in normalized_sources if src not in reference]

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
    show_timings = bool(config.get("displayResponseTimes", False))

    return SmartSearchOrchestrator(
        llm_client=llm_client,
        classifier_llm_client=classifier_client,
        routing_llm_client=routing_client,
        search_client=search_client,
        data_path=app.config['UPLOAD_FOLDER'],
        reranker=reranker,
        min_rerank_score=min_rerank_score,
        max_per_domain=max_per_domain,
        requested_search_sources=normalized_sources,
        active_search_sources=active_sources,
        active_search_source_labels=active_labels,
        missing_search_sources=missing_sources,
        configured_search_sources=configured_sources,
        show_timings=show_timings,
        google_api_key=google_key,
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
        seen_ids = set()
        
        # Get all models from providers (including available_models)
        for provider_name, provider_config in config.get("providers", {}).items():
            # Add the default model
            default_model = provider_config.get("model")
            if default_model and default_model not in seen_ids:
                models.append({
                    "id": default_model,
                    "provider": provider_name,
                    "display_name": f"{provider_name.upper()} - {default_model}"
                })
                seen_ids.add(default_model)
            
            # Add all available_models if present
            available_models = provider_config.get("available_models", [])
            if available_models:
                for model in available_models:
                    if model not in seen_ids:
                        models.append({
                            "id": model,
                            "provider": provider_name,
                            "display_name": f"{provider_name.upper()} - {model}"
                        })
                        seen_ids.add(model)
        
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

    search_sources: Optional[List[str]] = None
    if payload.get("search_sources") is not None:
        raw_sources = payload["search_sources"]
        if not isinstance(raw_sources, list):
            return jsonify({"error": "'search_sources' must be an array."}), 400
        allowed_sources = {"serp", "you", "mcp", "google"}
        normalized_sources: List[str] = []
        seen = set()
        for item in raw_sources:
            if not isinstance(item, str):
                return jsonify({"error": "Invalid search source value."}), 400
            token = item.strip().lower()
            if token not in allowed_sources:
                return jsonify({"error": f"Unsupported search source '{item}'."}), 400
            if token in seen:
                continue
            seen.add(token)
            normalized_sources.append(token)
        search_sources = normalized_sources

    search_pref = (payload.get("search") or "").strip().lower()
    if search_pref in {"on", "off"}:
        allow_search = search_pref == "on"
    else:
        legacy_mode = (payload.get("mode") or "search").strip().lower()
        allow_search = legacy_mode != "local"

    try:
        # Support both provider and model parameters for backward compatibility
        model = payload.get("model") or payload.get("provider")
        pipeline = build_pipeline(
            model_override=model,
            search_sources=search_sources if allow_search and search_sources else None,
        )
    except ConfigurationError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"error": f"Failed to build pipeline: {exc}"}), 500

    def _coerce_positive_int(raw_value: Any, field: str) -> Optional[int]:
        if raw_value is None:
            return None
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"'{field}' must be a positive integer.")
        if parsed <= 0:
            raise ValueError(f"'{field}' must be a positive integer.")
        return parsed

    try:
        legacy_num = _coerce_positive_int(payload.get("num_results"), "num_results")
        total_limit = _coerce_positive_int(payload.get("search_total_limit"), "search_total_limit")
        per_source_limit = _coerce_positive_int(payload.get("search_source_limit"), "search_source_limit")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    default_total = legacy_num if legacy_num is not None else 5
    if total_limit is None:
        total_limit = default_total
    if per_source_limit is None:
        per_source_limit = total_limit
    num_retrieved_docs = legacy_num if legacy_num is not None else total_limit

    try:
        print(f"[server] Processing query: {query[:50]}...")
        result = pipeline.answer(
            query,
            num_search_results=total_limit,
            per_source_search_results=per_source_limit,
            num_retrieved_docs=num_retrieved_docs,
            max_tokens=int(payload.get("max_tokens")) if payload.get("max_tokens") else 5000,
            temperature=float(payload.get("temperature")) if payload.get("temperature") else 0.3,
            allow_search=allow_search,
        )
        print(f"[server] Pipeline returned result with keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    except Exception as exc:  # pragma: no cover - propagate runtime issues
        import traceback
        error_msg = str(exc).encode('utf-8', errors='replace').decode('utf-8')
        print(f"[server] Pipeline execution error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({"error": f"Pipeline execution failed: {error_msg}"}), 500

    # Validate result structure
    if not isinstance(result, dict):
        print(f"[server] Invalid result type: {type(result)}")
        return jsonify({"error": "服务器返回数据格式错误"}), 500
    
    if "answer" not in result:
        print(f"[server] Missing 'answer' in result: {result.keys()}")
        result["answer"] = "未能生成答案"
    
    # Log answer length
    answer_len = len(result.get("answer", "")) if isinstance(result.get("answer"), str) else 0
    print(f"[server] Answer length: {answer_len} chars")
    
    # Ensure all values are JSON serializable
    try:
        result = ensure_json_serializable(result)
        print(f"[server] Serialization successful")
    except Exception as exc:
        print(f"[server] Failed to serialize result: {exc}")
        return jsonify({"error": "响应数据序列化失败"}), 500
    
    # Try to create JSON to verify it works
    try:
        test_json = json.dumps(result, ensure_ascii=False)
        print(f"[server] JSON creation successful, size: {len(test_json)} bytes")
    except Exception as exc:
        print(f"[server] JSON creation failed: {exc}")
        return jsonify({"error": f"JSON序列化失败: {str(exc)}"}), 500
    
    return jsonify(result)


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False  # 允许UTF-8字符
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
