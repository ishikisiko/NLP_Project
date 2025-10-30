import json
import os
from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """Universal client for various LLM API endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1",
        request_timeout: int = 300,
        provider: str = "openai"
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.provider = provider
        
        if not self.api_key:
            raise ValueError(f"{provider.upper()}_API_KEY must be provided.")
        
        # Set headers based on provider
        self.headers = {
            "Content-Type": "application/json",
        }
        
        if provider == "openai":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        elif provider == "anthropic":
            self.headers["x-api-key"] = self.api_key
            self.headers["anthropic-version"] = "2023-06-01"
        elif provider == "google":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        elif provider == "glm":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            # Default to Bearer token
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 5000,
        temperature: float = 0.7,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Chat method for OpenAI-compatible APIs."""
        endpoint = f"{self.base_url}/chat/completions"
        
        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # GLM-specific adjustments
        if self.provider == "glm":
            payload["stream"] = False

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return {"error": str(exc)}

        data = response.json()
        content = ""
        try:
            choices = data.get("choices", [])
            if choices:
                first = choices[0] if isinstance(choices[0], dict) else {}
                # Chat schema
                message = first.get("message") or {}
                content = (message.get("content") or "").strip()
                # Fallback to text-based schema
                if not content:
                    content = (first.get("text") or "").strip()
        except Exception:
            pass

        if not content:
            finish_reason = None
            try:
                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
            except Exception:
                pass
            return {
                "content": "",
                "warning": "Empty content returned. Possible causes: wrong endpoint for model, content filter, or max_tokens too small.",
                "finish_reason": finish_reason,
                "raw": data,
            }

        return {"content": content, "raw": data}


class HKGAIClient(LLMClient):
    """Legacy wrapper for HKGAI service."""
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(
            api_key=api_key,
            model_id="HKGAI-V1",
            base_url="https://oneapi.hkgai.net/v1",
            provider="hkgai"
        )


if __name__ == "__main__":
    client = HKGAIClient()
    system_prompt = "You are a helpful AI assistant providing concise and accurate responses."
    user_prompt = "what is the capital of China?"
    result = client.chat(system_prompt, user_prompt)
    print(json.dumps(result, indent=2))
