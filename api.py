import json
import os
from typing import Any, Dict, List, Optional

import requests


class HKGAIClient:
    """Thin wrapper around the HKGAI chat completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = "sk-iqA1pjC48rpFXdkU7cCaE3BfBc9145B4BfCbEe0912126646",
        model_id: str = "HKGAI-V1",
        base_url: str = "https://oneapi.hkgai.net/v1",
        request_timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("HKGAI_API_KEY")
        if not self.api_key:
            raise ValueError("HKGAI_API_KEY must be provided via argument or environment variable.")

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
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


if __name__ == "__main__":
    client = HKGAIClient()
    system_prompt = "You are a helpful AI assistant providing concise and accurate responses."
    user_prompt = "what is the capital of China?"
    result = client.chat(system_prompt, user_prompt)
    print(json.dumps(result, indent=2))
