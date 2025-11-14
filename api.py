import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, Timeout, RequestException
from urllib3.util.retry import Retry


class LLMClient:
    """Universal client for various LLM API endpoints with enhanced error handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1",
        request_timeout: int = 60,
        provider: str = "openai",
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.provider = provider
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
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
        elif provider in ("glm", "zai"):
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        elif provider == "openrouter":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            self.headers["HTTP-Referer"] = "https://your-site.com"
            # OpenRouter requires ASCII-safe header values; keep title simple to avoid encoding issues
            self.headers["X-Title"] = "Intelligent QA Assistant"
        elif provider == "minimax":
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            # Default to Bearer token
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 5000,
        temperature: float = 0.7,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Chat method for OpenAI-compatible APIs with enhanced error handling."""
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

        # GLM/Zai-specific adjustments
        if self.provider in ("glm", "zai"):
            payload["stream"] = False

        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                break
                
            except ConnectionError as exc:
                last_error = exc
                if "Connection aborted" in str(exc) or "RemoteDisconnected" in str(exc):
                    error_msg = f"连接被远程服务器断开: {str(exc)}"
                else:
                    error_msg = f"网络连接错误: {str(exc)}"
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                    
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    try:
                        print(f"网络连接失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    time.sleep(wait_time)
                else:
                    return {"error": f"经过 {self.max_retries + 1} 次重试后仍然失败: {error_msg}"}
                    
            except Timeout as exc:
                last_error = exc
                error_msg = f"请求超时 ({self.request_timeout}秒): {str(exc)}"
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    try:
                        print(f"请求超时，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1})")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    time.sleep(wait_time)
                else:
                    return {"error": f"经过 {self.max_retries + 1} 次重试后仍然超时: {error_msg}"}
                    
            except RequestException as exc:
                last_error = exc
                error_msg = f"请求错误: {str(exc)}"
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    try:
                        print(f"请求失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    time.sleep(wait_time)
                else:
                    return {"error": f"经过 {self.max_retries + 1} 次重试后仍然失败: {error_msg}"}
        else:
            return {"error": f"请求最终失败: {str(last_error)}"}

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return {"error": f"响应JSON解析失败: {str(exc)}".encode('utf-8', errors='replace').decode('utf-8')}

        def _coerce_content(value: Any) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                pieces = []
                for item in value:
                    if isinstance(item, str):
                        pieces.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str):
                            pieces.append(text)
                return "\n".join(piece for piece in pieces if piece).strip()
            if isinstance(value, dict):
                text = value.get("text") or value.get("content")
                if isinstance(text, str):
                    return text.strip()
            return ""

        content = ""
        try:
            choices = data.get("choices", [])
            if choices:
                first = choices[0] if isinstance(choices[0], dict) else {}
                # Chat schema
                message = first.get("message") or {}
                content = _coerce_content(message.get("content"))
                # Fallback to text-based schema
                if not content:
                    content = _coerce_content(first.get("text"))
                # Provider specific handling when standard fields are absent
                if not content:
                    if self.provider == "minimax":
                        # Minimax responses may return a messages array or output_text field
                        messages = first.get("messages") or []
                        for msg in reversed(messages):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                candidate = _coerce_content(msg.get("content"))
                                if candidate:
                                    content = candidate
                                    break
                        if not content:
                            output_text = first.get("output_text") or data.get("output_text")
                            content = _coerce_content(output_text)
                    elif self.provider == "openrouter":
                        # OpenRouter occasionally nests content inside a "delta" key
                        delta = first.get("delta") or {}
                        if isinstance(delta, dict):
                            content = _coerce_content(delta.get("content"))
        except Exception as exc:
            try:
                print(f"解析响应内容时出错: {exc}")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            return {"error": f"解析响应内容时出错: {str(exc)}".encode('utf-8', errors='replace').decode('utf-8')}

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

    def chat_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 5000,
        temperature: float = 0.7,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Chat method for OpenAI-compatible APIs with enhanced error handling."""
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
            "stream": True,
        }

        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                    stream=True,
                )
                response.raise_for_status()
                break
                
            except ConnectionError as exc:
                last_error = exc
                if "Connection aborted" in str(exc) or "RemoteDisconnected" in str(exc):
                    error_msg = f"连接被远程服务器断开: {str(exc)}"
                else:
                    error_msg = f"网络连接错误: {str(exc)}"
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                    
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    try:
                        print(f"网络连接失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    time.sleep(wait_time)
                else:
                    yield f"Error: 经过 {self.max_retries + 1} 次重试后仍然失败: {error_msg}"
                    return
                    
            except Timeout as exc:
                last_error = exc
                error_msg = f"请求超时 ({self.request_timeout}秒): {str(exc)}"
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    try:
                        print(f"请求超时，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1})")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    time.sleep(wait_time)
                else:
                    yield f"Error: 经过 {self.max_retries + 1} 次重试后仍然超时: {error_msg}"
                    return
                    
            except RequestException as exc:
                last_error = exc
                error_msg = f"请求错误: {str(exc)}"
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    try:
                        print(f"请求失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    time.sleep(wait_time)
                else:
                    yield f"Error: 经过 {self.max_retries + 1} 次重试后仍然失败: {error_msg}"
                    return
        else:
            yield f"Error: 请求最终失败: {str(last_error)}"
            return

        try:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[len('data: '):]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: Failed to process stream: {str(e)}"




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
    try:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(json.dumps(result, indent=2, ensure_ascii=False).encode('utf-8', errors='replace').decode('utf-8'))
