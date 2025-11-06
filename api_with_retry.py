import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RobustLLMClient:
    """增强版LLM客户端，包含重试机制和错误处理。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1",
        request_timeout: int = 300,
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
        
        # 设置headers
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
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # 配置重试策略
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
        """Chat方法，包含重试逻辑。"""
        endpoint = f"{self.base_url}/chat/completions"
        
        # 构建消息数组
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

        # GLM特定调整
        if self.provider == "glm":
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
                
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    print(f"请求失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1}): {exc}")
                    time.sleep(wait_time)
                else:
                    return {"error": f"经过 {self.max_retries + 1} 次重试后仍然失败: {str(exc)}"}
        else:
            return {"error": f"请求失败: {str(last_error)}"}

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return {"error": f"JSON解析失败: {str(exc)}"}

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


# 使用示例
if __name__ == "__main__":
    # 测试GLM API连接
    config = {
        "providers": {
            "glm": {
                "api_key": "2f7ee24f777f44e3a6ca78b537c4e315.AU1xVz8F5ttgGMXM",
                "model": "glm-4.6",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4"
            }
        }
    }
    
    client = RobustLLMClient(
        api_key=config["providers"]["glm"]["api_key"],
        model_id=config["providers"]["glm"]["model"],
        base_url=config["providers"]["glm"]["base_url"],
        provider="glm",
        max_retries=3,
        backoff_factor=2.0
    )
    
    result = client.chat(
        system_prompt="你是一个有用的AI助手。",
        user_prompt="你好，请简单介绍一下自己。",
        max_tokens=1000,
        temperature=0.7
    )
    
    print("结果:", json.dumps(result, indent=2, ensure_ascii=False))