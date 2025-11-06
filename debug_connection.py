#!/usr/bin/env python3
"""
GLM API连接诊断工具
用于诊断和解决连接问题
"""

import json
import os
import sys
import time
import requests
from typing import Dict, Any, Optional


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_api_key_validity(api_key: str, base_url: str) -> Dict[str, Any]:
    """测试API密钥的有效性"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 简单的API连接测试
    test_payload = {
        "model": "glm-4.6",
        "messages": [
            {"role": "user", "content": "hello"}
        ],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return {"status": "success", "message": "API密钥有效，连接正常"}
        elif response.status_code == 401:
            return {"status": "error", "message": "API密钥无效或已过期"}
        elif response.status_code == 429:
            return {"status": "error", "message": "API调用频率超限，请稍后重试"}
        else:
            return {"status": "error", "message": f"API返回错误状态码: {response.status_code}"}
            
    except requests.exceptions.ConnectionError as e:
        return {"status": "error", "message": f"网络连接错误: {str(e)}"}
    except requests.exceptions.Timeout as e:
        return {"status": "error", "message": f"连接超时: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"未知错误: {str(e)}"}


def test_network_connectivity(base_url: str) -> Dict[str, Any]:
    """测试网络连接性"""
    try:
        response = requests.get(
            base_url,
            timeout=10
        )
        return {"status": "success", "message": f"可以访问 {base_url}"}
    except requests.exceptions.ConnectionError as e:
        return {"status": "error", "message": f"无法访问 {base_url}: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"网络测试失败: {str(e)}"}


def test_with_retry(api_key: str, base_url: str, model: str, max_retries: int = 3) -> Dict[str, Any]:
    """使用重试机制测试API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    test_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "请简单回答：1+1等于几？"}
        ],
        "max_tokens": 50,
        "temperature": 0.3
    }
    
    for attempt in range(max_retries + 1):
        try:
            print(f"尝试 {attempt + 1}/{max_retries + 1}...")
            
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "status": "success",
                    "message": f"API调用成功",
                    "response": content,
                    "attempts": attempt + 1
                }
            else:
                error_msg = f"状态码 {response.status_code}: {response.text}"
                if attempt < max_retries:
                    print(f"失败，{2 ** attempt}秒后重试: {error_msg}")
                    time.sleep(2 ** attempt)
                else:
                    return {"status": "error", "message": f"API调用失败: {error_msg}"}
                    
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接错误: {str(e)}"
            if "Connection aborted" in error_msg or "RemoteDisconnected" in error_msg:
                print(f"检测到远程连接断开，这是您遇到的问题")
            if attempt < max_retries:
                print(f"失败，{2 ** attempt}秒后重试: {error_msg}")
                time.sleep(2 ** attempt)
            else:
                return {"status": "error", "message": f"连接测试失败: {error_msg}"}
                
        except requests.exceptions.Timeout as e:
            error_msg = f"超时: {str(e)}"
            if attempt < max_retries:
                print(f"超时，{2 ** attempt}秒后重试")
                time.sleep(2 ** attempt)
            else:
                return {"status": "error", "message": f"多次重试仍然超时: {error_msg}"}
                
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            if attempt < max_retries:
                print(f"失败，{2 ** attempt}秒后重试: {error_msg}")
                time.sleep(2 ** attempt)
            else:
                return {"status": "error", "message": f"测试失败: {error_msg}"}
    
    return {"status": "error", "message": "所有重试都失败了"}


def main():
    """主诊断函数"""
    print("=" * 60)
    print("GLM API 连接诊断工具")
    print("=" * 60)
    
    try:
        # 加载配置
        config = load_config()
        print("✓ 配置文件加载成功")
        
        # 获取GLM配置
        glm_config = config.get("providers", {}).get("glm", {})
        if not glm_config:
            print("✗ 在配置文件中找不到GLM配置")
            return
        
        api_key = glm_config.get("api_key")
        model = glm_config.get("model", "glm-4.6")
        base_url = glm_config.get("base_url")
        
        if not all([api_key, base_url]):
            print("✗ GLM配置不完整，缺少api_key或base_url")
            return
        
        print(f"✓ GLM配置: {model} @ {base_url}")
        
        # 1. 测试网络连接
        print("\n1. 测试网络连接...")
        network_result = test_network_connectivity(base_url)
        if network_result["status"] == "success":
            print(f"   ✓ {network_result['message']}")
        else:
            print(f"   ✗ {network_result['message']}")
        
        # 2. 测试API密钥
        print("\n2. 测试API密钥有效性...")
        key_result = test_api_key_validity(api_key, base_url)
        if key_result["status"] == "success":
            print(f"   ✓ {key_result['message']}")
        else:
            print(f"   ✗ {key_result['message']}")
        
        # 3. 完整API测试
        print("\n3. 完整API调用测试（含重试）...")
        test_result = test_with_retry(api_key, base_url, model)
        if test_result["status"] == "success":
            print(f"   ✓ {test_result['message']} (尝试了 {test_result['attempts']} 次)")
            print(f"   测试回答: {test_result['response']}")
        else:
            print(f"   ✗ {test_result['message']}")
        
        # 总结
        print("\n" + "=" * 60)
        print("诊断总结:")
        
        all_success = all([
            network_result["status"] == "success",
            key_result["status"] == "success", 
            test_result["status"] == "success"
        ])
        
        if all_success:
            print("✓ 所有测试都通过，您的API配置应该正常工作")
            print("  如果仍然遇到问题，可能是临时性网络问题")
        else:
            print("✗ 发现问题，请根据上述测试结果进行修复")
            
            if network_result["status"] == "error":
                print("  - 网络连接问题：检查网络设置或防火墙")
            if key_result["status"] == "error":
                print("  - API密钥问题：检查密钥是否有效或已过期")
            if test_result["status"] == "error":
                print("  - API调用问题：可能是服务器暂时不可用或配额超限")
        
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"✗ 文件错误: {e}")
    except Exception as e:
        print(f"✗ 诊断过程中发生错误: {e}")


if __name__ == "__main__":
    main()