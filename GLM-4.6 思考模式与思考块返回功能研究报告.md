# GLM-4.6 思考模式与思考块返回功能研究报告

## 核心结论

**GLM-4.6 完全支持思考模式和思考块返回功能。**

---

## 一、官方文档确认

根据智谱 AI 官方文档（[深度思考功能页面](https://zhipu-ef7018ed.mintlify.app/cn/guide/capabilities/thinking)），GLM-4.6 提供了完整的深度思考（Thinking）功能支持。

### 1.1 支持的模型

深度思考功能目前支持以下模型：
- **GLM-4.5**
- **GLM-4.6**
- **GLM-4.5v**

### 1.2 功能特性

通过启用深度思考，模型可以实现：

- **多步推理**：将复杂问题分解为多个步骤，逐步分析解决
- **逻辑分析**：提供清晰的推理过程和逻辑链条
- **提升准确性**：通过深度思考减少错误，提高回答质量
- **增强可解释性**：展示思考过程，让用户理解模型的推理逻辑
- **智能判断**：模型自动判断是否需要深度思考，优化响应效率

---

## 二、API 调用方式

### 2.1 核心参数

**thinking.type** 参数用于控制深度思考模式：

| 参数值 | 说明 |
|--------|------|
| `enabled` | （默认）启用动态思考，模型自动判断是否需要深度思考 |
| `disabled` | 禁用深度思考，直接给出回答 |

### 2.2 请求示例

**启用深度思考的请求：**

```bash
curl --location 'https://open.bigmodel.cn/api/paas/v4/chat/completions' \
--header 'Authorization: Bearer YOUR_API_KEY' \
--header 'Content-Type: application/json' \
--data '{
    "model": "glm-4.6",
    "messages": [
        {
            "role": "user",
            "content": "详细解释量子计算的基本原理，并分析其在密码学领域的潜在影响"
        }
    ],
    "thinking": {
        "type": "enabled"
    },
    "max_tokens": 4096,
    "temperature": 1.0
}'
```

**禁用深度思考的请求：**

```bash
curl --location 'https://open.bigmodel.cn/api/paas/v4/chat/completions' \
--header 'Authorization: Bearer YOUR_API_KEY' \
--header 'Content-Type: application/json' \
--data '{
    "model": "glm-4.6",
    "messages": [
        {
            "role": "user",
            "content": "今天天气怎么样？"
        }
    ],
    "thinking": {
        "type": "disabled"
    }
}'
```

### 2.3 流式调用

深度思考功能支持与流式输出结合使用：

```bash
curl --location 'https://open.bigmodel.cn/api/paas/v4/chat/completions' \
--header 'Authorization: Bearer YOUR_API_KEY' \
--header 'Content-Type: application/json' \
--data '{
    "model": "glm-4.6",
    "messages": [
        {
            "role": "user",
            "content": "设计一个电商网站的推荐系统架构"
        }
    ],
    "thinking": {
        "type": "enabled"
    },
    "stream": true,
    "max_tokens": 4096
}'
```

---

## 三、思考块返回格式

### 3.1 响应结构

启用深度思考后，API 响应中会包含 **`reasoning_content`** 字段，用于返回模型的思考过程：

```json
{
  "created": 1677652288,
  "model": "glm-4.6",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "人工智能在医疗诊断中具有巨大的应用前景...",
        "reasoning_content": "让我从多个角度来分析这个问题。首先，我需要考虑AI在医疗诊断中的技术优势..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "completion_tokens": 239,
    "prompt_tokens": 8,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "total_tokens": 247
  }
}
```

### 3.2 关键字段说明

| 字段 | 说明 |
|------|------|
| `content` | 模型的最终回答内容 |
| `reasoning_content` | 模型的思考过程和推理链条 |

---

## 四、应用场景建议

### 4.1 推荐启用深度思考的场景

- 复杂问题分析和解决
- 多步骤推理任务
- 技术方案设计
- 策略规划和决策
- 学术研究和分析
- 创意写作和内容创作

### 4.2 可以禁用深度思考的场景

- 简单事实查询
- 基础翻译任务
- 简单分类判断
- 快速问答需求

---

## 五、注意事项

### 5.1 性能影响

1. **响应时间**：启用深度思考会增加响应时间，特别是复杂任务
2. **Token 消耗**：思考过程会消耗额外的 Token，需合理规划使用
3. **流式输出**：结合流式输出可以实时查看思考过程，改善用户体验

### 5.2 参数名称差异

不同的 API 提供商可能使用不同的参数名称：

| 平台 | 参数名称 | 参数值 |
|------|---------|--------|
| **智谱 AI 官方** | `thinking.type` | `enabled` / `disabled` |
| **第三方平台**（如 SiliconFlow） | `enable_thinking` | `true` / `false` |

**重要提示**：在使用第三方 API 平台时，请参考对应平台的文档确认正确的参数名称和格式。

---

## 六、参考资料

1. [智谱 AI 官方文档 - 深度思考功能](https://zhipu-ef7018ed.mintlify.app/cn/guide/capabilities/thinking)
2. [智谱 AI 开放平台 API 文档](https://open.bigmodel.cn/dev/api)
3. [GitHub Issue: GLM-4.6 思考模式参数讨论](https://github.com/CherryHQ/cherry-studio/issues/10596)

---

## 七、总结

GLM-4.6 **完全支持**思考模式和思考块返回功能：

✅ 通过 `thinking.type` 参数控制思考模式  
✅ 通过 `reasoning_content` 字段返回思考过程  
✅ 支持动态思考和手动控制两种模式  
✅ 支持流式输出实时查看思考过程  
✅ 适用于复杂推理和多步骤分析任务  

该功能显著提升了模型在复杂任务中的准确性和可解释性，是 GLM-4.6 的重要特性之一。
