# 配置文件设置说明

## 快速开始

1. 复制配置模板文件：
```bash
cp config.example.json config.json
```

2. 编辑 `config.json` 文件，填入你的API密钥：
   - `SERPAPI_API_KEY`: 从 [SerpAPI](https://serpapi.com/) 获取
   - 各个LLM提供商的 `api_key`: 根据你使用的服务填写相应的密钥

## 支持的LLM提供商

### OpenAI
- 获取API密钥: https://platform.openai.com/api-keys
- 模型: gpt-3.5-turbo, gpt-4 等

### Anthropic Claude
- 获取API密钥: https://console.anthropic.com/
- 模型: claude-3-sonnet-20240229, claude-3-opus-20240229 等

### Google Gemini
- 获取API密钥: https://makersuite.google.com/app/apikey
- 模型: gemini-pro, gemini-pro-vision 等

### HKGAI
- 获取API密钥: https://oneapi.hkgai.net/
- 模型: HKGAI-V1

### GLM (智谱AI)
- 获取API密钥: https://open.bigmodel.cn/
- 模型: glm-4.6, glm-4 等

## 默认提供商

项目默认使用 `glm` 作为LLM提供商。你可以通过以下方式更改：

1. 修改 `config.json` 中的 `LLM_PROVIDER` 字段
2. 或使用命令行参数：`--provider openai`

## 搜索服务

项目使用 SerpAPI 作为搜索服务提供商。你需要：
1. 注册账户：https://serpapi.com/
2. 获取API密钥
3. 在 `config.json` 中设置 `SERPAPI_API_KEY`

## 安全提醒

- `config.json` 包含敏感信息，已添加到 `.gitignore`
- 不要将包含真实API密钥的 `config.json` 提交到版本控制系统
- 建议定期轮换API密钥