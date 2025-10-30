# NLP Project

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ishikisiko/NLP_Project.git
    cd NLP_Project
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `config.json` file in the root of the project.

2.  Configure your preferred LLM provider and add your API keys:
    ```json
    {
        "LLM_PROVIDER": "glm",
        "SERPAPI_API_KEY": "YOUR_SERPAPI_API_KEY_HERE",
        "providers": {
            "openai": {
                "api_key": "YOUR_OPENAI_API_KEY_HERE",
                "model": "gpt-3.5-turbo",
                "base_url": "https://api.openai.com/v1"
            },
            "anthropic": {
                "api_key": "YOUR_ANTHROPIC_API_KEY_HERE",
                "model": "claude-3-sonnet-20240229",
                "base_url": "https://api.anthropic.com/v1"
            },
            "google": {
                "api_key": "YOUR_GOOGLE_API_KEY_HERE",
                "model": "gemini-pro",
                "base_url": "https://generativelanguage.googleapis.com/v1beta"
            },
            "hkgai": {
                "api_key": "YOUR_HKGAI_API_KEY_HERE",
                "model": "HKGAI-V1",
                "base_url": "https://oneapi.hkgai.net/v1"
            },
            "glm": {
                "api_key": "YOUR_GLM_API_KEY_HERE",
                "model": "glm-4.6",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4/"
            }
        }
    }
    ```

### Supported LLM Providers

- **OpenAI**: Set `LLM_PROVIDER` to `"openai"` and provide your OpenAI API key
- **Anthropic Claude**: Set `LLM_PROVIDER` to `"anthropic"` and provide your Anthropic API key  
- **Google Gemini**: Set `LLM_PROVIDER` to `"google"` and provide your Google API key
- **GLM (智谱AI)**: Set `LLM_PROVIDER` to `"glm"` and provide your GLM API key. Supports GLM-4.6 and other models. (Default)
- **HKGAI**: Set `LLM_PROVIDER` to `"hkgai"` and provide your HKGAI API key

### Required Configuration

- `LLM_PROVIDER`: Choose your preferred LLM provider (`openai`, `anthropic`, `google`, `glm`, `hkgai`)
- `SERPAPI_API_KEY`: Your API key for the SerpAPI service (required for search functionality)
- Provider-specific API keys in the `providers` section

## Usage

### Basic Usage

Run the main script with your query (uses GLM by default):
```bash
python main.py "your query here"
```

### Override LLM Provider

You can temporarily override the LLM provider using the `--provider` flag:
```bash
# Use OpenAI
python main.py "your query here" --provider openai

# Use Anthropic Claude
python main.py "your query here" --provider anthropic

# Use Google Gemini
python main.py "your query here" --provider google

# Use HKGAI
python main.py "your query here" --provider hkgai

# Use GLM (智谱AI)
python main.py "your query here" --provider glm
```

### Additional Options

```bash
python main.py "your query here" \
    --max-tokens 1000 \
    --temperature 0.7 \
    --num-results 10 \
    --pretty
```

- `--max-tokens`: Maximum number of tokens for the LLM response
- `--temperature`: Sampling temperature (0.0-1.0)
- `--num-results`: Number of search results to include
- `--pretty`: Pretty print the JSON response
