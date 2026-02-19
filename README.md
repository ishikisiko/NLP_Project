# NLP Project

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline.

## LangChain integration

- Local and hybrid RAG now run on LangChain primitives (FAISS + HuggingFace embeddings) for chunking and retrieval.
- Install the refreshed dependencies (`langchain`, `langchain-community`, `faiss-cpu`) via `pip install -r requirements.txt`.

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
        "domainClassifier": {
            "provider": "glm",
            "model": "glm-4.6"
        },
        "RERANK_PROVIDER": "qwen3-rerank",
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
                "base_url": "https://open.bigmodel.cn/api/anthropic"
            Notes:
            - `minimax` provider now points to Minimax's Anthropic-compatible endpoint at `https://api.minimax.io/anthropic` and the example model is `MiniMax-M2`.
            - `zai` provider uses Anthropic-compatible URL `https://open.bigmodel.cn/api/anthropic` by default. The application detects Anthropic-compatible base_urls (containing `/anthropic`) and uses Anthropic-style headers and message endpoints.

            }
        },
        "rerank": {
            "min_score": 0.0,
            "max_per_domain": 1,
            "providers": {
                "qwen": {
                    "api_key": "YOUR_DASHSCOPE_API_KEY_HERE",
                    "model": "qwen3-rerank",
                    "base_url": "https://dashscope.aliyuncs.com/api/v1/services/rerank",
                    "timeout": 15
                }
            }
        }
    }
    ```

    Use the optional `domainClassifier` block to point domain routing at a lighter or cheaper model without impacting the primary answer generation client.



### Supported LLM Providers

- **OpenAI**: Set `LLM_PROVIDER` to `"openai"` and provide your OpenAI API key
- **Anthropic Claude**: Set `LLM_PROVIDER` to `"anthropic"` and provide your Anthropic API key  
- **Google Gemini**: Set `LLM_PROVIDER` to `"google"` and provide your Google API key
- **GLM (智谱AI)**: Set `LLM_PROVIDER` to `"glm"` and provide your GLM API key. Supports GLM-4.6 and other models. (Default)
- **HKGAI**: Set `LLM_PROVIDER` to `"hkgai"` and provide your HKGAI API key
- **MiniMax**: Set `LLM_PROVIDER` to `"minimax"` and provide your MiniMax API key. Supports MiniMax-M2 model with thinking mode.

#### MiniMax Thinking Mode

MiniMax M2 model supports a **thinking mode** that allows the model to show its reasoning process. You can configure this in `config.json`:

```json
"minimax": {
    "api_key": "YOUR_MINIMAX_API_KEY_HERE",
    "model": "MiniMax-M2",
    "base_url": "https://api.minimaxi.com/anthropic/v1",
    "thinking": {
        "enabled": true,
        "display_in_response": false
    }
}
```

- `thinking.enabled`: (boolean) Enable/disable the thinking mode. When `true`, the model will generate reasoning process internally.
- `thinking.display_in_response`: (boolean) When `true`, the thinking process will be included in the response text prefixed with `[思考过程]`. When `false`, only the final answer is returned.

**Note**: Thinking mode is only available for MiniMax M2 model via the Anthropic-compatible API endpoint.

### Required Configuration

- `LLM_PROVIDER`: Choose your preferred LLM provider (`openai`, `anthropic`, `google`, `glm`, `hkgai`)
- `SERPAPI_API_KEY`: Your API key for the SerpAPI service (required for search functionality)
- Provider-specific API keys in the `providers` section
- `RERANK_PROVIDER`: Optional reranking backend (`qwen3-rerank`). Provide the corresponding credentials under `rerank.providers`

## Usage

### Web Interface

Run the web server:
```bash
python server.py
```

Then open your browser to `http://localhost:8000`.

The web interface allows you to:
-   Switch between `search`, `local`, and `hybrid` RAG modes.
-   Upload local files for the `local` and `hybrid` modes.
-   Configure the LLM provider and other parameters.

### Command-Line Interface

#### Search RAG

Run the main script with your query (uses GLM by default):
```bash
python main.py "your query here"
```

#### Local RAG

Run the main script with your query and specify the path to your local files:
```bash
python main.py "your query here" --mode local --data-path ./data
```

#### Hybrid RAG

Combine web search with your local documents:
```bash
python main.py "your query here" --mode hybrid --data-path ./data
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
    --disable-rerank \
    --pretty
```

- `--max-tokens`: Maximum number of tokens for the LLM response
- `--temperature`: Sampling temperature (0.0-1.0)
- `--num-results`: Number of search results to include
- `--pretty`: Pretty print the JSON response
- `--disable-rerank`: Skip reranking even if configured

### Batch Testing Multiple Queries

You can exercise the orchestrator against a list of simple queries stored in a UTF-8 text file (one per line) using the batch runner:

```bash
# queries.txt contains one query per line; blank lines or lines starting with # are ignored
python batch_test.py --queries-file ./queries.txt --search on --pretty
```

Key flags:
- `--config`: optional path to an alternate `config.json` (otherwise `NLP_CONFIG_PATH` or the project root config is used)
- `--data-path`: directory for local RAG documents
- `--provider`: temporarily override the LLM provider for this batch
- `--disable-rerank`: force reranking off even if configured
- `--pretty`: pretty-print the full JSON output per query (omit for concise answers)

This is useful for quick regression checks before UI deployments or backend changes.
