# NLP_Project

Minimal baseline for the Intelligent Search Engine project.

## Quickstart

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Export the required API keys:

   ```bash
   export HKGAI_API_KEY="your-hkgai-api-key"
   export SERPAPI_API_KEY="your-serpapi-api-key"
   ```

3. Run the No-RAG baseline:

   ```bash
   python main.py "What is the capital of China?" --pretty
   ```

The pipeline fetches top search results from SerpAPI, injects the snippets into the prompt, and asks the HKGAI model for a concise answer.
