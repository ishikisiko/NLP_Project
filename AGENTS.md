# Repository Guidelines
Project should run in conda environment "env1"
## Project Structure & Module Organization
Core orchestration lives in `main.py`, which wires together search (`search.py`), embedding-based retrieval (`local_rag.py`), and model clients (`api.py`, `no_rag_baseline.py`). `server.py` hosts the Flask API that powers the browser UI under `frontend/` (plain HTML/CSS/JS). Persisted uploads land in `uploads/`; clean this directory when rotating documents. Configuration secrets belong in `config.json` (copy from `config.example.json`) or an alternate path exposed via `NLP_CONFIG_PATH`.

## Build, Test, and Development Commands
Install dependencies once per virtual environment: `pip install -r requirements.txt`. Launch the web app with `python server.py` and visit `http://localhost:8000`. The CLI fallback stays available through `python main.py "Your question" --mode search`. Switch to local retrieval with `python main.py "Your question" --mode local --data-path ./uploads`. Use `--provider` to override the default LLM for a single run. When debugging configuration, set `NLP_CONFIG_PATH=/full/path/config.json` before invoking either entrypoint.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive, snake_case names for Python modules, functions, and variables. Preserve type hints and concise docstrings where they already exist, and prefer early validation with informative errors. JavaScript in `frontend/script.js` is modern ES2015; keep functions arrow-based, use camelCase identifiers, and avoid introducing frameworks without discussion. Environment keys remain UPPER_SNAKE_CASE to match the existing config schema.

## Testing Guidelines
Automated tests are not yet committed; validate changes by exercising both the CLI and Flask routes. Run `python main.py "sanity check" --pretty` to inspect raw JSON, and submit sample questions through the UI to confirm search/local modes. When contributing test coverage, add a `tests/` package that uses `pytest`, update `requirements.txt`, and ensure new tests can be invoked with `python -m pytest`. Document any manual validation steps in the pull request until the automated suite matures.

## Commit & Pull Request Guidelines
Keep commit subjects short, imperative, and aligned with the existing history (for example, “Add simple web UI and API server”). Describe noteworthy implementation details and configuration impacts in the body. Pull requests should summarize intent, call out configuration changes (`config.json`, env vars), list manual or automated tests run, and attach UI screenshots when frontend behavior shifts. Link to tracking issues or tasks where applicable so reviewers understand context rapidly.
