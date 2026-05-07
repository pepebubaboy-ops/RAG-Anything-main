# Local-First Operation

The core workflow is designed to run on local files and local/on-prem services.
Do not send books, scans, OCR text, prompts, embeddings, model responses, or
logs to external services unless you explicitly configure that behavior.

Recommended defaults:

```bash
export GENEALOGY_RAG_OFFLINE=1
export RAGANYTHING_OFFLINE=1
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=ollama
export LLM_MODEL=qwen2.5:7b-instruct
```

Use pre-parsed `*_content_list.json` files for deterministic local runs. PDF
parsing is optional and should use locally installed parser tools and local
model artifacts.
