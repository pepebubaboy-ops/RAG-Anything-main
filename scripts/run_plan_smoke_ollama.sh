#!/usr/bin/env bash
set -euo pipefail

# Smoke run of the multi-model plan via Ollama.
# - Graph extract/summary: qwen2.5:32b
# - Query answer         : qwen2.5:32b
# - Embeddings           : nomic-embed-text:latest

export LLM_BINDING_HOST="${LLM_BINDING_HOST:-http://localhost:11434/v1}"
export LLM_BINDING_API_KEY="${LLM_BINDING_API_KEY:-ollama}"
export EMBEDDING_BINDING_HOST="${EMBEDDING_BINDING_HOST:-http://localhost:11434/v1}"
export EMBEDDING_BINDING_API_KEY="${EMBEDDING_BINDING_API_KEY:-ollama}"

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text:latest}"

export LLM_MODEL_GRAPH_EXTRACT="${LLM_MODEL_GRAPH_EXTRACT:-qwen2.5:32b}"
export LLM_MODEL_GRAPH_SUMMARY="${LLM_MODEL_GRAPH_SUMMARY:-qwen2.5:32b}"
export LLM_MODEL_QUERY="${LLM_MODEL_QUERY:-qwen2.5:32b}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:32b}"

export LLM_ROUTER_DEBUG="${LLM_ROUTER_DEBUG:-false}"

# Optional: ensure enough context for entity extraction prompts.
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"

# LightRAG keyword extraction is language-sensitive.
export RAG_LANGUAGE="${RAG_LANGUAGE:-Russian}"

CONTENT_LIST="${1:-output_lmstudio/Chehov_Kashtanka.358580.fb2/hybrid_auto/Chehov_Kashtanka.358580.fb2_content_list.json}"
QUESTION="${2:-Кто главный герой произведения и в чем завязка сюжета?}"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  PY="python3"
fi

"$PY" scripts/plan_ollama.py smoke \
  --content-list "$CONTENT_LIST" \
  --only-text \
  --max-items 120 \
  --question "$QUESTION" \
  --mode hybrid
