#!/usr/bin/env bash
set -euo pipefail

# Run living graph pipeline with text pre-cleaning enabled.
#
# Usage:
#   scripts/run_living_graph_prepared.sh [INPUT_CONTENT_LIST] [OUTPUT_DIR] [extra args...]
#
# Examples:
#   scripts/run_living_graph_prepared.sh
#   scripts/run_living_graph_prepared.sh output_lmstudio/romanovy/hybrid_auto/romanovy_content_list.json
#   DRY_RUN=true scripts/run_living_graph_prepared.sh ... ...
#   scripts/run_living_graph_prepared.sh ... ... --max-pages 20 --no-llm-merge-agent

export LLM_BINDING_HOST="${LLM_BINDING_HOST:-http://localhost:11434/v1}"
export LLM_BINDING_API_KEY="${LLM_BINDING_API_KEY:-ollama}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-8192}"

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4j}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-}"

INPUT="${1:-output_lmstudio/romanovy/hybrid_auto/romanovy_content_list.json}"
OUTPUT_DIR="${2:-output_living_graph/run_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift $#
fi

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  PY="python3"
fi

ARGS=(
  --input "$INPUT"
  --output-dir "$OUTPUT_DIR"
  --model "$LLM_MODEL"
  --llm-base-url "$LLM_BINDING_HOST"
  --llm-api-key "$LLM_BINDING_API_KEY"
  --llm-timeout "${LLM_TIMEOUT:-300}"
  --llm-max-tokens "${LLM_MAX_TOKENS:-900}"
  --prepare-text-for-llm
  --strip-repeated-page-lines
  --repeated-line-min-occurrences "${REPEATED_LINE_MIN_OCCURRENCES:-3}"
  --repeated-line-min-share "${REPEATED_LINE_MIN_SHARE:-0.35}"
  --pages-per-chunk "${PAGES_PER_CHUNK:-2}"
  --page-overlap "${PAGE_OVERLAP:-1}"
  --max-chars-per-chunk "${MAX_CHARS_PER_CHUNK:-4200}"
  --relation-candidate-window "${RELATION_CANDIDATE_WINDOW:-2}"
  --max-relation-candidates "${MAX_RELATION_CANDIDATES:-80}"
  --kinship-pass-max-candidates "${KINSHIP_PASS_MAX_CANDIDATES:-80}"
  --recall-pass-window-extra "${RECALL_PASS_WINDOW_EXTRA:-2}"
  --recall-pass-max-candidates "${RECALL_PASS_MAX_CANDIDATES:-60}"
  --min-relation-confidence "${MIN_RELATION_CONFIDENCE:-0.0}"
  --auto-merge-threshold "${AUTO_MERGE_THRESHOLD:-0.85}"
  --possible-merge-threshold "${POSSIBLE_MERGE_THRESHOLD:-0.65}"
  --llm-merge-min-confidence "${LLM_MERGE_MIN_CONFIDENCE:-0.78}"
  --llm-merge-max-candidates "${LLM_MERGE_MAX_CANDIDATES:-120}"
  --llm-merge-max-tokens "${LLM_MERGE_MAX_TOKENS:-320}"
  --llm-merge-derived-min-score "${LLM_MERGE_DERIVED_MIN_SCORE:-0.56}"
  --llm-merge-derived-max-pairs "${LLM_MERGE_DERIVED_MAX_PAIRS:-240}"
  --neo4j-uri "$NEO4J_URI"
  --neo4j-user "$NEO4J_USERNAME"
  --neo4j-password "$NEO4J_PASSWORD"
  --neo4j-db "$NEO4J_DATABASE"
)

if [[ "${RULE_KINSHIP_PASS:-true}" == "false" ]]; then
  ARGS+=(--no-rule-kinship-pass)
else
  ARGS+=(--rule-kinship-pass)
fi
if [[ "${STRICT_PARENT_CHILD_VALIDATION:-true}" == "false" ]]; then
  ARGS+=(--no-strict-parent-child-validation)
else
  ARGS+=(--strict-parent-child-validation)
fi

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  ARGS+=(--dry-run)
fi
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

"$PY" scripts/living_graph_pipeline.py "${ARGS[@]}"

echo
echo "Run completed."
echo "Output directory: $OUTPUT_DIR"
