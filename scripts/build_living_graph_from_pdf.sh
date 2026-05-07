#!/usr/bin/env bash
set -euo pipefail

# Build a living genealogy graph from any dynasty/family PDF.
#
# Usage:
#   scripts/build_living_graph_from_pdf.sh [--resume] /path/to/book.pdf [output_root]
#
# Outputs:
#   <output_root>/tree_seed      - parsed PDF + initial tree artifacts
#   <output_root>/living_raw     - raw living graph JSON from LLM pipeline
#   <output_root>/living_final   - filtered/visualized living graph (html/svg/json)

show_usage() {
  echo "Usage: $0 [--resume] /path/to/book.pdf [output_root]" >&2
}

RESUME="${RESUME:-0}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      show_usage
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -lt 1 || ${#ARGS[@]} -gt 2 ]]; then
  show_usage
  exit 1
fi

INPUT_PDF="${ARGS[0]}"
OUTPUT_ROOT="${ARGS[1]:-./output_living_from_pdf_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$INPUT_PDF" ]]; then
  echo "Input PDF not found: $INPUT_PDF" >&2
  exit 1
fi

if [[ -x ".venv312/bin/python" ]]; then
  PY=".venv312/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  PY="python3"
fi

# Ensure subprocess calls (e.g. `mineru`) resolve against the same env as $PY.
if [[ "$PY" == */* ]]; then
  PY_BIN_DIR="$(dirname "$PY")"
else
  PY_RESOLVED="$(command -v "$PY" || true)"
  PY_BIN_DIR="$(dirname "${PY_RESOLVED:-}")"
fi
if [[ -n "${PY_BIN_DIR:-}" && -d "$PY_BIN_DIR" ]]; then
  export PATH="$PY_BIN_DIR:$PATH"
fi

LLM_MODEL="${LLM_MODEL:-qwen2.5:7b-instruct}"
LLM_BINDING_HOST="${LLM_BINDING_HOST:-http://localhost:11434/v1}"
LLM_BINDING_API_KEY="${LLM_BINDING_API_KEY:-ollama}"
# MinerU defaults tuned for reliable local parsing.
# Override these via env if needed.
MINERU_BACKEND="${MINERU_BACKEND:-pipeline}"
MINERU_PARSE_METHOD="${MINERU_PARSE_METHOD:-txt}"
MINERU_LANG="${MINERU_LANG:-cyrillic}"
MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-modelscope}"
# Strict genealogy profile (generic for any dynasty/family).
STRICT_DYNASTY_MODE="${STRICT_DYNASTY_MODE:-1}"
DYNASTY_KEEP_REL_TYPES_RAW="${DYNASTY_KEEP_REL_TYPES_RAW:-parent_child,spouse,sibling}"
DYNASTY_MIN_COMPONENT_SIZE="${DYNASTY_MIN_COMPONENT_SIZE:-3}"
DYNASTY_MAX_COMPONENTS="${DYNASTY_MAX_COMPONENTS:-2}"
FINAL_INCLUDE_REL_TYPES="${FINAL_INCLUDE_REL_TYPES:-parent_child,parent_of,father_of,mother_of,child_of,spouse,sibling,sibling_of,brother_of,sister_of,grandparent_of,grandfather_of,grandmother_of,grandchild_of,grandson_of,granddaughter_of,aunt_uncle_of,aunt_of,uncle_of,nibling_of,nephew_of,niece_of,cousin_of}"

TREE_DIR="$OUTPUT_ROOT/tree_seed"
RAW_DIR="$OUTPUT_ROOT/living_raw"
FINAL_DIR="$OUTPUT_ROOT/living_final"
LOG_DIR="$OUTPUT_ROOT/logs"
PDF_TOTAL_PAGES=""
STEP3_STATIC_DETAILS=""
SOURCE_GRAPH_FOR_RENDER=""

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

format_elapsed_mmss() {
  local sec="${1:-0}"
  printf "%02d:%02d" "$((sec / 60))" "$((sec % 60))"
}

extract_pdf_total_pages() {
  local pdf_path="$1"
  local total
  total="$("$PY" - "$pdf_path" 2>/dev/null <<'PY'
import sys
from pathlib import Path

pdf_path = Path(sys.argv[1])
try:
    from pypdf import PdfReader
except Exception:
    print("")
    raise SystemExit(0)

try:
    r = PdfReader(str(pdf_path))
    print(len(r.pages))
except Exception:
    print("")
PY
)"
  printf "%s" "$total"
}

_normalize_log_for_scan() {
  local logfile="$1"
  if [[ ! -s "$logfile" ]]; then
    return 0
  fi
  tr '\r' '\n' <"$logfile" | tail -n 220
}

find_cached_content_list() {
  local parsed_dir="$1"
  if [[ ! -d "$parsed_dir" ]]; then
    return 0
  fi
  find "$parsed_dir" -type f -name '*_content_list.json' | head -n 1
}

_step_extra_progress() {
  local status_key="$1"
  local logfile="$2"

  case "$status_key" in
    parse_pdf)
      local lines cur total pair
      lines="$(_normalize_log_for_scan "$logfile")"
      pair="$(printf "%s\n" "$lines" | awk '
        {
          line = $0
          while (match(line, /([0-9]+)[[:space:]]*\/[[:space:]]*([0-9]+)/)) {
            frag = substr(line, RSTART, RLENGTH)
            split(frag, a, "/")
            gsub(/[[:space:]]/, "", a[1])
            gsub(/[[:space:]]/, "", a[2])
            c = a[1] + 0
            t = a[2] + 0
            if (t > best_t || (t == best_t && c > best_c)) {
              best_t = t
              best_c = c
            }
            line = substr(line, RSTART + RLENGTH)
          }
        }
        END {
          if (best_t > 0) {
            printf "%d/%d", best_c, best_t
          }
        }
      ')"
      if [[ -n "$pair" ]]; then
        cur="${pair%/*}"
        total="${pair#*/}"
      else
        cur=""
        total=""
      fi
      if [[ -z "$total" && -n "${PDF_TOTAL_PAGES:-}" ]]; then
        total="$PDF_TOTAL_PAGES"
      fi
      if [[ -n "$total" ]]; then
        if [[ -z "$cur" ]]; then
          printf " | pages ?/%s | init" "$total"
        else
          if [[ "$cur" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] && (( cur > total )); then
            cur="$total"
          fi
          if [[ "$cur" == "0" && ! -s "$logfile" ]]; then
            printf " | pages ?/%s | init" "$total"
          else
            printf " | pages %s/%s" "$cur" "$total"
          fi
        fi
      fi
      ;;
    living_raw)
      local lines chunks_total chunk_cur chunk_total mentions relations page_block page_nums page_span
      lines="$(_normalize_log_for_scan "$logfile")"
      chunks_total="$(printf "%s\n" "$lines" | sed -nE 's/.*chunks=([0-9]+).*/\1/p' | tail -n1)"
      chunk_cur="$(printf "%s\n" "$lines" | sed -nE 's/.*chunk ([0-9]+)\/([0-9]+).*/\1/p' | tail -n1)"
      chunk_total="$(printf "%s\n" "$lines" | sed -nE 's/.*chunk ([0-9]+)\/([0-9]+).*/\2/p' | tail -n1)"
      page_block="$(printf "%s\n" "$lines" | sed -nE 's/.*chunk [0-9]+\/[0-9]+ pages=\[([^]]*)\].*/\1/p' | tail -n1)"
      mentions="$(printf "%s\n" "$lines" | sed -nE 's/.*mentions=([0-9]+).*/\1/p' | tail -n1)"
      relations="$(printf "%s\n" "$lines" | sed -nE 's/.*relations=([0-9]+).*/\1/p' | tail -n1)"
      if [[ -n "$chunk_total" && -z "$chunks_total" ]]; then
        chunks_total="$chunk_total"
      fi
      if [[ -n "$chunks_total" ]]; then
        printf " | chunks %s/%s" "${chunk_cur:-0}" "$chunks_total"
      fi
      if [[ -n "$page_block" ]]; then
        page_nums="$(printf "%s\n" "$page_block" | tr ',' '\n' | sed -E 's/[^0-9-]//g' | awk 'NF')"
        if [[ -n "$page_nums" ]]; then
          page_span="$(printf "%s\n" "$page_nums" | awk '
            NR == 1 { min = $1 + 0; max = $1 + 0 }
            { x = $1 + 0; if (x < min) min = x; if (x > max) max = x }
            END {
              if (NR == 0) exit 0
              if (min == max) printf "%d", min
              else printf "%d-%d", min, max
            }
          ')"
          if [[ -n "$page_span" ]]; then
            printf " | pages %s" "$page_span"
          fi
        fi
      fi
      if [[ -n "$mentions" ]]; then
        printf " | mentions %s" "$mentions"
      fi
      if [[ -n "$relations" ]]; then
        printf " | relations %s" "$relations"
      fi
      ;;
    render_final)
      if [[ -n "${STEP3_STATIC_DETAILS:-}" ]]; then
        printf " | %s" "$STEP3_STATIC_DETAILS"
      fi
      ;;
  esac
}

step_skip() {
  local step="$1"
  local total="$2"
  local title="$3"
  local reason="$4"
  echo "[$step/$total] $title skipped ($reason)"
}

run_step() {
  local step="$1"
  local total="$2"
  local status_key="$3"
  local title="$4"
  local logfile="$5"
  shift 5

  local start_ts
  local end_ts
  local elapsed
  local elapsed_fmt
  local rc

  start_ts="$(date +%s)"

  if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo "[$step/$total] $title (verbose mode)"
    "$@" | tee "$logfile"
    rc="${PIPESTATUS[0]}"
  else
    "$@" >"$logfile" 2>&1 &
    local pid="$!"
    local spin='|/-\'
    local i=0

    if [[ -t 1 ]]; then
      while kill -0 "$pid" 2>/dev/null; do
        end_ts="$(date +%s)"
        elapsed="$((end_ts - start_ts))"
        elapsed_fmt="$(format_elapsed_mmss "$elapsed")"
        printf "\r[%s/%s] %s %c %s%s" "$step" "$total" "$title" "${spin:i++%${#spin}:1}" "$elapsed_fmt" "$(_step_extra_progress "$status_key" "$logfile")"
        sleep 0.2
      done
      wait "$pid"
      rc="$?"
      end_ts="$(date +%s)"
      elapsed="$((end_ts - start_ts))"
      elapsed_fmt="$(format_elapsed_mmss "$elapsed")"
      if [[ "$rc" -eq 0 ]]; then
        printf "\r[%s/%s] %s done %s%s\n" "$step" "$total" "$title" "$elapsed_fmt" "$(_step_extra_progress "$status_key" "$logfile")"
      else
        printf "\r[%s/%s] %s failed %s%s\n" "$step" "$total" "$title" "$elapsed_fmt" "$(_step_extra_progress "$status_key" "$logfile")" >&2
      fi
    else
      wait "$pid"
      rc="$?"
      if [[ "$rc" -eq 0 ]]; then
        echo "[$step/$total] $title done"
      else
        echo "[$step/$total] $title failed" >&2
      fi
    fi
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "Step [$step/$total] failed. Log: $logfile" >&2
    tail -n 120 "$logfile" >&2 || true
    exit "$rc"
  fi
}

if [[ "$RESUME" == "1" ]]; then
  echo "Resume mode: ON"
fi

CONTENT_LIST="$(find_cached_content_list "$TREE_DIR/_parsed_pdf")"
if [[ "$RESUME" == "1" && -n "$CONTENT_LIST" ]]; then
  if [[ -s "$TREE_DIR/tree.html" && -s "$TREE_DIR/people.json" && -s "$TREE_DIR/families.json" ]]; then
    step_skip "1" "3" "Parse PDF and build seed tree" "resume: found cached parse + tree artifacts"
  else
    run_step "1" "3" "seed_tree_from_cache" "Rebuild seed tree from cached content list" "$LOG_DIR/step_1_tree_rebuild.log" \
      "$PY" -m raganything.cli genealogy build \
        --input "$CONTENT_LIST" \
        --graph-mode tree \
        --output "$TREE_DIR"
  fi
else
  PDF_TOTAL_PAGES="$(extract_pdf_total_pages "$INPUT_PDF")"
  run_step "1" "3" "parse_pdf" "Parse PDF and build seed tree" "$LOG_DIR/step_1_parse.log" \
    env \
      MINERU_BACKEND="$MINERU_BACKEND" \
      MINERU_PARSE_METHOD="$MINERU_PARSE_METHOD" \
      MINERU_LANG="$MINERU_LANG" \
      MINERU_MODEL_SOURCE="$MINERU_MODEL_SOURCE" \
      "$PY" -m raganything.cli genealogy build \
        --input "$INPUT_PDF" \
        --parse-method mineru \
        --graph-mode tree \
        --output "$TREE_DIR"
fi

if [[ -z "$CONTENT_LIST" ]]; then
  CONTENT_LIST="$(find_cached_content_list "$TREE_DIR/_parsed_pdf")"
fi
if [[ -z "$CONTENT_LIST" ]]; then
  echo "Could not find parsed *_content_list.json under $TREE_DIR/_parsed_pdf" >&2
  exit 1
fi

if [[ "$RESUME" == "1" && -s "$RAW_DIR/living_graph.json" ]]; then
  step_skip "2" "3" "Build raw living graph from parsed content list" "resume: found living_graph.json"
else
  run_step "2" "3" "living_raw" "Build raw living graph from parsed content list" "$LOG_DIR/step_2_living_raw.log" \
    env PYTHONUNBUFFERED=1 "$PY" scripts/living_graph_pipeline.py \
      --input "$CONTENT_LIST" \
      --output-dir "$RAW_DIR" \
      --model "$LLM_MODEL" \
      --llm-base-url "$LLM_BINDING_HOST" \
      --llm-api-key "$LLM_BINDING_API_KEY" \
      --llm-timeout "${LLM_TIMEOUT:-180}" \
      --llm-max-tokens "${LLM_MAX_TOKENS:-500}" \
      --max-relation-candidates "${MAX_RELATION_CANDIDATES:-80}" \
      --kinship-pass-max-candidates "${KINSHIP_PASS_MAX_CANDIDATES:-80}" \
      --recall-pass-max-candidates "${RECALL_PASS_MAX_CANDIDATES:-60}" \
      --dry-run
fi

SOURCE_GRAPH_FOR_RENDER="$RAW_DIR/living_graph.json"
if [[ "$STRICT_DYNASTY_MODE" == "1" ]]; then
  FILTERED_SOURCE="$RAW_DIR/living_graph_genealogy_source.json"
  if [[ "$RESUME" == "1" && -s "$FILTERED_SOURCE" ]]; then
    echo "[strict] Reusing filtered genealogy source: $FILTERED_SOURCE"
  else
    "$PY" scripts/filter_living_graph_for_tree.py \
      --input "$RAW_DIR/living_graph.json" \
      --output "$FILTERED_SOURCE" \
      --keep-relation-types "$DYNASTY_KEEP_REL_TYPES_RAW" \
      --min-component-size "$DYNASTY_MIN_COMPONENT_SIZE" \
      --max-components "$DYNASTY_MAX_COMPONENTS"
  fi
  if [[ -s "$FILTERED_SOURCE" ]]; then
    SOURCE_GRAPH_FOR_RENDER="$FILTERED_SOURCE"
  fi
fi

STEP3_STATIC_DETAILS="$("$PY" - "$SOURCE_GRAPH_FOR_RENDER" 2>/dev/null <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print("")
    raise SystemExit(0)
try:
    obj = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
entities = len(obj.get("entities") or [])
relations = len(obj.get("relations") or [])
print(f"entities {entities} | relations {relations}")
PY
)"

if [[ "$RESUME" == "1" && -s "$FINAL_DIR/living_graph.html" && -s "$FINAL_DIR/relations.json" ]]; then
  step_skip "3" "3" "Render final living graph" "resume: found final living artifacts"
else
  run_step "3" "3" "render_final" "Render final living graph" "$LOG_DIR/step_3_render.log" \
    "$PY" -m raganything.cli genealogy build \
      --input "$SOURCE_GRAPH_FOR_RENDER" \
      --graph-mode living \
      --derive-kinship \
      --include-relation-types "$FINAL_INCLUDE_REL_TYPES" \
      --output "$FINAL_DIR"
fi

echo
echo "Done."
echo "Seed tree:      $TREE_DIR/tree.html"
echo "Raw living:     $RAW_DIR/living_graph_view.html"
echo "Final living:   $FINAL_DIR/living_graph.html"
echo "Logs:           $LOG_DIR"
