# genealogy-rag-core

`genealogy-rag-core` is a local-first genealogy extraction and RAG core for
private family-history books. It keeps parsing artifacts, prompts, OCR text,
candidate chunks, model outputs, embeddings, graph files, and logs local by
default.

## What Works

- `source_chunks` generation from pre-parsed content lists
- deterministic genealogy claim extraction from text
- local LLM claim extraction through an OpenAI-compatible endpoint
- JSON repair, schema validation, and evidence quote validation
- accepted/rejected/pending claim artifacts
- people, family, relationship, conflict, evidence, and mention artifacts
- lightweight retrieval context builders over local JSON/JSONL artifacts
- exports to DOT, JSON, GEDCOM, and HTML

## Not Yet

- production OCR
- vector database persistence
- production chat API or chat UI

## Install

```bash
pip install -e ".[dev,llm]"
```

The default install has no cloud dependency. The `llm` extra installs the
OpenAI Python client only so the CLI can talk to a local OpenAI-compatible
server such as Ollama.

## Local LLM

Example Ollama/OpenAI-compatible configuration:

```bash
export GENEALOGY_RAG_OFFLINE=1
export RAGANYTHING_OFFLINE=1
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_API_KEY=ollama
export LLM_MODEL=qwen2.5:7b-instruct
```

## CLI

```bash
genealogy-rag --help
genealogy-rag genealogy build --input ./inputs --output ./outputs
genealogy-rag genealogy llm-extract --input ./outputs --model qwen2.5:7b-instruct --llm-base-url http://localhost:11434/v1
genealogy-rag genealogy export --input ./outputs --format html
```

The temporary `raganything` console script remains as a compatibility alias.

## Security

No external APIs are used by default. The documented defaults point to local
or on-prem endpoints. Treat `inputs/`, `outputs/`, scans, OCR text, JSONL model
outputs, embeddings, and exported graphs as private data; they are ignored by
git by default.
