# genealogy-rag-core

## What it is

`genealogy-rag-core` is a local-first genealogy extraction and RAG core for
private family-history books.

## Local-first / closed-contour guarantee

No external APIs are used by default. Local artifacts, prompts, OCR text,
model outputs, embeddings, graph files, and logs should stay inside the
infrastructure boundary.

## What works now

- `source_chunks` generation from local parsed content
- deterministic genealogy claim extraction
- local LLM claim extraction through an OpenAI-compatible endpoint
- JSON repair and evidence quote validation
- people, family, relationship, conflict, evidence, and mention artifacts
- lightweight retrieval context builders over local JSON/JSONL artifacts
- exports to DOT, JSON, GEDCOM, and HTML

## Not implemented yet

- production OCR
- vector database persistence
- production chat API
- chat UI

## Install

```bash
pip install -e ".[dev,llm]"
```

## Local LLM example

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
genealogy-rag genealogy --help
genealogy-rag genealogy build --input ./inputs --output ./outputs
genealogy-rag genealogy llm-extract --input ./outputs --model qwen2.5:7b-instruct --llm-base-url http://localhost:11434/v1
genealogy-rag genealogy export --input ./outputs --format html
```

## Development checks

```bash
ruff check .
ruff format --check .
pytest -q
```

## Security notes

Do not commit books, scans, OCR outputs, generated JSONL claims, extracted
people/relations, embeddings, exported graphs, or chat logs.
