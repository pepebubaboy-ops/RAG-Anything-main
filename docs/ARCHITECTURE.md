# Architecture

The baseline local-first pipeline is:

```text
books / scans
  -> OCR / parsing
  -> source_chunks
  -> candidate chunks
  -> local LLM raw extraction
  -> JSON repair / schema validation / evidence quote validation
  -> accepted / rejected / pending claims
  -> people / relationships / conflicts
  -> retrieval / chat context
```

The current baseline stores intermediate artifacts as local JSON and JSONL files.

Production OCR, vector search, graph persistence, chat API, and UI are future
layers and must be configured explicitly.
