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
Claim lifecycle is part of the baseline: validated LLM claims are marked
`accepted`, `pending`, `rejected`, `conflict`, or `needs_review`. Only claims with
supported evidence quotes and confidence at or above `0.55` are written to
`claims.jsonl` and allowed into the accepted graph. Pending and rejected LLM rows
stay in the local audit artifact `llm_rejected_claims.jsonl` with reasons.

Production OCR, vector search, graph persistence, chat API, and UI are future layers and must be configured explicitly.
