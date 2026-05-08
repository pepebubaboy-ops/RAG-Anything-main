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

Claim validation has an explicit lifecycle. Only `accepted` claims enter
`claims.jsonl` and the final relationship graph. `pending`, `needs_review`,
`rejected`, and `conflict` rows are audit-only artifacts for inspection and
follow-up.

Co-parent relationships are not spouse relationships by themselves. The graph
may connect parents through a family node, but `spouse_of` is accepted only from
an explicit spouse claim.

Production OCR, vector search, graph persistence, chat API, and UI are future layers and must be configured explicitly.
