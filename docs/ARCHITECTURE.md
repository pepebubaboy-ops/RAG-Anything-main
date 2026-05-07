# Architecture

The local-first pipeline is:

```text
books/scans
-> OCR/parsing
-> source_chunks
-> candidate chunks
-> local LLM raw extraction
-> JSON repair/schema validation/evidence quote validation
-> accepted/rejected/pending claims
-> people/relationships/conflicts
-> retrieval/chat context
```

The baseline stores intermediate artifacts as local JSON and JSONL files. PDF
parsing, LLM calls, graph persistence, and chat serving are optional layers and
must be configured explicitly.
