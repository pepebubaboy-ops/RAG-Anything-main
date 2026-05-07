# Genealogy Pipeline Review Package

This archive is prepared for reviewing the current genealogy/RAG pipeline logic.

## What To Analyze

Please analyze how the project builds a genealogy knowledge graph from PDF text and local LLM extraction:

1. PDF/text parsing produces `source_chunks.jsonl`.
2. Deterministic candidate search finds genealogy-relevant chunks.
3. Local LLM interprets each candidate chunk and writes raw outputs.
4. Deterministic validation repairs/parses JSON, checks schema, checks evidence quotes, rejects unsafe claims.
5. Accepted claims build `people.json`, `relationships.json`, `conflicts.json`, and RAG documents.
6. Retrieval resolves query entities and uses accepted graph relationships before lexical fallback.

## Key Code Paths

- `raganything/cli.py`
  - CLI commands, including `genealogy build`, `genealogy export`, and `genealogy llm-extract`.
- `raganything/genealogy/build.py`
  - Main non-LLM genealogy build flow.
- `raganything/genealogy/llm_claim_extraction.py`
  - New LLM extraction pipeline: candidate search, raw LLM output, tolerant JSON parsing, validation, graph build.
- `raganything/genealogy/knowledge_graph.py`
  - Canonical relationship graph and conflict validation.
- `raganything/genealogy/mentions.py`
  - Mention extraction.
- `raganything/genealogy/resolution.py`
  - Mention/person resolution report.
- `raganything/genealogy/rag_index.py`
  - RAG document construction.
- `raganything/genealogy/retrieval.py`
  - Entity-aware retrieval from accepted graph facts.
- `raganything/parser.py`
  - MinerU parser integration and fallback discovery of `.venv/bin/mineru`.

## Important Output Artifacts

See `outputs/output_genealogy_romanovy_pdf_llm_claim_graph/`.

Important files:

- `candidate_chunks.jsonl`
- `llm_extraction_raw.jsonl`
- `llm_claim_candidates.jsonl`
- `llm_rejected_claims.jsonl`
- `claims.jsonl`
- `people.json`
- `relationships.json`
- `conflicts.json`
- `person_resolution.json`
- `tree.dot`
- `tree.html`

The omitted heavy artifacts are:

- original PDF/images
- MinerU intermediate files
- `rag_documents.jsonl`
- rendered PNG/SVG

## Current Run Summary

Full LLM extraction on the Romanov PDF text layer:

- candidate chunks: 22
- raw LLM responses: 22
- parsed raw responses: 13
- invalid JSON raw responses: 9
- accepted claims: 14
- rejected claim candidates: 13
- people: 20
- relationships: 16
- conflicts: 0

Known issues to inspect:

- local LLM still emits invalid JSON for some chunks;
- Russian name normalization is weak;
- case inflection breaks resolution (`Михаила Федоровича` vs `Михаил Федорович`);
- generic/title names can leak from LLM output (`император Павла I`);
- partial names appear (`Петр` instead of `Петр I`);
- retrieval still depends on canonicalized names.

## Suggested Review Questions

1. Is the separation `raw LLM output -> validated claims -> graph` robust enough?
2. Are rejected claims handled safely and transparently?
3. Where should Russian name canonicalization happen?
4. Should graph building use only accepted claims, or include `pending/needs_review` relations?
5. How should retry/repair of invalid JSON be implemented?
6. What schema changes would make genealogy claims more reliable?
