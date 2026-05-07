# Roadmap

## Cleanup baseline

- Keep the package local-first by default.
- Remove legacy runtime code and generated artifacts.
- Keep CI, tests, linting, and formatting green.

## Claim schema and status

- Stabilize accepted, rejected, pending, conflict, and needs_review states.
- Prevent low-confidence claims from entering the accepted graph.

## General genealogy normalization

- Normalize common person-name spelling and whitespace variants.
- Normalize patronymics, surnames, initials, dates, and place strings where possible.
- Keep normalization conservative, data-driven, and test-backed.
- Do not add old-orthography or nobility/title-specific rules unless real input data requires them.

## Entity resolution

- Separate mentions, candidates, and canonical people.
- Add manual review paths for ambiguous merges.
- Avoid automatic merges based only on normalized names.

## Persistence

- Add optional PostgreSQL, Qdrant, and Neo4j/Memgraph adapters.
- Keep local file artifacts as the default baseline.

## Chat API

- Add a local retrieval-backed API for genealogy answers.
- Require explicit local model and storage configuration.

## UI

- Build a private local review interface for claims, evidence, and graph edits.

## Eval / gold dataset

- Create small synthetic and permission-safe fixtures.
- Add regression tests for extraction, validation, and retrieval quality.
