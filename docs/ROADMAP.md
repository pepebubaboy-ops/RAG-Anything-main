# Roadmap

## Cleanup Baseline

- Keep the package local-first by default.
- Remove legacy runtime code and generated artifacts.
- Keep CI, tests, linting, and formatting green.

## Russian Name Normalization

- Improve patronymic, surname, title, and orthographic handling.
- Keep changes data-driven and test-backed.

## Claim Schema And Status

- Stabilize accepted, rejected, and pending claim states.
- Tighten validation around evidence quotes and provenance.

## Entity Resolution

- Add explicit resolution workflows for duplicate people and ambiguous mentions.
- Keep manual review paths available.

## PostgreSQL/Qdrant/Neo4j Persistence

- Add optional persistence adapters behind extras.
- Keep local file artifacts as the default baseline.

## Chat API

- Add a local API for retrieval-backed genealogy answers.
- Require explicit model and storage configuration.

## UI

- Build a private local review interface for claims, evidence, and graph edits.
- Keep export and audit trails visible.

## Eval/Gold Dataset

- Create small synthetic and permission-safe fixtures.
- Add regression tests for extraction, validation, and retrieval quality.
