# RAGAnything (Offline-First Genealogy)

This repository is now optimized for local/offline genealogy workflows.

## Install

Minimal (genealogy + CLI):

```bash
pip install -e .
```

With PDF parsing (MinerU):

```bash
pip install -e '.[mineru]'
```

With full optional stack:

```bash
pip install -e '.[all]'
```

## CLI

Show help:

```bash
raganything --help
```

Build a family tree from `*_content_list.json` (or a directory containing them):

```bash
raganything genealogy build --input ./inputs --output ./output_genealogy
```

Build from PDF (requires parser extras/tools):

```bash
raganything genealogy build --input ./input.pdf --parse-method mineru --output ./output_genealogy
```

Export existing artifacts:

```bash
raganything genealogy export --input ./output_genealogy --format dot
raganything genealogy export --input ./output_genealogy --format json
raganything genealogy export --input ./output_genealogy --format gedcom
```

## Offline Mode

Strict offline mode:

```bash
export RAGANYTHING_OFFLINE=1
```

In this mode, operations that may require downloads are blocked with a clear error.
Use pre-parsed `*_content_list.json` for deterministic local runs.

## Notes

- Public `RAGAnything` and `RAGAnythingConfig` exports remain available via lazy import.
- Heavy dependencies are now in extras (`rag`, `mineru`, `neo4j`, `dotenv`, `all`).
- Legacy script entrypoints were reduced to thin wrappers in `scripts/`.
