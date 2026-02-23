# Offline Quick Guide

## 1) Minimal install

```bash
pip install -e .
```

## 2) Optional parser install

```bash
pip install -e '.[mineru]'
```

## 3) Enable strict offline mode

```bash
export RAGANYTHING_OFFLINE=1
```

## 4) Build family tree from content list

```bash
raganything genealogy build --input ./path/to/content_lists --output ./output_genealogy
```

Artifacts produced:

- `tree.dot`
- `claims.jsonl`
- `people.json`
- `families.json`

## 5) Export formats

```bash
raganything genealogy export --input ./output_genealogy --format dot
raganything genealogy export --input ./output_genealogy --format json
raganything genealogy export --input ./output_genealogy --format gedcom
```
