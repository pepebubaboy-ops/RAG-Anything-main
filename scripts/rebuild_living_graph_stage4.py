#!/usr/bin/env python3
"""
Rebuild living graph (stage 4/5) from an existing run directory.

Input run directory must contain:
- raw_extractions_general.json
- optional run_summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _load_pipeline_module():
    root = Path(__file__).resolve().parent
    pipeline_path = root / "living_graph_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "living_graph_pipeline_rebuild", str(pipeline_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["living_graph_pipeline_rebuild"] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild living graph from raw extractions (stage 4/5 only)"
    )
    p.add_argument(
        "--run-dir", required=True, help="Directory with raw_extractions_general.json"
    )
    p.add_argument(
        "--output-dir", default=None, help="Output dir (default: <run-dir>_rebuild)"
    )
    p.add_argument("--model", default=None, help="LLM model override")
    p.add_argument(
        "--llm-base-url",
        default=os.getenv("LLM_BINDING_HOST", "http://localhost:11434/v1"),
    )
    p.add_argument("--llm-api-key", default=os.getenv("LLM_BINDING_API_KEY", "ollama"))
    p.add_argument(
        "--llm-timeout", type=int, default=int(os.getenv("LLM_TIMEOUT", "300"))
    )
    p.add_argument(
        "--min-relation-confidence",
        type=float,
        default=float(os.getenv("MIN_RELATION_CONFIDENCE", "0.0")),
    )
    p.add_argument(
        "--auto-merge-threshold",
        type=float,
        default=float(os.getenv("AUTO_MERGE_THRESHOLD", "0.85")),
    )
    p.add_argument(
        "--possible-merge-threshold",
        type=float,
        default=float(os.getenv("POSSIBLE_MERGE_THRESHOLD", "0.65")),
    )
    p.add_argument(
        "--llm-merge-min-confidence",
        type=float,
        default=float(os.getenv("LLM_MERGE_MIN_CONFIDENCE", "0.78")),
    )
    p.add_argument(
        "--llm-merge-max-candidates",
        type=int,
        default=int(os.getenv("LLM_MERGE_MAX_CANDIDATES", "120")),
    )
    p.add_argument(
        "--llm-merge-max-tokens",
        type=int,
        default=int(os.getenv("LLM_MERGE_MAX_TOKENS", "320")),
    )
    p.add_argument(
        "--llm-merge-derived-min-score",
        type=float,
        default=float(os.getenv("LLM_MERGE_DERIVED_MIN_SCORE", "0.56")),
    )
    p.add_argument(
        "--llm-merge-derived-max-pairs",
        type=int,
        default=int(os.getenv("LLM_MERGE_DERIVED_MAX_PAIRS", "240")),
    )
    p.add_argument(
        "--llm-merge-agent", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--llm-merge-derive-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--persist-neo4j", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    p.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--neo4j-db", default=os.getenv("NEO4J_DATABASE", ""))
    return p


async def _run(args: argparse.Namespace) -> int:
    lg = _load_pipeline_module()

    run_dir = Path(args.run_dir).resolve()
    raw_path = run_dir / "raw_extractions_general.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path}")
    parsed_chunks = json.loads(raw_path.read_text(encoding="utf-8"))

    summary_path = run_dir / "run_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    model = args.model or summary.get("model") or os.getenv("LLM_MODEL", "qwen2.5:32b")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(f"{run_dir}_rebuild").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = await lg._build_graph(
        parsed_chunks,
        min_relation_confidence=float(args.min_relation_confidence),
        auto_merge_threshold=float(args.auto_merge_threshold),
        possible_merge_threshold=float(args.possible_merge_threshold),
        llm_merge_agent=bool(args.llm_merge_agent),
        llm_merge_min_confidence=float(args.llm_merge_min_confidence),
        llm_merge_max_candidates=int(args.llm_merge_max_candidates),
        llm_merge_max_tokens=int(args.llm_merge_max_tokens),
        llm_merge_derive_candidates=bool(args.llm_merge_derive_candidates),
        llm_merge_derived_min_score=float(args.llm_merge_derived_min_score),
        llm_merge_derived_max_pairs=int(args.llm_merge_derived_max_pairs),
        model=str(model),
        base_url=str(args.llm_base_url),
        api_key=str(args.llm_api_key),
        timeout=int(args.llm_timeout),
    )
    graph["model"] = model

    (output_dir / "mentions_resolved.json").write_text(
        json.dumps(lg._serialize(graph.get("mentions")), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "living_graph.json").write_text(
        json.dumps(lg._serialize(graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    view_path = lg._write_living_graph_view(output_dir, graph, title="Living Graph")

    print("DONE")
    print(f"out_dir={output_dir}")
    print(f"web_view={view_path}")
    print(
        f"entities={len(graph.get('entities') or [])} "
        f"relations={len(graph.get('relations') or [])} "
        f"possible_same={len(graph.get('possible_same') or [])} "
        f"title_typo_merges={int(graph.get('title_typo_merged_entities') or 0)}"
    )

    if bool(args.persist_neo4j):
        source_file = str(summary.get("input_content_list") or "")
        neo4j_stats = lg._persist_graph_to_neo4j(
            graph,
            source_file=source_file,
            neo4j_uri=str(args.neo4j_uri),
            neo4j_user=str(args.neo4j_user),
            neo4j_password=str(args.neo4j_password),
            neo4j_db=str(args.neo4j_db or ""),
            reset_db=False,
        )
        print(f"neo4j={neo4j_stats}")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
