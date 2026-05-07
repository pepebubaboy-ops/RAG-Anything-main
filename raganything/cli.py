from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from . import __version__
from .genealogy.build import build_genealogy_tree
from .genealogy.export import export_genealogy
from .genealogy.living_graph import (
    build_living_graph,
    looks_like_living_graph_json,
    parse_relation_types_arg,
)
from .genealogy.llm_claim_extraction import run_llm_claim_pipeline


def _cmd_genealogy_build(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if (
        args.graph_mode == "living"
        or args.graph_mode == "auto"
        and input_path.is_file()
        and looks_like_living_graph_json(input_path)
    ):
        result = build_living_graph(
            input_path,
            output_dir,
            include_relation_types=parse_relation_types_arg(
                args.include_relation_types
            ),
            exclude_relation_types=parse_relation_types_arg(
                args.exclude_relation_types
            ),
            derive_kinship=bool(args.derive_kinship),
            max_relations=int(args.max_relations),
        )
        print(
            f"Living graph built: entities={result.entities_count}, "
            f"relations={result.relations_count}. "
            f"Output: {result.output_dir}{result.message_suffix}"
        )
        return 0

    reference_people = (
        Path(args.reference_people).expanduser().resolve()
        if args.reference_people
        else None
    )
    result = build_genealogy_tree(
        input_path,
        output_dir,
        parse_method=args.parse_method,
        reference_people=reference_people,
    )

    reconcile_suffix = ""
    if result.reconciliation_stats is not None:
        stats = result.reconciliation_stats
        reconcile_suffix = (
            " "
            f"Reconciled with reference people: matched={stats['matched']}, "
            f"renamed={stats['renamed']}, "
            f"enriched_fields={stats['enriched_fields']}."
        )

    print(
        f"Build complete: people={result.people_count}, "
        f"families={result.families_count}, claims={result.claims_count}. "
        f"Output: {result.output_dir}{reconcile_suffix}"
    )
    return 0


def _cmd_genealogy_export(args: argparse.Namespace) -> int:
    output_path = export_genealogy(
        Path(args.input),
        args.format,
        Path(args.output) if args.output else None,
    )
    labels = {"dot": "DOT", "json": "JSON", "gedcom": "GEDCOM", "html": "HTML"}
    print(f"{labels.get(args.format, args.format.upper())} exported: {output_path}")
    return 0


def _cmd_genealogy_llm_extract(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / "llm_claim_graph"
    )
    result = run_llm_claim_pipeline(
        input_dir,
        output_dir,
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout=int(args.llm_timeout),
        max_tokens=int(args.llm_max_tokens),
        max_candidates=(
            int(args.max_candidates) if int(args.max_candidates) > 0 else None
        ),
        context_window=int(args.context_window),
    )
    print(
        f"LLM extraction complete: candidates={result.details.get('candidate_chunks_count')}, "
        f"claims={result.claims_count}, people={result.people_count}, "
        f"relations={result.details.get('relationships_count')}. "
        f"Output: {result.output_dir}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genealogy-rag",
        description="Local-first genealogy extraction and RAG helper commands.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    genealogy_parser = subparsers.add_parser("genealogy", help="Genealogy commands")
    genealogy_subparsers = genealogy_parser.add_subparsers(dest="genealogy_command")

    build_parser = genealogy_subparsers.add_parser(
        "build",
        help="Build a local family tree from content_list JSON or PDF",
    )
    build_parser.add_argument(
        "--input",
        required=True,
        help="Path to *_content_list.json, directory with content lists, or PDF",
    )
    build_parser.add_argument(
        "--output",
        default="./outputs",
        help="Output directory for generated tree files",
    )
    build_parser.add_argument(
        "--parse-method",
        choices=["none", "mineru", "docling"],
        default="none",
        help="PDF parsing backend; default is none for offline safety",
    )
    build_parser.add_argument(
        "--graph-mode",
        choices=["auto", "tree", "living"],
        default="auto",
        help="auto: detect by input type, tree: family tree from content_list, living: relation graph from living_graph.json",
    )
    build_parser.add_argument(
        "--max-relations",
        type=int,
        default=400,
        help="Maximum number of relations in living graph DOT output",
    )
    build_parser.add_argument(
        "--include-relation-types",
        default="",
        help="Comma-separated relation types to keep in living graph mode",
    )
    build_parser.add_argument(
        "--exclude-relation-types",
        default="",
        help="Comma-separated relation types to drop in living graph mode",
    )
    build_parser.add_argument(
        "--derive-kinship",
        action="store_true",
        help=(
            "Derive extra kinship relations in living mode "
            "(brother/sister, aunt/uncle, niece/nephew, grandparent, cousin)"
        ),
    )
    build_parser.add_argument(
        "--reference-people",
        default="",
        help=(
            "Optional canonical people.json path to reconcile names and enrich "
            "birth/death fields after tree build"
        ),
    )
    build_parser.set_defaults(handler=_cmd_genealogy_build)

    export_parser = genealogy_subparsers.add_parser(
        "export",
        help="Export existing people/families artifacts to another format",
    )
    export_parser.add_argument(
        "--input",
        required=True,
        help="Directory containing people.json and families.json",
    )
    export_parser.add_argument(
        "--format",
        required=True,
        choices=["dot", "json", "gedcom", "html"],
        help="Target export format",
    )
    export_parser.add_argument(
        "--output",
        help="Output file path (optional)",
    )
    export_parser.set_defaults(handler=_cmd_genealogy_export)

    llm_parser = genealogy_subparsers.add_parser(
        "llm-extract",
        help=(
            "Run deterministic candidate search, local LLM interpretation, "
            "and validated claim graph build from source_chunks.jsonl"
        ),
    )
    llm_parser.add_argument(
        "--input",
        required=True,
        help="Directory containing source_chunks.jsonl from genealogy build",
    )
    llm_parser.add_argument(
        "--output",
        default="",
        help="Output directory; default is <input>/llm_claim_graph",
    )
    llm_parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
        help="OpenAI-compatible local model name",
    )
    llm_parser.add_argument(
        "--llm-base-url",
        default=os.getenv("LLM_BASE_URL")
        or os.getenv("LLM_BINDING_HOST")
        or "http://localhost:11434/v1",
        help="OpenAI-compatible local LLM base URL",
    )
    llm_parser.add_argument(
        "--llm-api-key",
        default=os.getenv("LLM_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or "ollama",
        help="API key for local OpenAI-compatible endpoint",
    )
    llm_parser.add_argument(
        "--llm-timeout",
        type=int,
        default=int(os.getenv("LLM_TIMEOUT", "300")),
        help="LLM request timeout in seconds",
    )
    llm_parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_TOKENS", "900")),
        help="Maximum response tokens per candidate chunk",
    )
    llm_parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Limit candidate chunks for experiments; 0 means all candidates",
    )
    llm_parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="Number of neighboring chunks to include before and after each candidate",
    )
    llm_parser.set_defaults(handler=_cmd_genealogy_llm_extract)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        return int(handler(args))
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
