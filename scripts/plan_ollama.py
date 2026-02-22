#!/usr/bin/env python3
"""
Plan runner for RAG-Anything using Ollama (OpenAI-compatible /v1 API).

Features:
- Insert pre-parsed MinerU content_list JSON (no parser needed)
- Query the created workspace
- Route models by step (graph_extract / graph_summary / query_answer)

Examples:
  .venv/bin/python scripts/plan_ollama.py smoke \\
    --content-list output_lmstudio/..._content_list.json \\
    --max-items 80 \\
    --question "Кто главный герой и в чем завязка сюжета?"

  .venv/bin/python scripts/plan_ollama.py query \\
    --working-dir ./rag_storage_plan/<uuid> \\
    --question "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc


GRAPH_EXTRACT_MARKERS = (
    "extract entities and relationships from the input text",
    "knowledge graph specialist responsible for extracting entities and relationships",
    "<entity_types>",
)
GRAPH_SUMMARY_MARKERS = (
    "synthesize a list of descriptions",
    "description list:",
    "merged description",
)
QUERY_ANSWER_MARKERS = (
    "reference document list",
    "### references",
    "knowledge graph data",
    "document chunks",
)


def _contains_any(text: str, markers: Tuple[str, ...]) -> bool:
    return any(m in text for m in markers)


def _classify_llm_task(prompt: str, system_prompt: Optional[str]) -> str:
    prompt_text = (prompt or "").lower()
    system_text = (system_prompt or "").lower()
    full_text = f"{system_text}\n{prompt_text}"

    if _contains_any(full_text, GRAPH_EXTRACT_MARKERS):
        return "graph_extract"
    if _contains_any(full_text, GRAPH_SUMMARY_MARKERS):
        return "graph_summary"
    if _contains_any(full_text, QUERY_ANSWER_MARKERS):
        return "query_answer"
    return "default"


def _pick_model(task: str) -> str:
    # Defaults are chosen for the "plan":
    # - Graph: qwen2.5:32b
    # - Query: qwen2.5:32b (fast), optionally post-refine with llama3.3:70b
    fallback = os.getenv("LLM_MODEL", "qwen2.5:32b")
    graph = os.getenv("LLM_MODEL_GRAPH", fallback)
    graph_extract = os.getenv("LLM_MODEL_GRAPH_EXTRACT", graph)
    graph_summary = os.getenv("LLM_MODEL_GRAPH_SUMMARY", graph_extract)
    query = os.getenv("LLM_MODEL_QUERY", graph_extract)

    if task == "graph_extract":
        return graph_extract
    if task == "graph_summary":
        return graph_summary
    if task == "query_answer":
        return query
    return fallback


def _ollama_extra_body() -> Dict[str, Any]:
    # Ollama supports "options" in the OpenAI-compatible API.
    # Keep it conservative; users can override via env.
    extra_body: Dict[str, Any] = {}

    options: Dict[str, Any] = {}
    num_ctx = os.getenv("OLLAMA_NUM_CTX")
    if num_ctx:
        try:
            options["num_ctx"] = int(num_ctx)
        except ValueError:
            pass

    if options:
        extra_body["options"] = options
    return extra_body


async def _make_llm_func() -> Any:
    base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_BINDING_API_KEY", "ollama")
    router_debug = os.getenv("LLM_ROUTER_DEBUG", "false").lower() == "true"

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        task = _classify_llm_task(prompt, system_prompt)
        model = _pick_model(task)
        if router_debug:
            print(f"🧭 llm_router task='{task}' model='{model}'")

        # Inject Ollama-specific options via OpenAI's extra_body hook.
        # Preserve any caller-provided extra_body.
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        injected = _ollama_extra_body()
        if injected:
            merged_options = dict(extra_body.get("options", {}) or {})
            merged_options.update(injected.get("options", {}) or {})
            extra_body["options"] = merged_options
            kwargs["extra_body"] = extra_body

        return await openai_complete_if_cache(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

    return llm_model_func


async def _make_embedding_func() -> EmbeddingFunc:
    base_url = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434/v1")
    api_key = os.getenv("EMBEDDING_BINDING_API_KEY", "ollama")
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

    async def embed_async(texts: List[str]) -> np.ndarray:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        try:
            resp = await client.embeddings.create(model=model, input=texts)
            vectors: List[List[float]] = []
            for item in resp.data:
                emb = item.embedding
                if not isinstance(emb, list):
                    raise TypeError(f"Unsupported embedding payload type: {type(emb)}")
                vectors.append([float(v) for v in emb])
            return np.array(vectors, dtype=np.float32)
        finally:
            await client.close()

    probe = await embed_async(["dimension probe"])
    if probe.ndim != 2 or probe.shape[0] < 1:
        raise RuntimeError(f"Unexpected embedding output shape: {probe.shape}")
    embedding_dim = int(probe.shape[1])

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
        func=embed_async,
    )


def _load_content_list(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"content_list JSON must be a list, got: {type(obj)}")
    return obj


def _select_items(
    content_list: List[Dict[str, Any]],
    max_items: int,
    only_text: bool,
) -> List[Dict[str, Any]]:
    if max_items <= 0:
        return []

    if not only_text:
        return content_list[:max_items]

    selected: List[Dict[str, Any]] = []
    for item in content_list:
        if len(selected) >= max_items:
            break
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        if not str(item.get("text", "")).strip():
            continue
        selected.append(item)
    return selected


async def _new_rag(working_dir: str) -> RAGAnything:
    llm_model_func = await _make_llm_func()
    embedding_func = await _make_embedding_func()

    rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="txt",
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
            display_content_stats=True,
        ),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            # Keep it conservative for local inference; can be overridden via env if needed.
            "addon_params": {
                # LightRAG keyword extraction is language-sensitive; default to Russian for RU corpora.
                "language": os.getenv("RAG_LANGUAGE", "Russian"),
            },
            "chunk_token_size": int(os.getenv("CHUNK_TOKEN_SIZE", "800")),
            "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP_TOKENS", "80")),
            "llm_model_max_async": int(os.getenv("LLM_MAX_ASYNC", "1")),
            "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "1")),
            "default_llm_timeout": int(os.getenv("LLM_TIMEOUT", "600")),
            "default_embedding_timeout": int(os.getenv("EMBED_TIMEOUT", "120")),
            "entity_extract_max_gleaning": int(os.getenv("ENTITY_GLEANING", "0")),
        },
    )

    # We are inserting pre-parsed content_list; don't block on external parser installs.
    rag._parser_installation_checked = True

    # Compatibility for different LightRAG doc_status schemas.
    async def _noop_mark_multimodal(_: str) -> None:
        return None

    rag._mark_multimodal_processing_complete = _noop_mark_multimodal
    return rag


async def cmd_insert(args: argparse.Namespace) -> int:
    content_path = Path(args.content_list)
    content_list = _load_content_list(content_path)
    selected = _select_items(content_list, args.max_items, args.only_text)
    if not selected:
        raise SystemExit("No items selected to insert.")

    working_dir = args.working_dir or f"./rag_storage_plan/{uuid.uuid4()}"
    doc_id = args.doc_id or f"doc-{uuid.uuid4()}"
    file_path = args.file_path or content_path.name

    print(f"📁 working_dir: {working_dir}")
    print(f"🧾 doc_id     : {doc_id}")
    print(f"📄 file_path  : {file_path}")
    print(f"🧩 items      : {len(selected)} (only_text={args.only_text})")

    rag = await _new_rag(working_dir)
    try:
        await rag.insert_content_list(
            content_list=selected,
            file_path=file_path,
            split_by_character="\\n",
            # Allow fallback token-based splitting when a line/segment is too large.
            split_by_character_only=False,
            doc_id=doc_id,
            display_stats=True,
        )
    finally:
        await rag.finalize_storages()

    print("✅ insert: done")
    return 0


async def cmd_query(args: argparse.Namespace) -> int:
    if not args.working_dir:
        raise SystemExit("--working-dir is required for query")

    rag = await _new_rag(args.working_dir)
    try:
        init = await rag._ensure_lightrag_initialized()
        if isinstance(init, dict) and not init.get("success", False):
            raise SystemExit(init.get("error") or "Failed to initialize LightRAG")
        enable_rerank = os.getenv("ENABLE_RERANK", "false").lower() == "true"
        system_prompt = os.getenv("QUERY_SYSTEM_PROMPT")
        if system_prompt is None:
            # Keep default behavior unless the user signals a target language.
            lang = (os.getenv("RAG_LANGUAGE") or "").strip().lower()
            if lang in ("ru", "russian", "русский", "русский язык"):
                system_prompt = "Отвечай по-русски. Не добавляй выдумок."

        answer = await rag.aquery(
            args.question,
            mode=args.mode,
            system_prompt=system_prompt,
            enable_rerank=enable_rerank,
        )
    finally:
        await rag.finalize_storages()

    print("\n=== ANSWER ===")
    print(answer)

    final_model = args.final_model
    if final_model:
        base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:11434/v1")
        api_key = os.getenv("LLM_BINDING_API_KEY", "ollama")
        refined = await openai_complete_if_cache(
            model=final_model,
            prompt=(
                "Улучши ответ, сохранив факты и не добавляя выдумок.\\n\\n"
                f"Вопрос: {args.question}\\n\\n"
                f"Черновик ответа:\\n{answer}"
            ),
            system_prompt="Ты редактор. Улучши формулировки и структуру.",
            base_url=base_url,
            api_key=api_key,
            max_tokens=int(os.getenv("FINAL_MAX_TOKENS", "800")),
            temperature=float(os.getenv("FINAL_TEMPERATURE", "0.2")),
            extra_body=_ollama_extra_body() or None,
        )
        print("\n=== FINAL PASS ===")
        print(refined)

    return 0


async def cmd_smoke(args: argparse.Namespace) -> int:
    # One-shot: insert + query in the same working dir.
    working_dir = args.working_dir or f"./rag_storage_plan/{uuid.uuid4()}"
    insert_ns = argparse.Namespace(
        content_list=args.content_list,
        max_items=args.max_items,
        only_text=args.only_text,
        working_dir=working_dir,
        doc_id=args.doc_id,
        file_path=args.file_path,
    )
    await cmd_insert(insert_ns)

    query_ns = argparse.Namespace(
        working_dir=working_dir,
        question=args.question,
        mode=args.mode,
        final_model=args.final_model,
    )
    await cmd_query(query_ns)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG plan runner (Ollama)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_insert(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--content-list",
            required=True,
            help="Path to *_content_list.json",
        )
        sp.add_argument(
            "--max-items",
            type=int,
            default=80,
            help="How many items from content_list to use (default: 80)",
        )
        sp.add_argument(
            "--only-text",
            action="store_true",
            help="Insert only text items (recommended for first run)",
        )
        sp.add_argument(
            "--working-dir",
            default=None,
            help="LightRAG workspace dir (default: ./rag_storage_plan/<uuid>)",
        )
        sp.add_argument("--doc-id", default=None)
        sp.add_argument("--file-path", default=None)

    sp_insert = sub.add_parser(
        "insert", help="Insert content_list into a new/existing working_dir"
    )
    add_common_insert(sp_insert)

    sp_query = sub.add_parser("query", help="Query an existing working_dir")
    sp_query.add_argument("--working-dir", required=True)
    sp_query.add_argument("--question", required=True)
    sp_query.add_argument(
        "--mode",
        default="hybrid",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
    )
    sp_query.add_argument(
        "--final-model",
        default=None,
        help="Optional final refinement model (e.g., llama3.3:70b)",
    )

    sp_smoke = sub.add_parser("smoke", help="Insert + query in one run")
    add_common_insert(sp_smoke)
    sp_smoke.add_argument("--question", required=True)
    sp_smoke.add_argument(
        "--mode",
        default="hybrid",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
    )
    sp_smoke.add_argument(
        "--final-model",
        default=None,
        help="Optional final refinement model (e.g., llama3.3:70b)",
    )

    return p


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "insert":
        return asyncio.run(cmd_insert(args))
    if args.cmd == "query":
        return asyncio.run(cmd_query(args))
    if args.cmd == "smoke":
        return asyncio.run(cmd_smoke(args))
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
