#!/usr/bin/env python3
"""
Genealogy expansion runner using:
- RAGAnything/LightRAG as evidence retrieval
- A canonical Neo4j store for people/families/claims/evidence

Design notes:
- The canonical tree is stored in Neo4j. LightRAG's KG is treated as assistive/noisy.
- We write "claims" with provenance instead of directly trusting a single answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig
from raganything.genealogy import (
    Claim,
    Evidence,
    GenealogyPipeline,
    GenealogyPipelineConfig,
    InMemoryGenealogyStore,
    Neo4jGenealogyStore,
    PersonSpec,
)
from raganything.genealogy.extractors import ClaimExtractor
from raganything.genealogy.json_utils import robust_json_loads


def _to_ollama_host(base_url: str) -> str:
    # http://localhost:11434/v1 -> http://localhost:11434
    return base_url.replace("/v1", "").rstrip("/")


def _ollama_extra_body() -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    options: Dict[str, Any] = {}
    num_ctx = os.getenv("OLLAMA_NUM_CTX")
    if num_ctx:
        try:
            options["num_ctx"] = int(num_ctx)
        except ValueError:
            pass
    if options:
        extra["options"] = options
    return extra


async def _make_llm_func() -> Any:
    base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_BINDING_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:32b")

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
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
    binding = os.getenv("EMBEDDING_BINDING", "ollama").lower()

    async def embed_async(texts: List[str]) -> np.ndarray:
        if binding == "ollama":
            from lightrag.llm.ollama import ollama_embed

            embs = await ollama_embed.func(
                texts=texts,
                embed_model=model,
                host=_to_ollama_host(base_url),
                api_key=api_key,
            )
            return np.array(embs, dtype=np.float32)

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
    dim = int(probe.shape[1])

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
        func=embed_async,
    )


async def _new_rag(working_dir: str) -> RAGAnything:
    llm_model_func = await _make_llm_func()
    embedding_func = await _make_embedding_func()

    rag = RAGAnything(
        config=RAGAnythingConfig(
            working_dir=working_dir,
            parser=os.getenv("PARSER", "mineru"),
            parse_method=os.getenv("PARSE_METHOD", "txt"),
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
            display_content_stats=False,
        ),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "addon_params": {"language": os.getenv("RAG_LANGUAGE", "English")},
            "chunk_token_size": int(os.getenv("CHUNK_TOKEN_SIZE", "800")),
            "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP_TOKENS", "80")),
            "llm_model_max_async": int(os.getenv("LLM_MAX_ASYNC", "1")),
            "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "1")),
            "default_llm_timeout": int(os.getenv("LLM_TIMEOUT", "600")),
            "default_embedding_timeout": int(os.getenv("EMBED_TIMEOUT", "120")),
            "entity_extract_max_gleaning": int(os.getenv("ENTITY_GLEANING", "0")),
        },
    )

    # In genealogy runs we are querying an existing working_dir; don't block on parser install.
    rag._parser_installation_checked = True
    return rag


_SYSTEM_PROMPT = """You are a careful genealogist extracting structured facts from evidence.

Rules:
- Do not invent. If evidence is insufficient, return an empty list of claims.
- Output ONLY valid JSON (no prose).
- Confidence must be 0..1.
- Evidence quotes must be verbatim snippets from the provided context (short).
"""


def _task_prompt(task_type: str, subject: Dict[str, Any], retrieved_context: str) -> str:
    if task_type == "find_parents":
        person: PersonSpec = subject["person"]
        return f"""
Extract the parents of the target person from the context.

Target person:
- name: {person.name}
- birth_year: {person.birth_year}
- birth_place: {person.birth_place}

Return JSON:
{{
  "claims": [
    {{
      "claim_type": "parent_child",
      "confidence": 0.0,
      "parents": [{{"name": "...", "birth_year": null, "birth_place": null}}, {{"name": "...", "birth_year": null, "birth_place": null}}],
      "child": {{"name": "{person.name}"}},
      "evidence": [{{"file_path": null, "page_idx": null, "quote": "..."}}],
      "notes": "optional"
    }}
  ]
}}

Context:
{retrieved_context}
""".strip()

    if task_type == "find_children":
        parents: List[PersonSpec] = subject.get("parents") or []
        pnames = ", ".join(p.name for p in parents) if parents else "(unknown)"
        return f"""
Extract children for the parental couple from the context.

Parents: {pnames}

Return JSON:
{{
  "claims": [
    {{
      "claim_type": "parent_child",
      "confidence": 0.0,
      "parents": [{", ".join([json.dumps({'name': p.name}) for p in parents])}],
      "child": {{"name": "...", "birth_year": null, "birth_place": null}},
      "evidence": [{{"file_path": null, "page_idx": null, "quote": "..."}}],
      "notes": "optional"
    }}
  ]
}}

Context:
{retrieved_context}
""".strip()

    if task_type == "find_spouses":
        person: PersonSpec = subject["person"]
        return f"""
Extract spouses/partners of the target person from the context.

Target person:
- name: {person.name}

Return JSON:
{{
  "claims": [
    {{
      "claim_type": "spouse",
      "confidence": 0.0,
      "person1": {{"name": "{person.name}"}},
      "person2": {{"name": "..."}},
      "evidence": [{{"file_path": null, "page_idx": null, "quote": "..."}}],
      "notes": "optional"
    }}
  ]
}}

Context:
{retrieved_context}
""".strip()

    if task_type == "find_profile":
        person: PersonSpec = subject["person"]
        return f"""
Extract biographical details for the target person from the context.

Target person:
- name: {person.name}
- birth_date: {person.birth_date}
- birth_year: {person.birth_year}
- birth_place: {person.birth_place}
- death_date: {person.death_date}
- death_year: {person.death_year}
- death_place: {person.death_place}

Return JSON:
{{
  "claims": [
    {{
      "claim_type": "person_profile",
      "confidence": 0.0,
      "attributes": {{
        "birth_date": null,
        "death_date": null,
        "birth_place": null,
        "death_place": null,
        "occupation": null,
        "gender": null,
        "aliases": [],
        "biography": null,
        "media": [
          {{"kind": "photo", "path": "/absolute/path/to/image.jpg", "caption": "optional"}}
        ]
      }},
      "evidence": [{{"file_path": null, "page_idx": null, "quote": "...", "image_path": null}}],
      "notes": "optional"
    }}
  ]
}}

Context:
{retrieved_context}
""".strip()

    return f'{{"claims":[]}}'


class OllamaGenealogyExtractor(ClaimExtractor):
    def __init__(self, rag: RAGAnything, mode: str = "hybrid", debug: bool = False) -> None:
        self.rag = rag
        self.mode = mode
        self.debug = debug

    async def _retrieve_context(self, query: str) -> str:
        from lightrag import QueryParam

        # We need raw retrieved context (chunks/graph), not the rendered answer prompt template.
        qp = QueryParam(mode=self.mode, only_need_context=True, enable_rerank=False)
        return await self.rag.lightrag.aquery(query, param=qp)

    async def _repair_claim_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        repair_prompt = f"""
Convert the following extraction output into strict JSON.

Requirements:
- Return ONLY valid JSON object with top-level key "claims".
- Schema: {{"claims":[{{"claim_type":"parent_child|spouse|person_profile","confidence":0.0,"evidence":[]}}]}}
- Keep only supported claim fields; drop unknown fields.
- If no usable claims exist, return {{"claims":[]}}.

Raw extraction output:
{raw_output}
""".strip()
        repaired = await self.rag.llm_model_func(
            repair_prompt,
            system_prompt="You normalize outputs to strict JSON. Output JSON only.",
        )
        return robust_json_loads(repaired)

    async def extract(self, task_type: str, subject: Dict[str, Any]) -> List[Claim]:
        # Build a retrieval query.
        if task_type == "find_parents":
            person: PersonSpec = subject["person"]
            q = f"Кто родители {person.name}?"
        elif task_type == "find_children":
            parents: List[PersonSpec] = subject.get("parents") or []
            if len(parents) >= 2:
                q = f"Перечисли детей {parents[0].name} и {parents[1].name}."
            elif len(parents) == 1:
                q = f"Перечисли детей {parents[0].name}."
            else:
                return []
        elif task_type == "find_spouses":
            person = subject["person"]
            q = f"Кто супруг(а) {person.name}?"
        elif task_type == "find_profile":
            person = subject["person"]
            q = (
                "Извлеки биографические данные: дата/место рождения, дата/место смерти, "
                f"профессия, пол, алиасы и упоминания фото для {person.name}."
            )
        else:
            return []

        retrieved = await self._retrieve_context(q)
        prompt = _task_prompt(task_type, subject, retrieved)
        if self.debug:
            print("\n=== RETRIEVED PROMPT ===")
            print(retrieved[:2000])
            print("\n=== CLAIM PROMPT ===")
            print(prompt[:2000])

        resp = await self.rag.llm_model_func(prompt, system_prompt=_SYSTEM_PROMPT)
        obj = robust_json_loads(resp)
        if not obj:
            obj = await self._repair_claim_json(resp)
        if not obj:
            if self.debug:
                print("\n=== RAW MODEL OUTPUT (UNPARSED) ===")
                print((resp or "")[:4000])
            return []
        raw_claims = obj.get("claims") or []
        if not isinstance(raw_claims, list):
            return []

        out: List[Claim] = []
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            ctype = str(c.get("claim_type") or c.get("type") or "").strip()
            if not ctype:
                continue
            conf = c.get("confidence", 0.5)
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.5

            evidence_list: List[Evidence] = []
            raw_evs = c.get("evidence") or []
            if isinstance(raw_evs, list):
                for e in raw_evs:
                    if not isinstance(e, dict):
                        continue
                    evidence_list.append(
                        Evidence(
                            file_path=e.get("file_path"),
                            page_idx=e.get("page_idx"),
                            quote=e.get("quote"),
                            doc_id=e.get("doc_id"),
                            chunk_id=e.get("chunk_id"),
                            image_path=e.get("image_path") or e.get("img_path"),
                        )
                    )

            if ctype == "parent_child":
                data = {
                    "parents": c.get("parents") or [],
                    "child": c.get("child") or {},
                }
            elif ctype == "spouse":
                data = {
                    "person1": c.get("person1") or {},
                    "person2": c.get("person2") or {},
                }
            elif ctype == "person_profile":
                # Force the profile to apply to the current subject person_id.
                data = {
                    "person_id": subject.get("person_id"),
                    "attributes": c.get("attributes") or {},
                }
            else:
                # Ignore unsupported types for now.
                continue

            out.append(
                Claim(
                    claim_type=ctype,
                    confidence=max(0.0, min(1.0, conf_f)),
                    data=data,
                    evidence=evidence_list,
                    notes=c.get("notes"),
                    raw=c,
                )
            )

        return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Genealogy expansion runner (Neo4j + RAGAnything)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_run = sub.add_parser("run", help="Seed + expand into Neo4j")
    sp_run.add_argument("--rag-working-dir", required=True, help="Existing LightRAG working_dir with indexed sources")
    sp_run.add_argument("--name", required=True, help="Seed person name")
    sp_run.add_argument("--birth-date", default=None)
    sp_run.add_argument("--birth-year", type=int, default=None)
    sp_run.add_argument("--birth-place", default=None)
    sp_run.add_argument("--death-date", default=None)
    sp_run.add_argument("--death-year", type=int, default=None)
    sp_run.add_argument("--death-place", default=None)
    sp_run.add_argument("--occupation", default=None)
    sp_run.add_argument("--gender", default=None)
    sp_run.add_argument("--max-depth", type=int, default=3)
    sp_run.add_argument("--max-tasks", type=int, default=80)
    sp_run.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    sp_run.add_argument("--no-spouses", action="store_true", help="Disable spouse/descendant expansion")
    sp_run.add_argument("--no-profiles", action="store_true", help="Disable profile extraction (birth/death/occupation/etc)")
    sp_run.add_argument("--debug", action="store_true")

    sp_run.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    sp_run.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    sp_run.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    sp_run.add_argument("--neo4j-db", default=os.getenv("NEO4J_DATABASE", None))

    sp_mock = sub.add_parser("smoke-mock", help="Offline smoke run (no Neo4j, no LLM)")
    sp_mock.add_argument("--max-depth", type=int, default=3)
    sp_mock.add_argument("--max-tasks", type=int, default=50)

    return p


async def cmd_smoke_mock(args: argparse.Namespace) -> int:
    store = InMemoryGenealogyStore()

    # Hardcoded toy graph:
    # Alice -> parents Bob+Carol; Bob+Carol children Alice+David; David spouse Emma; David+Emma child Frank.
    seed = PersonSpec(name="Alice Doe", birth_year=1980, birth_place="Springfield")

    class ToyExtractor(ClaimExtractor):
        async def extract(self, task_type: str, subject: Dict[str, Any]) -> List[Claim]:
            if task_type == "find_parents" and subject["person"].name == "Alice Doe":
                return [
                    Claim(
                        claim_type="parent_child",
                        confidence=0.9,
                        data={
                            "parents": [{"name": "Bob Doe"}, {"name": "Carol Smith"}],
                            "child": {"name": "Alice Doe", "birth_year": 1980},
                        },
                        evidence=[Evidence(file_path="toy.txt", quote="Alice, daughter of Bob and Carol")],
                    )
                ]

            if task_type == "find_children":
                parents = [p.name for p in (subject.get("parents") or [])]
                if set(parents) == {"Bob Doe", "Carol Smith"}:
                    return [
                        Claim(
                            claim_type="parent_child",
                            confidence=0.8,
                            data={
                                "parents": [{"name": "Bob Doe"}, {"name": "Carol Smith"}],
                                "child": {"name": "David Doe", "birth_year": 1982},
                            },
                            evidence=[Evidence(file_path="toy.txt", quote="Bob and Carol had a son David")],
                        )
                    ]
                if set(parents) == {"David Doe", "Emma Roe"}:
                    return [
                        Claim(
                            claim_type="parent_child",
                            confidence=0.85,
                            data={
                                "parents": [{"name": "David Doe"}, {"name": "Emma Roe"}],
                                "child": {"name": "Frank Doe", "birth_year": 2010},
                            },
                            evidence=[Evidence(file_path="toy.txt", quote="Frank, child of David and Emma")],
                        )
                    ]
            if task_type == "find_spouses" and subject["person"].name == "David Doe":
                return [
                    Claim(
                        claim_type="spouse",
                        confidence=0.7,
                        data={"person1": {"name": "David Doe"}, "person2": {"name": "Emma Roe"}},
                        evidence=[Evidence(file_path="toy.txt", quote="David married Emma")],
                    )
                ]
            if task_type == "find_profile" and subject["person"].name == "David Doe":
                return [
                    Claim(
                        claim_type="person_profile",
                        confidence=0.6,
                        data={
                            "person_id": subject.get("person_id"),
                            "attributes": {
                                "occupation": "Engineer",
                                "aliases": ["Dave Doe"],
                                "media": [
                                    {
                                        "kind": "photo",
                                        "path": "/tmp/david_doe_portrait.jpg",
                                        "caption": "Portrait photo",
                                    }
                                ],
                            },
                        },
                        evidence=[
                            Evidence(
                                file_path="toy.txt",
                                quote="David Doe (also known as Dave Doe) worked as an engineer.",
                                image_path="/tmp/david_doe_portrait.jpg",
                            )
                        ],
                    )
                ]
            return []

    pipeline = GenealogyPipeline(
        store=store,
        extractor=ToyExtractor(),
        config=GenealogyPipelineConfig(
            max_depth=args.max_depth,
            max_tasks=args.max_tasks,
            enable_spouse_search=True,
            enable_descendant_expansion=True,
        ),
    )
    seed_rec = pipeline.seed_person(seed)
    stats = await pipeline.expand(seed_rec.person_id)

    print("=== SMOKE MOCK SUMMARY ===")
    print(json.dumps(stats, indent=2))
    print(
        "People:",
        [
            {
                "name": p.spec.name,
                "occupation": p.spec.occupation,
                "birth_year": p.spec.birth_year,
                "aliases": p.spec.aliases,
            }
            for p in store.people.values()
        ],
    )
    print("Families:", len(store.families))
    if store.media:
        print("Media:", [{"kind": m.kind, "path": m.path} for m in store.media.values()])
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    store = Neo4jGenealogyStore(
        uri=args.neo4j_uri,
        username=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_db,
    )
    store.ensure_schema()

    rag = await _new_rag(args.rag_working_dir)
    try:
        init = await rag._ensure_lightrag_initialized()
        if isinstance(init, dict) and not init.get("success", True):
            raise SystemExit(init.get("error") or "Failed to initialize LightRAG")

        extractor = OllamaGenealogyExtractor(rag=rag, mode=args.mode, debug=args.debug)
        pipeline = GenealogyPipeline(
            store=store,
            extractor=extractor,
            config=GenealogyPipelineConfig(
                max_depth=args.max_depth,
                max_tasks=args.max_tasks,
                enable_spouse_search=not args.no_spouses,
                enable_descendant_expansion=not args.no_spouses,
                enable_profile_search=not args.no_profiles,
            ),
        )

        seed_spec = PersonSpec(
            name=args.name,
            birth_date=args.birth_date,
            birth_year=args.birth_year,
            birth_place=args.birth_place,
            death_date=args.death_date,
            death_year=args.death_year,
            death_place=args.death_place,
            occupation=args.occupation,
            gender=args.gender,
        )
        seed_rec = pipeline.seed_person(seed_spec)
        stats = await pipeline.expand(seed_rec.person_id)

        print("=== RUN SUMMARY ===")
        print(json.dumps({"seed_person_id": seed_rec.person_id, **stats}, indent=2))
        return 0

    finally:
        try:
            await rag.finalize_storages()
        except Exception:
            pass
        store.close()


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "smoke-mock":
        return asyncio.run(cmd_smoke_mock(args))
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
