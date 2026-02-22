#!/usr/bin/env python3
"""
Family tree pipeline:
1) Parse content_list into text pages
2) Named entity + kinship extraction (LLM)
3) Build a people/relations knowledge graph
4) Reform/filter it into a family tree and persist into Neo4j
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from lightrag.llm.openai import openai_complete_if_cache
from raganything.genealogy.json_utils import robust_json_loads
from raganything.genealogy.models import Evidence, PersonSpec
from raganything.genealogy.normalize import coerce_year, normalize_name
from raganything.genealogy.stores import Neo4jGenealogyStore


PERSON_NOISE_SUBSTRINGS = {
    "династия",
    "дом романовых",
    "романовы",
    "книга",
    "изд",
    "том",
    "страниц",
    "глава",
    "редактор",
    "издание",
    "история",
    "семья романовых",
}


@dataclass(frozen=True)
class PageItem:
    page_idx: int
    text: str


@dataclass(frozen=True)
class ChunkItem:
    chunk_id: int
    page_indices: List[int]
    text: str


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _to_ollama_options() -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    num_ctx = os.getenv("OLLAMA_NUM_CTX")
    if num_ctx:
        try:
            options["num_ctx"] = int(num_ctx)
        except ValueError:
            pass
    if not options:
        return {}
    return {"options": options}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return max(0.0, min(1.0, v))


def _resolve_content_list_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    candidates = sorted(input_path.rglob("*_content_list.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_content_list.json found under: {input_path}")
    return candidates[0]


def _load_pages(
    content_list_path: Path, max_pages: Optional[int] = None
) -> Tuple[List[PageItem], str]:
    obj = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"content_list must be a list, got: {type(obj)}")

    page_map: Dict[int, List[str]] = {}
    source_file = content_list_path.name

    for item in obj:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        pidx_raw = item.get("page_idx")
        try:
            page_idx = int(pidx_raw) if pidx_raw is not None else 0
        except Exception:
            page_idx = 0
        page_map.setdefault(page_idx, []).append(text)
        sf = item.get("source_file")
        if sf:
            source_file = str(sf)

    pages: List[PageItem] = []
    for p in sorted(page_map.keys()):
        merged = "\n".join(x for x in page_map[p] if x)
        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
        if merged:
            pages.append(PageItem(page_idx=p, text=merged))

    if max_pages is not None:
        pages = pages[: max(0, max_pages)]

    if not pages:
        raise RuntimeError(f"No text pages found in: {content_list_path}")
    return pages, source_file


def _write_stage1_artifacts(output_dir: Path, pages: Sequence[PageItem]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_jsonl = output_dir / "pages.jsonl"
    book_txt = output_dir / "book.txt"

    with pages_jsonl.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(
                json.dumps({"page_idx": p.page_idx, "text": p.text}, ensure_ascii=False)
            )
            f.write("\n")

    with book_txt.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"[PAGE {p.page_idx}]\n")
            f.write(p.text)
            f.write("\n\n")


def _chunk_pages(
    pages: Sequence[PageItem],
    pages_per_chunk: int,
    max_chars: int,
    max_chunks: Optional[int] = None,
) -> List[ChunkItem]:
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be > 0")
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    chunks: List[ChunkItem] = []
    cur_pages: List[int] = []
    cur_texts: List[str] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_pages, cur_texts, cur_len
        if not cur_pages:
            return
        chunk_id = len(chunks)
        text = "\n\n".join(cur_texts).strip()
        chunks.append(
            ChunkItem(chunk_id=chunk_id, page_indices=list(cur_pages), text=text)
        )
        cur_pages = []
        cur_texts = []
        cur_len = 0

    for p in pages:
        piece = f"[PAGE {p.page_idx}]\n{p.text.strip()}"
        if not piece.strip():
            continue
        projected_len = cur_len + len(piece) + 2
        if cur_pages and (
            len(cur_pages) >= pages_per_chunk or projected_len > max_chars
        ):
            flush()
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        cur_pages.append(p.page_idx)
        cur_texts.append(piece)
        cur_len += len(piece) + 2

    if max_chunks is None or len(chunks) < max_chunks:
        flush()

    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return chunks


def _extract_name(v: Any) -> Optional[str]:
    if isinstance(v, dict):
        for key in ("name", "full_name", "person", "value"):
            if key in v and str(v[key]).strip():
                return str(v[key]).strip()
        return None
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _normalize_name_surface(name: str) -> str:
    s = re.sub(r"\s+", " ", name).strip(" \n\t,.;:!?()[]{}\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def _is_valid_person_name(name: str) -> bool:
    s = _normalize_name_surface(name)
    if len(s) < 2 or len(s) > 80:
        return False
    if "\n" in s:
        return False
    if not re.search(r"[A-Za-zА-Яа-яЁё]", s):
        return False
    if len(s.split()) > 8:
        return False
    low = s.lower()
    for noise in PERSON_NOISE_SUBSTRINGS:
        if noise in low:
            return False
    # Avoid overly generic all-lowercase fragments.
    if not re.search(r"[A-ZА-ЯЁ]", s):
        return False
    return True


def _extract_evidence(
    rel: Dict[str, Any], fallback_page: Optional[int]
) -> Dict[str, Any]:
    quote: Optional[str] = None
    page_idx: Optional[int] = None
    evidence_obj = rel.get("evidence")
    if isinstance(evidence_obj, dict):
        quote = str(evidence_obj.get("quote") or "").strip() or None
        page_idx = evidence_obj.get("page_idx")
    elif isinstance(evidence_obj, list) and evidence_obj:
        first = evidence_obj[0]
        if isinstance(first, dict):
            quote = str(first.get("quote") or "").strip() or None
            page_idx = first.get("page_idx")
    if quote is None:
        quote = str(rel.get("quote") or "").strip() or None
    if page_idx is None:
        page_idx = rel.get("page_idx")
    try:
        page_val = int(page_idx) if page_idx is not None else fallback_page
    except Exception:
        page_val = fallback_page
    if quote and len(quote) > 320:
        quote = quote[:320]
    return {"quote": quote, "page_idx": page_val}


def _canonical_relation_type(v: Any) -> Optional[str]:
    s = str(v or "").strip().lower()
    if not s:
        return None
    if s in {"parent_child", "parent-child", "parentchild", "child_of"}:
        return "parent_child"
    if s in {"spouse", "spouses", "married", "husband_wife", "wife_husband"}:
        return "spouse"
    if "parent" in s or "родител" in s or "father" in s or "mother" in s:
        return "parent_child"
    if "spouse" in s or "marri" in s or "супруг" in s or "жена" in s or "муж" in s:
        return "spouse"
    return None


def _parse_parent_child_relation(
    rel: Dict[str, Any],
) -> Tuple[List[str], Optional[str]]:
    parents: List[str] = []
    child: Optional[str] = None

    if isinstance(rel.get("parents"), list):
        for p in rel["parents"]:
            name = _extract_name(p)
            if name:
                parents.append(name)

    for key in ("parent", "father", "mother"):
        name = _extract_name(rel.get(key))
        if name:
            parents.append(name)

    other_parent = _extract_name(rel.get("other_parent"))
    if other_parent:
        parents.append(other_parent)

    child = _extract_name(rel.get("child"))
    if not child:
        child = (
            _extract_name(rel.get("person2"))
            if str(rel.get("direction", "")).lower() == "parent_to_child"
            else child
        )

    deduped: List[str] = []
    seen = set()
    for p in parents:
        n = _normalize_name_surface(p)
        key = normalize_name(n)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(n)
    child_norm = _normalize_name_surface(child) if child else None
    return deduped, child_norm


def _parse_spouse_relation(rel: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    p1 = (
        _extract_name(rel.get("person1"))
        or _extract_name(rel.get("spouse1"))
        or _extract_name(rel.get("a"))
    )
    p2 = (
        _extract_name(rel.get("person2"))
        or _extract_name(rel.get("spouse2"))
        or _extract_name(rel.get("b"))
    )
    if not p1 and isinstance(rel.get("people"), list):
        people = rel.get("people") or []
        if len(people) >= 2:
            p1 = _extract_name(people[0])
            p2 = _extract_name(people[1])
    return (
        _normalize_name_surface(p1) if p1 else None,
        _normalize_name_surface(p2) if p2 else None,
    )


def _merge_person(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(existing)
    if len(str(incoming.get("name") or "")) > len(str(out.get("name") or "")):
        out["name"] = incoming.get("name")
    for key in (
        "birth_date",
        "death_date",
        "birth_place",
        "death_place",
        "gender",
        "occupation",
        "biography",
    ):
        if not out.get(key) and incoming.get(key):
            out[key] = incoming[key]
    if not out.get("birth_year"):
        out["birth_year"] = incoming.get("birth_year")
    if not out.get("death_year"):
        out["death_year"] = incoming.get("death_year")
    out["confidence"] = max(
        float(out.get("confidence") or 0.0), float(incoming.get("confidence") or 0.0)
    )
    aliases = set(out.get("aliases") or [])
    aliases.update(incoming.get("aliases") or [])
    aliases.discard(out.get("name"))
    out["aliases"] = sorted(a for a in aliases if a)
    return out


def _build_extraction_prompt(chunk: ChunkItem) -> str:
    return f"""
Извлеки факты генеалогии из текста.

Требования:
- Верни ТОЛЬКО валидный JSON.
- Извлекай только людей и семейные отношения.
- Если нет уверенных фактов, верни пустые списки.
- Не выдумывай.

JSON schema:
{{
  "people": [
    {{
      "name": "string",
      "aliases": ["string"],
      "birth_date": null,
      "death_date": null,
      "birth_year": null,
      "death_year": null,
      "birth_place": null,
      "death_place": null,
      "gender": null,
      "occupation": null,
      "biography": null,
      "confidence": 0.0
    }}
  ],
  "relations": [
    {{
      "type": "parent_child",
      "parents": [{{"name":"..."}}],
      "child": {{"name":"..."}},
      "confidence": 0.0,
      "evidence": {{"quote":"...", "page_idx": 0}}
    }},
    {{
      "type": "spouse",
      "person1": {{"name":"..."}},
      "person2": {{"name":"..."}},
      "confidence": 0.0,
      "evidence": {{"quote":"...", "page_idx": 0}}
    }}
  ]
}}

Текст чанка (страницы {chunk.page_indices}):
{chunk.text}
""".strip()


async def _llm_complete(
    prompt: str,
    system_prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
) -> str:
    return await openai_complete_if_cache(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body=_to_ollama_options() or None,
    )


async def _repair_json(
    raw_output: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    repair_prompt = f"""
Normalize the following model output into STRICT JSON with this exact top-level schema:
{{
  "people": [],
  "relations": []
}}

Rules:
- Output JSON only.
- Keep only person entities and family relations.
- Supported relation types: "parent_child", "spouse".
- If no useful data is present, output {{"people":[],"relations":[]}}.

Raw output:
{raw_output}
""".strip()
    repaired = await _llm_complete(
        repair_prompt,
        "You are a strict JSON normalizer. Output valid JSON only.",
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    return robust_json_loads(repaired)


async def _extract_chunk(
    chunk: ChunkItem,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
) -> Dict[str, Any]:
    raw = await _llm_complete(
        _build_extraction_prompt(chunk),
        (
            "Ты аккуратный историк-генеалог. "
            "Извлекай только достоверные упоминания людей и семейных связей. "
            "Верни только JSON."
        ),
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    obj = robust_json_loads(raw)
    repaired = False
    if not obj:
        repaired = True
        obj = await _repair_json(
            raw,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    if not obj:
        obj = {"people": [], "relations": []}
    return {"json": obj, "raw_preview": (raw or "")[:2000], "repaired": repaired}


def _parse_extracted_chunk(chunk: ChunkItem, payload: Dict[str, Any]) -> Dict[str, Any]:
    obj = payload.get("json") or {}
    people_raw = obj.get("people") if isinstance(obj, dict) else []
    relations_raw = obj.get("relations") if isinstance(obj, dict) else []

    parsed_people: List[Dict[str, Any]] = []
    if isinstance(people_raw, list):
        for p in people_raw:
            if not isinstance(p, dict):
                continue
            name = _extract_name(p.get("name") or p.get("person"))
            if not name:
                continue
            name = _normalize_name_surface(name)
            if not _is_valid_person_name(name):
                continue
            aliases_val = p.get("aliases") or []
            aliases: List[str]
            if isinstance(aliases_val, list):
                aliases = [
                    _normalize_name_surface(str(a))
                    for a in aliases_val
                    if str(a).strip()
                ]
            elif isinstance(aliases_val, str):
                aliases = [
                    _normalize_name_surface(x)
                    for x in aliases_val.split(",")
                    if x.strip()
                ]
            else:
                aliases = []
            parsed_people.append(
                {
                    "name": name,
                    "aliases": sorted(
                        set(a for a in aliases if _is_valid_person_name(a))
                    ),
                    "birth_date": p.get("birth_date"),
                    "death_date": p.get("death_date"),
                    "birth_year": coerce_year(
                        p.get("birth_year") or p.get("birth_date")
                    ),
                    "death_year": coerce_year(
                        p.get("death_year") or p.get("death_date")
                    ),
                    "birth_place": p.get("birth_place"),
                    "death_place": p.get("death_place"),
                    "gender": p.get("gender"),
                    "occupation": p.get("occupation"),
                    "biography": p.get("biography"),
                    "confidence": _safe_float(p.get("confidence"), 0.5),
                }
            )

    parsed_relations: List[Dict[str, Any]] = []
    if isinstance(relations_raw, list):
        for rel in relations_raw:
            if not isinstance(rel, dict):
                continue
            rtype = _canonical_relation_type(
                rel.get("type") or rel.get("relation_type")
            )
            if not rtype:
                continue
            confidence = _safe_float(rel.get("confidence"), 0.5)
            fallback_page = chunk.page_indices[0] if chunk.page_indices else None
            ev = _extract_evidence(rel, fallback_page)
            if rtype == "parent_child":
                parents, child = _parse_parent_child_relation(rel)
                parents = [p for p in parents if _is_valid_person_name(p)]
                if child:
                    child = _normalize_name_surface(child)
                if not child or not _is_valid_person_name(child) or not parents:
                    continue
                parsed_relations.append(
                    {
                        "type": "parent_child",
                        "parents": parents[:2],
                        "child": child,
                        "confidence": confidence,
                        "evidence": ev,
                    }
                )
            elif rtype == "spouse":
                p1, p2 = _parse_spouse_relation(rel)
                if not p1 or not p2:
                    continue
                if not _is_valid_person_name(p1) or not _is_valid_person_name(p2):
                    continue
                if normalize_name(p1) == normalize_name(p2):
                    continue
                parsed_relations.append(
                    {
                        "type": "spouse",
                        "person1": p1,
                        "person2": p2,
                        "confidence": confidence,
                        "evidence": ev,
                    }
                )

    return {
        "chunk_id": chunk.chunk_id,
        "pages": chunk.page_indices,
        "repaired": bool(payload.get("repaired")),
        "raw_preview": payload.get("raw_preview"),
        "people": parsed_people,
        "relations": parsed_relations,
    }


def _append_evidence(dst: List[Dict[str, Any]], ev: Dict[str, Any]) -> None:
    key = (ev.get("quote"), ev.get("page_idx"))
    existing = {(x.get("quote"), x.get("page_idx")) for x in dst}
    if key not in existing:
        dst.append({"quote": ev.get("quote"), "page_idx": ev.get("page_idx")})


def _build_knowledge_graph(
    parsed_chunks: Sequence[Dict[str, Any]],
    *,
    min_relation_confidence: float,
) -> Dict[str, Any]:
    people: Dict[str, Dict[str, Any]] = {}
    spouse_rel: Dict[Tuple[str, str], Dict[str, Any]] = {}
    parent_child_rel: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def upsert_person(person: Dict[str, Any]) -> Optional[str]:
        name = _normalize_name_surface(str(person.get("name") or ""))
        if not name or not _is_valid_person_name(name):
            return None
        key = normalize_name(name)
        if not key:
            return None
        record = {
            "key": key,
            "name": name,
            "aliases": person.get("aliases") or [],
            "birth_date": person.get("birth_date"),
            "death_date": person.get("death_date"),
            "birth_year": coerce_year(
                person.get("birth_year") or person.get("birth_date")
            ),
            "death_year": coerce_year(
                person.get("death_year") or person.get("death_date")
            ),
            "birth_place": person.get("birth_place"),
            "death_place": person.get("death_place"),
            "gender": person.get("gender"),
            "occupation": person.get("occupation"),
            "biography": person.get("biography"),
            "confidence": _safe_float(person.get("confidence"), 0.5),
        }
        if key in people:
            people[key] = _merge_person(people[key], record)
        else:
            people[key] = record
        return key

    for chunk in parsed_chunks:
        for p in chunk.get("people") or []:
            upsert_person(p)

        for rel in chunk.get("relations") or []:
            conf = _safe_float(rel.get("confidence"), 0.0)
            if conf < min_relation_confidence:
                continue
            if rel.get("type") == "spouse":
                p1_key = upsert_person({"name": rel.get("person1"), "confidence": conf})
                p2_key = upsert_person({"name": rel.get("person2"), "confidence": conf})
                if not p1_key or not p2_key or p1_key == p2_key:
                    continue
                a, b = sorted([p1_key, p2_key])
                edge = spouse_rel.setdefault(
                    (a, b),
                    {
                        "person1_key": a,
                        "person2_key": b,
                        "confidence": conf,
                        "evidence": [],
                    },
                )
                edge["confidence"] = max(edge["confidence"], conf)
                _append_evidence(edge["evidence"], rel.get("evidence") or {})
            elif rel.get("type") == "parent_child":
                child_key = upsert_person(
                    {"name": rel.get("child"), "confidence": conf}
                )
                parents = rel.get("parents") or []
                parent_keys: List[str] = []
                for p_name in parents[:2]:
                    pk = upsert_person({"name": p_name, "confidence": conf})
                    if pk:
                        parent_keys.append(pk)
                if not child_key or not parent_keys:
                    continue
                for idx, pk in enumerate(parent_keys):
                    co_parent = None
                    if len(parent_keys) == 2:
                        co_parent = parent_keys[1 - idx]
                    edge_key = (pk, child_key)
                    edge = parent_child_rel.setdefault(
                        edge_key,
                        {
                            "parent_key": pk,
                            "child_key": child_key,
                            "co_parent_key": co_parent,
                            "confidence": conf,
                            "evidence": [],
                        },
                    )
                    if not edge.get("co_parent_key") and co_parent:
                        edge["co_parent_key"] = co_parent
                    edge["confidence"] = max(edge["confidence"], conf)
                    _append_evidence(edge["evidence"], rel.get("evidence") or {})

    connected = set()
    for a, b in spouse_rel:
        connected.add(a)
        connected.add(b)
    for p, c in parent_child_rel:
        connected.add(p)
        connected.add(c)

    filtered_people = {k: v for k, v in people.items() if k in connected}
    filtered_spouse = {
        k: v
        for k, v in spouse_rel.items()
        if k[0] in filtered_people and k[1] in filtered_people
    }
    filtered_parent_child = {
        k: v
        for k, v in parent_child_rel.items()
        if k[0] in filtered_people and k[1] in filtered_people
    }

    return {
        "people": sorted(filtered_people.values(), key=lambda x: x["name"]),
        "relations": {
            "spouse": sorted(
                filtered_spouse.values(),
                key=lambda x: (x["person1_key"], x["person2_key"]),
            ),
            "parent_child": sorted(
                filtered_parent_child.values(),
                key=lambda x: (x["parent_key"], x["child_key"]),
            ),
        },
    }


def _build_family_tree(kg: Dict[str, Any]) -> Dict[str, Any]:
    people = kg.get("people") or []
    spouse_rel = kg.get("relations", {}).get("spouse") or []
    parent_child_rel = kg.get("relations", {}).get("parent_child") or []

    family_children: Dict[Tuple[str, ...], set] = {}
    for rel in parent_child_rel:
        p1 = rel.get("parent_key")
        p2 = rel.get("co_parent_key")
        child = rel.get("child_key")
        if not p1 or not child:
            continue
        parents = tuple(sorted([x for x in [p1, p2] if x]))
        family_children.setdefault(parents, set()).add(child)

    for rel in spouse_rel:
        parents = tuple(sorted([rel.get("person1_key"), rel.get("person2_key")]))
        if parents and all(parents):
            family_children.setdefault(parents, set())

    families = []
    for parent_keys, children in sorted(family_children.items(), key=lambda x: x[0]):
        families.append(
            {
                "parent_keys": list(parent_keys),
                "children_keys": sorted(children),
            }
        )

    return {
        "persons": people,
        "relations": {
            "spouse": spouse_rel,
            "parent_child": parent_child_rel,
        },
        "families": families,
    }


def _reset_neo4j(
    uri: str, username: str, password: str, database: Optional[str]
) -> None:
    from neo4j import GraphDatabase  # type: ignore

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as s:
            s.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()


def _persist_family_tree(
    tree: Dict[str, Any],
    *,
    source_file: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: Optional[str],
    reset_db: bool,
) -> Dict[str, Any]:
    if reset_db:
        _reset_neo4j(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)

    store = Neo4jGenealogyStore(
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )
    store.ensure_schema()

    person_id_map: Dict[str, str] = {}
    try:
        for p in tree.get("persons") or []:
            spec = PersonSpec(
                name=p.get("name"),
                birth_date=p.get("birth_date"),
                death_date=p.get("death_date"),
                birth_year=coerce_year(p.get("birth_year") or p.get("birth_date")),
                death_year=coerce_year(p.get("death_year") or p.get("death_date")),
                birth_place=p.get("birth_place"),
                death_place=p.get("death_place"),
                gender=p.get("gender"),
                occupation=p.get("occupation"),
                biography=p.get("biography"),
                aliases=list(p.get("aliases") or []),
            )
            rec = store.upsert_person(spec)
            person_id_map[p["key"]] = rec.person_id

        spouse_written = 0
        parent_child_written = 0
        evidence_written = 0
        single_parent_family: Dict[str, str] = {}

        for rel in tree.get("relations", {}).get("spouse", []):
            k1 = rel.get("person1_key")
            k2 = rel.get("person2_key")
            if k1 not in person_id_map or k2 not in person_id_map:
                continue
            id1 = person_id_map[k1]
            id2 = person_id_map[k2]
            fam = store.upsert_family([id1, id2], family_type="couple")
            store.link_spouses(
                fam.family_id, id1, id2, props={"confidence": rel.get("confidence")}
            )
            claim_id = store.create_claim(
                "spouse",
                float(rel.get("confidence") or 0.0),
                {
                    "person1": {"name": k1},
                    "person2": {"name": k2},
                },
            )
            store.link_claim_to_person(claim_id, id1, role="spouse")
            store.link_claim_to_person(claim_id, id2, role="spouse")
            store.link_claim_to_family(claim_id, fam.family_id, role="family")
            for ev in (rel.get("evidence") or [])[:3]:
                evidence_written += 1
                store.attach_evidence(
                    claim_id,
                    Evidence(
                        file_path=source_file,
                        page_idx=ev.get("page_idx"),
                        quote=ev.get("quote"),
                    ),
                )
            spouse_written += 1

        for rel in tree.get("relations", {}).get("parent_child", []):
            pk = rel.get("parent_key")
            ck = rel.get("child_key")
            opk = rel.get("co_parent_key")
            if pk not in person_id_map or ck not in person_id_map:
                continue
            parent_ids = [person_id_map[pk]]
            if opk and opk in person_id_map:
                parent_ids.append(person_id_map[opk])
            child_id = person_id_map[ck]

            family_id: Optional[str] = None
            if len(parent_ids) >= 2:
                fam = store.upsert_family(parent_ids[:2], family_type="parents")
                family_id = fam.family_id
                store.link_parents_to_family(fam.family_id, parent_ids[:2])
                store.link_child_to_family(
                    fam.family_id, child_id, props={"confidence": rel.get("confidence")}
                )
            else:
                pid = parent_ids[0]
                if pid not in single_parent_family:
                    fam = store.upsert_family([pid], family_type="single_parent")
                    single_parent_family[pid] = fam.family_id
                    store.link_parents_to_family(fam.family_id, [pid])
                family_id = single_parent_family[pid]
                store.link_child_to_family(
                    family_id, child_id, props={"confidence": rel.get("confidence")}
                )

            claim_data = {
                "parents": [{"name": pk}] + ([{"name": opk}] if opk else []),
                "child": {"name": ck},
            }
            claim_id = store.create_claim(
                "parent_child",
                float(rel.get("confidence") or 0.0),
                claim_data,
            )
            store.link_claim_to_person(claim_id, person_id_map[pk], role="parent")
            if opk and opk in person_id_map:
                store.link_claim_to_person(claim_id, person_id_map[opk], role="parent")
            store.link_claim_to_person(claim_id, child_id, role="child")
            if family_id:
                store.link_claim_to_family(claim_id, family_id, role="family")
            for ev in (rel.get("evidence") or [])[:3]:
                evidence_written += 1
                store.attach_evidence(
                    claim_id,
                    Evidence(
                        file_path=source_file,
                        page_idx=ev.get("page_idx"),
                        quote=ev.get("quote"),
                    ),
                )
            parent_child_written += 1

        return {
            "people_written": len(person_id_map),
            "spouse_claims_written": spouse_written,
            "parent_child_claims_written": parent_child_written,
            "evidences_written": evidence_written,
        }
    finally:
        store.close()


def _serialize_for_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _serialize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize_for_json(v) for v in data]
    if isinstance(data, set):
        return sorted(_serialize_for_json(v) for v in data)
    return data


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Family-tree extraction pipeline from content_list"
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to content_list JSON file or a directory containing *_content_list.json",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for pipeline artifacts",
    )
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--pages-per-chunk", type=int, default=4)
    p.add_argument("--max-chars-per-chunk", type=int, default=7000)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--min-relation-confidence", type=float, default=0.35)

    p.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen2.5:32b"))
    p.add_argument(
        "--llm-base-url",
        default=os.getenv("LLM_BINDING_HOST", "http://localhost:11434/v1"),
    )
    p.add_argument("--llm-api-key", default=os.getenv("LLM_BINDING_API_KEY", "ollama"))
    p.add_argument("--llm-timeout", type=int, default=300)
    p.add_argument("--llm-max-tokens", type=int, default=1600)

    p.add_argument("--dry-run", action="store_true", help="Skip writing to Neo4j")
    p.add_argument(
        "--reset-neo4j", action="store_true", help="Clear Neo4j before writing results"
    )
    p.add_argument(
        "--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    p.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--neo4j-db", default=os.getenv("NEO4J_DATABASE", None))
    return p


async def _run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    content_list_path = _resolve_content_list_path(input_path)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"./output_family_tree/run_{_now_ts()}")
    )
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] parse_to_txt: {content_list_path}")
    pages, source_file = _load_pages(content_list_path, max_pages=args.max_pages)
    _write_stage1_artifacts(out_dir, pages)
    print(f"  pages: {len(pages)}  source_file: {source_file}")

    print("[2/4] ner_and_relation_extraction")
    chunks = _chunk_pages(
        pages,
        pages_per_chunk=args.pages_per_chunk,
        max_chars=args.max_chars_per_chunk,
        max_chunks=args.max_chunks,
    )
    print(f"  chunks: {len(chunks)}")
    parsed_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"  - chunk {idx}/{len(chunks)} pages={chunk.page_indices}")
        raw_chunk = await _extract_chunk(
            chunk,
            model=args.model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
        )
        parsed = _parse_extracted_chunk(chunk, raw_chunk)
        parsed_chunks.append(parsed)
        print(
            f"    people={len(parsed['people'])} relations={len(parsed['relations'])} repaired={parsed['repaired']}"
        )

    (out_dir / "raw_extractions.json").write_text(
        json.dumps(_serialize_for_json(parsed_chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[3/4] knowledge_graph_build")
    kg = _build_knowledge_graph(
        parsed_chunks,
        min_relation_confidence=float(args.min_relation_confidence),
    )
    (out_dir / "knowledge_graph.json").write_text(
        json.dumps(_serialize_for_json(kg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"  people={len(kg.get('people') or [])} "
        f"parent_child={len(kg.get('relations', {}).get('parent_child') or [])} "
        f"spouse={len(kg.get('relations', {}).get('spouse') or [])}"
    )

    print("[4/4] family_tree_filter")
    tree = _build_family_tree(kg)
    (out_dir / "family_tree.json").write_text(
        json.dumps(_serialize_for_json(tree), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"  family_nodes={len(tree.get('persons') or [])} "
        f"families={len(tree.get('families') or [])}"
    )

    neo4j_stats: Dict[str, Any] = {"skipped": True}
    if not args.dry_run:
        print("  writing_to_neo4j...")
        neo4j_stats = _persist_family_tree(
            tree,
            source_file=source_file,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_db,
            reset_db=bool(args.reset_neo4j),
        )
        print(f"  neo4j: {neo4j_stats}")

    summary = {
        "input_content_list": str(content_list_path),
        "output_dir": str(out_dir),
        "pages": len(pages),
        "chunks": len(chunks),
        "parsed_people_total": sum(len(c.get("people") or []) for c in parsed_chunks),
        "parsed_relations_total": sum(
            len(c.get("relations") or []) for c in parsed_chunks
        ),
        "kg_people": len(kg.get("people") or []),
        "kg_parent_child": len(kg.get("relations", {}).get("parent_child") or []),
        "kg_spouse": len(kg.get("relations", {}).get("spouse") or []),
        "tree_people": len(tree.get("persons") or []),
        "tree_families": len(tree.get("families") or []),
        "neo4j": neo4j_stats,
        "model": args.model,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(_serialize_for_json(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
