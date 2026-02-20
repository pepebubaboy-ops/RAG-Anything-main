#!/usr/bin/env python3
"""
General living-entities graph pipeline:
1) parse content_list into text pages/chunks
2) extract mentions + relations with LLM (NER + relation extraction)
3) resolve mentions into canonical entities (merge / possible_same / split)
4) build and persist a common relationship graph to Neo4j
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from raganything.genealogy.json_utils import robust_json_loads
from raganything.genealogy.normalize import coerce_year, normalize_name


NOISE_NAME_SUBSTRINGS = {
    "страница",
    "глава",
    "оглавление",
    "издательство",
    "isbn",
    "литература",
    "примечание",
    "автор",
    "рисунок",
    "таблица",
    "copyright",
    "москва",
    "санкт-петербург",
}

ENTITY_TYPE_ALIASES = {
    "human": {"human", "person", "человек", "персона", "личность"},
    "animal": {"animal", "pet", "животное", "питомец", "зверь"},
    "group": {"group", "family", "organization", "dynasty", "семья", "группа", "династия", "род"},
    "other_living": {"other_living", "living", "organism", "существо", "организм"},
}

RELATION_TYPE_SYNONYMS = {
    "parent_child": {"parent_child", "child_of", "parent-of", "родитель", "родители", "отец", "мать", "сын", "дочь"},
    "spouse": {"spouse", "married", "husband_wife", "wife_husband", "супруг", "жена", "муж", "брак"},
    "friend": {"friend", "друг", "друзья", "friend_of"},
    "enemy": {"enemy", "foe", "враг", "противник", "enemy_of"},
    "partner": {"partner", "партнер", "союзник", "пара"},
    "owner_pet": {"owner_pet", "owner_of_pet", "хозяин", "владелец", "питомец"},
    "ally": {"ally", "ally_of", "союзник", "альянс"},
    "rival": {"rival", "rival_of", "соперник", "конкурент"},
    "mentor": {"mentor", "наставник", "учитель", "mentor_of"},
    "student": {"student", "ученик", "student_of"},
    "sibling": {"sibling", "brother_sister", "брат", "сестра"},
    "relative": {"relative", "родственник", "родня"},
    "works_with": {"works_with", "colleague", "коллега", "сотрудничает"},
    "knows": {"knows", "знает", "знаком"},
    "member_of": {"member_of", "член", "состоит_в"},
    "leader_of": {"leader_of", "лидер", "глава", "правитель"},
    "associated_with": {"associated_with", "related_to", "связан", "отношение"},
}

RELATION_DEFAULT_POLARITY = {
    "friend": "positive",
    "ally": "positive",
    "partner": "positive",
    "spouse": "positive",
    "parent_child": "neutral",
    "owner_pet": "neutral",
    "mentor": "positive",
    "student": "positive",
    "sibling": "neutral",
    "relative": "neutral",
    "works_with": "neutral",
    "knows": "neutral",
    "member_of": "neutral",
    "leader_of": "neutral",
    "enemy": "negative",
    "rival": "negative",
    "associated_with": "neutral",
}

SYMMETRIC_RELATIONS = {
    "friend",
    "enemy",
    "partner",
    "spouse",
    "ally",
    "rival",
    "sibling",
    "relative",
    "works_with",
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


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def _safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v or "").strip().lower()
    if s in {"1", "true", "yes", "y", "да"}:
        return True
    if s in {"0", "false", "no", "n", "нет"}:
        return False
    return default


def _resolve_content_list_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    candidates = sorted(input_path.rglob("*_content_list.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_content_list.json found under: {input_path}")
    return candidates[0]


def _load_pages(content_list_path: Path, max_pages: Optional[int] = None) -> Tuple[List[PageItem], str]:
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
        raw_page = item.get("page_idx")
        try:
            page_idx = int(raw_page) if raw_page is not None else 0
        except Exception:
            page_idx = 0
        page_map.setdefault(page_idx, []).append(text)
        sf = item.get("source_file")
        if sf:
            source_file = str(sf)

    pages: List[PageItem] = []
    for p in sorted(page_map.keys()):
        merged = "\n".join(page_map[p]).strip()
        if merged:
            pages.append(PageItem(page_idx=p, text=merged))
    if max_pages is not None:
        pages = pages[: max(0, max_pages)]
    if not pages:
        raise RuntimeError(f"No text pages found in: {content_list_path}")
    return pages, source_file


def _write_stage1_artifacts(output_dir: Path, pages: Sequence[PageItem]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps({"page_idx": p.page_idx, "text": p.text}, ensure_ascii=False) + "\n")
    with (output_dir / "book.txt").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"[PAGE {p.page_idx}]\n{p.text}\n\n")


def _chunk_pages(
    pages: Sequence[PageItem],
    pages_per_chunk: int,
    max_chars: int,
    max_chunks: Optional[int] = None,
    page_overlap: int = 0,
) -> List[ChunkItem]:
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be > 0")
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    overlap = max(0, min(int(page_overlap), pages_per_chunk - 1))
    step = max(1, pages_per_chunk - overlap)

    chunks: List[ChunkItem] = []
    i = 0
    total_pages = len(pages)

    while i < total_pages:
        selected: List[PageItem] = []
        cur_len = 0
        j = i
        while j < total_pages and len(selected) < pages_per_chunk:
            p = pages[j]
            piece = f"[PAGE {p.page_idx}]\n{p.text}"
            projected = cur_len + len(piece) + 2
            if selected and projected > max_chars:
                break
            selected.append(p)
            cur_len = projected
            j += 1

        # Always keep at least one page in a chunk.
        if not selected:
            selected = [pages[i]]

        chunks.append(
            ChunkItem(
                chunk_id=len(chunks),
                page_indices=[p.page_idx for p in selected],
                text="\n\n".join(f"[PAGE {p.page_idx}]\n{p.text}" for p in selected).strip(),
            )
        )
        if max_chunks is not None and len(chunks) >= max_chunks:
            break

        i += step

    return chunks[:max_chunks] if max_chunks is not None else chunks


def _normalize_surface_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name or "").strip())
    s = s.strip(" \n\t,.;:!?()[]{}\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def _line_signature(line: str) -> str:
    s = _normalize_surface_name(line).lower()
    if not s:
        return ""
    # Make signatures robust to changing page numbers.
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_boundary_line_candidate(line: str) -> bool:
    s = _normalize_surface_name(line)
    if not s:
        return False
    if len(s) < 2 or len(s) > 120:
        return False
    if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", s):
        return True
    if re.fullmatch(r"[\W_]+", s):
        return False
    return bool(re.search(r"[A-Za-zА-Яа-яЁё]", s))


def _detect_repeated_boundary_signatures(
    pages: Sequence[PageItem],
    *,
    min_occurrences: int,
    min_share: float,
) -> set[str]:
    total_pages = len(pages)
    if total_pages <= 1:
        return set()

    counts: Dict[str, int] = {}
    for p in pages:
        lines = [ln.strip() for ln in str(p.text or "").replace("\r", "\n").split("\n") if ln.strip()]
        if not lines:
            continue
        boundary_lines = [lines[0]]
        if len(lines) > 1:
            boundary_lines.append(lines[-1])
        for line in boundary_lines:
            if not _is_boundary_line_candidate(line):
                continue
            sig = _line_signature(line)
            if not sig:
                continue
            counts[sig] = counts.get(sig, 0) + 1

    if not counts:
        return set()

    share = max(0.0, min(1.0, float(min_share)))
    threshold = max(int(min_occurrences), int(total_pages * share + 0.9999))
    return {sig for sig, cnt in counts.items() if cnt >= threshold}


def _remove_repeated_boundary_lines(
    text: str,
    repeated_signatures: set[str],
) -> Tuple[str, bool, bool]:
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    removed_first = False
    removed_last = False

    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        sig = _line_signature(raw)
        if sig and sig in repeated_signatures:
            del lines[i]
            removed_first = True
        break

    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].strip():
            continue
        sig = _line_signature(lines[i])
        if sig and sig in repeated_signatures:
            del lines[i]
            removed_last = True
        break

    return "\n".join(lines), removed_first, removed_last


def _normalize_text_for_llm(text: str) -> str:
    s = str(text or "")
    s = s.replace("\u00ad", "")  # soft hyphen from OCR/PDF extraction
    s = s.replace("\xa0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Join words broken by line-wrap hyphenation.
    s = re.sub(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", s)
    s = re.sub(r"[ \t\f\v]+\n", "\n", s)
    s = re.sub(r"\n[ \t\f\v]+", "\n", s)

    # Convert soft newlines inside sentences to spaces; keep paragraph breaks.
    s = re.sub(r"(?<![.!?…:;])\n(?!\n|[-*•]\s)", " ", s)

    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


def _prepare_pages_for_llm(
    pages: Sequence[PageItem],
    *,
    strip_repeated_page_lines: bool,
    repeated_line_min_occurrences: int,
    repeated_line_min_share: float,
) -> Tuple[List[PageItem], Dict[str, Any]]:
    repeated_signatures: set[str] = set()
    if strip_repeated_page_lines:
        repeated_signatures = _detect_repeated_boundary_signatures(
            pages,
            min_occurrences=max(1, int(repeated_line_min_occurrences)),
            min_share=float(repeated_line_min_share),
        )

    prepared: List[PageItem] = []
    changed_pages = 0
    removed_first_total = 0
    removed_last_total = 0
    chars_before = 0
    chars_after = 0

    for p in pages:
        original = str(p.text or "")
        chars_before += len(original)
        working = original

        removed_first = False
        removed_last = False
        if repeated_signatures:
            working, removed_first, removed_last = _remove_repeated_boundary_lines(
                working,
                repeated_signatures,
            )

        normalized = _normalize_text_for_llm(working)
        if not normalized:
            # Preserve non-empty input pages with a light fallback cleanup.
            normalized = _normalize_surface_name(working)

        if normalized != original:
            changed_pages += 1
        if removed_first:
            removed_first_total += 1
        if removed_last:
            removed_last_total += 1
        chars_after += len(normalized)
        if normalized:
            prepared.append(PageItem(page_idx=p.page_idx, text=normalized))

    stats = {
        "enabled": True,
        "pages_in": len(pages),
        "pages_out": len(prepared),
        "changed_pages": changed_pages,
        "chars_before": chars_before,
        "chars_after": chars_after,
        "removed_first_lines": removed_first_total,
        "removed_last_lines": removed_last_total,
        "repeated_boundary_signatures": len(repeated_signatures),
    }
    return prepared, stats


def _extract_name(x: Any) -> Optional[str]:
    if isinstance(x, dict):
        for k in ("name", "full_name", "value", "entity"):
            if k in x and str(x[k]).strip():
                return str(x[k]).strip()
        return None
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _is_valid_living_name(name: str) -> bool:
    s = _normalize_surface_name(name)
    if not s or len(s) < 2 or len(s) > 100:
        return False
    if len(s.split()) > 10:
        return False
    if not re.search(r"[A-Za-zА-Яа-яЁё]", s):
        return False
    low = s.lower()
    for noise in NOISE_NAME_SUBSTRINGS:
        if noise in low:
            return False
    if re.match(r"^\d+$", s):
        return False
    return True


def _canonical_entity_type(raw: Any, name: Optional[str], species: Optional[str]) -> str:
    s = str(raw or "").strip().lower()
    for canon, aliases in ENTITY_TYPE_ALIASES.items():
        if s in aliases:
            return canon
    if species:
        sp = str(species).lower()
        if any(k in sp for k in ("dog", "cat", "horse", "bird", "волк", "собак", "кот", "конь", "пес")):
            return "animal"
    n = (name or "").lower()
    if any(k in n for k in ("семья", "династия", "род", "клан", "группа")):
        return "group"
    return "human" if s == "" else "other_living"


def _canonical_relation_type(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "associated_with"
    for canon, aliases in RELATION_TYPE_SYNONYMS.items():
        if s in aliases:
            return canon
    if any(x in s for x in ("родител", "father", "mother", "child")):
        return "parent_child"
    if any(x in s for x in ("супруг", "marri", "wife", "husband", "spouse")):
        return "spouse"
    if any(x in s for x in ("друг", "friend")):
        return "friend"
    if any(x in s for x in ("враг", "enemy", "foe")):
        return "enemy"
    if any(x in s for x in ("партнер", "partner")):
        return "partner"
    if any(x in s for x in ("питом", "owner", "pet", "хозя")):
        return "owner_pet"
    if any(x in s for x in ("союз", "ally")):
        return "ally"
    if any(x in s for x in ("сопер", "rival")):
        return "rival"
    if any(x in s for x in ("настав", "mentor", "учител")):
        return "mentor"
    if any(x in s for x in ("ученик", "student")):
        return "student"
    if any(x in s for x in ("брат", "сестра", "sibling")):
        return "sibling"
    if any(x in s for x in ("родств", "relative")):
        return "relative"
    if any(x in s for x in ("коллег", "works_with", "cooperat")):
        return "works_with"
    if any(x in s for x in ("знает", "knows")):
        return "knows"
    if any(x in s for x in ("member", "член")):
        return "member_of"
    if any(x in s for x in ("лидер", "глава", "leader")):
        return "leader_of"
    return "associated_with"


def _extract_evidence(rel: Dict[str, Any], fallback_page: Optional[int]) -> Dict[str, Any]:
    quote: Optional[str] = None
    page_idx: Optional[int] = None
    ev = rel.get("evidence")
    if isinstance(ev, dict):
        quote = str(ev.get("quote") or "").strip() or None
        page_idx = ev.get("page_idx")
    elif isinstance(ev, list) and ev:
        first = ev[0]
        if isinstance(first, dict):
            quote = str(first.get("quote") or "").strip() or None
            page_idx = first.get("page_idx")
    if quote is None:
        quote = str(rel.get("quote") or "").strip() or None
    if page_idx is None:
        page_idx = rel.get("page_idx")
    try:
        page = int(page_idx) if page_idx is not None else fallback_page
    except Exception:
        page = fallback_page
    if quote and len(quote) > 320:
        quote = quote[:320]
    return {"quote": quote, "page_idx": page}


def _extract_endpoint(
    rel: Dict[str, Any],
    role: str,
    mention_map: Dict[str, str],
) -> Dict[str, Optional[str]]:
    # role: "source" or "target"
    raw_obj = rel.get(role) or rel.get("from" if role == "source" else "to")
    name: Optional[str] = None
    local_id: Optional[str] = None

    if isinstance(raw_obj, dict):
        name = _extract_name(raw_obj.get("name") or raw_obj.get("entity") or raw_obj)
        local_id = str(raw_obj.get("id") or raw_obj.get("mention_id") or "").strip() or None
    elif isinstance(raw_obj, str):
        name = _extract_name(raw_obj)

    if local_id is None:
        for key in (
            [f"{role}_id", "from_id", "subject_id", "person1_id", "parent_id", "owner_id"]
            if role == "source"
            else [f"{role}_id", "to_id", "object_id", "person2_id", "child_id", "pet_id"]
        ):
            if key in rel and str(rel.get(key)).strip():
                local_id = str(rel.get(key)).strip()
                break
    if name is None:
        for key in (
            [f"{role}_name", "from_name", "subject", "person1", "parent", "owner", "a"]
            if role == "source"
            else [f"{role}_name", "to_name", "object", "person2", "child", "pet", "b"]
        ):
            if key in rel and _extract_name(rel.get(key)):
                name = _extract_name(rel.get(key))
                break

    mention_id = mention_map.get(local_id) if local_id else None
    return {"mention_id": mention_id, "name": _normalize_surface_name(name) if name else None}


def _build_extraction_prompt(chunk: ChunkItem) -> str:
    return f"""
Извлеки упоминания живых сущностей и их отношения.

Правила:
- Возвращай ТОЛЬКО JSON.
- Извлекай только живые сущности: люди, животные, группы живых существ.
- Не выдумывай факты.
- Если данных нет, верни пустые массивы.

Схема JSON:
{{
  "mentions": [
    {{
      "id": "m1",
      "name": "string",
      "entity_type": "human|animal|group|other_living",
      "species": null,
      "aliases": [],
      "birth_year": null,
      "death_year": null,
      "birth_place": null,
      "death_place": null,
      "gender": null,
      "occupation": null,
      "description": null,
      "confidence": 0.0
    }}
  ],
  "relations": [
    {{
      "type": "friend|enemy|partner|spouse|parent_child|owner_pet|ally|rival|mentor|student|sibling|relative|works_with|knows|member_of|leader_of|associated_with",
      "source": {{"id":"m1","name":"..."}},
      "target": {{"id":"m2","name":"..."}},
      "polarity": "positive|negative|neutral",
      "confidence": 0.0,
      "evidence": {{"quote":"...", "page_idx": 0}}
    }}
  ]
}}

Текст (страницы {chunk.page_indices}):
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
    prompt = f"""
Normalize to strict JSON with top-level keys:
{{
  "mentions": [],
  "relations": []
}}

Keep only living entities and relation records.
Output JSON only.

Raw output:
{raw_output}
""".strip()
    repaired = await _llm_complete(
        prompt,
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
            "Ты извлекаешь граф живых сущностей из текста. "
            "Фиксируй только явно поддержанные факты. Верни только JSON."
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
        obj = {"mentions": [], "relations": []}
    return {"json": obj, "repaired": repaired, "raw_preview": (raw or "")[:1200]}


def _parse_extracted_chunk(chunk: ChunkItem, payload: Dict[str, Any]) -> Dict[str, Any]:
    obj = payload.get("json") or {}
    mentions_raw = obj.get("mentions") if isinstance(obj, dict) else []
    relations_raw = obj.get("relations") if isinstance(obj, dict) else []

    mentions: List[Dict[str, Any]] = []
    local_to_global: Dict[str, str] = {}

    if isinstance(mentions_raw, list):
        for idx, m in enumerate(mentions_raw, start=1):
            if not isinstance(m, dict):
                continue
            name = _extract_name(m.get("name") or m.get("entity"))
            if not name:
                continue
            name = _normalize_surface_name(name)
            if not _is_valid_living_name(name):
                continue
            local_id = str(m.get("id") or m.get("mention_id") or f"m{idx}").strip()
            global_id = f"ch{chunk.chunk_id}:{local_id}"
            local_to_global[local_id] = global_id
            entity_type = _canonical_entity_type(m.get("entity_type"), name, m.get("species"))
            aliases_val = m.get("aliases") or []
            if isinstance(aliases_val, str):
                aliases = [_normalize_surface_name(x) for x in aliases_val.split(",") if x.strip()]
            elif isinstance(aliases_val, list):
                aliases = [_normalize_surface_name(str(x)) for x in aliases_val if str(x).strip()]
            else:
                aliases = []
            aliases = [a for a in aliases if _is_valid_living_name(a)]
            mention = {
                "mention_id": global_id,
                "local_id": local_id,
                "chunk_id": chunk.chunk_id,
                "page_idx": chunk.page_indices[0] if chunk.page_indices else None,
                "name": name,
                "normalized_name": normalize_name(name),
                "entity_type": entity_type,
                "species": m.get("species"),
                "aliases": sorted(set(aliases)),
                "birth_year": coerce_year(m.get("birth_year")),
                "death_year": coerce_year(m.get("death_year")),
                "birth_place": m.get("birth_place"),
                "death_place": m.get("death_place"),
                "gender": m.get("gender"),
                "occupation": m.get("occupation"),
                "description": m.get("description") or m.get("biography"),
                "confidence": _safe_float(m.get("confidence"), 0.5),
            }
            mentions.append(mention)

    relations: List[Dict[str, Any]] = []
    if isinstance(relations_raw, list):
        for i, rel in enumerate(relations_raw, start=1):
            if not isinstance(rel, dict):
                continue
            rtype = _canonical_relation_type(rel.get("type") or rel.get("relation_type"))
            source = _extract_endpoint(rel, "source", local_to_global)
            target = _extract_endpoint(rel, "target", local_to_global)
            if not source.get("mention_id") and source.get("name") and not _is_valid_living_name(str(source["name"])):
                source["name"] = None
            if not target.get("mention_id") and target.get("name") and not _is_valid_living_name(str(target["name"])):
                target["name"] = None
            if not (source.get("mention_id") or source.get("name")):
                continue
            if not (target.get("mention_id") or target.get("name")):
                continue
            polarity = str(rel.get("polarity") or "").strip().lower()
            if polarity not in {"positive", "negative", "neutral"}:
                polarity = RELATION_DEFAULT_POLARITY.get(rtype, "neutral")
            confidence = _safe_float(rel.get("confidence"), 0.5)
            ev = _extract_evidence(rel, chunk.page_indices[0] if chunk.page_indices else None)
            relations.append(
                {
                    "relation_local_id": f"ch{chunk.chunk_id}:r{i}",
                    "chunk_id": chunk.chunk_id,
                    "type": rtype,
                    "source": source,
                    "target": target,
                    "polarity": polarity,
                    "confidence": confidence,
                    "evidence": ev,
                }
            )

    return {
        "chunk_id": chunk.chunk_id,
        "pages": chunk.page_indices,
        "repaired": bool(payload.get("repaired")),
        "raw_preview": payload.get("raw_preview"),
        "mentions": mentions,
        "relations": relations,
    }


def _extract_roman_numeral(name: str) -> Optional[str]:
    tokens = re.findall(r"\b[IVXLCDM]+\b", name.upper())
    return tokens[-1] if tokens else None


def _strip_roman(name: str) -> str:
    return re.sub(r"\b[IVXLCDM]+\b", "", name.upper()).strip()


def _extract_patronymic(name: str) -> Optional[str]:
    tokens = name.lower().split()
    for t in tokens:
        if t.endswith(("ович", "евич", "ична", "овна", "евна", "оглы", "кызы")):
            return t
    return None


def _last_token(name: str) -> Optional[str]:
    parts = [p for p in name.lower().split() if p]
    return parts[-1] if parts else None


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _name_match_keys(name: str) -> List[str]:
    s = _normalize_surface_name(name)
    if not s:
        return []
    variants = {
        s,
        re.sub(r"\([^)]*\)", " ", s),
        re.sub(r"\s[-–—]\s.*$", "", s),
        s.split(",", 1)[0],
    }
    extra = {re.sub(r"[\"'«»“”„‟‘’`]", " ", x) for x in variants}
    variants.update(extra)
    keys = {normalize_name(x) for x in variants}
    return sorted(k for k in keys if k and len(k) >= 2)


def _entity_name_keys(entity: Dict[str, Any]) -> List[str]:
    keys: set[str] = set()
    base = str(entity.get("canonical_name") or "").strip()
    if base:
        keys.update(_name_match_keys(base))
    for alias in entity.get("aliases") or []:
        a = str(alias).strip()
        if a:
            keys.update(_name_match_keys(a))
    return sorted(keys)


def _mention_name_keys(mention: Dict[str, Any]) -> List[str]:
    keys: set[str] = set()
    base = str(mention.get("name") or "").strip()
    if base:
        keys.update(_name_match_keys(base))
    for alias in mention.get("aliases") or []:
        a = str(alias).strip()
        if a:
            keys.update(_name_match_keys(a))
    return sorted(keys)


def _identity_match_signals(mention: Dict[str, Any], entity: Dict[str, Any]) -> List[str]:
    signals: List[str] = []

    m_by = coerce_year(mention.get("birth_year"))
    e_by = coerce_year(entity.get("birth_year"))
    if m_by is not None and e_by is not None and abs(m_by - e_by) <= 10:
        signals.append("birth_year_match")

    m_dy = coerce_year(mention.get("death_year"))
    e_dy = coerce_year(entity.get("death_year"))
    if m_dy is not None and e_dy is not None and abs(m_dy - e_dy) <= 10:
        signals.append("death_year_match")

    m_gender = str(mention.get("gender") or "").strip().lower()
    e_gender = str(entity.get("gender") or "").strip().lower()
    if m_gender and e_gender and m_gender == e_gender:
        signals.append("gender_match")

    m_bp = str(mention.get("birth_place") or "").strip().lower()
    e_bp = str(entity.get("birth_place") or "").strip().lower()
    if m_bp and e_bp and m_bp == e_bp:
        signals.append("birth_place_match")

    m_dp = str(mention.get("death_place") or "").strip().lower()
    e_dp = str(entity.get("death_place") or "").strip().lower()
    if m_dp and e_dp and m_dp == e_dp:
        signals.append("death_place_match")

    m_occ = normalize_name(str(mention.get("occupation") or ""))
    e_occ = normalize_name(str(entity.get("occupation") or ""))
    if m_occ and e_occ and (m_occ == e_occ or m_occ in e_occ or e_occ in m_occ):
        signals.append("occupation_match")

    m_species = str(mention.get("species") or "").strip().lower()
    e_species = str(entity.get("species") or "").strip().lower()
    if m_species and e_species and m_species == e_species:
        signals.append("species_match")

    return signals


def _is_identity_confirmed(mention: Dict[str, Any], entity: Dict[str, Any]) -> bool:
    return len(_identity_match_signals(mention, entity)) > 0


def _compatible_and_score(
    mention: Dict[str, Any],
    entity: Dict[str, Any],
) -> Tuple[bool, float, List[str], List[str]]:
    reasons: List[str] = []
    blockers: List[str] = []
    score = 0.0

    m_name = str(mention.get("name") or "")
    e_name = str(entity.get("canonical_name") or "")
    m_norm = str(mention.get("normalized_name") or normalize_name(m_name))
    e_norm = str(entity.get("normalized_name") or normalize_name(e_name))

    m_type = str(mention.get("entity_type") or "")
    e_type = str(entity.get("entity_type") or "")
    if m_type and e_type and m_type != e_type and {m_type, e_type} != {"other_living", "human"}:
        blockers.append("entity_type_mismatch")

    mr = _extract_roman_numeral(m_name)
    er = _extract_roman_numeral(e_name)
    if mr and er and mr != er:
        blockers.append("roman_numeral_mismatch")

    mp = _extract_patronymic(m_name)
    ep = _extract_patronymic(e_name)
    if mp and ep and mp != ep:
        blockers.append("patronymic_mismatch")

    m_by = coerce_year(mention.get("birth_year"))
    e_by = coerce_year(entity.get("birth_year"))
    if m_by is not None and e_by is not None and abs(m_by - e_by) > 15:
        blockers.append("birth_year_conflict")

    m_dy = coerce_year(mention.get("death_year"))
    e_dy = coerce_year(entity.get("death_year"))
    if m_dy is not None and e_dy is not None and abs(m_dy - e_dy) > 15:
        blockers.append("death_year_conflict")

    m_gender = str(mention.get("gender") or "").lower()
    e_gender = str(entity.get("gender") or "").lower()
    if m_gender and e_gender and m_gender != e_gender:
        blockers.append("gender_conflict")

    m_species = str(mention.get("species") or "").lower()
    e_species = str(entity.get("species") or "").lower()
    if m_species and e_species and m_species != e_species:
        blockers.append("species_conflict")

    if blockers:
        return False, 0.0, reasons, blockers

    if m_norm and e_norm and m_norm == e_norm:
        score += 0.55
        reasons.append("exact_normalized_name")
    elif _strip_roman(m_name) and _strip_roman(e_name) and _strip_roman(m_name) == _strip_roman(e_name):
        score += 0.35
        reasons.append("same_name_without_roman")

    sim = _name_similarity(m_norm, e_norm)
    if sim >= 0.95:
        score += 0.25
        reasons.append("high_name_similarity")
    elif sim >= 0.85:
        score += 0.15
        reasons.append("medium_name_similarity")

    aliases = [normalize_name(x) for x in (entity.get("aliases") or [])]
    if m_norm in aliases or e_norm in [normalize_name(x) for x in (mention.get("aliases") or [])]:
        score += 0.2
        reasons.append("alias_overlap")

    m_keys = set(_mention_name_keys(mention))
    e_keys = set(_entity_name_keys(entity))
    if m_keys and e_keys and (m_keys & e_keys):
        score += 0.25
        reasons.append("name_key_overlap")

    if m_type and e_type and m_type == e_type:
        score += 0.1
        reasons.append("same_entity_type")

    if m_by is not None and e_by is not None:
        diff = abs(m_by - e_by)
        if diff <= 2:
            score += 0.2
            reasons.append("close_birth_year")
        elif diff <= 10:
            score += 0.1
            reasons.append("near_birth_year")

    if m_dy is not None and e_dy is not None:
        diff = abs(m_dy - e_dy)
        if diff <= 2:
            score += 0.1
            reasons.append("close_death_year")
        elif diff <= 10:
            score += 0.05
            reasons.append("near_death_year")

    if m_gender and e_gender and m_gender == e_gender:
        score += 0.05
        reasons.append("same_gender")

    m_bp = str(mention.get("birth_place") or "").strip().lower()
    e_bp = str(entity.get("birth_place") or "").strip().lower()
    if m_bp and e_bp and m_bp == e_bp:
        score += 0.05
        reasons.append("same_birth_place")

    if _last_token(m_name) and _last_token(e_name) and _last_token(m_name) == _last_token(e_name):
        score += 0.08
        reasons.append("same_last_token")

    return True, min(score, 1.0), reasons, blockers


def _merge_mention_into_entity(entity: Dict[str, Any], mention: Dict[str, Any]) -> None:
    if len(str(mention.get("name") or "")) > len(str(entity.get("canonical_name") or "")):
        entity["canonical_name"] = mention.get("name")
    entity["normalized_name"] = normalize_name(str(entity.get("canonical_name") or ""))
    for key in ("entity_type", "species", "birth_year", "death_year", "birth_place", "death_place", "gender", "occupation", "description"):
        if not entity.get(key) and mention.get(key):
            entity[key] = mention[key]
    entity["confidence"] = max(float(entity.get("confidence") or 0.0), float(mention.get("confidence") or 0.0))
    aliases = set(entity.get("aliases") or [])
    aliases.update(mention.get("aliases") or [])
    n = mention.get("name")
    if n and n != entity.get("canonical_name"):
        aliases.add(n)
    entity["aliases"] = sorted(a for a in aliases if a and a != entity.get("canonical_name"))
    pages = set(entity.get("source_pages") or [])
    if mention.get("page_idx") is not None:
        pages.add(int(mention["page_idx"]))
    entity["source_pages"] = sorted(pages)
    mention_ids = set(entity.get("mention_ids") or [])
    mention_ids.add(str(mention.get("mention_id")))
    entity["mention_ids"] = sorted(mention_ids)


def _new_entity_from_mention(entity_id: str, mention: Dict[str, Any]) -> Dict[str, Any]:
    pages = []
    if mention.get("page_idx") is not None:
        pages = [int(mention["page_idx"])]
    return {
        "entity_id": entity_id,
        "canonical_name": mention.get("name"),
        "normalized_name": normalize_name(str(mention.get("name") or "")),
        "entity_type": mention.get("entity_type"),
        "species": mention.get("species"),
        "aliases": sorted(set(mention.get("aliases") or [])),
        "birth_year": mention.get("birth_year"),
        "death_year": mention.get("death_year"),
        "birth_place": mention.get("birth_place"),
        "death_place": mention.get("death_place"),
        "gender": mention.get("gender"),
        "occupation": mention.get("occupation"),
        "description": mention.get("description"),
        "confidence": float(mention.get("confidence") or 0.0),
        "source_pages": pages,
        "mention_ids": [str(mention.get("mention_id"))],
    }


def _resolve_mentions_to_entities(
    mentions: Sequence[Dict[str, Any]],
    *,
    auto_merge_threshold: float,
    possible_merge_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    entities: List[Dict[str, Any]] = []
    mention_to_entity: Dict[str, str] = {}
    possible_same: List[Dict[str, Any]] = []
    next_idx = 1
    entity_key_cache: Dict[str, set[str]] = {}

    def _cached_entity_keys(entity: Dict[str, Any]) -> set[str]:
        eid = str(entity.get("entity_id"))
        keys = entity_key_cache.get(eid)
        if keys is not None:
            return keys
        computed = set(_entity_name_keys(entity))
        entity_key_cache[eid] = computed
        return computed

    for m in mentions:
        mention_id = str(m.get("mention_id"))
        mention_keys = set(_mention_name_keys(m))
        best: Optional[Tuple[float, Dict[str, Any], List[str], List[str]]] = None
        name_key_candidates: List[Tuple[float, Dict[str, Any], List[str], List[str]]] = []
        for e in entities:
            ok, score, reasons, blockers = _compatible_and_score(m, e)
            if not ok:
                continue
            if best is None or score > best[0]:
                best = (score, e, reasons, blockers)
            if mention_keys and (mention_keys & _cached_entity_keys(e)):
                name_key_candidates.append((score, e, reasons, blockers))

        if len(name_key_candidates) == 1:
            _, best_e, _, _ = name_key_candidates[0]
            if _is_identity_confirmed(m, best_e):
                _merge_mention_into_entity(best_e, m)
                mention_to_entity[mention_id] = best_e["entity_id"]
                entity_key_cache.pop(str(best_e.get("entity_id")), None)
                continue
        if len(name_key_candidates) > 1:
            name_key_candidates.sort(key=lambda x: x[0], reverse=True)
            top = name_key_candidates[0]
            second = name_key_candidates[1]
            if (
                top[0] >= auto_merge_threshold
                and (top[0] - second[0]) >= 0.15
                and _is_identity_confirmed(m, top[1])
            ):
                _, best_e, _, _ = top
                _merge_mention_into_entity(best_e, m)
                mention_to_entity[mention_id] = best_e["entity_id"]
                entity_key_cache.pop(str(best_e.get("entity_id")), None)
                continue

        if best and best[0] >= auto_merge_threshold and _is_identity_confirmed(m, best[1]):
            _, best_e, _, _ = best
            _merge_mention_into_entity(best_e, m)
            mention_to_entity[mention_id] = best_e["entity_id"]
            entity_key_cache.pop(str(best_e.get("entity_id")), None)
            continue

        new_id = f"ent-{next_idx:06d}"
        next_idx += 1
        new_entity = _new_entity_from_mention(new_id, m)
        entities.append(new_entity)
        mention_to_entity[mention_id] = new_id
        entity_key_cache.pop(new_id, None)

        if best and best[0] >= possible_merge_threshold:
            score, best_e, reasons, _ = best
            a, b = sorted([best_e["entity_id"], new_id])
            possible_same.append(
                {
                    "entity_a": a,
                    "entity_b": b,
                    "score": float(score),
                    "reasons": reasons,
                    "trigger_mention_id": mention_id,
                }
            )

    # dedup possible links
    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in possible_same:
        key = (p["entity_a"], p["entity_b"])
        if key not in uniq or float(p.get("score") or 0.0) > float(uniq[key].get("score") or 0.0):
            uniq[key] = p
    return entities, mention_to_entity, list(uniq.values())


def _resolve_name_to_entity_id(name: str, entities: Sequence[Dict[str, Any]]) -> Optional[str]:
    n = normalize_name(name)
    if not n:
        return None
    exact = [e for e in entities if normalize_name(str(e.get("canonical_name") or "")) == n]
    if len(exact) == 1:
        return exact[0]["entity_id"]
    if len(exact) > 1:
        # pick most confident if duplicate canonical names exist
        exact.sort(key=lambda x: float(x.get("confidence") or 0.0), reverse=True)
        return exact[0]["entity_id"]
    alias_matches = [e for e in entities if n in [normalize_name(a) for a in (e.get("aliases") or [])]]
    if alias_matches:
        alias_matches.sort(key=lambda x: float(x.get("confidence") or 0.0), reverse=True)
        return alias_matches[0]["entity_id"]
    # variant exact match: tolerate epithets/parentheses, but require best unique overlap
    query_keys = set(_name_match_keys(name))
    if query_keys:
        variant_matches: List[Tuple[int, float, Dict[str, Any]]] = []
        for e in entities:
            overlap = len(query_keys & set(_entity_name_keys(e)))
            if overlap > 0:
                variant_matches.append((overlap, float(e.get("confidence") or 0.0), e))
        if variant_matches:
            variant_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if len(variant_matches) == 1:
                return variant_matches[0][2]["entity_id"]
            best = variant_matches[0]
            second = variant_matches[1]
            if best[0] > second[0] or best[1] > second[1]:
                return best[2]["entity_id"]
    # fuzzy fallback
    best_score = 0.0
    best_id: Optional[str] = None
    for e in entities:
        en = normalize_name(str(e.get("canonical_name") or ""))
        score = _name_similarity(n, en)
        if score > best_score:
            best_score = score
            best_id = e["entity_id"]
    if best_score >= 0.93:
        return best_id
    return None


def _append_ev(dst: List[Dict[str, Any]], ev: Dict[str, Any]) -> None:
    key = (ev.get("quote"), ev.get("page_idx"))
    existing = {(x.get("quote"), x.get("page_idx")) for x in dst}
    if key not in existing:
        dst.append({"quote": ev.get("quote"), "page_idx": ev.get("page_idx")})


def _entity_summary_for_agent(entity: Dict[str, Any], mention_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mids = [str(x) for x in (entity.get("mention_ids") or [])]
    mention_samples: List[Dict[str, Any]] = []
    for mid in mids[:6]:
        m = mention_by_id.get(mid)
        if not m:
            continue
        mention_samples.append(
            {
                "mention_id": mid,
                "name": m.get("name"),
                "page_idx": m.get("page_idx"),
                "birth_year": m.get("birth_year"),
                "death_year": m.get("death_year"),
                "gender": m.get("gender"),
                "occupation": m.get("occupation"),
            }
        )
    return {
        "entity_id": entity.get("entity_id"),
        "canonical_name": entity.get("canonical_name"),
        "entity_type": entity.get("entity_type"),
        "aliases": list(entity.get("aliases") or [])[:8],
        "birth_year": entity.get("birth_year"),
        "death_year": entity.get("death_year"),
        "birth_place": entity.get("birth_place"),
        "death_place": entity.get("death_place"),
        "gender": entity.get("gender"),
        "occupation": entity.get("occupation"),
        "species": entity.get("species"),
        "source_pages": list(entity.get("source_pages") or [])[:16],
        "mention_count": len(mids),
        "mention_samples": mention_samples,
    }


async def _llm_decide_possible_same_pair(
    candidate: Dict[str, Any],
    entity_a: Dict[str, Any],
    entity_b: Dict[str, Any],
    mention_by_id: Dict[str, Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    payload = {
        "candidate": {
            "entity_a": candidate.get("entity_a"),
            "entity_b": candidate.get("entity_b"),
            "score": float(candidate.get("score") or 0.0),
            "reasons": list(candidate.get("reasons") or []),
            "trigger_mention_id": candidate.get("trigger_mention_id"),
        },
        "entity_a": _entity_summary_for_agent(entity_a, mention_by_id),
        "entity_b": _entity_summary_for_agent(entity_b, mention_by_id),
    }
    prompt = (
        "Нужно решить, описывают ли две сущности одно и то же живое существо.\n"
        "Разрешай слияние только при высокой уверенности.\n"
        "Если есть сомнение, ответь same_entity=false.\n\n"
        "Верни ТОЛЬКО JSON:\n"
        "{\n"
        '  "same_entity": true,\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short"\n'
        "}\n\n"
        "Данные:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    raw = await _llm_complete(
        prompt,
        (
            "Ты арбитр entity-resolution. "
            "Используй только данные из входа. "
            "Если недостаточно оснований, выбирай same_entity=false. "
            "Выводи только валидный JSON."
        ),
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    obj = robust_json_loads(raw)
    if not obj:
        return None
    same_entity = _safe_bool(obj.get("same_entity"), False)
    confidence = _safe_float(obj.get("confidence"), 0.0)
    reason = str(obj.get("reason") or obj.get("rationale") or "").strip()
    return {
        "same_entity": same_entity,
        "confidence": confidence,
        "reason": reason[:280] if reason else None,
    }


async def _llm_review_possible_same(
    possible_same: Sequence[Dict[str, Any]],
    entities: Sequence[Dict[str, Any]],
    mentions: Sequence[Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
    min_confidence: float,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    if not possible_same:
        return []
    entity_by_id = {str(e.get("entity_id")): e for e in entities}
    mention_by_id = {str(m.get("mention_id")): m for m in mentions}

    ranked = sorted(possible_same, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    if max_candidates > 0:
        ranked = ranked[:max_candidates]

    reviews: List[Dict[str, Any]] = []
    for cand in ranked:
        a_id = str(cand.get("entity_a") or "")
        b_id = str(cand.get("entity_b") or "")
        if not a_id or not b_id or a_id == b_id:
            continue
        ea = entity_by_id.get(a_id)
        eb = entity_by_id.get(b_id)
        if not ea or not eb:
            continue
        decision = await _llm_decide_possible_same_pair(
            cand,
            ea,
            eb,
            mention_by_id,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        if not decision:
            continue
        reviews.append(
            {
                "entity_a": a_id,
                "entity_b": b_id,
                "base_score": float(cand.get("score") or 0.0),
                "base_reasons": list(cand.get("reasons") or []),
                "same_entity": bool(decision.get("same_entity")),
                "confidence": _safe_float(decision.get("confidence"), 0.0),
                "accepted": bool(decision.get("same_entity")) and _safe_float(decision.get("confidence"), 0.0) >= min_confidence,
                "reason": decision.get("reason"),
            }
        )
    return reviews


def _prefer_entity_type(a: Optional[str], b: Optional[str]) -> Optional[str]:
    rank = {"human": 4, "animal": 3, "group": 2, "other_living": 1}
    aa = str(a or "").strip()
    bb = str(b or "").strip()
    if not aa:
        return bb or None
    if not bb:
        return aa
    return aa if rank.get(aa, 0) >= rank.get(bb, 0) else bb


def _merge_entity_into_entity(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst_name = str(dst.get("canonical_name") or "").strip()
    src_name = str(src.get("canonical_name") or "").strip()
    aliases = set(dst.get("aliases") or [])
    aliases.update(src.get("aliases") or [])
    if dst_name and src_name and dst_name != src_name:
        aliases.add(dst_name)
        aliases.add(src_name)
    if len(src_name) > len(dst_name):
        dst["canonical_name"] = src_name
    dst["normalized_name"] = normalize_name(str(dst.get("canonical_name") or ""))

    dst["entity_type"] = _prefer_entity_type(dst.get("entity_type"), src.get("entity_type"))
    if not dst.get("species") and src.get("species"):
        dst["species"] = src.get("species")
    for key in ("birth_year", "death_year", "birth_place", "death_place", "gender", "occupation", "description"):
        if not dst.get(key) and src.get(key):
            dst[key] = src.get(key)
    dst["confidence"] = max(float(dst.get("confidence") or 0.0), float(src.get("confidence") or 0.0))
    dst["source_pages"] = sorted({int(x) for x in (dst.get("source_pages") or []) + (src.get("source_pages") or [])})
    dst["mention_ids"] = sorted({str(x) for x in (dst.get("mention_ids") or []) + (src.get("mention_ids") or []) if str(x)})
    canonical = str(dst.get("canonical_name") or "")
    dst["aliases"] = sorted(a for a in aliases if a and a != canonical)


def _apply_llm_entity_merges(
    entities: Sequence[Dict[str, Any]],
    mention_to_entity: Dict[str, str],
    possible_same: Sequence[Dict[str, Any]],
    reviews: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    accepted_pairs = [
        (str(r.get("entity_a") or ""), str(r.get("entity_b") or ""))
        for r in reviews
        if bool(r.get("accepted"))
    ]
    accepted_pairs = [(a, b) for a, b in accepted_pairs if a and b and a != b]
    if not accepted_pairs:
        return list(entities), dict(mention_to_entity), list(possible_same)

    entity_by_id = {str(e.get("entity_id")): dict(e) for e in entities}
    parent: Dict[str, str] = {eid: eid for eid in entity_by_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        parent[rb] = ra

    for a, b in accepted_pairs:
        if a in parent and b in parent:
            union(a, b)

    groups: Dict[str, List[str]] = {}
    for eid in entity_by_id:
        root = find(eid)
        groups.setdefault(root, []).append(eid)

    def _entity_rank(e: Dict[str, Any]) -> Tuple[int, int, float, int, str]:
        mention_count = len(e.get("mention_ids") or [])
        page_count = len(e.get("source_pages") or [])
        conf = float(e.get("confidence") or 0.0)
        name_len = len(str(e.get("canonical_name") or ""))
        eid = str(e.get("entity_id") or "")
        return (mention_count, page_count, conf, name_len, eid)

    remap: Dict[str, str] = {}
    merged_entities: List[Dict[str, Any]] = []
    for member_ids in groups.values():
        if len(member_ids) == 1:
            eid = member_ids[0]
            remap[eid] = eid
            merged_entities.append(entity_by_id[eid])
            continue
        ranked = sorted(member_ids, key=lambda x: _entity_rank(entity_by_id[x]), reverse=True)
        rep_id = ranked[0]
        rep = dict(entity_by_id[rep_id])
        for oid in ranked[1:]:
            _merge_entity_into_entity(rep, entity_by_id[oid])
        rep["entity_id"] = rep_id
        rep["normalized_name"] = normalize_name(str(rep.get("canonical_name") or ""))
        merged_entities.append(rep)
        for oid in member_ids:
            remap[oid] = rep_id

    remapped_mentions = {k: remap.get(v, v) for k, v in mention_to_entity.items()}

    possible_rows: List[Dict[str, Any]] = []
    for p in possible_same:
        a = remap.get(str(p.get("entity_a") or ""), str(p.get("entity_a") or ""))
        b = remap.get(str(p.get("entity_b") or ""), str(p.get("entity_b") or ""))
        if not a or not b or a == b:
            continue
        aa, bb = sorted([a, b])
        row = dict(p)
        row["entity_a"] = aa
        row["entity_b"] = bb
        possible_rows.append(row)
    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in possible_rows:
        key = (row["entity_a"], row["entity_b"])
        if key not in uniq or float(row.get("score") or 0.0) > float(uniq[key].get("score") or 0.0):
            uniq[key] = row

    merged_entities.sort(key=lambda x: str(x.get("entity_id") or ""))
    return merged_entities, remapped_mentions, list(uniq.values())


async def _build_graph(
    parsed_chunks: Sequence[Dict[str, Any]],
    *,
    min_relation_confidence: float,
    auto_merge_threshold: float,
    possible_merge_threshold: float,
    llm_merge_agent: bool,
    llm_merge_min_confidence: float,
    llm_merge_max_candidates: int,
    llm_merge_max_tokens: int,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
) -> Dict[str, Any]:
    mentions: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    for ch in parsed_chunks:
        mentions.extend(ch.get("mentions") or [])
        relations.extend(ch.get("relations") or [])

    entities, mention_to_entity, possible_same = _resolve_mentions_to_entities(
        mentions,
        auto_merge_threshold=auto_merge_threshold,
        possible_merge_threshold=possible_merge_threshold,
    )
    llm_merge_reviews: List[Dict[str, Any]] = []
    if llm_merge_agent and possible_same:
        llm_merge_reviews = await _llm_review_possible_same(
            possible_same,
            entities,
            mentions,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=llm_merge_max_tokens,
            min_confidence=llm_merge_min_confidence,
            max_candidates=llm_merge_max_candidates,
        )
        entities, mention_to_entity, possible_same = _apply_llm_entity_merges(
            entities,
            mention_to_entity,
            possible_same,
            llm_merge_reviews,
        )

    entity_by_id: Dict[str, Dict[str, Any]] = {e["entity_id"]: e for e in entities}
    next_entity_idx = len(entities) + 1

    def _ensure_entity_by_name(name: Optional[str]) -> Optional[str]:
        nonlocal next_entity_idx
        if not name:
            return None
        n = _normalize_surface_name(name)
        if not _is_valid_living_name(n):
            return None
        found_id = _resolve_name_to_entity_id(n, entities)
        if found_id:
            return found_id
        ent_id = f"ent-{next_entity_idx:06d}"
        next_entity_idx += 1
        new_e = {
            "entity_id": ent_id,
            "canonical_name": n,
            "normalized_name": normalize_name(n),
            "entity_type": _canonical_entity_type(None, n, None),
            "species": None,
            "aliases": [],
            "birth_year": None,
            "death_year": None,
            "birth_place": None,
            "death_place": None,
            "gender": None,
            "occupation": None,
            "description": None,
            "confidence": 0.3,
            "source_pages": [],
            "mention_ids": [],
        }
        entities.append(new_e)
        entity_by_id[ent_id] = new_e
        return ent_id

    # attach resolved entity_id to mention outputs
    mentions_resolved = []
    for m in mentions:
        mr = dict(m)
        mr["entity_id"] = mention_to_entity.get(str(m.get("mention_id")))
        mentions_resolved.append(mr)

    agg: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    claim_rows: List[Dict[str, Any]] = []
    claim_idx = 1

    for rel in relations:
        conf = _safe_float(rel.get("confidence"), 0.0)
        if conf < min_relation_confidence:
            continue
        src_ep = rel.get("source") or {}
        dst_ep = rel.get("target") or {}

        src_id = src_ep.get("mention_id")
        dst_id = dst_ep.get("mention_id")

        src_entity = mention_to_entity.get(str(src_id)) if src_id else None
        dst_entity = mention_to_entity.get(str(dst_id)) if dst_id else None

        if not src_entity and src_ep.get("name"):
            src_entity = _resolve_name_to_entity_id(str(src_ep["name"]), entities)
        if not dst_entity and dst_ep.get("name"):
            dst_entity = _resolve_name_to_entity_id(str(dst_ep["name"]), entities)
        if not src_entity and src_ep.get("name"):
            src_entity = _ensure_entity_by_name(str(src_ep["name"]))
        if not dst_entity and dst_ep.get("name"):
            dst_entity = _ensure_entity_by_name(str(dst_ep["name"]))
        if not src_entity or not dst_entity or src_entity == dst_entity:
            continue

        rtype = _canonical_relation_type(rel.get("type"))
        pol = str(rel.get("polarity") or RELATION_DEFAULT_POLARITY.get(rtype, "neutral")).lower()
        if pol not in {"positive", "negative", "neutral"}:
            pol = RELATION_DEFAULT_POLARITY.get(rtype, "neutral")
        symmetric = rtype in SYMMETRIC_RELATIONS
        directed = not symmetric

        src_final = src_entity
        dst_final = dst_entity
        if symmetric and src_final > dst_final:
            src_final, dst_final = dst_final, src_final

        key = (src_final, dst_final, rtype, pol)
        ev = rel.get("evidence") or {}

        row = agg.setdefault(
            key,
            {
                "relation_id": f"rel-{_md5('|'.join(key))}",
                "source_entity_id": src_final,
                "target_entity_id": dst_final,
                "relation_type": rtype,
                "polarity": pol,
                "confidence": 0.0,
                "support_count": 0,
                "directed": directed,
                "symmetric": symmetric,
                "evidence": [],
            },
        )
        row["confidence"] = max(float(row["confidence"]), conf)
        row["support_count"] = int(row["support_count"]) + 1
        _append_ev(row["evidence"], ev)

        claim_id = f"claim-{claim_idx:06d}"
        claim_idx += 1
        claim_rows.append(
            {
                "claim_id": claim_id,
                "relation_id": row["relation_id"],
                "relation_type": rtype,
                "source_entity_id": src_final,
                "target_entity_id": dst_final,
                "polarity": pol,
                "confidence": conf,
                "chunk_id": rel.get("chunk_id"),
                "evidence": ev,
                "source_relation_local_id": rel.get("relation_local_id"),
            }
        )

    return {
        "entities": sorted(entities, key=lambda x: x["entity_id"]),
        "mentions": sorted(mentions_resolved, key=lambda x: x["mention_id"]),
        "possible_same": sorted(possible_same, key=lambda x: (x["entity_a"], x["entity_b"])),
        "relations": sorted(agg.values(), key=lambda x: (x["relation_type"], x["source_entity_id"], x["target_entity_id"])),
        "relation_claims": claim_rows,
        "llm_merge_reviews": llm_merge_reviews,
        "llm_merge_accepted_total": sum(1 for r in llm_merge_reviews if r.get("accepted")),
    }


def _serialize(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(v) for v in data]
    if isinstance(data, set):
        return sorted(_serialize(v) for v in data)
    return data


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_living_graph_view_html(
    graph: Dict[str, Any],
    *,
    title: str,
) -> str:
    entity_color = {
        "human": "#2E86DE",
        "animal": "#17A589",
        "group": "#8E44AD",
        "other_living": "#5D6D7E",
    }
    polarity_color = {
        "positive": "#1E8449",
        "negative": "#B03A2E",
        "neutral": "#7F8C8D",
    }

    nodes: List[Dict[str, Any]] = []
    for e in graph.get("entities") or []:
        et = str(e.get("entity_type") or "other_living")
        name = str(e.get("canonical_name") or e.get("entity_id") or "")
        label = name if len(name) <= 42 else f"{name[:39]}..."
        tip = (
            f"<b>{_html_escape(name)}</b><br>"
            f"type: {_html_escape(et)}<br>"
            f"confidence: {e.get('confidence')}<br>"
            f"birth_year: {e.get('birth_year')}<br>"
            f"death_year: {e.get('death_year')}"
        )
        nodes.append(
            {
                "id": e.get("entity_id"),
                "label": label,
                "group": et,
                "title": tip,
                "color": entity_color.get(et, "#5D6D7E"),
            }
        )

    edges: List[Dict[str, Any]] = []
    for r in graph.get("relations") or []:
        rt = str(r.get("relation_type") or "associated_with")
        pol = str(r.get("polarity") or "neutral")
        directed = bool(r.get("directed", True))
        edges.append(
            {
                "from": r.get("source_entity_id"),
                "to": r.get("target_entity_id"),
                "label": rt,
                "relation_type": rt,
                "title": (
                    f"polarity: {_html_escape(pol)}<br>"
                    f"confidence: {r.get('confidence')}<br>"
                    f"support: {r.get('support_count')}"
                ),
                "arrows": "to" if directed else "",
                "color": polarity_color.get(pol, "#7F8C8D"),
                "smooth": {"type": "dynamic"},
            }
        )

    relation_types = sorted(
        {
            str(r.get("relation_type") or "associated_with")
            for r in (graph.get("relations") or [])
        }
    )
    model_name = str(graph.get("model") or "")
    heading = f"{title} ({model_name})" if model_name else title

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{_html_escape(heading)}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d8dee9;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif; background: radial-gradient(circle at 20% 10%, #f0fdfa, #f6f7fb 45%, #eef2ff 100%); color: var(--text); }}
    #top {{ display: grid; gap: 10px; padding: 12px 14px; background: linear-gradient(90deg, #ffffff, #f8fafc); border-bottom: 1px solid var(--line); }}
    #title {{ font-weight: 700; }}
    #meta {{ color: var(--muted); font-size: 13px; }}
    #controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .chip {{ border: 1px solid var(--line); padding: 4px 8px; border-radius: 999px; background: var(--panel); font-size: 12px; color: #334155; }}
    #q {{ border: 1px solid var(--line); border-radius: 10px; padding: 7px 10px; min-width: 280px; background: var(--panel); }}
    #relations {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    #relations label {{ font-size: 12px; color: #334155; border: 1px solid var(--line); background: #fff; padding: 5px 8px; border-radius: 10px; }}
    #mynetwork {{ width: 100vw; height: calc(100vh - 170px); min-height: 420px; }}
    @media (max-width: 900px) {{
      #q {{ min-width: 100%; }}
      #mynetwork {{ height: calc(100vh - 240px); }}
    }}
  </style>
</head>
<body>
  <div id="top">
    <div id="title">{_html_escape(heading)}</div>
    <div id="meta">nodes: {len(nodes)} | edges: {len(edges)}</div>
    <div id="controls">
      <input id="q" placeholder="Search person/group..."/>
      <span class="chip">toggle relation types below</span>
    </div>
    <div id="relations"></div>
  </div>
  <div id="mynetwork"></div>
  <script>
    const baseNodes = {json.dumps(nodes, ensure_ascii=False)};
    const baseEdges = {json.dumps(edges, ensure_ascii=False)};
    const relationTypes = {json.dumps(relation_types, ensure_ascii=False)};
    const selectedRelations = new Set(relationTypes);

    const nodes = new vis.DataSet(baseNodes);
    const edges = new vis.DataSet(baseEdges);

    const container = document.getElementById("mynetwork");
    const network = new vis.Network(container, {{ nodes, edges }}, {{
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
      physics: {{ stabilization: false, barnesHut: {{ gravitationalConstant: -19000, springLength: 165 }} }},
      nodes: {{ shape: "dot", size: 17, font: {{ size: 13 }} }},
      edges: {{ font: {{ align: "middle", size: 10 }}, width: 1.8 }}
    }});

    const relationsEl = document.getElementById("relations");
    relationTypes.forEach((t) => {{
      const id = `rt_${{t}}`;
      const wrap = document.createElement("label");
      wrap.innerHTML = `<input type="checkbox" id="${{id}}" checked/> ${{t}}`;
      const cb = wrap.querySelector("input");
      cb.addEventListener("change", () => {{
        if (cb.checked) selectedRelations.add(t); else selectedRelations.delete(t);
        applyFilters();
      }});
      relationsEl.appendChild(wrap);
    }});

    document.getElementById("q").addEventListener("input", applyFilters);

    function applyFilters() {{
      const q = (document.getElementById("q").value || "").trim().toLowerCase();
      const filteredEdges = baseEdges.filter(e => selectedRelations.has(e.relation_type));
      const touched = new Set();
      filteredEdges.forEach(e => {{ touched.add(e.from); touched.add(e.to); }});

      const filteredNodes = baseNodes.filter(n => {{
        if (q && !String(n.label || "").toLowerCase().includes(q)) return false;
        if (q) return true;
        return touched.has(n.id);
      }});

      const visibleIds = new Set(filteredNodes.map(n => n.id));
      const finalEdges = filteredEdges.filter(e => visibleIds.has(e.from) && visibleIds.has(e.to));

      nodes.clear();
      edges.clear();
      nodes.add(filteredNodes);
      edges.add(finalEdges);
      network.fit({{ animation: {{ duration: 300 }} }});
    }}
  </script>
</body>
</html>
"""


def _write_living_graph_view(
    output_dir: Path,
    graph: Dict[str, Any],
    *,
    title: str,
) -> Path:
    html = _build_living_graph_view_html(graph, title=title)
    path = output_dir / "living_graph_view.html"
    path.write_text(html, encoding="utf-8")
    return path


def _reset_neo4j(uri: str, user: str, password: str, db: Optional[str]) -> None:
    from neo4j import GraphDatabase  # type: ignore

    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with drv.session(database=db) as s:
            s.run("MATCH (n) DETACH DELETE n")
    finally:
        drv.close()


def _persist_graph_to_neo4j(
    graph: Dict[str, Any],
    *,
    source_file: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_db: Optional[str],
    reset_db: bool,
) -> Dict[str, Any]:
    from neo4j import GraphDatabase  # type: ignore

    if reset_db:
        _reset_neo4j(neo4j_uri, neo4j_user, neo4j_password, neo4j_db)

    drv = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with drv.session(database=neo4j_db) as s:
            schema = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT mention_id IF NOT EXISTS FOR (m:Mention) REQUIRE m.mention_id IS UNIQUE",
                "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR (r:Relation) REQUIRE r.relation_id IS UNIQUE",
                "CREATE CONSTRAINT relation_claim_id IF NOT EXISTS FOR (c:RelationClaim) REQUIRE c.claim_id IS UNIQUE",
                "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
            ]
            for q in schema:
                s.run(q)

            for e in graph.get("entities") or []:
                props = {
                    "canonical_name": e.get("canonical_name"),
                    "normalized_name": e.get("normalized_name"),
                    "entity_type": e.get("entity_type"),
                    "species": e.get("species"),
                    "aliases": list(e.get("aliases") or []),
                    "birth_year": e.get("birth_year"),
                    "death_year": e.get("death_year"),
                    "birth_place": e.get("birth_place"),
                    "death_place": e.get("death_place"),
                    "gender": e.get("gender"),
                    "occupation": e.get("occupation"),
                    "description": e.get("description"),
                    "confidence": float(e.get("confidence") or 0.0),
                    "source_pages": list(e.get("source_pages") or []),
                    "mention_count": len(e.get("mention_ids") or []),
                    "updated_at": int(time.time() * 1000),
                }
                props = {k: v for k, v in props.items() if v is not None}
                s.run(
                    """
                    MERGE (e:Entity {entity_id: $entity_id})
                    ON CREATE SET e.created_at = timestamp()
                    SET e += $props
                    """,
                    {"entity_id": e["entity_id"], "props": props},
                )

            for m in graph.get("mentions") or []:
                ent_id = m.get("entity_id")
                if not ent_id:
                    continue
                props = {
                    "name": m.get("name"),
                    "normalized_name": m.get("normalized_name"),
                    "entity_type": m.get("entity_type"),
                    "species": m.get("species"),
                    "aliases": list(m.get("aliases") or []),
                    "page_idx": m.get("page_idx"),
                    "chunk_id": m.get("chunk_id"),
                    "confidence": float(m.get("confidence") or 0.0),
                    "source_file": source_file,
                    "updated_at": int(time.time() * 1000),
                }
                props = {k: v for k, v in props.items() if v is not None}
                s.run(
                    """
                    MERGE (m:Mention {mention_id: $mention_id})
                    ON CREATE SET m.created_at = timestamp()
                    SET m += $props
                    WITH m
                    MATCH (e:Entity {entity_id: $entity_id})
                    MERGE (m)-[:REFERS_TO]->(e)
                    """,
                    {"mention_id": m["mention_id"], "entity_id": ent_id, "props": props},
                )

            for p in graph.get("possible_same") or []:
                a = p.get("entity_a")
                b = p.get("entity_b")
                if not a or not b or a == b:
                    continue
                s.run(
                    """
                    MATCH (a:Entity {entity_id: $a})
                    MATCH (b:Entity {entity_id: $b})
                    MERGE (a)-[r:POSSIBLE_SAME_AS]->(b)
                    SET r.score = $score,
                        r.reasons = $reasons,
                        r.trigger_mention_id = $trigger_mention_id,
                        r.updated_at = timestamp()
                    """,
                    {
                        "a": a,
                        "b": b,
                        "score": float(p.get("score") or 0.0),
                        "reasons": list(p.get("reasons") or []),
                        "trigger_mention_id": p.get("trigger_mention_id"),
                    },
                )

            for r in graph.get("relations") or []:
                props = {
                    "relation_type": r.get("relation_type"),
                    "polarity": r.get("polarity"),
                    "confidence": float(r.get("confidence") or 0.0),
                    "support_count": int(r.get("support_count") or 0),
                    "directed": bool(r.get("directed")),
                    "symmetric": bool(r.get("symmetric")),
                    "updated_at": int(time.time() * 1000),
                }
                s.run(
                    """
                    MERGE (r:Relation {relation_id: $relation_id})
                    ON CREATE SET r.created_at = timestamp()
                    SET r += $props
                    WITH r
                    MATCH (s:Entity {entity_id: $source_entity_id})
                    MATCH (t:Entity {entity_id: $target_entity_id})
                    MERGE (s)-[:RELATION_SOURCE]->(r)
                    MERGE (r)-[:RELATION_TARGET]->(t)
                    """,
                    {
                        "relation_id": r["relation_id"],
                        "source_entity_id": r["source_entity_id"],
                        "target_entity_id": r["target_entity_id"],
                        "props": props,
                    },
                )

            for c in graph.get("relation_claims") or []:
                rel_id = c.get("relation_id")
                if not rel_id:
                    continue
                ev = c.get("evidence") or {}
                props = {
                    "relation_type": c.get("relation_type"),
                    "polarity": c.get("polarity"),
                    "confidence": float(c.get("confidence") or 0.0),
                    "source_file": source_file,
                    "chunk_id": c.get("chunk_id"),
                    "page_idx": ev.get("page_idx"),
                    "source_relation_local_id": c.get("source_relation_local_id"),
                    "updated_at": int(time.time() * 1000),
                }
                props = {k: v for k, v in props.items() if v is not None}
                s.run(
                    """
                    MERGE (c:RelationClaim {claim_id: $claim_id})
                    ON CREATE SET c.created_at = timestamp()
                    SET c += $props
                    WITH c
                    MATCH (r:Relation {relation_id: $relation_id})
                    MERGE (c)-[:CLAIMS]->(r)
                    """,
                    {"claim_id": c["claim_id"], "relation_id": rel_id, "props": props},
                )
                if ev.get("quote") or ev.get("page_idx") is not None:
                    ev_key = _md5(
                        "|".join(
                            [
                                str(source_file),
                                str(ev.get("page_idx")),
                                str(ev.get("quote") or ""),
                            ]
                        )
                    )
                    evidence_id = f"evidence-{ev_key}"
                    s.run(
                        """
                        MERGE (e:Evidence {evidence_id: $evidence_id})
                        ON CREATE SET e.created_at = timestamp()
                        SET e.source_file = $source_file,
                            e.page_idx = $page_idx,
                            e.quote = $quote,
                            e.updated_at = timestamp()
                        WITH e
                        MATCH (c:RelationClaim {claim_id: $claim_id})
                        MERGE (c)-[:SUPPORTED_BY]->(e)
                        """,
                        {
                            "evidence_id": evidence_id,
                            "source_file": source_file,
                            "page_idx": ev.get("page_idx"),
                            "quote": ev.get("quote"),
                            "claim_id": c["claim_id"],
                        },
                    )

        return {
            "entities_written": len(graph.get("entities") or []),
            "mentions_written": len(graph.get("mentions") or []),
            "possible_same_written": len(graph.get("possible_same") or []),
            "relations_written": len(graph.get("relations") or []),
            "claims_written": len(graph.get("relation_claims") or []),
        }
    finally:
        drv.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="General living-entities graph pipeline")
    p.add_argument("--input", required=True, help="content_list file or directory")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument(
        "--prepare-text-for-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-clean extracted text before LLM extraction",
    )
    p.add_argument(
        "--strip-repeated-page-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove repeated header/footer-like lines before LLM extraction",
    )
    p.add_argument(
        "--repeated-line-min-occurrences",
        type=int,
        default=3,
        help="Minimum occurrences of a boundary line to treat it as repeated noise",
    )
    p.add_argument(
        "--repeated-line-min-share",
        type=float,
        default=0.35,
        help="Minimum page share (0..1) for repeated boundary line detection",
    )
    p.add_argument("--pages-per-chunk", type=int, default=1)
    p.add_argument("--max-chars-per-chunk", type=int, default=3000)
    p.add_argument("--max-chunks", type=int, default=None)

    p.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"))
    p.add_argument("--llm-base-url", default=os.getenv("LLM_BINDING_HOST", "http://localhost:11434/v1"))
    p.add_argument("--llm-api-key", default=os.getenv("LLM_BINDING_API_KEY", "ollama"))
    p.add_argument("--llm-timeout", type=int, default=300)
    p.add_argument("--llm-max-tokens", type=int, default=900)

    p.add_argument("--min-relation-confidence", type=float, default=0.0)
    p.add_argument("--auto-merge-threshold", type=float, default=0.85)
    p.add_argument("--possible-merge-threshold", type=float, default=0.65)
    p.add_argument("--llm-merge-agent", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--llm-merge-min-confidence", type=float, default=0.78)
    p.add_argument("--llm-merge-max-candidates", type=int, default=40)
    p.add_argument("--llm-merge-max-tokens", type=int, default=320)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--strict-neo4j",
        action="store_true",
        help="Fail run if Neo4j persistence step errors",
    )
    p.add_argument("--reset-neo4j", action="store_true")
    p.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    p.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--neo4j-db", default=os.getenv("NEO4J_DATABASE", None))
    return p


async def _run(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    content_list_path = _resolve_content_list_path(input_path)
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"./output_living_graph/run_{_now_ts()}")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] parse_to_txt: {content_list_path}")
    pages, source_file = _load_pages(content_list_path, max_pages=args.max_pages)
    pages_raw_count = len(pages)
    print(f"  pages_raw={pages_raw_count} source_file={source_file}")

    prep_stats: Dict[str, Any] = {
        "enabled": False,
        "pages_in": pages_raw_count,
        "pages_out": pages_raw_count,
        "changed_pages": 0,
        "chars_before": sum(len(str(p.text or "")) for p in pages),
        "chars_after": sum(len(str(p.text or "")) for p in pages),
        "removed_first_lines": 0,
        "removed_last_lines": 0,
        "repeated_boundary_signatures": 0,
    }
    print("[2/5] prepare_text_for_llm")
    if args.prepare_text_for_llm:
        pages, prep_stats = _prepare_pages_for_llm(
            pages,
            strip_repeated_page_lines=bool(args.strip_repeated_page_lines),
            repeated_line_min_occurrences=int(args.repeated_line_min_occurrences),
            repeated_line_min_share=float(args.repeated_line_min_share),
        )
        if not pages:
            raise RuntimeError("Text preparation removed all pages; adjust preprocessing settings")
        print(
            "  "
            f"pages_out={prep_stats['pages_out']} "
            f"changed_pages={prep_stats['changed_pages']} "
            f"headers_removed={prep_stats['removed_first_lines']} "
            f"footers_removed={prep_stats['removed_last_lines']} "
            f"chars={prep_stats['chars_before']}->{prep_stats['chars_after']}"
        )
    else:
        print("  skipped")

    _write_stage1_artifacts(out_dir, pages)
    print(f"  pages_for_extraction={len(pages)}")

    print("[3/5] mention_and_relation_extraction")
    chunks = _chunk_pages(
        pages,
        pages_per_chunk=args.pages_per_chunk,
        max_chars=args.max_chars_per_chunk,
        max_chunks=args.max_chunks,
    )
    print(f"  chunks={len(chunks)}")
    parsed_chunks: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks, start=1):
        print(f"  - chunk {idx}/{len(chunks)} pages={ch.page_indices}")
        raw = await _extract_chunk(
            ch,
            model=args.model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
        )
        parsed = _parse_extracted_chunk(ch, raw)
        parsed_chunks.append(parsed)
        print(
            f"    mentions={len(parsed['mentions'])} relations={len(parsed['relations'])} repaired={parsed['repaired']}"
        )
    (out_dir / "raw_extractions_general.json").write_text(
        json.dumps(_serialize(parsed_chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/5] entity_resolution_and_graph_build")
    graph = await _build_graph(
        parsed_chunks,
        min_relation_confidence=float(args.min_relation_confidence),
        auto_merge_threshold=float(args.auto_merge_threshold),
        possible_merge_threshold=float(args.possible_merge_threshold),
        llm_merge_agent=bool(args.llm_merge_agent),
        llm_merge_min_confidence=float(args.llm_merge_min_confidence),
        llm_merge_max_candidates=int(args.llm_merge_max_candidates),
        llm_merge_max_tokens=int(args.llm_merge_max_tokens),
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout=int(args.llm_timeout),
    )
    graph["model"] = args.model
    (out_dir / "mentions_resolved.json").write_text(
        json.dumps(_serialize(graph.get("mentions")), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "living_graph.json").write_text(
        json.dumps(_serialize(graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    view_path = _write_living_graph_view(out_dir, graph, title="Living Graph")
    print(
        f"  entities={len(graph.get('entities') or [])} "
        f"relations={len(graph.get('relations') or [])} "
        f"possible_same={len(graph.get('possible_same') or [])} "
        f"llm_merges={int(graph.get('llm_merge_accepted_total') or 0)}"
    )
    print(f"  web_view={view_path}")

    neo4j_stats: Dict[str, Any] = {"skipped": True}
    print("[5/5] persist_graph")
    if not args.dry_run:
        try:
            neo4j_stats = _persist_graph_to_neo4j(
                graph,
                source_file=source_file,
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
                neo4j_db=args.neo4j_db,
                reset_db=bool(args.reset_neo4j),
            )
            print(f"  neo4j={neo4j_stats}")
        except Exception as e:
            neo4j_stats = {
                "failed": True,
                "error": str(e),
                "uri": args.neo4j_uri,
                "db": args.neo4j_db,
            }
            print(f"  neo4j=failed ({e})")
            if args.strict_neo4j:
                raise
    else:
        print("  neo4j=skipped (dry-run)")

    summary = {
        "input_content_list": str(content_list_path),
        "output_dir": str(out_dir),
        "pages": len(pages),
        "pages_raw": pages_raw_count,
        "pages_prepared": len(pages),
        "text_preparation": prep_stats,
        "chunking": {
            "pages_per_chunk": int(args.pages_per_chunk),
            "max_chars_per_chunk": int(args.max_chars_per_chunk),
            "max_chunks": int(args.max_chunks) if args.max_chunks is not None else None,
        },
        "chunks": len(chunks),
        "mentions_total": len(graph.get("mentions") or []),
        "entities_total": len(graph.get("entities") or []),
        "possible_same_total": len(graph.get("possible_same") or []),
        "relations_total": len(graph.get("relations") or []),
        "relation_claims_total": len(graph.get("relation_claims") or []),
        "llm_merge_agent_enabled": bool(args.llm_merge_agent),
        "llm_merge_reviews_total": len(graph.get("llm_merge_reviews") or []),
        "llm_merge_accepted_total": int(graph.get("llm_merge_accepted_total") or 0),
        "neo4j": neo4j_stats,
        "model": args.model,
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(_serialize(summary), ensure_ascii=False, indent=2),
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
