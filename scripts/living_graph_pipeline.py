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

NAME_DECORATOR_TOKENS = {
    "государь",
    "государыня",
    "император",
    "императрица",
    "царь",
    "царевна",
    "царевич",
    "король",
    "королева",
    "князь",
    "княгиня",
    "княжна",
    "великий",
    "великая",
    "великие",
    "граф",
    "герцог",
    "принц",
    "принцесса",
    "mr",
    "mrs",
    "ms",
    "sir",
    "lord",
    "lady",
    "king",
    "queen",
    "tsar",
    "emperor",
    "empress",
    "цесаревич",
    "цесаревна",
}

NAME_EPITHET_TOKENS = {
    "освободитель",
    "миротворец",
    "благословенный",
    "тишайший",
    "грозный",
    "великий",
    "великая",
    "великие",
    "the",
    "great",
    "peacekeeper",
    "liberator",
}

GENDER_MALE_TOKENS = {
    "m",
    "male",
    "man",
    "masculine",
    "м",
    "муж",
    "мужской",
    "мужчина",
    "мужчинам",
}

GENDER_FEMALE_TOKENS = {
    "f",
    "female",
    "woman",
    "feminine",
    "ж",
    "жен",
    "женский",
    "женщина",
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


def _canonical_gender(value: Any) -> Optional[str]:
    raw = normalize_name(str(value or ""))
    if not raw:
        return None
    token = raw.split(" ", 1)[0]
    if token in GENDER_MALE_TOKENS:
        return "male"
    if token in GENDER_FEMALE_TOKENS:
        return "female"
    return raw


CONTEXT_TITLE_DECORATORS = (
    "цесаревич",
    "царевич",
    "цесаревна",
    "царевна",
)


def _contextual_title_for_single_token_name(name: str, text: str) -> Optional[str]:
    nm = normalize_name(str(name or ""))
    if not nm:
        return None
    tokens = [t for t in nm.split() if t]
    if len(tokens) != 1:
        return None
    tnorm = normalize_name(str(text or ""))
    if not tnorm:
        return None
    base = tokens[0]
    base_stem = _soft_stem_token(base)
    for dec in CONTEXT_TITLE_DECORATORS:
        pat = rf"(?<!\w){re.escape(dec)}(?:[а-яёa-z]+)?\s+([а-яёa-z]+)(?!\w)"
        for m in re.finditer(pat, tnorm):
            cand = str(m.group(1) or "")
            if _soft_stem_token(cand) == base_stem:
                return dec
    return None


def _apply_contextual_title_to_mention_name(name: str, text: str) -> str:
    base = _normalize_surface_name(name)
    if not base:
        return base
    if _normalize_surface_name(base).split(" ", 1)[0].lower() in CONTEXT_TITLE_DECORATORS:
        return base
    dec = _contextual_title_for_single_token_name(base, text)
    if not dec:
        return base
    return _normalize_surface_name(f"{dec} {base}")


def _normalize_name_core_for_merge(name: str, *, drop_roman: bool) -> str:
    s = _normalize_surface_name(name).lower()
    if not s:
        return ""
    s = re.sub(r"[\"'«»“”„‟‘’`]", " ", s)
    s = re.sub(r"[()\[\]{}.,;:!?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split(" ") if t]
    if not tokens:
        return ""

    while tokens and tokens[0] in NAME_DECORATOR_TOKENS:
        tokens = tokens[1:]
    cleaned: List[str] = []
    for t in tokens:
        if t in NAME_EPITHET_TOKENS:
            continue
        if drop_roman and re.fullmatch(r"[ivxlcdm]+", t, flags=re.IGNORECASE):
            continue
        cleaned.append(t)
    return normalize_name(" ".join(cleaned))


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


def _build_mentions_extraction_prompt(chunk: ChunkItem) -> str:
    return f"""
Извлеки только упоминания живых сущностей.

Правила:
- Возвращай ТОЛЬКО JSON.
- Извлекай только живые сущности: люди, животные, группы живых существ.
- Не выдумывай.
- Не извлекай отношения на этом шаге.

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
  ]
}}

Текст (страницы {chunk.page_indices}):
{chunk.text}
""".strip()


def _normalize_mentions_for_extraction(mentions_raw: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(mentions_raw, list):
        return rows

    for m in mentions_raw:
        if not isinstance(m, dict):
            continue
        name = _extract_name(m.get("name") or m.get("entity"))
        if not name:
            continue
        name = _normalize_surface_name(name)
        if not _is_valid_living_name(name):
            continue

        aliases_val = m.get("aliases") or []
        if isinstance(aliases_val, str):
            aliases = [_normalize_surface_name(x) for x in aliases_val.split(",") if x.strip()]
        elif isinstance(aliases_val, list):
            aliases = [_normalize_surface_name(str(x)) for x in aliases_val if str(x).strip()]
        else:
            aliases = []
        aliases = [a for a in aliases if _is_valid_living_name(a)]

        rows.append(
            {
                "id": "",  # assigned below
                "name": name,
                "entity_type": _canonical_entity_type(m.get("entity_type"), name, m.get("species")),
                "species": m.get("species"),
                "aliases": sorted(set(aliases)),
                "birth_year": coerce_year(m.get("birth_year")),
                "death_year": coerce_year(m.get("death_year")),
                "birth_place": m.get("birth_place"),
                "death_place": m.get("death_place"),
                "gender": _canonical_gender(m.get("gender")),
                "occupation": m.get("occupation"),
                "description": m.get("description") or m.get("biography"),
                "confidence": _safe_float(m.get("confidence"), 0.5),
            }
        )

    # Reassign stable local ids for downstream relation linking.
    for i, row in enumerate(rows, start=1):
        row["id"] = f"m{i}"
    return rows


def _text_paragraphs_for_candidates(text: str) -> List[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if paras:
        return paras
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    return lines


_SOFT_STEM_SUFFIXES_RU = (
    "иями",
    "ями",
    "ами",
    "иях",
    "ах",
    "ях",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ов",
    "ев",
    "ой",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ом",
    "ем",
    "ам",
    "ям",
    "ую",
    "юю",
    "ея",
    "ей",
    "ия",
    "а",
    "я",
    "е",
    "и",
    "ы",
    "у",
    "ю",
    "й",
)

KINSHIP_PARENT_ROLE_KEYWORDS = (
    "сын",
    "сыном",
    "сына",
    "сынов",
    "сыновья",
    "сыновей",
    "дочь",
    "дочерью",
    "дочери",
    "дочерей",
    "дочер",
    "дети",
    "детей",
    "ребенок",
    "ребёнок",
    "родил",
    "родила",
    "родился",
    "родилась",
    "родилось",
    "родились",
    "child",
    "children",
)

KINSHIP_GRANDCHILD_ROLE_KEYWORDS = (
    "внук",
    "внука",
    "внуком",
    "внучка",
    "внучки",
    "внучкой",
    "правнук",
    "правнучка",
)

KINSHIP_SPOUSE_ROLE_KEYWORDS = (
    "супруг",
    "супруга",
    "жена",
    "муж",
    "женат",
    "замуж",
    "брак",
    "женился",
)

KINSHIP_PARENT_FORWARD_HINTS = (
    "имел",
    "имела",
    "родил",
    "родила",
    "родились",
    "родилось",
    "дети",
    "детей",
)

KINSHIP_FORWARD_CONTEXT_KEYWORDS = (
    "у пары",
    "у супруг",
    "в браке",
    "родилось",
    "родились",
    "родила",
    "родил",
    "имел",
    "имела",
    "дети",
    "детей",
)

KINSHIP_COLLATERAL_KEYWORDS = (
    "племянник",
    "племянница",
    "племянниц",
    "дядя",
    "тетя",
    "тётя",
    "брат",
    "сестра",
    "кузен",
    "кузина",
    "nephew",
    "niece",
    "uncle",
    "aunt",
    "brother",
    "sister",
    "cousin",
)


def _soft_stem_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if len(t) <= 3:
        return t
    for suffix in _SOFT_STEM_SUFFIXES_RU:
        if t.endswith(suffix) and len(t) - len(suffix) >= 3:
            return t[: -len(suffix)]
    return t


def _soft_stem_text(text: str) -> str:
    tokens = [_soft_stem_token(t) for t in normalize_name(str(text or "")).split() if t]
    return " ".join(t for t in tokens if t)


def _single_token_case_variants(token: str) -> set[str]:
    t = normalize_name(str(token or ""))
    if not t:
        return set()
    out = {t}
    if len(t) < 4:
        return out
    if t.endswith("ей"):
        out.add(f"{t[:-2]}ея")
    elif t.endswith("ий"):
        out.add(f"{t[:-2]}ия")
    elif t.endswith("ел"):
        # Павел -> Павла
        out.add(f"{t[:-2]}ла")
    elif t.endswith("а"):
        out.add(f"{t[:-1]}ы")
    elif t.endswith("я"):
        out.add(f"{t[:-1]}и")
    elif re.fullmatch(r"[а-яёa-z]{4,}", t):
        out.add(f"{t}а")
    return out


def _mention_search_forms(mention: Dict[str, Any]) -> List[str]:
    forms: set[str] = set()
    name = str(mention.get("name") or "").strip()
    name_roman = _extract_roman_numeral(name) if name else None
    if name:
        forms.update(_name_match_keys(name))
    for a in mention.get("aliases") or []:
        forms.update(_name_match_keys(str(a)))

    # Extra robust forms for Russian case variants and names with regnal numbers.
    extra: set[str] = set()
    for f in list(forms):
        tokens = [t for t in normalize_name(f).split() if t]
        if len(tokens) >= 2:
            stem = _soft_stem_text(f)
            if stem and stem != f:
                extra.add(stem)
            extra.add(" ".join(tokens[:2]))
        elif len(tokens) == 1:
            extra.update(_single_token_case_variants(tokens[0]))
        roman = _extract_roman_numeral(f)
        if roman and tokens:
            extra.add(normalize_name(f"{tokens[0]} {roman}"))
            for v in _single_token_case_variants(tokens[0]):
                extra.add(normalize_name(f"{v} {roman}"))
    forms.update(x for x in extra if x)

    if name_roman:
        # For regnal names keep numeral-aware forms to avoid "Петр I" matching "Петр II".
        filtered_regnal: set[str] = set()
        for f in forms:
            token_count = len([t for t in normalize_name(f).split() if t])
            if token_count <= 1 and _extract_roman_numeral(f) is None:
                continue
            filtered_regnal.add(f)
        if filtered_regnal:
            forms = filtered_regnal

    # Avoid noisy single-letter matches.
    return sorted(f for f in forms if len(f) >= 2)


def _first_keyword_position(norm_text: str, keywords: Sequence[str]) -> Tuple[Optional[int], Optional[str]]:
    s = normalize_name(norm_text)
    best_pos: Optional[int] = None
    best_kw: Optional[str] = None
    for kw in keywords:
        k = normalize_name(str(kw))
        if not k:
            continue
        pos = s.find(k)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_kw = k
    return best_pos, best_kw


def _find_mention_positions_in_span(
    span_norm: str,
    mention_forms_by_id: Dict[str, List[str]],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not span_norm:
        return out
    span_soft = _soft_stem_text(span_norm)
    for mid, forms in mention_forms_by_id.items():
        best: Optional[int] = None
        for f in forms:
            ff = normalize_name(f)
            if not ff:
                continue
            match = re.search(rf"(?<!\w){re.escape(ff)}(?!\w)", span_norm)
            pos = match.start() if match else -1
            if pos >= 0 and (best is None or pos < best):
                best = pos
        if best is None and span_soft:
            for f in forms:
                if len([t for t in normalize_name(f).split() if t]) <= 1:
                    continue
                fs = _soft_stem_text(f)
                if not fs:
                    continue
                match = re.search(rf"(?<!\w){re.escape(fs)}(?!\w)", span_soft)
                pos = match.start() if match else -1
                if pos >= 0 and (best is None or pos < best):
                    best = pos
        if best is not None:
            out[mid] = best
    return out


def _name_position_in_quote(name: str, quote_norm: str) -> Optional[int]:
    if not name or not quote_norm:
        return None
    mention_stub = {"name": name, "aliases": []}
    forms = _mention_search_forms(mention_stub)
    best: Optional[int] = None
    for f in forms:
        ff = normalize_name(f)
        if len(ff) < 3:
            continue
        match = re.search(rf"(?<!\w){re.escape(ff)}(?!\w)", quote_norm)
        if not match:
            continue
        pos = int(match.start())
        if best is None or pos < best:
            best = pos
    if best is not None:
        return best

    # Fallback for Russian inflection/cases: compare soft stems on multi-token forms.
    quote_soft = _soft_stem_text(quote_norm)
    if quote_soft:
        for f in forms:
            ff = normalize_name(f)
            tokens = [t for t in ff.split() if t]
            has_roman = _extract_roman_numeral(ff) is not None
            if len(tokens) < 2 and not has_roman:
                continue
            fs = _soft_stem_text(ff)
            if len(fs) < 3:
                continue
            match = re.search(rf"(?<!\w){re.escape(fs)}(?!\w)", quote_soft)
            if not match:
                continue
            pos = int(match.start())
            if best is None or pos < best:
                best = pos
    return best


def _apply_parent_child_guardrails(relations: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    adjusted = 0
    out: List[Dict[str, Any]] = []
    for rel in relations:
        allow_grandchild_parent_orientation = False
        if not isinstance(rel, dict):
            continue
        row = dict(rel)
        rtype = _canonical_relation_type(row.get("type") or row.get("relation_type"))
        row["type"] = rtype
        if rtype == "parent_child":
            evidence = row.get("evidence")
            quote = ""
            if isinstance(evidence, dict):
                quote = str(evidence.get("quote") or "")
            quote_norm = normalize_name(quote)
            src = row.get("source") if isinstance(row.get("source"), dict) else {}
            dst = row.get("target") if isinstance(row.get("target"), dict) else {}
            src_pos = _name_position_in_quote(str(src.get("name") or ""), quote_norm)
            dst_pos = _name_position_in_quote(str(dst.get("name") or ""), quote_norm)
            parent_pos, _ = _first_keyword_position(quote_norm, KINSHIP_PARENT_ROLE_KEYWORDS)
            has_spouse_kw = any(k in quote_norm for k in KINSHIP_SPOUSE_ROLE_KEYWORDS)
            has_grandchild_kw = any(k in quote_norm for k in KINSHIP_GRANDCHILD_ROLE_KEYWORDS)
            grand_pos, _ = _first_keyword_position(quote_norm, KINSHIP_GRANDCHILD_ROLE_KEYWORDS)
            has_parent_kw = parent_pos is not None
            has_any_endpoint_mention = src_pos is not None or dst_pos is not None

            if not has_any_endpoint_mention:
                # If neither endpoint is present in quote, relation likely leaked from context.
                row["type"] = "relative"
                row["confidence"] = min(0.86, _safe_float(row.get("confidence"), 0.5))
                adjusted += 1
                out.append(row)
                continue

            if has_grandchild_kw:
                # Keep parent-child only when explicit child-of fragment supports this pair,
                # e.g. "..., сын царевича Алексея" where parent is after parent keyword.
                valid_grandchild_parent = False
                if parent_pos is not None:
                    checks = ((src_pos, dst_pos), (dst_pos, src_pos))
                    for parent_side, other_side in checks:
                        if parent_side is None or parent_side <= parent_pos:
                            continue
                        if other_side is None:
                            valid_grandchild_parent = True
                            break
                        if grand_pos is not None and other_side < grand_pos:
                            valid_grandchild_parent = True
                            break
                if valid_grandchild_parent:
                    allow_grandchild_parent_orientation = True
                else:
                    row["type"] = "relative"
                    row["confidence"] = min(0.9, _safe_float(row.get("confidence"), 0.5))
                    adjusted += 1
                    out.append(row)
                    continue

            if not has_parent_kw:
                # No explicit parenthood cue in evidence: keep weakly as relative/spouse.
                row["type"] = "spouse" if has_spouse_kw else "relative"
                row["confidence"] = min(0.9, _safe_float(row.get("confidence"), 0.5))
                adjusted += 1
                out.append(row)
                continue

            # If both entities are on the same side of child-marker keyword,
            # it is typically a sibling list or two-parent mention, not parent->child.
            if parent_pos is not None and src_pos is not None and dst_pos is not None:
                src_after = src_pos > parent_pos
                dst_after = dst_pos > parent_pos
                if src_after == dst_after:
                    row["type"] = "spouse" if has_spouse_kw else "relative"
                    row["confidence"] = min(0.92, _safe_float(row.get("confidence"), 0.5))
                    adjusted += 1
                    out.append(row)
                    continue

            # Collateral kinship terms (niece/nephew/uncle/aunt/sibling) frequently
            # produce false parent-child from weak local models.
            has_collateral = any(k in quote_norm for k in KINSHIP_COLLATERAL_KEYWORDS)
            if has_collateral:
                valid_parent_signal = (
                    parent_pos is not None
                    and src_pos is not None
                    and dst_pos is not None
                    and dst_pos < parent_pos < src_pos
                )
                if not valid_parent_signal:
                    row["type"] = "relative"
                    row["confidence"] = min(0.92, _safe_float(row.get("confidence"), 0.5))
                    adjusted += 1
                    out.append(row)
                    continue

        evidence = row.get("evidence")
        quote = ""
        if isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "")
        quote_norm = normalize_name(quote)
        has_grandchild = any(k in quote_norm for k in KINSHIP_GRANDCHILD_ROLE_KEYWORDS)
        parent_pos, _ = _first_keyword_position(quote_norm, KINSHIP_PARENT_ROLE_KEYWORDS)
        if parent_pos is not None and (not has_grandchild or allow_grandchild_parent_orientation):
            src = row.get("source") if isinstance(row.get("source"), dict) else {}
            dst = row.get("target") if isinstance(row.get("target"), dict) else {}
            src_pos = _name_position_in_quote(str(src.get("name") or ""), quote_norm)
            dst_pos = _name_position_in_quote(str(dst.get("name") or ""), quote_norm)
            forward_context = any(k in quote_norm for k in KINSHIP_FORWARD_CONTEXT_KEYWORDS)
            has_collateral = any(k in quote_norm for k in KINSHIP_COLLATERAL_KEYWORDS)
            has_spouse_kw = any(k in quote_norm for k in KINSHIP_SPOUSE_ROLE_KEYWORDS)
            parent_side: Optional[str] = None
            if (src_pos is not None or dst_pos is not None) and not has_collateral and not has_spouse_kw:
                if src_pos is not None and dst_pos is not None:
                    src_before = src_pos < parent_pos
                    dst_before = dst_pos < parent_pos
                    src_after = src_pos > parent_pos
                    dst_after = dst_pos > parent_pos
                    if forward_context:
                        # "У пары ... сын X": parents before marker, children after marker.
                        if src_before and dst_after:
                            parent_side = "source"
                        elif dst_before and src_after:
                            parent_side = "target"
                    else:
                        # "X — сын Y": children before marker, parents after marker.
                        if src_after and dst_before:
                            parent_side = "source"
                        elif dst_after and src_before:
                            parent_side = "target"
                elif src_pos is not None:
                    if forward_context:
                        parent_side = "source" if src_pos < parent_pos else "target"
                    else:
                        parent_side = "source" if src_pos > parent_pos else "target"
                elif dst_pos is not None:
                    if forward_context:
                        parent_side = "target" if dst_pos < parent_pos else "source"
                    else:
                        parent_side = "target" if dst_pos > parent_pos else "source"

            if parent_side in {"source", "target"}:
                prev_type = _canonical_relation_type(row.get("type") or row.get("relation_type"))
                prev_src = row.get("source")
                prev_dst = row.get("target")
                prev_conf = _safe_float(row.get("confidence"), 0.5)
                row["type"] = "parent_child"
                if parent_side == "target":
                    row["source"] = dict(dst)
                    row["target"] = dict(src)
                infer_floor = 0.86 if (src_pos is not None and dst_pos is not None) else 0.8
                row["confidence"] = max(_safe_float(row.get("confidence"), 0.5), infer_floor)
                if row["type"] != prev_type or row.get("source") != prev_src or row.get("target") != prev_dst or _safe_float(row.get("confidence"), 0.5) != prev_conf:
                    adjusted += 1
        out.append(row)
    return out, adjusted


def _build_rule_based_kinship_relations(
    mentions: Sequence[Dict[str, Any]],
    text: str,
    *,
    page_idx: Optional[int],
) -> List[Dict[str, Any]]:
    mention_by_id: Dict[str, Dict[str, Any]] = {}
    forms_by_id: Dict[str, List[str]] = {}
    for m in mentions:
        mid = str(m.get("id") or m.get("mention_id") or "").strip()
        if not mid:
            continue
        forms = _mention_search_forms(m)
        if not forms:
            continue
        mention_by_id[mid] = m
        forms_by_id[mid] = forms
    if len(mention_by_id) < 2:
        return []

    # Drop ambiguous one-token forms shared by multiple mentions ("петр", "алексей", ...).
    form_counts: Dict[str, int] = {}
    for forms in forms_by_id.values():
        for f in set(forms):
            form_counts[f] = int(form_counts.get(f, 0)) + 1
    for mid, forms in list(forms_by_id.items()):
        filtered: List[str] = []
        for f in forms:
            token_count = len([t for t in normalize_name(f).split() if t])
            if token_count <= 1 and int(form_counts.get(f, 0)) > 1:
                continue
            filtered.append(f)
        forms_by_id[mid] = filtered or forms

    out: List[Dict[str, Any]] = []
    last_subject_id: Optional[str] = None

    for span in _text_spans_for_relation_hints(text):
        span_norm = normalize_name(span)
        if not span_norm:
            continue
        prev_subject_id = last_subject_id
        pos_map = _find_mention_positions_in_span(span_norm, forms_by_id)
        current_subject_id: Optional[str] = None
        if pos_map:
            ordered = sorted(pos_map.items(), key=lambda x: x[1])
            current_subject_id = ordered[0][0]
        ordered_before_all = sorted(pos_map.items(), key=lambda x: x[1])

        parent_pos, parent_kw = _first_keyword_position(span_norm, KINSHIP_PARENT_ROLE_KEYWORDS)
        if parent_pos is not None and parent_kw:
            before = [x for x in ordered_before_all if x[1] < parent_pos]
            after = [x for x in ordered_before_all if x[1] > parent_pos]
            parent_id: Optional[str] = None
            child_id: Optional[str] = None
            prefix = span_norm[max(0, parent_pos - 28) : parent_pos]
            is_forward_parenting = parent_kw.startswith("дет") or any(h in prefix for h in KINSHIP_PARENT_FORWARD_HINTS)
            has_grandchild_context = any(k in span_norm for k in KINSHIP_GRANDCHILD_ROLE_KEYWORDS)
            if is_forward_parenting:
                parent_id = before[-1][0] if before else None
                child_id = after[0][0] if after else None
            else:
                child_id = before[0][0] if before else None
                parent_id = after[0][0] if after else None
                if has_grandchild_context and prev_subject_id and prev_subject_id != parent_id:
                    child_id = prev_subject_id
                elif not child_id and prev_subject_id and prev_subject_id != parent_id:
                    child_id = prev_subject_id
            if parent_id and child_id and parent_id != child_id:
                out.append(
                    {
                        "type": "parent_child",
                        "source": {"id": parent_id, "name": mention_by_id[parent_id].get("name")},
                        "target": {"id": child_id, "name": mention_by_id[child_id].get("name")},
                        "polarity": "positive",
                        "confidence": 0.98,
                        "evidence": {"quote": span[:320], "page_idx": page_idx},
                    }
                )

        grand_pos, _ = _first_keyword_position(span_norm, KINSHIP_GRANDCHILD_ROLE_KEYWORDS)
        if grand_pos is not None:
            before = [x for x in ordered_before_all if x[1] < grand_pos]
            after = [x for x in ordered_before_all if x[1] > grand_pos]
            child_id = before[0][0] if before else prev_subject_id
            grandparent_id = after[0][0] if after else None
            if child_id and grandparent_id and child_id != grandparent_id:
                out.append(
                    {
                        "type": "relative",
                        "source": {"id": child_id, "name": mention_by_id[child_id].get("name")},
                        "target": {"id": grandparent_id, "name": mention_by_id[grandparent_id].get("name")},
                        "polarity": "neutral",
                        "confidence": 0.94,
                        "evidence": {"quote": span[:320], "page_idx": page_idx},
                    }
                )

        spouse_pos, _ = _first_keyword_position(span_norm, KINSHIP_SPOUSE_ROLE_KEYWORDS)
        if spouse_pos is not None and len(ordered_before_all) >= 2:
            before = [x for x in ordered_before_all if x[1] < spouse_pos]
            after = [x for x in ordered_before_all if x[1] > spouse_pos]
            left_id = before[-1][0] if before else ordered_before_all[0][0]
            right_id = after[0][0] if after else ordered_before_all[1][0]
            if left_id and right_id and left_id != right_id:
                out.append(
                    {
                        "type": "spouse",
                        "source": {"id": left_id, "name": mention_by_id[left_id].get("name")},
                        "target": {"id": right_id, "name": mention_by_id[right_id].get("name")},
                        "polarity": "positive",
                        "confidence": 0.88,
                        "evidence": {"quote": span[:320], "page_idx": page_idx},
                    }
                )

        if current_subject_id:
            last_subject_id = current_subject_id

    return _dedup_raw_relations(out)


KINSHIP_RELATION_HINT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "parent_child": (
        "родител",
        "отец",
        "мать",
        "мама",
        "папа",
        "сын",
        "дочь",
        "дети",
        "family",
        "parent",
        "child",
        "father",
        "mother",
        "son",
        "daughter",
    ),
    "spouse": (
        "супруг",
        "супруга",
        "жена",
        "муж",
        "брак",
        "женат",
        "замуж",
        "married",
        "spouse",
        "wife",
        "husband",
    ),
    "sibling": (
        "брат",
        "сестра",
        "siblings",
        "sibling",
        "brother",
        "sister",
    ),
    "relative": (
        "родствен",
        "родня",
        "семья",
        "family",
        "relative",
        "kin",
    ),
}


def _candidate_pair_key_from_ids(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _candidate_pair_key(pair: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    a = str(pair.get("source_id") or "").strip()
    b = str(pair.get("target_id") or "").strip()
    if not a or not b:
        return None
    return _candidate_pair_key_from_ids(a, b)


def _relation_pair_key(rel: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    source = rel.get("source")
    target = rel.get("target")
    s_id = ""
    t_id = ""
    if isinstance(source, dict):
        s_id = str(source.get("id") or source.get("mention_id") or "").strip()
    if isinstance(target, dict):
        t_id = str(target.get("id") or target.get("mention_id") or "").strip()
    if not s_id or not t_id:
        return None
    return _candidate_pair_key_from_ids(s_id, t_id)


def _merge_candidate_pairs(
    primary: Sequence[Dict[str, Any]],
    secondary: Sequence[Dict[str, Any]],
    *,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in [*primary, *secondary]:
        if not isinstance(row, dict):
            continue
        key = _candidate_pair_key(row)
        if not key:
            continue
        score = _safe_float(row.get("score"), 0.0)
        dist = row.get("min_paragraph_distance")
        hints = set(str(x).strip() for x in (row.get("relation_hints") or []) if str(x).strip())
        prev = rows.get(key)
        if prev is None:
            rows[key] = {
                "source_id": row.get("source_id"),
                "target_id": row.get("target_id"),
                "source_name": row.get("source_name"),
                "target_name": row.get("target_name"),
                "min_paragraph_distance": dist,
                "score": score,
                "relation_hints": sorted(hints),
            }
            continue
        prev_hints = set(str(x).strip() for x in (prev.get("relation_hints") or []) if str(x).strip())
        prev_hints.update(hints)
        prev["relation_hints"] = sorted(prev_hints)
        if score > _safe_float(prev.get("score"), 0.0):
            prev["score"] = score
            if row.get("source_name"):
                prev["source_name"] = row.get("source_name")
            if row.get("target_name"):
                prev["target_name"] = row.get("target_name")
        prev_dist = prev.get("min_paragraph_distance")
        if isinstance(dist, int):
            if not isinstance(prev_dist, int) or dist < prev_dist:
                prev["min_paragraph_distance"] = dist

    ranked = sorted(
        rows.values(),
        key=lambda x: (_safe_float(x.get("score"), 0.0), len(x.get("relation_hints") or [])),
        reverse=True,
    )
    lim = max(0, int(max_candidates))
    return ranked if lim <= 0 else ranked[:lim]


def _text_spans_for_relation_hints(text: str) -> List[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    spans = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw) if s.strip()]
    if spans:
        return spans
    return _text_paragraphs_for_candidates(text)


def _build_kinship_relation_candidates(
    mentions: Sequence[Dict[str, Any]],
    text: str,
    *,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    if len(mentions) < 2:
        return []

    mention_by_id: Dict[str, Dict[str, Any]] = {}
    forms_by_id: Dict[str, List[str]] = {}
    for m in mentions:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        forms = _mention_search_forms(m)
        if not forms:
            continue
        mention_by_id[mid] = m
        forms_by_id[mid] = forms
    if len(forms_by_id) < 2:
        return []

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for span in _text_spans_for_relation_hints(text):
        span_norm = normalize_name(span)
        if not span_norm:
            continue
        rel_hints = {
            rel_type
            for rel_type, kws in KINSHIP_RELATION_HINT_KEYWORDS.items()
            if any(k in span_norm for k in kws)
        }
        if not rel_hints:
            continue

        present_ids: List[str] = []
        for mid, forms in forms_by_id.items():
            if any(f in span_norm for f in forms):
                present_ids.append(mid)
        if len(present_ids) < 2:
            continue

        for i in range(len(present_ids)):
            for j in range(i + 1, len(present_ids)):
                a_id = present_ids[i]
                b_id = present_ids[j]
                key = _candidate_pair_key_from_ids(a_id, b_id)
                a_conf = _safe_float(mention_by_id[a_id].get("confidence"), 0.5)
                b_conf = _safe_float(mention_by_id[b_id].get("confidence"), 0.5)
                score = 1.25 + a_conf + b_conf + (0.1 * len(rel_hints))
                prev = rows.get(key)
                if prev is None:
                    rows[key] = {
                        "source_id": a_id,
                        "target_id": b_id,
                        "source_name": mention_by_id[a_id].get("name"),
                        "target_name": mention_by_id[b_id].get("name"),
                        "min_paragraph_distance": 0,
                        "score": score,
                        "relation_hints": sorted(rel_hints),
                        "hint_span": span[:220],
                    }
                    continue
                prev_hints = set(prev.get("relation_hints") or [])
                prev_hints.update(rel_hints)
                prev["relation_hints"] = sorted(prev_hints)
                if score > _safe_float(prev.get("score"), 0.0):
                    prev["score"] = score
                    prev["hint_span"] = span[:220]

    ranked = sorted(
        rows.values(),
        key=lambda x: (_safe_float(x.get("score"), 0.0), len(x.get("relation_hints") or [])),
        reverse=True,
    )
    lim = max(0, int(max_candidates))
    return ranked if lim <= 0 else ranked[:lim]


def _dedup_raw_relations(relations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rtype = _canonical_relation_type(rel.get("type") or rel.get("relation_type"))
        source = rel.get("source")
        target = rel.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            continue
        s_id = str(source.get("id") or source.get("mention_id") or "").strip()
        t_id = str(target.get("id") or target.get("mention_id") or "").strip()
        s_name = normalize_name(_normalize_surface_name(_extract_name(source.get("name") or "")))
        t_name = normalize_name(_normalize_surface_name(_extract_name(target.get("name") or "")))
        if not (s_id or s_name) or not (t_id or t_name):
            continue
        left = s_id or s_name
        right = t_id or t_name
        if rtype in SYMMETRIC_RELATIONS and str(left) > str(right):
            left, right = right, left
            source, target = target, source
            rel = dict(rel)
            rel["source"] = source
            rel["target"] = target
        key = (rtype, left, right)
        conf = _safe_float(rel.get("confidence"), 0.5)
        prev = dedup.get(key)
        if prev is None or conf > _safe_float(prev.get("confidence"), 0.5):
            dedup[key] = rel
            continue
        prev_ev = prev.get("evidence")
        cur_ev = rel.get("evidence")
        if not prev_ev and cur_ev:
            prev["evidence"] = cur_ev
    return list(dedup.values())


def _build_relation_candidates(
    mentions: Sequence[Dict[str, Any]],
    text: str,
    *,
    paragraph_window: int,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    if len(mentions) < 2:
        return []

    paras = _text_paragraphs_for_candidates(text)
    para_norm = [normalize_name(p) for p in paras]
    hits: Dict[str, set[int]] = {}
    for m in mentions:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        forms = _mention_search_forms(m)
        pos: set[int] = set()
        if forms and para_norm:
            for i, p in enumerate(para_norm):
                if not p:
                    continue
                if any(f in p for f in forms):
                    pos.add(i)
        hits[mid] = pos

    window = max(0, int(paragraph_window))
    rows: List[Tuple[float, Dict[str, Any]]] = []
    n = len(mentions)
    for i in range(n):
        a = mentions[i]
        a_id = str(a.get("id") or "")
        for j in range(i + 1, n):
            b = mentions[j]
            b_id = str(b.get("id") or "")
            if not a_id or not b_id:
                continue
            a_hits = hits.get(a_id, set())
            b_hits = hits.get(b_id, set())
            if not a_hits or not b_hits:
                continue
            min_dist = min(abs(x - y) for x in a_hits for y in b_hits)
            if min_dist > window:
                continue
            score = (1.0 / (1.0 + float(min_dist))) + float(a.get("confidence") or 0.0) + float(
                b.get("confidence") or 0.0
            )
            rows.append(
                (
                    score,
                    {
                        "source_id": a_id,
                        "target_id": b_id,
                        "source_name": a.get("name"),
                        "target_name": b.get("name"),
                        "min_paragraph_distance": min_dist,
                        "score": score,
                        "relation_hints": [],
                    },
                )
            )

    fallback: List[Tuple[float, Dict[str, Any]]] = []
    for i in range(n):
        a = mentions[i]
        a_id = str(a.get("id") or "")
        for j in range(i + 1, n):
            b = mentions[j]
            b_id = str(b.get("id") or "")
            if not a_id or not b_id:
                continue
            score = float(a.get("confidence") or 0.0) + float(b.get("confidence") or 0.0)
            fallback.append(
                (
                    score,
                    {
                        "source_id": a_id,
                        "target_id": b_id,
                        "source_name": a.get("name"),
                        "target_name": b.get("name"),
                        "min_paragraph_distance": None,
                        "score": score,
                        "relation_hints": [],
                    },
                )
            )
    # If proximity found nothing, fallback to confidence pairs.
    if not rows:
        rows = fallback
    else:
        lim = max(0, int(max_candidates))
        if lim == 0 or len(rows) < lim:
            existing = {(_candidate_pair_key(row) or ("", "")) for _, row in rows}
            for score, row in fallback:
                key = _candidate_pair_key(row)
                if not key or key in existing:
                    continue
                rows.append((score, row))
                existing.add(key)

    rows.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    lim = max(0, int(max_candidates))
    for _, row in rows:
        a = str(row.get("source_id") or "")
        b = str(row.get("target_id") or "")
        key = tuple(sorted((a, b)))
        if not a or not b or key in seen:
            continue
        seen.add(key)
        out.append(row)
        if lim and len(out) >= lim:
            break
    return out


def _build_relations_extraction_prompt(
    chunk: ChunkItem,
    mentions: Sequence[Dict[str, Any]],
    candidate_pairs: Sequence[Dict[str, Any]],
) -> str:
    mention_view = [
        {
            "id": m.get("id"),
            "name": m.get("name"),
            "entity_type": m.get("entity_type"),
            "aliases": list(m.get("aliases") or [])[:4],
        }
        for m in mentions
    ]
    pair_view = [
        {
            "source_id": p.get("source_id"),
            "target_id": p.get("target_id"),
            "source_name": p.get("source_name"),
            "target_name": p.get("target_name"),
            "min_paragraph_distance": p.get("min_paragraph_distance"),
            "relation_hints": list(p.get("relation_hints") or []),
        }
        for p in candidate_pairs
    ]
    return f"""
Извлеки только отношения между уже заданными парами сущностей.

Правила:
- Возвращай ТОЛЬКО JSON.
- Рассматривай ТОЛЬКО пары из candidate_pairs.
- Не создавай новые сущности и новые id.
- Если для пары нет явной связи, не добавляй ее в результат.
- Поддерживаемые типы: friend|enemy|partner|spouse|parent_child|owner_pet|ally|rival|mentor|student|sibling|relative|works_with|knows|member_of|leader_of|associated_with

Сущности:
{json.dumps(mention_view, ensure_ascii=False)}

Кандидатные пары:
{json.dumps(pair_view, ensure_ascii=False)}

Схема JSON:
{{
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


def _build_kinship_relations_extraction_prompt(
    chunk: ChunkItem,
    mentions: Sequence[Dict[str, Any]],
    candidate_pairs: Sequence[Dict[str, Any]],
) -> str:
    mention_view = [
        {
            "id": m.get("id"),
            "name": m.get("name"),
            "entity_type": m.get("entity_type"),
            "aliases": list(m.get("aliases") or [])[:4],
        }
        for m in mentions
    ]
    pair_view = [
        {
            "source_id": p.get("source_id"),
            "target_id": p.get("target_id"),
            "source_name": p.get("source_name"),
            "target_name": p.get("target_name"),
            "relation_hints": list(p.get("relation_hints") or []),
            "hint_span": p.get("hint_span"),
        }
        for p in candidate_pairs
    ]
    return f"""
Извлеки только семейные отношения и отношения брака между заданными парами.

Правила:
- Возвращай ТОЛЬКО JSON.
- Рассматривай ТОЛЬКО пары из candidate_pairs.
- Не создавай новые сущности и id.
- Поддерживаемые типы: spouse|parent_child|sibling|relative
- Если связи нет, не добавляй ее.

Сущности:
{json.dumps(mention_view, ensure_ascii=False)}

Кандидатные пары:
{json.dumps(pair_view, ensure_ascii=False)}

Схема JSON:
{{
  "relations": [
    {{
      "type": "spouse|parent_child|sibling|relative",
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


def _build_recall_relations_extraction_prompt(
    chunk: ChunkItem,
    mentions: Sequence[Dict[str, Any]],
    candidate_pairs: Sequence[Dict[str, Any]],
) -> str:
    mention_view = [
        {
            "id": m.get("id"),
            "name": m.get("name"),
            "entity_type": m.get("entity_type"),
            "aliases": list(m.get("aliases") or [])[:4],
        }
        for m in mentions
    ]
    pair_view = [
        {
            "source_id": p.get("source_id"),
            "target_id": p.get("target_id"),
            "source_name": p.get("source_name"),
            "target_name": p.get("target_name"),
            "min_paragraph_distance": p.get("min_paragraph_distance"),
            "relation_hints": list(p.get("relation_hints") or []),
        }
        for p in candidate_pairs
    ]
    return f"""
Извлеки дополнительные отношения в режиме recall для заданных пар.

Правила:
- Возвращай ТОЛЬКО JSON.
- Рассматривай ТОЛЬКО пары из candidate_pairs.
- Не создавай новые сущности и id.
- Разрешенные типы: friend|enemy|partner|owner_pet|ally|rival|mentor|student|works_with|knows|member_of|leader_of|associated_with
- Допустимы слабые связи knows/associated_with при явном совместном упоминании в контексте.
- Если связи нет, не добавляй ее.

Сущности:
{json.dumps(mention_view, ensure_ascii=False)}

Кандидатные пары:
{json.dumps(pair_view, ensure_ascii=False)}

Схема JSON:
{{
  "relations": [
    {{
      "type": "friend|enemy|partner|owner_pet|ally|rival|mentor|student|works_with|knows|member_of|leader_of|associated_with",
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
    top_level_schema: Optional[Dict[str, Any]] = None,
    keep_instruction: str = "Keep only living entities and relation records.",
) -> Optional[Dict[str, Any]]:
    schema = top_level_schema or {"mentions": [], "relations": []}
    prompt = f"""
Normalize to strict JSON with top-level schema:
{json.dumps(schema, ensure_ascii=False)}

{keep_instruction}
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
    relation_candidate_window: int,
    max_relation_candidates: int,
    enable_kinship_marriage_pass: bool,
    kinship_pass_max_candidates: int,
    enable_recall_pass: bool,
    recall_pass_window_extra: int,
    recall_pass_max_candidates: int,
    enable_rule_kinship_pass: bool,
    strict_parent_child_validation: bool,
) -> Dict[str, Any]:
    raw_mentions = await _llm_complete(
        _build_mentions_extraction_prompt(chunk),
        (
            "Ты извлекаешь только упоминания живых сущностей из текста. "
            "Не извлекай отношения на этом шаге. Верни только JSON."
        ),
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    mentions_obj = robust_json_loads(raw_mentions)
    mentions_repaired = False
    if not mentions_obj:
        mentions_repaired = True
        mentions_obj = await _repair_json(
            raw_mentions,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            top_level_schema={"mentions": []},
            keep_instruction="Keep only living entity mention records.",
        )
    mentions_clean = _normalize_mentions_for_extraction(
        (mentions_obj or {}).get("mentions") if isinstance(mentions_obj, dict) else []
    )

    base_candidate_pairs = _build_relation_candidates(
        mentions_clean,
        chunk.text,
        paragraph_window=relation_candidate_window,
        max_candidates=max_relation_candidates,
    )
    kinship_candidate_pairs: List[Dict[str, Any]] = []
    if enable_kinship_marriage_pass and mentions_clean:
        kinship_candidate_pairs = _build_kinship_relation_candidates(
            mentions_clean,
            chunk.text,
            max_candidates=max(0, int(kinship_pass_max_candidates)),
        )
    candidate_pairs = _merge_candidate_pairs(
        base_candidate_pairs,
        kinship_candidate_pairs,
        max_candidates=max_relation_candidates,
    )

    async def _run_relations_pass(
        prompt: str,
        system_prompt: str,
        keep_instruction: str,
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        raw = await _llm_complete(
            prompt,
            system_prompt,
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
                top_level_schema={"relations": []},
                keep_instruction=keep_instruction,
            )
        rows: List[Dict[str, Any]] = []
        if isinstance(obj, dict) and isinstance(obj.get("relations"), list):
            rows = [r for r in (obj.get("relations") or []) if isinstance(r, dict)]
        return rows, raw, repaired

    relations_main: List[Dict[str, Any]] = []
    raw_relations_main = ""
    relations_main_repaired = False
    if mentions_clean and candidate_pairs:
        relations_main, raw_relations_main, relations_main_repaired = await _run_relations_pass(
            _build_relations_extraction_prompt(chunk, mentions_clean, candidate_pairs),
            (
                "Ты извлекаешь только отношения между заданными парами упоминаний. "
                "Не создавай новые id и новые сущности. Верни только JSON."
            ),
            "Keep only relation records between already provided mention ids.",
        )

    relations_kinship: List[Dict[str, Any]] = []
    raw_relations_kinship = ""
    relations_kinship_repaired = False
    if enable_kinship_marriage_pass and mentions_clean and kinship_candidate_pairs:
        kinship_pairs = _merge_candidate_pairs(
            kinship_candidate_pairs,
            [],
            max_candidates=max(0, int(kinship_pass_max_candidates)),
        )
        relations_kinship, raw_relations_kinship, relations_kinship_repaired = await _run_relations_pass(
            _build_kinship_relations_extraction_prompt(chunk, mentions_clean, kinship_pairs),
            (
                "Ты извлекаешь только родственные отношения и брак. "
                "Используй только заданные пары и верни только JSON."
            ),
            "Keep only kinship/marriage relation records between already provided mention ids.",
        )

    covered_pair_keys: set[Tuple[str, str]] = set()
    for rel in [*relations_main, *relations_kinship]:
        key = _relation_pair_key(rel)
        if key:
            covered_pair_keys.add(key)

    recall_candidate_pairs: List[Dict[str, Any]] = []
    if enable_recall_pass and mentions_clean:
        recall_limit = max(0, int(recall_pass_max_candidates))
        recall_pool_limit = max(recall_limit * 2, int(max_relation_candidates) * 2, 0)
        recall_pool = _build_relation_candidates(
            mentions_clean,
            chunk.text,
            paragraph_window=max(0, int(relation_candidate_window) + max(0, int(recall_pass_window_extra))),
            max_candidates=recall_pool_limit,
        )
        seen_pairs: set[Tuple[str, str]] = set()
        for pair in recall_pool:
            key = _candidate_pair_key(pair)
            if not key or key in covered_pair_keys or key in seen_pairs:
                continue
            seen_pairs.add(key)
            recall_candidate_pairs.append(pair)
            if recall_limit > 0 and len(recall_candidate_pairs) >= recall_limit:
                break

    relations_recall: List[Dict[str, Any]] = []
    raw_relations_recall = ""
    relations_recall_repaired = False
    if enable_recall_pass and mentions_clean and recall_candidate_pairs:
        relations_recall, raw_relations_recall, relations_recall_repaired = await _run_relations_pass(
            _build_recall_relations_extraction_prompt(chunk, mentions_clean, recall_candidate_pairs),
            (
                "Ты извлекаешь дополнительные отношения в режиме recall. "
                "Можно вернуть слабые связи knows/associated_with при явной текстовой опоре."
            ),
            "Keep only relation records between already provided mention ids.",
        )

    combined_relations = [*relations_main, *relations_kinship, *relations_recall]
    strict_parent_child_adjusted = 0

    rule_kinship_relations: List[Dict[str, Any]] = []
    if enable_rule_kinship_pass and mentions_clean:
        rule_kinship_relations = _build_rule_based_kinship_relations(
            mentions_clean,
            chunk.text,
            page_idx=chunk.page_indices[0] if chunk.page_indices else None,
        )

    relations_raw = _dedup_raw_relations([*combined_relations, *rule_kinship_relations])
    if strict_parent_child_validation and relations_raw:
        relations_raw, strict_parent_child_adjusted = _apply_parent_child_guardrails(relations_raw)
        relations_raw = _dedup_raw_relations(relations_raw)

    preview = (raw_mentions or "")[:500]
    if raw_relations_main:
        preview += f"\n---RELATIONS_MAIN---\n{(raw_relations_main or '')[:500]}"
    if raw_relations_kinship:
        preview += f"\n---RELATIONS_KINSHIP---\n{(raw_relations_kinship or '')[:500]}"
    if raw_relations_recall:
        preview += f"\n---RELATIONS_RECALL---\n{(raw_relations_recall or '')[:500]}"

    obj = {"mentions": mentions_clean, "relations": relations_raw}
    relations_repaired = bool(
        relations_main_repaired or relations_kinship_repaired or relations_recall_repaired
    )
    return {
        "json": obj,
        "repaired": bool(mentions_repaired or relations_repaired),
        "mentions_repaired": bool(mentions_repaired),
        "relations_repaired": bool(relations_repaired),
        "relations_main_repaired": bool(relations_main_repaired),
        "relations_kinship_repaired": bool(relations_kinship_repaired),
        "relations_recall_repaired": bool(relations_recall_repaired),
        "strict_parent_child_adjusted": int(strict_parent_child_adjusted),
        "rule_kinship_relations": int(len(rule_kinship_relations)),
        "candidate_pairs": len(candidate_pairs),
        "base_candidate_pairs": len(base_candidate_pairs),
        "kinship_candidate_pairs": len(kinship_candidate_pairs),
        "recall_candidate_pairs": len(recall_candidate_pairs),
        "raw_preview": preview,
    }


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
            contextual_name = _apply_contextual_title_to_mention_name(name, chunk.text)
            local_id = str(m.get("id") or m.get("mention_id") or f"m{idx}").strip()
            global_id = f"ch{chunk.chunk_id}:{local_id}"
            local_to_global[local_id] = global_id
            entity_type = _canonical_entity_type(m.get("entity_type"), contextual_name, m.get("species"))
            aliases_val = m.get("aliases") or []
            if isinstance(aliases_val, str):
                aliases = [_normalize_surface_name(x) for x in aliases_val.split(",") if x.strip()]
            elif isinstance(aliases_val, list):
                aliases = [_normalize_surface_name(str(x)) for x in aliases_val if str(x).strip()]
            else:
                aliases = []
            aliases = [a for a in aliases if _is_valid_living_name(a)]
            if contextual_name != name:
                aliases.append(name)
            mention = {
                "mention_id": global_id,
                "local_id": local_id,
                "chunk_id": chunk.chunk_id,
                "page_idx": chunk.page_indices[0] if chunk.page_indices else None,
                "name": contextual_name,
                "normalized_name": normalize_name(contextual_name),
                "entity_type": entity_type,
                "species": m.get("species"),
                "aliases": sorted(set(aliases)),
                "birth_year": coerce_year(m.get("birth_year")),
                "death_year": coerce_year(m.get("death_year")),
                "birth_place": m.get("birth_place"),
                "death_place": m.get("death_place"),
                "gender": _canonical_gender(m.get("gender")),
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
        "mentions_repaired": bool(payload.get("mentions_repaired")),
        "relations_repaired": bool(payload.get("relations_repaired")),
        "relations_main_repaired": bool(payload.get("relations_main_repaired")),
        "relations_kinship_repaired": bool(payload.get("relations_kinship_repaired")),
        "relations_recall_repaired": bool(payload.get("relations_recall_repaired")),
        "strict_parent_child_adjusted": int(payload.get("strict_parent_child_adjusted") or 0),
        "rule_kinship_relations": int(payload.get("rule_kinship_relations") or 0),
        "candidate_pairs": int(payload.get("candidate_pairs") or 0),
        "base_candidate_pairs": int(payload.get("base_candidate_pairs") or 0),
        "kinship_candidate_pairs": int(payload.get("kinship_candidate_pairs") or 0),
        "recall_candidate_pairs": int(payload.get("recall_candidate_pairs") or 0),
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


def _name_identity_token_count(name: str) -> int:
    core = _normalize_name_core_for_merge(name, drop_roman=True)
    base = core or normalize_name(str(name or ""))
    return len([t for t in base.split() if t])


def _tokens_close(a: str, b: str, *, min_ratio: float = 0.84) -> bool:
    aa = normalize_name(str(a or ""))
    bb = normalize_name(str(b or ""))
    if not aa or not bb:
        return False
    return aa == bb or _name_similarity(aa, bb) >= float(min_ratio)


def _name_match_keys(name: str) -> List[str]:
    s = _normalize_surface_name(name)
    if not s:
        return []
    variants = {
        s,
        re.sub(r"\([^)]*\)", " ", s),
        re.sub(r"\s[-–—]\s.*$", "", s),
        s.split(",", 1)[0],
        _normalize_name_core_for_merge(s, drop_roman=False),
        _normalize_name_core_for_merge(s, drop_roman=True),
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

    m_gender = _canonical_gender(mention.get("gender"))
    e_gender = _canonical_gender(entity.get("gender"))
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
    signals = _identity_match_signals(mention, entity)
    if not signals:
        return False

    m_name = str(mention.get("name") or "")
    e_name = str(entity.get("canonical_name") or "")
    min_tokens = min(_name_identity_token_count(m_name), _name_identity_token_count(e_name))
    if min_tokens <= 1:
        strong = {
            "birth_year_match",
            "death_year_match",
            "birth_place_match",
            "death_place_match",
            "occupation_match",
            "species_match",
        }
        if any(s in strong for s in signals):
            return True
        m_roman = _extract_roman_numeral(m_name)
        e_roman = _extract_roman_numeral(e_name)
        if m_roman and e_roman and m_roman == e_roman and "gender_match" in signals:
            return True
        return False

    return True


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

    m_gender = _canonical_gender(mention.get("gender"))
    e_gender = _canonical_gender(entity.get("gender"))
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
        '  "merged_name": "optional",\n'
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
    merged_name = _normalize_surface_name(str(obj.get("merged_name") or "").strip())
    if merged_name and not _is_valid_living_name(merged_name):
        merged_name = ""
    reason = str(obj.get("reason") or obj.get("rationale") or "").strip()
    return {
        "same_entity": same_entity,
        "confidence": confidence,
        "merged_name": merged_name or None,
        "reason": reason[:280] if reason else None,
    }


def _score_possible_same_candidate_pair(
    entity_a: Dict[str, Any],
    entity_b: Dict[str, Any],
) -> Tuple[float, List[str], List[str]]:
    blockers = _entity_merge_blockers(entity_a, entity_b)
    hard_blockers = {
        "entity_type_mismatch",
        "roman_numeral_mismatch",
        "patronymic_mismatch",
        "birth_year_conflict",
        "death_year_conflict",
        "gender_conflict",
        "species_conflict",
        "single_token_ambiguous",
    }
    if any(b in hard_blockers for b in blockers):
        return 0.0, [], blockers

    score = 0.0
    reasons: List[str] = []

    a_name = str(entity_a.get("canonical_name") or "")
    b_name = str(entity_b.get("canonical_name") or "")
    a_norm = normalize_name(str(entity_a.get("normalized_name") or a_name))
    b_norm = normalize_name(str(entity_b.get("normalized_name") or b_name))
    a_core = _normalize_name_core_for_merge(a_name, drop_roman=False)
    b_core = _normalize_name_core_for_merge(b_name, drop_roman=False)
    a_no_roman = _normalize_name_without_roman(a_name)
    b_no_roman = _normalize_name_without_roman(b_name)
    a_nr_tokens = [t for t in a_no_roman.split() if t]
    b_nr_tokens = [t for t in b_no_roman.split() if t]

    if a_norm and b_norm and a_norm == b_norm:
        score += 0.42
        reasons.append("exact_normalized_name")
    if a_core and b_core and a_core == b_core:
        score += 0.35
        reasons.append("same_core_name")
    if a_no_roman and b_no_roman and a_no_roman == b_no_roman:
        score += 0.16
        reasons.append("same_name_without_roman")

    sim_full = _name_similarity(a_norm, b_norm)
    sim_core = _name_similarity(a_core, b_core) if a_core and b_core else 0.0
    sim_nr = _name_similarity(a_no_roman, b_no_roman) if a_no_roman and b_no_roman else 0.0
    best_sim = max(sim_full, sim_core, sim_nr)
    if best_sim >= 0.95:
        score += 0.24
        reasons.append("high_name_similarity")
    elif best_sim >= 0.9:
        score += 0.16
        reasons.append("good_name_similarity")
    elif best_sim >= 0.85:
        score += 0.08
        reasons.append("medium_name_similarity")

    ar = _extract_roman_numeral(a_name)
    br = _extract_roman_numeral(b_name)
    if ar and br and ar == br:
        score += 0.08
        reasons.append("same_roman_numeral")
    elif (ar and not br) or (br and not ar):
        score += 0.03
        reasons.append("partial_roman_match")

    a_by = coerce_year(entity_a.get("birth_year"))
    b_by = coerce_year(entity_b.get("birth_year"))
    same_birth_year = False
    if a_by is not None and b_by is not None:
        if a_by == b_by:
            same_birth_year = True
            score += 0.22
            reasons.append("same_birth_year")
        elif abs(a_by - b_by) <= 2:
            score += 0.08
            reasons.append("near_birth_year")

    a_dy = coerce_year(entity_a.get("death_year"))
    b_dy = coerce_year(entity_b.get("death_year"))
    same_death_year = False
    if a_dy is not None and b_dy is not None:
        if a_dy == b_dy:
            same_death_year = True
            score += 0.08
            reasons.append("same_death_year")
        elif abs(a_dy - b_dy) <= 2:
            score += 0.03
            reasons.append("near_death_year")

    if same_birth_year and same_death_year:
        score += 0.35
        reasons.append("exact_lifespan_match")

    if a_nr_tokens and b_nr_tokens and _tokens_close(a_nr_tokens[0], b_nr_tokens[0], min_ratio=0.88):
        score += 0.1
        reasons.append("same_first_name_token")

    a_gender = _canonical_gender(entity_a.get("gender"))
    b_gender = _canonical_gender(entity_b.get("gender"))
    if a_gender and b_gender and a_gender == b_gender:
        score += 0.05
        reasons.append("same_gender")

    alias_overlap = _entity_alias_keys(entity_a) & _entity_alias_keys(entity_b)
    if alias_overlap:
        score += 0.08
        reasons.append("alias_overlap")

    return min(score, 1.0), reasons, blockers


def _merge_possible_same_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        a = str(row.get("entity_a") or "").strip()
        b = str(row.get("entity_b") or "").strip()
        if not a or not b or a == b:
            continue
        aa, bb = sorted([a, b])
        key = (aa, bb)
        score = float(row.get("score") or 0.0)
        if key not in uniq:
            clean = dict(row)
            clean["entity_a"] = aa
            clean["entity_b"] = bb
            clean["score"] = score
            clean["reasons"] = list(dict.fromkeys(str(x) for x in (row.get("reasons") or []) if str(x)))
            uniq[key] = clean
            continue
        prev = uniq[key]
        prev["reasons"] = list(dict.fromkeys(list(prev.get("reasons") or []) + list(row.get("reasons") or [])))
        if score > float(prev.get("score") or 0.0):
            prev["score"] = score
            if row.get("trigger_mention_id"):
                prev["trigger_mention_id"] = row.get("trigger_mention_id")
            if row.get("source"):
                prev["source"] = row.get("source")
    return list(uniq.values())


def _build_additional_possible_same_candidates(
    entities: Sequence[Dict[str, Any]],
    base_candidates: Sequence[Dict[str, Any]],
    *,
    min_score: float,
    max_pairs: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_keys = {
        tuple(sorted((str(x.get("entity_a") or ""), str(x.get("entity_b") or ""))))
        for x in base_candidates
    }
    n = len(entities)
    for i in range(n):
        a = entities[i]
        a_id = str(a.get("entity_id") or "")
        if not a_id:
            continue
        for j in range(i + 1, n):
            b = entities[j]
            b_id = str(b.get("entity_id") or "")
            if not b_id:
                continue
            key = tuple(sorted((a_id, b_id)))
            score, reasons, blockers = _score_possible_same_candidate_pair(a, b)
            if score < float(min_score):
                continue
            row = {
                "entity_a": key[0],
                "entity_b": key[1],
                "score": float(score),
                "reasons": list(reasons),
                "trigger_mention_id": None,
                "source": "derived" if key not in base_keys else "base+derived",
                "blockers": list(blockers),
            }
            rows.append(row)
    rows = _merge_possible_same_rows(rows)
    rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    lim = max(0, int(max_pairs))
    return rows if lim == 0 else rows[:lim]


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
                "merged_name": decision.get("merged_name"),
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
    merged_name = _compose_merged_canonical_name(dst_name, src_name)
    if merged_name:
        dst["canonical_name"] = merged_name
    dst["normalized_name"] = normalize_name(str(dst.get("canonical_name") or ""))

    dst["entity_type"] = _prefer_entity_type(dst.get("entity_type"), src.get("entity_type"))
    if not dst.get("species") and src.get("species"):
        dst["species"] = src.get("species")
    for key in ("birth_year", "death_year", "birth_place", "death_place", "gender"):
        if not dst.get(key) and src.get(key):
            dst[key] = src.get(key)
    dst["gender"] = _canonical_gender(dst.get("gender") or src.get("gender"))
    dst["occupation"] = _merge_text_field(dst.get("occupation"), src.get("occupation"), max_len=300)
    dst["description"] = _merge_text_field(dst.get("description"), src.get("description"), max_len=700)
    dst["confidence"] = max(float(dst.get("confidence") or 0.0), float(src.get("confidence") or 0.0))
    dst["source_pages"] = sorted({int(x) for x in (dst.get("source_pages") or []) + (src.get("source_pages") or [])})
    dst["mention_ids"] = sorted({str(x) for x in (dst.get("mention_ids") or []) + (src.get("mention_ids") or []) if str(x)})
    canonical = str(dst.get("canonical_name") or "")
    dst["aliases"] = sorted(a for a in aliases if a and a != canonical)


def _entity_rank_for_merge(e: Dict[str, Any]) -> Tuple[int, int, float, int, str]:
    mention_count = len(e.get("mention_ids") or [])
    page_count = len(e.get("source_pages") or [])
    conf = float(e.get("confidence") or 0.0)
    name_len = len(str(e.get("canonical_name") or ""))
    eid = str(e.get("entity_id") or "")
    return (mention_count, page_count, conf, name_len, eid)


def _entity_alias_keys(entity: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    base = str(entity.get("canonical_name") or "").strip()
    if base:
        n = normalize_name(base)
        if n:
            keys.add(n)
    for a in entity.get("aliases") or []:
        aa = normalize_name(str(a))
        if aa:
            keys.add(aa)
    return keys


def _name_first_token(name: str) -> str:
    tokens = [t for t in normalize_name(str(name or "")).split() if t]
    return tokens[0] if tokens else ""


def _leading_decorator_tokens(name: str) -> List[str]:
    tokens = [t for t in normalize_name(str(name or "")).split() if t]
    out: List[str] = []
    for t in tokens:
        if t in NAME_DECORATOR_TOKENS:
            out.append(t)
            continue
        break
    return out


def _merge_text_field(a: Any, b: Any, *, max_len: int = 600) -> Optional[str]:
    aa = _normalize_surface_name(str(a or ""))
    bb = _normalize_surface_name(str(b or ""))
    if not aa and not bb:
        return None
    if not aa:
        return bb
    if not bb:
        return aa
    aa_norm = normalize_name(aa)
    bb_norm = normalize_name(bb)
    if aa_norm == bb_norm:
        return aa if len(aa) >= len(bb) else bb
    if bb_norm and bb_norm in aa_norm:
        return aa
    if aa_norm and aa_norm in bb_norm:
        return bb
    merged = f"{aa}; {bb}"
    return merged[:max_len]


def _compose_merged_canonical_name(a_name: str, b_name: str) -> str:
    a = _normalize_surface_name(a_name)
    b = _normalize_surface_name(b_name)
    if not a:
        return b
    if not b:
        return a
    if normalize_name(a) == normalize_name(b):
        return a if len(a) >= len(b) else b

    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    if b_norm in a_norm:
        return a
    if a_norm in b_norm:
        return b

    a_regnal = _extract_roman_numeral(a) is not None
    b_regnal = _extract_roman_numeral(b) is not None
    a_first = _name_first_token(_normalize_name_without_roman(a))
    b_first = _name_first_token(_normalize_name_without_roman(b))
    same_first = bool(a_first and b_first and a_first == b_first)

    a_has_details = len([t for t in _normalize_name_without_roman(a).split() if t]) >= 2
    b_has_details = len([t for t in _normalize_name_without_roman(b).split() if t]) >= 2

    if same_first and a_regnal and b_has_details:
        return f"{a} ({b})"
    if same_first and b_regnal and a_has_details:
        return f"{b} ({a})"

    return a if len(a) >= len(b) else b


def _normalize_name_without_roman(name: str) -> str:
    core = _normalize_name_core_for_merge(name, drop_roman=True)
    if core:
        return core
    raw = str(name or "")
    stripped = re.sub(r"\b[IVXLCDM]+\b", " ", raw, flags=re.IGNORECASE)
    return normalize_name(stripped)


def _entity_merge_blockers(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    blockers: List[str] = []

    a_type = str(a.get("entity_type") or "")
    b_type = str(b.get("entity_type") or "")
    if a_type and b_type and a_type != b_type and {a_type, b_type} != {"other_living", "human"}:
        blockers.append("entity_type_mismatch")

    a_name = str(a.get("canonical_name") or "")
    b_name = str(b.get("canonical_name") or "")
    ar = _extract_roman_numeral(a_name)
    br = _extract_roman_numeral(b_name)
    if ar and br and ar != br:
        blockers.append("roman_numeral_mismatch")

    ap = _extract_patronymic(a_name)
    bp = _extract_patronymic(b_name)
    if ap and bp and ap != bp:
        blockers.append("patronymic_mismatch")

    a_by = coerce_year(a.get("birth_year"))
    b_by = coerce_year(b.get("birth_year"))
    if a_by is not None and b_by is not None and abs(a_by - b_by) > 15:
        blockers.append("birth_year_conflict")

    a_dy = coerce_year(a.get("death_year"))
    b_dy = coerce_year(b.get("death_year"))
    if a_dy is not None and b_dy is not None and abs(a_dy - b_dy) > 15:
        blockers.append("death_year_conflict")

    a_gender = _canonical_gender(a.get("gender"))
    b_gender = _canonical_gender(b.get("gender"))
    if a_gender and b_gender and a_gender != b_gender:
        blockers.append("gender_conflict")

    a_species = str(a.get("species") or "").strip().lower()
    b_species = str(b.get("species") or "").strip().lower()
    if a_species and b_species and a_species != b_species:
        blockers.append("species_conflict")

    a_core = _normalize_name_without_roman(a_name)
    b_core = _normalize_name_without_roman(b_name)
    a_tokens = len([t for t in a_core.split() if t]) if a_core else _name_identity_token_count(a_name)
    b_tokens = len([t for t in b_core.split() if t]) if b_core else _name_identity_token_count(b_name)
    if max(a_tokens, b_tokens) <= 1:
        if (
            str(a.get("entity_type") or "").strip().lower() == "group"
            and str(b.get("entity_type") or "").strip().lower() == "group"
            and normalize_name(a_name) == normalize_name(b_name)
        ):
            return blockers
        same_exact_name = normalize_name(a_name) == normalize_name(b_name)
        if same_exact_name and (not a_gender or not b_gender or a_gender == b_gender):
            a_pages = [int(x) for x in (a.get("source_pages") or []) if x is not None]
            b_pages = [int(x) for x in (b.get("source_pages") or []) if x is not None]
            if a_pages and b_pages:
                min_diff = min(abs(x - y) for x in a_pages for y in b_pages)
                if min_diff <= 8:
                    return blockers
        same_birth = a_by is not None and b_by is not None and a_by == b_by
        same_death = a_dy is not None and b_dy is not None and a_dy == b_dy
        same_regnal = bool(ar and br and ar == br)
        alias_overlap = _entity_alias_keys(a) & _entity_alias_keys(b)
        alias_anchor = any(len([t for t in x.split() if t]) >= 2 for x in alias_overlap)
        a_first = _name_first_token(a_core or a_name)
        b_first = _name_first_token(b_core or b_name)
        a_decor = set(_leading_decorator_tokens(a_name))
        b_decor = set(_leading_decorator_tokens(b_name))
        same_title_context = bool(a_decor and b_decor and a_decor == b_decor)
        close_single_token = bool(a_first and b_first and _tokens_close(a_first, b_first, min_ratio=0.9))
        if not (same_birth or same_death or same_regnal or alias_anchor or (same_title_context and close_single_token)):
            blockers.append("single_token_ambiguous")

    return blockers


def _finalize_entity_remap(
    entity_by_id: Dict[str, Dict[str, Any]],
    remap: Dict[str, str],
    mention_to_entity: Dict[str, str],
    possible_same: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    merged_entities: List[Dict[str, Any]] = []
    for eid, e in entity_by_id.items():
        if remap.get(eid, eid) == eid:
            e["normalized_name"] = normalize_name(str(e.get("canonical_name") or e.get("normalized_name") or ""))
            merged_entities.append(e)
    merged_entities.sort(key=lambda x: str(x.get("entity_id") or ""))

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
    return merged_entities, remapped_mentions, list(uniq.values())


def _can_auto_merge_exact_name_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, List[str]]:
    a_name = str(a.get("canonical_name") or "")
    b_name = str(b.get("canonical_name") or "")
    a_norm = normalize_name(str(a.get("normalized_name") or a_name))
    b_norm = normalize_name(str(b.get("normalized_name") or b_name))
    a_core = _normalize_name_core_for_merge(a_name, drop_roman=False)
    b_core = _normalize_name_core_for_merge(b_name, drop_roman=False)
    a_no_roman = _normalize_name_without_roman(a_name)
    b_no_roman = _normalize_name_without_roman(b_name)

    exact_match = bool(a_norm and b_norm and a_norm == b_norm)
    core_match = bool(a_core and b_core and a_core == b_core)
    near_match = False
    if not exact_match and not core_match:
        same_title_context = bool(
            set(_leading_decorator_tokens(a_name))
            and set(_leading_decorator_tokens(a_name)) == set(_leading_decorator_tokens(b_name))
        )
        sim = max(
            _name_similarity(a_norm, b_norm),
            _name_similarity(a_core, b_core) if a_core and b_core else 0.0,
            _name_similarity(a_no_roman, b_no_roman) if a_no_roman and b_no_roman else 0.0,
        )
        if same_title_context and sim >= 0.92:
            near_match = True
        else:
            return False, ["normalized_name_mismatch"]

    blockers = _entity_merge_blockers(a, b)
    if blockers:
        return False, blockers

    # Single-token names are highly ambiguous; merge them only with extra identity signals.
    token_count = len([t for t in (a_core or a_norm).split() if t])
    if token_count <= 1:
        a_type = str(a.get("entity_type") or "").strip().lower()
        b_type = str(b.get("entity_type") or "").strip().lower()
        if exact_match and a_type == "group" and b_type == "group":
            return True, []
        signals = 0
        a_gender = _canonical_gender(a.get("gender"))
        b_gender = _canonical_gender(b.get("gender"))
        if a_gender and b_gender and a_gender == b_gender:
            signals += 1
        a_by = coerce_year(a.get("birth_year"))
        b_by = coerce_year(b.get("birth_year"))
        if a_by is not None and b_by is not None and abs(a_by - b_by) <= 10:
            signals += 1
        a_dy = coerce_year(a.get("death_year"))
        b_dy = coerce_year(b.get("death_year"))
        if a_dy is not None and b_dy is not None and abs(a_dy - b_dy) <= 10:
            signals += 1
        if _entity_alias_keys(a) & _entity_alias_keys(b):
            signals += 1
        page_close = False
        a_pages = [int(x) for x in (a.get("source_pages") or []) if x is not None]
        b_pages = [int(x) for x in (b.get("source_pages") or []) if x is not None]
        if a_pages and b_pages:
            min_diff = min(abs(x - y) for x in a_pages for y in b_pages)
            page_close = min_diff <= 8
        if signals < 2 and not near_match:
            if not (exact_match and signals >= 1 and page_close):
                return False, ["single_token_name_without_identity_signals"]

    return True, []


def _can_auto_merge_birth_year_name_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, List[str]]:
    a_type = str(a.get("entity_type") or "")
    b_type = str(b.get("entity_type") or "")
    humanish = {"human", "other_living", ""}
    if a_type not in humanish or b_type not in humanish:
        return False, ["non_human_entity_type"]

    a_by = coerce_year(a.get("birth_year"))
    b_by = coerce_year(b.get("birth_year"))
    if a_by is None or b_by is None:
        return False, ["missing_birth_year"]
    if a_by != b_by:
        return False, ["birth_year_mismatch"]
    a_dy = coerce_year(a.get("death_year"))
    b_dy = coerce_year(b.get("death_year"))
    exact_lifespan = a_dy is not None and b_dy is not None and a_dy == b_dy

    blockers = _entity_merge_blockers(a, b)
    if blockers:
        return False, blockers

    a_name = str(a.get("canonical_name") or "")
    b_name = str(b.get("canonical_name") or "")
    a_norm = normalize_name(a_name)
    b_norm = normalize_name(b_name)
    if not a_norm or not b_norm:
        return False, ["missing_name"]

    a_core = _normalize_name_core_for_merge(a_name, drop_roman=False)
    b_core = _normalize_name_core_for_merge(b_name, drop_roman=False)
    if a_core and b_core and a_core == b_core:
        return True, []

    a_no_roman = _normalize_name_without_roman(a_name)
    b_no_roman = _normalize_name_without_roman(b_name)
    a_first = _name_first_token(a_no_roman)
    b_first = _name_first_token(b_no_roman)
    first_name_close = bool(a_first and b_first and _tokens_close(a_first, b_first, min_ratio=0.88))
    if a_no_roman and b_no_roman and a_no_roman == b_no_roman and len(a_no_roman) >= 4:
        return True, []

    if exact_lifespan and first_name_close:
        # Strong criterion from user: exact lifespan + same first token.
        return True, []

    sim_full = _name_similarity(a_norm, b_norm)
    sim_core = _name_similarity(a_core, b_core) if a_core and b_core else 0.0
    sim_no_roman = _name_similarity(a_no_roman, b_no_roman) if a_no_roman and b_no_roman else 0.0
    if max(sim_full, sim_core, sim_no_roman) >= (0.82 if exact_lifespan else 0.9):
        return True, []

    return False, ["name_similarity_too_low"]


def _can_auto_merge_title_typo_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, List[str]]:
    a_name = str(a.get("canonical_name") or "")
    b_name = str(b.get("canonical_name") or "")
    a_decor = tuple(_leading_decorator_tokens(a_name))
    b_decor = tuple(_leading_decorator_tokens(b_name))
    if not a_decor or a_decor != b_decor:
        return False, ["decorator_context_mismatch"]

    blockers = _entity_merge_blockers(a, b)
    if blockers:
        return False, blockers

    a_no_roman = _normalize_name_without_roman(a_name)
    b_no_roman = _normalize_name_without_roman(b_name)
    a_first = _name_first_token(a_no_roman)
    b_first = _name_first_token(b_no_roman)
    if not (a_first and b_first and _tokens_close(a_first, b_first, min_ratio=0.9)):
        return False, ["first_token_not_close"]
    ar = _extract_roman_numeral(a_name)
    br = _extract_roman_numeral(b_name)
    same_regnal = bool(ar and br and ar == br)

    a_norm = normalize_name(str(a.get("normalized_name") or a_name))
    b_norm = normalize_name(str(b.get("normalized_name") or b_name))
    sim = max(
        _name_similarity(a_norm, b_norm),
        _name_similarity(a_no_roman, b_no_roman) if a_no_roman and b_no_roman else 0.0,
    )
    if sim < 0.9 and not (same_regnal and sim >= 0.62):
        return False, ["name_similarity_too_low"]

    return True, []


def _auto_merge_title_typo_entities(
    entities: Sequence[Dict[str, Any]],
    mention_to_entity: Dict[str, str],
    possible_same: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    entity_by_id = {str(e.get("entity_id")): dict(e) for e in entities}
    if not entity_by_id:
        return [], dict(mention_to_entity), list(possible_same), {"groups": 0, "merged_entities": 0}

    by_decorator: Dict[Tuple[str, ...], List[str]] = {}
    for eid, e in entity_by_id.items():
        decor = tuple(_leading_decorator_tokens(str(e.get("canonical_name") or "")))
        if not decor:
            continue
        by_decorator.setdefault(decor, []).append(eid)

    remap: Dict[str, str] = {eid: eid for eid in entity_by_id}
    merged_entities_count = 0
    merged_groups_count = 0

    for member_ids in by_decorator.values():
        if len(member_ids) < 2:
            continue
        ranked = sorted(member_ids, key=lambda x: _entity_rank_for_merge(entity_by_id[x]), reverse=True)
        reps: List[str] = []
        group_merged = 0
        for eid in ranked:
            if not reps:
                reps.append(eid)
                remap[eid] = eid
                continue
            candidate = entity_by_id[eid]
            merged_into: Optional[str] = None
            for rid in reps:
                ok, _ = _can_auto_merge_title_typo_pair(candidate, entity_by_id[rid])
                if ok:
                    merged_into = rid
                    break
            if merged_into:
                _merge_entity_into_entity(entity_by_id[merged_into], candidate)
                remap[eid] = merged_into
                group_merged += 1
                merged_entities_count += 1
            else:
                reps.append(eid)
                remap[eid] = eid
        if group_merged > 0:
            merged_groups_count += 1

    merged_entities, remapped_mentions, remapped_possible_same = _finalize_entity_remap(
        entity_by_id,
        remap,
        mention_to_entity,
        possible_same,
    )
    stats = {
        "groups": int(merged_groups_count),
        "merged_entities": int(merged_entities_count),
    }
    return merged_entities, remapped_mentions, remapped_possible_same, stats


def _auto_merge_exact_name_entities(
    entities: Sequence[Dict[str, Any]],
    mention_to_entity: Dict[str, str],
    possible_same: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    entity_by_id = {str(e.get("entity_id")): dict(e) for e in entities}
    if not entity_by_id:
        return [], dict(mention_to_entity), list(possible_same), {"groups": 0, "merged_entities": 0}

    by_norm: Dict[str, List[str]] = {}
    for eid, e in entity_by_id.items():
        n = normalize_name(str(e.get("normalized_name") or e.get("canonical_name") or ""))
        if not n:
            continue
        by_norm.setdefault(n, []).append(eid)

    remap: Dict[str, str] = {eid: eid for eid in entity_by_id}
    merged_entities_count = 0
    merged_groups_count = 0

    for member_ids in by_norm.values():
        if len(member_ids) < 2:
            continue
        ranked = sorted(member_ids, key=lambda x: _entity_rank_for_merge(entity_by_id[x]), reverse=True)
        reps: List[str] = []
        group_merged = 0
        for eid in ranked:
            if not reps:
                reps.append(eid)
                remap[eid] = eid
                continue
            candidate = entity_by_id[eid]
            merged_into: Optional[str] = None
            for rid in reps:
                ok, _ = _can_auto_merge_exact_name_pair(candidate, entity_by_id[rid])
                if ok:
                    merged_into = rid
                    break
            if merged_into:
                _merge_entity_into_entity(entity_by_id[merged_into], candidate)
                remap[eid] = merged_into
                group_merged += 1
                merged_entities_count += 1
            else:
                reps.append(eid)
                remap[eid] = eid
        if group_merged > 0:
            merged_groups_count += 1

    merged_entities, remapped_mentions, remapped_possible_same = _finalize_entity_remap(
        entity_by_id,
        remap,
        mention_to_entity,
        possible_same,
    )

    stats = {
        "groups": int(merged_groups_count),
        "merged_entities": int(merged_entities_count),
    }
    return merged_entities, remapped_mentions, remapped_possible_same, stats


def _auto_merge_birth_year_name_entities(
    entities: Sequence[Dict[str, Any]],
    mention_to_entity: Dict[str, str],
    possible_same: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    entity_by_id = {str(e.get("entity_id")): dict(e) for e in entities}
    if not entity_by_id:
        return [], dict(mention_to_entity), list(possible_same), {"groups": 0, "merged_entities": 0}

    by_birth_year: Dict[int, List[str]] = {}
    for eid, e in entity_by_id.items():
        by = coerce_year(e.get("birth_year"))
        if by is None:
            continue
        by_birth_year.setdefault(by, []).append(eid)

    remap: Dict[str, str] = {eid: eid for eid in entity_by_id}
    merged_entities_count = 0
    merged_groups_count = 0

    for member_ids in by_birth_year.values():
        if len(member_ids) < 2:
            continue
        ranked = sorted(member_ids, key=lambda x: _entity_rank_for_merge(entity_by_id[x]), reverse=True)
        reps: List[str] = []
        group_merged = 0
        for eid in ranked:
            if not reps:
                reps.append(eid)
                remap[eid] = eid
                continue
            candidate = entity_by_id[eid]
            merged_into: Optional[str] = None
            for rid in reps:
                ok, _ = _can_auto_merge_birth_year_name_pair(candidate, entity_by_id[rid])
                if ok:
                    merged_into = rid
                    break
            if merged_into:
                _merge_entity_into_entity(entity_by_id[merged_into], candidate)
                remap[eid] = merged_into
                group_merged += 1
                merged_entities_count += 1
            else:
                reps.append(eid)
                remap[eid] = eid
        if group_merged > 0:
            merged_groups_count += 1

    merged_entities, remapped_mentions, remapped_possible_same = _finalize_entity_remap(
        entity_by_id,
        remap,
        mention_to_entity,
        possible_same,
    )
    stats = {
        "groups": int(merged_groups_count),
        "merged_entities": int(merged_entities_count),
    }
    return merged_entities, remapped_mentions, remapped_possible_same, stats


def _apply_llm_entity_merges(
    entities: Sequence[Dict[str, Any]],
    mention_to_entity: Dict[str, str],
    possible_same: Sequence[Dict[str, Any]],
    reviews: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    accepted_pairs = [
        (str(r.get("entity_a") or ""), str(r.get("entity_b") or ""))
        for r in reviews
        if bool(r.get("accepted"))
    ]
    accepted_pairs = [(a, b) for a, b in accepted_pairs if a and b and a != b]
    if not accepted_pairs:
        return (
            list(entities),
            dict(mention_to_entity),
            list(possible_same),
            {"merged_entities": 0, "name_suggestions_applied": 0},
        )

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
    merged_entities_count = 0
    name_suggestions_applied = 0
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
            merged_entities_count += 1

        member_set = set(member_ids)
        suggested_names: List[str] = []
        for r in reviews:
            if not bool(r.get("accepted")):
                continue
            a = str(r.get("entity_a") or "")
            b = str(r.get("entity_b") or "")
            if a not in member_set or b not in member_set:
                continue
            s_name = _normalize_surface_name(str(r.get("merged_name") or ""))
            if s_name and _is_valid_living_name(s_name):
                suggested_names.append(s_name)
        if suggested_names:
            freq: Dict[str, int] = {}
            for n in suggested_names:
                freq[n] = freq.get(n, 0) + 1
            chosen = sorted(freq.keys(), key=lambda n: (freq[n], len(n)), reverse=True)[0]
            prev_name = str(rep.get("canonical_name") or "").strip()
            if chosen and chosen != prev_name:
                aliases = set(rep.get("aliases") or [])
                if prev_name:
                    aliases.add(prev_name)
                rep["canonical_name"] = chosen
                rep["aliases"] = sorted(a for a in aliases if a and a != chosen)
                name_suggestions_applied += 1

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
    return (
        merged_entities,
        remapped_mentions,
        list(uniq.values()),
        {
            "merged_entities": int(merged_entities_count),
            "name_suggestions_applied": int(name_suggestions_applied),
        },
    )


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
    llm_merge_derive_candidates: bool,
    llm_merge_derived_min_score: float,
    llm_merge_derived_max_pairs: int,
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
    relations, graph_parent_child_adjusted = _apply_parent_child_guardrails(relations)

    entities, mention_to_entity, possible_same = _resolve_mentions_to_entities(
        mentions,
        auto_merge_threshold=auto_merge_threshold,
        possible_merge_threshold=possible_merge_threshold,
    )
    llm_possible_same = list(possible_same)
    llm_derived_candidates: List[Dict[str, Any]] = []
    if llm_merge_agent and llm_merge_derive_candidates:
        llm_derived_candidates = _build_additional_possible_same_candidates(
            entities,
            possible_same,
            min_score=float(llm_merge_derived_min_score),
            max_pairs=int(llm_merge_derived_max_pairs),
        )
        llm_possible_same = _merge_possible_same_rows([*possible_same, *llm_derived_candidates])

    llm_merge_reviews: List[Dict[str, Any]] = []
    llm_merge_apply_stats: Dict[str, int] = {"merged_entities": 0, "name_suggestions_applied": 0}
    if llm_merge_agent and llm_possible_same:
        llm_merge_reviews = await _llm_review_possible_same(
            llm_possible_same,
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
        entities, mention_to_entity, possible_same, llm_merge_apply_stats = _apply_llm_entity_merges(
            entities,
            mention_to_entity,
            possible_same,
            llm_merge_reviews,
        )

    entities, mention_to_entity, possible_same, exact_name_merge_stats = _auto_merge_exact_name_entities(
        entities,
        mention_to_entity,
        possible_same,
    )
    entities, mention_to_entity, possible_same, birth_year_name_merge_stats = _auto_merge_birth_year_name_entities(
        entities,
        mention_to_entity,
        possible_same,
    )
    entities, mention_to_entity, possible_same, title_typo_merge_stats = _auto_merge_title_typo_entities(
        entities,
        mention_to_entity,
        possible_same,
    )

    entity_by_id: Dict[str, Dict[str, Any]] = {str(e["entity_id"]): e for e in entities if e.get("entity_id")}
    max_entity_idx = 0
    for eid in entity_by_id:
        m = re.match(r"^ent-(\d+)$", str(eid))
        if m:
            max_entity_idx = max(max_entity_idx, int(m.group(1)))
    next_entity_idx = max_entity_idx + 1 if max_entity_idx > 0 else len(entity_by_id) + 1

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
        while ent_id in entity_by_id:
            next_entity_idx += 1
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
    age_parent_child_reversed_total = 0
    age_parent_child_downgraded_total = 0

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
        if rtype == "parent_child":
            src_e = entity_by_id.get(str(src_entity))
            dst_e = entity_by_id.get(str(dst_entity))
            src_by = coerce_year(src_e.get("birth_year")) if src_e else None
            dst_by = coerce_year(dst_e.get("birth_year")) if dst_e else None
            if src_by is not None and dst_by is not None:
                # parent should be at least 13 years older than child
                if src_by <= (dst_by - 13):
                    pass
                elif dst_by <= (src_by - 13):
                    src_entity, dst_entity = dst_entity, src_entity
                    age_parent_child_reversed_total += 1
                else:
                    rtype = "relative"
                    age_parent_child_downgraded_total += 1

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
        "llm_merge_entities_merged_total": int(llm_merge_apply_stats.get("merged_entities") or 0),
        "llm_merge_name_suggestions_applied": int(llm_merge_apply_stats.get("name_suggestions_applied") or 0),
        "llm_possible_same_total": len(llm_possible_same),
        "llm_derived_candidates_total": len(llm_derived_candidates),
        "exact_name_merge_groups": int(exact_name_merge_stats.get("groups") or 0),
        "exact_name_merged_entities": int(exact_name_merge_stats.get("merged_entities") or 0),
        "birth_year_name_merge_groups": int(birth_year_name_merge_stats.get("groups") or 0),
        "birth_year_name_merged_entities": int(birth_year_name_merge_stats.get("merged_entities") or 0),
        "title_typo_merge_groups": int(title_typo_merge_stats.get("groups") or 0),
        "title_typo_merged_entities": int(title_typo_merge_stats.get("merged_entities") or 0),
        "graph_parent_child_adjusted_total": int(graph_parent_child_adjusted),
        "age_parent_child_reversed_total": int(age_parent_child_reversed_total),
        "age_parent_child_downgraded_total": int(age_parent_child_downgraded_total),
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

    node_by_id: Dict[str, Dict[str, Any]] = {}
    type_rank = {"human": 4, "animal": 3, "group": 2, "other_living": 1}
    for e in graph.get("entities") or []:
        eid = str(e.get("entity_id") or "").strip()
        if not eid:
            continue
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
        node = {
            "id": eid,
            "label": label,
            "group": et,
            "title": tip,
            "color": entity_color.get(et, "#5D6D7E"),
        }
        prev = node_by_id.get(eid)
        if prev is None:
            node_by_id[eid] = node
            continue
        # Dedup entity nodes for vis.DataSet (duplicate ids cause rendering failure).
        if len(str(node.get("label") or "")) > len(str(prev.get("label") or "")):
            prev["label"] = node["label"]
            prev["title"] = node["title"]
        if type_rank.get(et, 0) > type_rank.get(str(prev.get("group") or ""), 0):
            prev["group"] = et
            prev["color"] = entity_color.get(et, "#5D6D7E")

    nodes: List[Dict[str, Any]] = sorted(node_by_id.values(), key=lambda x: str(x.get("id") or ""))

    edges: List[Dict[str, Any]] = []
    node_ids = {str(n.get("id") or "") for n in nodes}
    for r in graph.get("relations") or []:
        src_id = str(r.get("source_entity_id") or "")
        dst_id = str(r.get("target_entity_id") or "")
        if not src_id or not dst_id or src_id not in node_ids or dst_id not in node_ids:
            continue
        rt = str(r.get("relation_type") or "associated_with")
        pol = str(r.get("polarity") or "neutral")
        directed = bool(r.get("directed", True))
        edges.append(
            {
                "from": src_id,
                "to": dst_id,
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
    #mynetwork {{ width: 100vw; height: calc(100vh - 170px); min-height: 420px; position: relative; overflow: hidden; background: linear-gradient(180deg, #f8fafc, #eff6ff); }}
    #graph-canvas {{ width: 100%; height: 100%; display: block; cursor: grab; }}
    #graph-canvas.dragging {{ cursor: grabbing; }}
    #graph-tooltip {{ position: absolute; pointer-events: none; max-width: 320px; display: none; z-index: 10; white-space: pre-line; background: rgba(15, 23, 42, 0.92); color: #f8fafc; font-size: 12px; line-height: 1.35; border-radius: 8px; padding: 8px 10px; box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35); }}
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

    const container = document.getElementById("mynetwork");
    const canvas = document.createElement("canvas");
    canvas.id = "graph-canvas";
    container.appendChild(canvas);
    const tooltip = document.createElement("div");
    tooltip.id = "graph-tooltip";
    container.appendChild(tooltip);
    const ctx = canvas.getContext("2d");
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const camera = {{ x: 0, y: 0, zoom: 1.0 }};
    const layout = new Map();
    let filteredNodes = [];
    let filteredEdges = [];
    let nodeById = new Map();
    let hoveredNode = null;
    let dragNode = null;
    let isPanning = false;
    let lastMouse = null;

    function htmlToText(raw) {{
      return String(raw || "")
        .replace(/<br\\s*\\/?>/gi, "\\n")
        .replace(/<[^>]*>/g, "")
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&amp;/g, "&");
    }}

    function resizeCanvas() {{
      const w = Math.max(320, container.clientWidth || 0);
      const h = Math.max(240, container.clientHeight || 0);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${{w}}px`;
      canvas.style.height = `${{h}}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function worldToScreen(x, y) {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      return {{
        x: (x - camera.x) * camera.zoom + w / 2,
        y: (y - camera.y) * camera.zoom + h / 2,
      }};
    }}

    function screenToWorld(x, y) {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      return {{
        x: (x - w / 2) / camera.zoom + camera.x,
        y: (y - h / 2) / camera.zoom + camera.y,
      }};
    }}

    function getNodeRadius(node) {{
      if (String(node.group || "") === "group") return 10;
      if (String(node.group || "") === "animal") return 9;
      return 8;
    }}

    function ensureNodeLayout(node, idx, total) {{
      const id = String(node.id || "");
      let state = layout.get(id);
      if (!state) {{
        const angle = (Math.PI * 2 * idx) / Math.max(1, total);
        const radius = 80 + Math.sqrt(Math.max(1, total)) * 30;
        state = {{
          x: Math.cos(angle) * radius + (Math.random() - 0.5) * 16,
          y: Math.sin(angle) * radius + (Math.random() - 0.5) * 16,
          vx: 0,
          vy: 0,
          fx: 0,
          fy: 0,
        }};
        layout.set(id, state);
      }}
      node._layout = state;
      return state;
    }}

    function applyFilters() {{
      const q = (document.getElementById("q").value || "").trim().toLowerCase();
      const edgesByType = baseEdges.filter((e) => selectedRelations.has(e.relation_type));
      const touched = new Set();
      edgesByType.forEach((e) => {{
        touched.add(String(e.from || ""));
        touched.add(String(e.to || ""));
      }});

      filteredNodes = baseNodes.filter((n) => {{
        const label = String(n.label || "").toLowerCase();
        const nid = String(n.id || "");
        if (q && !label.includes(q)) return false;
        if (q) return true;
        return touched.has(nid);
      }});

      nodeById = new Map(filteredNodes.map((n) => [String(n.id || ""), n]));
      filteredEdges = edgesByType.filter((e) => {{
        const from = String(e.from || "");
        const to = String(e.to || "");
        return nodeById.has(from) && nodeById.has(to);
      }});

      filteredNodes.forEach((n, idx) => ensureNodeLayout(n, idx, filteredNodes.length));
      const meta = document.getElementById("meta");
      if (meta) {{
        meta.textContent = `nodes: ${{filteredNodes.length}}/${{baseNodes.length}} | edges: ${{filteredEdges.length}}/${{baseEdges.length}}`;
      }}
    }}

    function stepPhysics() {{
      if (!filteredNodes.length) return;
      const repulsion = 12000;
      const springK = 0.0036;
      const targetLen = 120;
      const centerPull = 0.0016;
      const damping = 0.86;

      for (const node of filteredNodes) {{
        const s = node._layout;
        s.fx = 0;
        s.fy = 0;
      }}

      for (let i = 0; i < filteredNodes.length; i += 1) {{
        const a = filteredNodes[i];
        const sa = a._layout;
        for (let j = i + 1; j < filteredNodes.length; j += 1) {{
          const b = filteredNodes[j];
          const sb = b._layout;
          const dx = sa.x - sb.x;
          const dy = sa.y - sb.y;
          const d2 = dx * dx + dy * dy + 0.01;
          const d = Math.sqrt(d2);
          const f = repulsion / d2;
          const fx = (dx / d) * f;
          const fy = (dy / d) * f;
          sa.fx += fx;
          sa.fy += fy;
          sb.fx -= fx;
          sb.fy -= fy;
        }}
      }}

      for (const edge of filteredEdges) {{
        const a = nodeById.get(String(edge.from || ""));
        const b = nodeById.get(String(edge.to || ""));
        if (!a || !b) continue;
        const sa = a._layout;
        const sb = b._layout;
        const dx = sb.x - sa.x;
        const dy = sb.y - sa.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1.0;
        const f = springK * (d - targetLen);
        const fx = (dx / d) * f;
        const fy = (dy / d) * f;
        sa.fx += fx;
        sa.fy += fy;
        sb.fx -= fx;
        sb.fy -= fy;
      }}

      for (const node of filteredNodes) {{
        if (node === dragNode) continue;
        const s = node._layout;
        s.vx = (s.vx + s.fx - s.x * centerPull) * damping;
        s.vy = (s.vy + s.fy - s.y * centerPull) * damping;
        s.x += s.vx;
        s.y += s.vy;
      }}
    }}

    function drawArrow(fromX, fromY, toX, toY, color) {{
      const headLen = 7;
      const dx = toX - fromX;
      const dy = toY - fromY;
      const len = Math.sqrt(dx * dx + dy * dy) || 1.0;
      const ux = dx / len;
      const uy = dy / len;
      const tipX = toX - ux * 10;
      const tipY = toY - uy * 10;
      const leftX = tipX - ux * headLen - uy * headLen * 0.7;
      const leftY = tipY - uy * headLen + ux * headLen * 0.7;
      const rightX = tipX - ux * headLen + uy * headLen * 0.7;
      const rightY = tipY - uy * headLen - ux * headLen * 0.7;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(leftX, leftY);
      ctx.lineTo(rightX, rightY);
      ctx.closePath();
      ctx.fill();
    }}

    function draw() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);

      ctx.lineWidth = 1.5;
      for (const edge of filteredEdges) {{
        const src = nodeById.get(String(edge.from || ""));
        const dst = nodeById.get(String(edge.to || ""));
        if (!src || !dst) continue;
        const a = worldToScreen(src._layout.x, src._layout.y);
        const b = worldToScreen(dst._layout.x, dst._layout.y);
        const color = String(edge.color || "#7F8C8D");
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        if (String(edge.arrows || "") === "to") {{
          drawArrow(a.x, a.y, b.x, b.y, color);
        }}
      }}

      for (const node of filteredNodes) {{
        const s = node._layout;
        const p = worldToScreen(s.x, s.y);
        const radius = getNodeRadius(node);
        ctx.beginPath();
        ctx.fillStyle = String(node.color || "#5D6D7E");
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
        if (node === hoveredNode) {{
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2.2;
          ctx.stroke();
        }}

        const label = String(node.label || "");
        if (label) {{
          ctx.font = "12px ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif";
          ctx.fillStyle = "#1f2937";
          ctx.fillText(label, p.x + radius + 4, p.y - radius - 2);
        }}
      }}
    }}

    function pickNode(screenX, screenY) {{
      for (let i = filteredNodes.length - 1; i >= 0; i -= 1) {{
        const node = filteredNodes[i];
        const s = node._layout;
        const p = worldToScreen(s.x, s.y);
        const r = getNodeRadius(node) + 4;
        const dx = screenX - p.x;
        const dy = screenY - p.y;
        if (dx * dx + dy * dy <= r * r) return node;
      }}
      return null;
    }}

    function pointerPos(ev) {{
      const rect = canvas.getBoundingClientRect();
      return {{
        x: ev.clientX - rect.left,
        y: ev.clientY - rect.top,
      }};
    }}

    function updateTooltip(node, x, y) {{
      if (!node) {{
        tooltip.style.display = "none";
        return;
      }}
      const title = htmlToText(node.title);
      tooltip.textContent = title ? `${{String(node.label || "")}}\\n${{title}}` : String(node.label || "");
      tooltip.style.left = `${{Math.round(x + 12)}}px`;
      tooltip.style.top = `${{Math.round(y + 12)}}px`;
      tooltip.style.display = "block";
    }}

    canvas.addEventListener("mousedown", (ev) => {{
      const p = pointerPos(ev);
      lastMouse = p;
      dragNode = pickNode(p.x, p.y);
      isPanning = !dragNode;
      canvas.classList.add("dragging");
    }});

    window.addEventListener("mousemove", (ev) => {{
      const p = pointerPos(ev);
      if (dragNode) {{
        const w = screenToWorld(p.x, p.y);
        dragNode._layout.x = w.x;
        dragNode._layout.y = w.y;
        dragNode._layout.vx = 0;
        dragNode._layout.vy = 0;
      }} else if (isPanning && lastMouse) {{
        const dx = p.x - lastMouse.x;
        const dy = p.y - lastMouse.y;
        camera.x -= dx / camera.zoom;
        camera.y -= dy / camera.zoom;
      }} else {{
        hoveredNode = pickNode(p.x, p.y);
        canvas.style.cursor = hoveredNode ? "pointer" : "grab";
      }}
      updateTooltip(hoveredNode, p.x, p.y);
      lastMouse = p;
    }});

    window.addEventListener("mouseup", () => {{
      dragNode = null;
      isPanning = false;
      lastMouse = null;
      canvas.classList.remove("dragging");
    }});

    canvas.addEventListener(
      "wheel",
      (ev) => {{
        ev.preventDefault();
        const p = pointerPos(ev);
        const before = screenToWorld(p.x, p.y);
        const factor = ev.deltaY < 0 ? 1.1 : 0.9;
        camera.zoom = Math.max(0.2, Math.min(4.0, camera.zoom * factor));
        const after = screenToWorld(p.x, p.y);
        camera.x += before.x - after.x;
        camera.y += before.y - after.y;
      }},
      {{ passive: false }}
    );

    const relationsEl = document.getElementById("relations");
    relationTypes.forEach((t) => {{
      const id = `rt_${{t}}`;
      const wrap = document.createElement("label");
      wrap.innerHTML = `<input type="checkbox" id="${{id}}" checked/> ${{t}}`;
      const cb = wrap.querySelector("input");
      cb.addEventListener("change", () => {{
        if (cb.checked) selectedRelations.add(t);
        else selectedRelations.delete(t);
        applyFilters();
      }});
      relationsEl.appendChild(wrap);
    }});

    document.getElementById("q").addEventListener("input", () => {{
      applyFilters();
    }});

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    applyFilters();

    function frame() {{
      stepPhysics();
      draw();
      requestAnimationFrame(frame);
    }}
    frame();
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
    p.add_argument("--pages-per-chunk", type=int, default=2)
    p.add_argument("--page-overlap", type=int, default=1)
    p.add_argument("--max-chars-per-chunk", type=int, default=4200)
    p.add_argument(
        "--relation-candidate-window",
        type=int,
        default=2,
        help="Max paragraph distance for candidate mention pairs in relation extraction",
    )
    p.add_argument(
        "--max-relation-candidates",
        type=int,
        default=80,
        help="Max candidate mention pairs passed to relation extraction per chunk",
    )
    p.add_argument(
        "--kinship-marriage-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable focused extra pass for kinship/marriage relations",
    )
    p.add_argument(
        "--kinship-pass-max-candidates",
        type=int,
        default=80,
        help="Max candidate pairs for kinship/marriage relation pass per chunk",
    )
    p.add_argument(
        "--rule-kinship-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic rule-based kinship/marriage extraction from chunk text",
    )
    p.add_argument(
        "--strict-parent-child-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downgrade suspicious parent_child claims (e.g. grandchild wording) to relative",
    )
    p.add_argument(
        "--recall-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable additional high-recall relation extraction pass",
    )
    p.add_argument(
        "--recall-pass-window-extra",
        type=int,
        default=2,
        help="Extra paragraph window for recall pass candidate generation",
    )
    p.add_argument(
        "--recall-pass-max-candidates",
        type=int,
        default=60,
        help="Max additional candidate pairs for recall pass per chunk",
    )
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
    p.add_argument("--llm-merge-max-candidates", type=int, default=120)
    p.add_argument("--llm-merge-max-tokens", type=int, default=320)
    p.add_argument(
        "--llm-merge-derive-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add heuristic derived candidate pairs for LLM merge review",
    )
    p.add_argument(
        "--llm-merge-derived-min-score",
        type=float,
        default=0.56,
        help="Minimum heuristic score for adding derived merge candidates",
    )
    p.add_argument(
        "--llm-merge-derived-max-pairs",
        type=int,
        default=240,
        help="Maximum count of derived merge candidate pairs",
    )

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
        page_overlap=args.page_overlap,
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
            relation_candidate_window=int(args.relation_candidate_window),
            max_relation_candidates=int(args.max_relation_candidates),
            enable_kinship_marriage_pass=bool(args.kinship_marriage_pass),
            kinship_pass_max_candidates=int(args.kinship_pass_max_candidates),
            enable_recall_pass=bool(args.recall_pass),
            recall_pass_window_extra=int(args.recall_pass_window_extra),
            recall_pass_max_candidates=int(args.recall_pass_max_candidates),
            enable_rule_kinship_pass=bool(args.rule_kinship_pass),
            strict_parent_child_validation=bool(args.strict_parent_child_validation),
        )
        parsed = _parse_extracted_chunk(ch, raw)
        parsed_chunks.append(parsed)
        print(
            "    "
            f"mentions={len(parsed['mentions'])} "
            f"candidate_pairs={parsed['candidate_pairs']} "
            f"(base={parsed['base_candidate_pairs']} "
            f"kinship={parsed['kinship_candidate_pairs']} "
            f"recall={parsed['recall_candidate_pairs']}) "
            f"relations={len(parsed['relations'])} "
            f"rule_kinship={parsed['rule_kinship_relations']} "
            f"pc_adjusted={parsed['strict_parent_child_adjusted']} "
            f"mentions_repaired={parsed['mentions_repaired']} "
            f"relations_repaired={parsed['relations_repaired']}"
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
        llm_merge_derive_candidates=bool(args.llm_merge_derive_candidates),
        llm_merge_derived_min_score=float(args.llm_merge_derived_min_score),
        llm_merge_derived_max_pairs=int(args.llm_merge_derived_max_pairs),
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
        f"pc_graph_adjusted={int(graph.get('graph_parent_child_adjusted_total') or 0)} "
        f"pc_age_reversed={int(graph.get('age_parent_child_reversed_total') or 0)} "
        f"pc_age_downgraded={int(graph.get('age_parent_child_downgraded_total') or 0)} "
        f"llm_candidates={int(graph.get('llm_possible_same_total') or 0)} "
        f"llm_merges={int(graph.get('llm_merge_accepted_total') or 0)} "
        f"llm_name_suggestions={int(graph.get('llm_merge_name_suggestions_applied') or 0)} "
        f"exact_name_merges={int(graph.get('exact_name_merged_entities') or 0)} "
        f"birth_year_name_merges={int(graph.get('birth_year_name_merged_entities') or 0)} "
        f"title_typo_merges={int(graph.get('title_typo_merged_entities') or 0)}"
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
            "page_overlap": int(args.page_overlap),
            "max_chars_per_chunk": int(args.max_chars_per_chunk),
            "relation_candidate_window": int(args.relation_candidate_window),
            "max_relation_candidates": int(args.max_relation_candidates),
            "kinship_marriage_pass": bool(args.kinship_marriage_pass),
            "kinship_pass_max_candidates": int(args.kinship_pass_max_candidates),
            "rule_kinship_pass": bool(args.rule_kinship_pass),
            "strict_parent_child_validation": bool(args.strict_parent_child_validation),
            "recall_pass": bool(args.recall_pass),
            "recall_pass_window_extra": int(args.recall_pass_window_extra),
            "recall_pass_max_candidates": int(args.recall_pass_max_candidates),
            "max_chunks": int(args.max_chunks) if args.max_chunks is not None else None,
        },
        "chunks": len(chunks),
        "candidate_pairs_total": int(sum(int(c.get("candidate_pairs") or 0) for c in parsed_chunks)),
        "base_candidate_pairs_total": int(sum(int(c.get("base_candidate_pairs") or 0) for c in parsed_chunks)),
        "kinship_candidate_pairs_total": int(sum(int(c.get("kinship_candidate_pairs") or 0) for c in parsed_chunks)),
        "recall_candidate_pairs_total": int(sum(int(c.get("recall_candidate_pairs") or 0) for c in parsed_chunks)),
        "mentions_repaired_chunks": int(sum(1 for c in parsed_chunks if c.get("mentions_repaired"))),
        "relations_repaired_chunks": int(sum(1 for c in parsed_chunks if c.get("relations_repaired"))),
        "relations_main_repaired_chunks": int(sum(1 for c in parsed_chunks if c.get("relations_main_repaired"))),
        "relations_kinship_repaired_chunks": int(sum(1 for c in parsed_chunks if c.get("relations_kinship_repaired"))),
        "relations_recall_repaired_chunks": int(sum(1 for c in parsed_chunks if c.get("relations_recall_repaired"))),
        "strict_parent_child_adjusted_total": int(sum(int(c.get("strict_parent_child_adjusted") or 0) for c in parsed_chunks)),
        "rule_kinship_relations_total": int(sum(int(c.get("rule_kinship_relations") or 0) for c in parsed_chunks)),
        "mentions_total": len(graph.get("mentions") or []),
        "entities_total": len(graph.get("entities") or []),
        "possible_same_total": len(graph.get("possible_same") or []),
        "relations_total": len(graph.get("relations") or []),
        "relation_claims_total": len(graph.get("relation_claims") or []),
        "llm_merge_agent_enabled": bool(args.llm_merge_agent),
        "llm_merge_reviews_total": len(graph.get("llm_merge_reviews") or []),
        "llm_merge_accepted_total": int(graph.get("llm_merge_accepted_total") or 0),
        "llm_merge_entities_merged_total": int(graph.get("llm_merge_entities_merged_total") or 0),
        "llm_merge_name_suggestions_applied": int(graph.get("llm_merge_name_suggestions_applied") or 0),
        "llm_possible_same_total": int(graph.get("llm_possible_same_total") or 0),
        "llm_derived_candidates_total": int(graph.get("llm_derived_candidates_total") or 0),
        "llm_merge_derive_candidates": bool(args.llm_merge_derive_candidates),
        "llm_merge_derived_min_score": float(args.llm_merge_derived_min_score),
        "llm_merge_derived_max_pairs": int(args.llm_merge_derived_max_pairs),
        "exact_name_merge_groups": int(graph.get("exact_name_merge_groups") or 0),
        "exact_name_merged_entities": int(graph.get("exact_name_merged_entities") or 0),
        "birth_year_name_merge_groups": int(graph.get("birth_year_name_merge_groups") or 0),
        "birth_year_name_merged_entities": int(graph.get("birth_year_name_merged_entities") or 0),
        "title_typo_merge_groups": int(graph.get("title_typo_merge_groups") or 0),
        "title_typo_merged_entities": int(graph.get("title_typo_merged_entities") or 0),
        "graph_parent_child_adjusted_total": int(graph.get("graph_parent_child_adjusted_total") or 0),
        "age_parent_child_reversed_total": int(graph.get("age_parent_child_reversed_total") or 0),
        "age_parent_child_downgraded_total": int(graph.get("age_parent_child_downgraded_total") or 0),
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
