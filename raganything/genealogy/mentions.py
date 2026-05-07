from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .claim_extraction import _clean_name_with_years
from .normalize import normalize_name
from .rag_index import write_jsonl

_MENTION_TOKEN_PATTERN = r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9'’\-]*"
_MENTION_CORE_PATTERN = (
    rf"{_MENTION_TOKEN_PATTERN}(?:\s+{_MENTION_TOKEN_PATTERN}){{0,3}}"
)
_MENTION_YEAR_SUFFIX_PATTERN = (
    r"(?:\s*(?:\(|\[)\s*(?:1[0-9]{3}|20[0-9]{2})\s*[-–—/]\s*"
    r"(?:1[0-9]{3}|20[0-9]{2})\s*(?:\)|\]))?"
)
_MENTION_PATTERN = re.compile(rf"{_MENTION_CORE_PATTERN}{_MENTION_YEAR_SUFFIX_PATTERN}")
_MENTION_NOISE_NORMALIZED = {
    "claim",
    "claims",
    "confidence",
    "conflict",
    "evidence",
    "family",
    "genealogy",
    "parent",
    "parents",
    "person",
    "relationship",
    "relationships",
    "source",
    "status",
    "child",
    "children",
    "spouse",
    "spouses",
    "утверждение",
    "уверенность",
    "конфликт",
    "доказательство",
    "семья",
    "родитель",
    "родители",
    "человек",
    "персона",
    "связь",
    "связи",
    "источник",
    "статус",
    "ребенок",
    "дети",
    "супруг",
    "супруга",
}


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


@dataclass(slots=True)
class MentionRecord:
    mention_id: str
    source_id: str
    chunk_id: str
    surface: str
    normalized_name: str
    page_idx: int | None = None
    span_start: int | None = None
    span_end: int | None = None
    mention_type: str = "person"
    attributes: dict[str, Any] = field(default_factory=dict)
    candidate_person_ids: list[str] = field(default_factory=list)


def _is_person_like_mention(surface: str, normalized_name: str) -> bool:
    if not surface or not normalized_name:
        return False
    if normalized_name in _MENTION_NOISE_NORMALIZED:
        return False
    if normalized_name.isdigit():
        return False
    return any(ch.isalpha() for ch in surface)


def extract_mentions_from_text(
    text: str,
    *,
    source_id: str,
    chunk_id: str,
    page_idx: int | None = None,
) -> list[MentionRecord]:
    mentions: list[MentionRecord] = []
    seen: set[tuple[int, int, str]] = set()
    for match in _MENTION_PATTERN.finditer(str(text or "")):
        raw_surface = match.group(0)
        surface, birth_year, death_year = _clean_name_with_years(raw_surface)
        normalized_name = normalize_name(surface)
        if not _is_person_like_mention(surface, normalized_name):
            continue

        key = (match.start(), match.end(), normalized_name)
        if key in seen:
            continue
        seen.add(key)

        attributes: dict[str, Any] = {}
        if birth_year is not None:
            attributes["birth_year"] = birth_year
        if death_year is not None:
            attributes["death_year"] = death_year

        mentions.append(
            MentionRecord(
                mention_id=_stable_id(
                    "mention",
                    source_id,
                    chunk_id,
                    match.start(),
                    match.end(),
                    surface,
                ),
                source_id=source_id,
                chunk_id=chunk_id,
                surface=surface,
                normalized_name=normalized_name,
                page_idx=page_idx,
                span_start=match.start(),
                span_end=match.end(),
                attributes=attributes,
            )
        )
    return mentions


def write_mentions(output_dir: Path, mentions: Sequence[MentionRecord]) -> Path:
    path = output_dir / "mentions.jsonl"
    write_jsonl(path, (asdict(mention) for mention in mentions))
    return path
