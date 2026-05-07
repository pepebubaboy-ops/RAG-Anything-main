from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Iterator, Tuple

from .models import Claim, Evidence


_NAME_CORE_PATTERN = (
    r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'’\-]*(?:\s+[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё'’\-]*){0,3}"
)
_NAME_YEAR_SUFFIX_PATTERN = (
    r"(?:\s*(?:\(|\[)\s*(?:1[0-9]{3}|20[0-9]{2})\s*[-–—/]\s*"
    r"(?:1[0-9]{3}|20[0-9]{2})\s*(?:\)|\]))?"
)
_NAME_PATTERN = rf"(?-i:{_NAME_CORE_PATTERN}{_NAME_YEAR_SUFFIX_PATTERN})"
_PARENT_PATTERNS = [
    re.compile(
        rf"(?P<child>{_NAME_PATTERN})\s+(?:is|was)\s+(?:the\s+)?(?:son|daughter|child)\s+of\s+(?P<parent1>{_NAME_PATTERN})(?:\s+and\s+(?P<parent2>{_NAME_PATTERN}))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<child>{_NAME_PATTERN})\s*(?:—|-)?\s*(?:сын|дочь|ребенок)\s+(?P<parent1>{_NAME_PATTERN})(?:\s+и\s+(?P<parent2>{_NAME_PATTERN}))?",
        re.IGNORECASE,
    ),
]
_SPOUSE_PATTERNS = [
    re.compile(
        rf"(?P<person1>{_NAME_PATTERN})\s+(?:is|was)?\s*(?:married\s+to|spouse\s+of)\s+(?P<person2>{_NAME_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<person1>{_NAME_PATTERN})\s*(?:—|-)?\s*(?:муж|жена|супруг|супруга)\s+(?P<person2>{_NAME_PATTERN})",
        re.IGNORECASE,
    ),
]
_GENEALOGY_NAME_NOISE_TOKENS = {
    "line",
    "branch",
    "dynasty",
    "lineage",
    "house",
    "family",
    "род",
    "ветвь",
    "династия",
    "линия",
    "дом",
    "семья",
}
_YEAR_RANGE_INLINE_PATTERN = re.compile(
    r"(?:\(|\[)?\s*(?P<birth>(?:1[0-9]{3}|20[0-9]{2}))\s*[-–—/]\s*"
    r"(?P<death>(?:1[0-9]{3}|20[0-9]{2}))\s*(?:\)|\])?"
)
_YEAR_RANGE_ANNOTATION_PATTERN = re.compile(
    r"(?:\(|\[)\s*(?:ca?\.?\s*)?(?:1[0-9]{3}|20[0-9]{2})\s*[-–—/]\s*"
    r"(?:ca?\.?\s*)?(?:1[0-9]{3}|20[0-9]{2})\s*(?:\)|\])",
    re.IGNORECASE,
)



def _coerce_claim_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year <= 0 or year > 2500:
        return None
    return year


def _extract_year_range(value: Any) -> Tuple[int | None, int | None]:
    text = str(value or "")
    match = _YEAR_RANGE_INLINE_PATTERN.search(text)
    if not match:
        return None, None

    birth = _coerce_claim_year(match.group("birth"))
    death = _coerce_claim_year(match.group("death"))
    if birth is None or death is None:
        return None, None
    if birth > death:
        birth, death = death, birth
    return birth, death


def _clean_name(value: str) -> str:
    text = str(value or "")
    text = _YEAR_RANGE_ANNOTATION_PATTERN.sub(" ", text)
    text = text.strip(" ,.;:!?\t\n\"'`«»“”()[]{}")
    tokens = [token for token in text.split() if token]
    if not tokens:
        return ""

    while len(tokens) > 1 and tokens[-1].strip(".,;:!?").lower() in _GENEALOGY_NAME_NOISE_TOKENS:
        tokens.pop()

    return " ".join(tokens)


def _clean_name_with_years(value: Any) -> Tuple[str, int | None, int | None]:
    birth_year, death_year = _extract_year_range(value)
    return _clean_name(str(value or "")), birth_year, death_year


def _extract_claims_from_text(
    text: str,
    *,
    source: str,
    page_idx: int | None,
) -> Iterator[Claim]:
    compact_text = " ".join(text.split())

    for pattern in _PARENT_PATTERNS:
        for match in pattern.finditer(compact_text):
            child, child_birth_year, child_death_year = _clean_name_with_years(
                match.group("child")
            )
            parent1, parent1_birth_year, parent1_death_year = _clean_name_with_years(
                match.group("parent1")
            )
            parent2, parent2_birth_year, parent2_death_year = _clean_name_with_years(
                match.group("parent2") or ""
            )

            if not child or not parent1:
                continue

            parents = [{"name": parent1}]
            if parent1_birth_year is not None:
                parents[0]["birth_year"] = parent1_birth_year
            if parent1_death_year is not None:
                parents[0]["death_year"] = parent1_death_year
            if parent2:
                parent2_payload: Dict[str, Any] = {"name": parent2}
                if parent2_birth_year is not None:
                    parent2_payload["birth_year"] = parent2_birth_year
                if parent2_death_year is not None:
                    parent2_payload["death_year"] = parent2_death_year
                parents.append(parent2_payload)

            child_payload: Dict[str, Any] = {"name": child}
            if child_birth_year is not None:
                child_payload["birth_year"] = child_birth_year
            if child_death_year is not None:
                child_payload["death_year"] = child_death_year

            yield Claim(
                claim_type="parent_child",
                confidence=0.8,
                data={"parents": parents, "child": child_payload},
                evidence=[
                    Evidence(file_path=source, page_idx=page_idx, quote=match.group(0))
                ],
            )

    for pattern in _SPOUSE_PATTERNS:
        for match in pattern.finditer(compact_text):
            person1, person1_birth_year, person1_death_year = _clean_name_with_years(
                match.group("person1")
            )
            person2, person2_birth_year, person2_death_year = _clean_name_with_years(
                match.group("person2")
            )
            if not person1 or not person2:
                continue

            person1_payload: Dict[str, Any] = {"name": person1}
            if person1_birth_year is not None:
                person1_payload["birth_year"] = person1_birth_year
            if person1_death_year is not None:
                person1_payload["death_year"] = person1_death_year

            person2_payload: Dict[str, Any] = {"name": person2}
            if person2_birth_year is not None:
                person2_payload["birth_year"] = person2_birth_year
            if person2_death_year is not None:
                person2_payload["death_year"] = person2_death_year

            yield Claim(
                claim_type="spouse",
                confidence=0.75,
                data={"person1": person1_payload, "person2": person2_payload},
                evidence=[
                    Evidence(file_path=source, page_idx=page_idx, quote=match.group(0))
                ],
            )


def _iter_claims_from_content_items(
    items: Iterable[Dict[str, Any]],
    source: str,
) -> Iterator[Claim]:
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue

        text = str(item.get("text") or "").strip()
        if not text:
            continue

        page_raw = item.get("page_idx")
        page_idx = int(page_raw) if isinstance(page_raw, int) else None
        yield from _extract_claims_from_text(text, source=source, page_idx=page_idx)


extract_claims_from_text = _extract_claims_from_text
iter_claims_from_content_items = _iter_claims_from_content_items
clean_name = _clean_name
coerce_claim_year = _coerce_claim_year
