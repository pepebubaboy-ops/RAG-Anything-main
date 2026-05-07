from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .normalize import normalize_name


@dataclass(frozen=True, slots=True)
class ResolvedGenealogyQuery:
    person_id: str
    name: str
    intent: str
    matched_text: str
    confidence: float


_PARENT_INTENT_TOKENS = {
    "parent",
    "parents",
    "father",
    "mother",
    "родитель",
    "родители",
    "отец",
    "мать",
    "папа",
    "мама",
}
_CHILD_INTENT_TOKENS = {
    "child",
    "children",
    "son",
    "sons",
    "daughter",
    "daughters",
    "descendant",
    "descendants",
    "kids",
    "ребенок",
    "ребёнок",
    "дети",
    "сын",
    "сыновья",
    "дочь",
    "дочери",
    "потомок",
    "потомки",
}
_SPOUSE_INTENT_TOKENS = {
    "spouse",
    "spouses",
    "wife",
    "wives",
    "husband",
    "husbands",
    "married",
    "marriage",
    "супруг",
    "супруга",
    "супруги",
    "муж",
    "жена",
    "брак",
    "женат",
    "замужем",
}
_RELATIONSHIP_INTENT_TOKENS = {
    "connected",
    "connection",
    "connections",
    "related",
    "relation",
    "relations",
    "relationship",
    "relationships",
    "связан",
    "связана",
    "связаны",
    "связь",
    "связи",
    "родство",
}


def _artifact_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        return resolved
    return resolved.parent


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload is not None else default


def detect_genealogy_intent(query: str) -> str:
    tokens = set(normalize_name(query).split())
    if tokens & _PARENT_INTENT_TOKENS:
        return "parents"
    if tokens & _CHILD_INTENT_TOKENS:
        return "children"
    if tokens & _SPOUSE_INTENT_TOKENS:
        return "spouses"
    if tokens & _RELATIONSHIP_INTENT_TOKENS:
        return "relationships"
    return "person"


def _person_variants(
    person: dict[str, Any],
    resolution_surfaces: dict[str, set[str]],
) -> set[str]:
    variants = {
        str(person.get("name") or ""),
        str(person.get("normalized_name") or ""),
    }
    for alias in person.get("aliases") or []:
        variants.add(str(alias or ""))
    variants.update(resolution_surfaces.get(str(person.get("person_id") or ""), set()))
    return {variant for variant in variants if normalize_name(variant)}


def _resolution_surfaces_by_person(resolution: dict[str, Any]) -> dict[str, set[str]]:
    surfaces_by_person: dict[str, set[str]] = {}
    for row in resolution.get("resolved") or []:
        if not isinstance(row, dict):
            continue
        person_id = str(row.get("person_id") or "")
        if not person_id:
            continue
        surfaces = surfaces_by_person.setdefault(person_id, set())
        for surface in row.get("surfaces") or []:
            if str(surface or "").strip():
                surfaces.add(str(surface))
    return surfaces_by_person


def _contains_normalized_phrase(query: str, phrase: str) -> bool:
    query_tokens = normalize_name(query).split()
    phrase_tokens = normalize_name(phrase).split()
    if not query_tokens or not phrase_tokens:
        return False
    phrase_size = len(phrase_tokens)
    if phrase_size > len(query_tokens):
        return False
    for index in range(0, len(query_tokens) - phrase_size + 1):
        if query_tokens[index : index + phrase_size] == phrase_tokens:
            return True
    return False


def resolve_query_person(
    query: str,
    people: Sequence[dict[str, Any]],
    resolution: dict[str, Any] | None = None,
) -> ResolvedGenealogyQuery | None:
    resolution_surfaces = _resolution_surfaces_by_person(resolution or {})
    best: tuple[float, int, str, dict[str, Any]] | None = None

    for person in people:
        person_id = str(person.get("person_id") or "")
        if not person_id:
            continue
        for variant in _person_variants(person, resolution_surfaces):
            normalized_variant = normalize_name(variant)
            if not _contains_normalized_phrase(query, normalized_variant):
                continue
            token_count = len(normalized_variant.split())
            score = 0.7 + min(0.25, 0.04 * token_count)
            candidate = (score, token_count, normalized_variant, person)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    if best is None:
        return None

    score, _token_count, matched_text, person = best
    return ResolvedGenealogyQuery(
        person_id=str(person.get("person_id") or ""),
        name=str(person.get("name") or person.get("person_id") or ""),
        intent=detect_genealogy_intent(query),
        matched_text=matched_text,
        confidence=score,
    )


def resolve_genealogy_query(
    query: str,
    artifact_path: Path,
) -> ResolvedGenealogyQuery | None:
    directory = _artifact_dir(artifact_path)
    people = _load_json(directory / "people.json", [])
    if not isinstance(people, list):
        return None
    resolution = _load_json(directory / "person_resolution.json", {})
    if not isinstance(resolution, dict):
        resolution = {}
    return resolve_query_person(query, people, resolution)
