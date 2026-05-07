from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .mentions import MentionRecord
from .normalize import normalize_name


def _year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year <= 0 or year > 2500:
        return None
    return year


def _person_id(person: dict[str, Any]) -> str:
    return str(person.get("person_id") or "")


def _person_name(person: dict[str, Any]) -> str:
    return str(person.get("name") or person.get("person_id") or "")


def _person_variants(person: dict[str, Any]) -> set[str]:
    variants = {
        normalize_name(str(person.get("name") or "")),
        normalize_name(str(person.get("normalized_name") or "")),
    }
    for alias in person.get("aliases") or []:
        variants.add(normalize_name(str(alias or "")))
    return {variant for variant in variants if variant}


def _index_people_by_variant(
    people: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for person in people:
        if not _person_id(person):
            continue
        for variant in _person_variants(person):
            index.setdefault(variant, []).append(person)
    for candidates in index.values():
        candidates.sort(key=_person_id)
    return index


def _iter_payload_names(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        name = payload.get("name")
        if isinstance(name, str) and name.strip():
            yield name
        for value in payload.values():
            yield from _iter_payload_names(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_payload_names(item)


def _claim_ids_by_normalized_name(
    claims: Sequence[dict[str, Any]],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id:
            continue
        for name in _iter_payload_names(claim.get("data") or {}):
            normalized = normalize_name(name)
            if normalized:
                index.setdefault(normalized, set()).add(claim_id)
    return index


def _candidate_years_compatible(
    mention: MentionRecord,
    person: dict[str, Any],
) -> bool:
    mention_birth = _year(mention.attributes.get("birth_year"))
    mention_death = _year(mention.attributes.get("death_year"))
    person_birth = _year(person.get("birth_year"))
    person_death = _year(person.get("death_year"))
    if mention_birth is not None and person_birth is not None and mention_birth != person_birth:
        return False
    if mention_death is not None and person_death is not None and mention_death != person_death:
        return False
    return True


def _mention_has_years(mention: MentionRecord) -> bool:
    return (
        _year(mention.attributes.get("birth_year")) is not None
        or _year(mention.attributes.get("death_year")) is not None
    )


def resolve_mentions_to_people(
    mentions: Sequence[MentionRecord],
    people: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    people_by_variant = _index_people_by_variant(people)
    claims_by_name = _claim_ids_by_normalized_name(claims)
    resolved_by_person: dict[str, dict[str, Any]] = {}
    unresolved_mentions: list[dict[str, Any]] = []
    ambiguous_mentions: list[dict[str, Any]] = []

    for mention in mentions:
        candidates = list(people_by_variant.get(mention.normalized_name) or [])
        compatible_candidates = [
            candidate
            for candidate in candidates
            if _candidate_years_compatible(mention, candidate)
        ]
        mention.candidate_person_ids = [_person_id(candidate) for candidate in compatible_candidates]

        if not candidates:
            unresolved_mentions.append(
                {
                    "mention_id": mention.mention_id,
                    "surface": mention.surface,
                    "normalized_name": mention.normalized_name,
                    "candidate_person_ids": [],
                    "reason": "no_matching_person",
                }
            )
            continue

        if not compatible_candidates:
            unresolved_mentions.append(
                {
                    "mention_id": mention.mention_id,
                    "surface": mention.surface,
                    "normalized_name": mention.normalized_name,
                    "candidate_person_ids": [_person_id(candidate) for candidate in candidates],
                    "reason": "year_mismatch",
                }
            )
            continue

        if len(compatible_candidates) > 1:
            ambiguous_mentions.append(
                {
                    "mention_id": mention.mention_id,
                    "surface": mention.surface,
                    "normalized_name": mention.normalized_name,
                    "candidate_person_ids": [
                        _person_id(candidate) for candidate in compatible_candidates
                    ],
                    "reason": "multiple_matching_people",
                }
            )
            continue

        person = compatible_candidates[0]
        person_id = _person_id(person)
        reasons = {"normalized_name_match"}
        if _mention_has_years(mention):
            reasons.add("year_compatible")

        claim_ids: set[str] = set()
        for variant in _person_variants(person):
            claim_ids.update(claims_by_name.get(variant, set()))

        row = resolved_by_person.setdefault(
            person_id,
            {
                "person_id": person_id,
                "name": _person_name(person),
                "normalized_name": normalize_name(str(person.get("normalized_name") or "")),
                "mention_ids": [],
                "surfaces": [],
                "claim_ids": [],
                "reasons": [],
            },
        )
        row["mention_ids"].append(mention.mention_id)
        if mention.surface not in row["surfaces"]:
            row["surfaces"].append(mention.surface)
        row["claim_ids"] = sorted(set(row["claim_ids"]) | claim_ids)
        row["reasons"] = sorted(set(row["reasons"]) | reasons)

    resolved = sorted(
        resolved_by_person.values(),
        key=lambda row: str(row.get("person_id") or ""),
    )
    for row in resolved:
        row["mention_ids"] = sorted(set(row["mention_ids"]))
        row["surfaces"] = sorted(set(row["surfaces"]))
        row["claim_ids"] = sorted(set(row["claim_ids"]))
        row["reasons"] = sorted(set(row["reasons"]))

    unresolved_mentions.sort(key=lambda row: str(row.get("mention_id") or ""))
    ambiguous_mentions.sort(key=lambda row: str(row.get("mention_id") or ""))
    return {
        "resolved": resolved,
        "unresolved_mentions": unresolved_mentions,
        "ambiguous_mentions": ambiguous_mentions,
        "summary": {
            "mentions_count": len(mentions),
            "resolved_mentions_count": sum(len(row["mention_ids"]) for row in resolved),
            "unresolved_mentions_count": len(unresolved_mentions),
            "ambiguous_mentions_count": len(ambiguous_mentions),
        },
    }


def write_person_resolution(output_dir: Path, resolution: dict[str, Any]) -> Path:
    path = output_dir / "person_resolution.json"
    path.write_text(
        json.dumps(resolution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
