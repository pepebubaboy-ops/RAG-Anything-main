from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .normalize import normalize_name
from .rag_index import write_jsonl


PARENT_MIN_AGE_GAP = 12
PARENT_MAX_AGE_GAP = 80


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class KnowledgeGraphArtifact:
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


def claim_id_for_row(row: Dict[str, Any]) -> str:
    return stable_id(
        "claim",
        row.get("claim_type"),
        json.dumps(row.get("data") or {}, ensure_ascii=False, sort_keys=True),
        json.dumps(row.get("evidence") or [], ensure_ascii=False, sort_keys=True),
    )


def evidence_id_for_row(row: Dict[str, Any]) -> str:
    return stable_id(
        "evidence",
        row.get("file_path"),
        row.get("doc_id"),
        row.get("chunk_id"),
        row.get("page_idx"),
        row.get("quote"),
        row.get("image_path"),
    )


def write_evidences(output_dir: Path, claim_rows: Sequence[Dict[str, Any]]) -> Path:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for claim in claim_rows:
        claim_id = str(claim.get("claim_id") or "")
        for evidence in claim.get("evidence") or []:
            if not isinstance(evidence, dict):
                continue
            evidence_id = str(
                evidence.get("evidence_id") or evidence_id_for_row(evidence)
            )
            key = (claim_id, evidence_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "claim_id": claim_id,
                    "source_id": evidence.get("doc_id"),
                    "chunk_id": evidence.get("chunk_id"),
                    "file_path": evidence.get("file_path"),
                    "page_idx": evidence.get("page_idx"),
                    "quote": evidence.get("quote"),
                    "image_path": evidence.get("image_path"),
                }
            )

    path = output_dir / "evidences.jsonl"
    write_jsonl(path, rows)
    return path


def _year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year <= 0 or year > 2500:
        return None
    return year


def _person_key_from_payload(
    payload: Dict[str, Any],
) -> tuple[str, int | None, int | None]:
    return (
        normalize_name(str(payload.get("name") or "")),
        _year(payload.get("birth_year")),
        _year(payload.get("death_year")),
    )


def _index_people(people: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for person in people:
        names = {
            str(person.get("name") or ""),
            str(person.get("normalized_name") or ""),
            *[str(alias) for alias in (person.get("aliases") or [])],
        }
        for name in names:
            normalized = normalize_name(name)
            if not normalized:
                continue
            index.setdefault(normalized, []).append(person)
    return index


def _resolve_person(
    payload: Dict[str, Any],
    person_index: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any] | None:
    normalized, birth_year, death_year = _person_key_from_payload(payload)
    candidates = list(person_index.get(normalized) or [])
    if not candidates:
        return None

    def compatible(candidate: Dict[str, Any]) -> bool:
        candidate_birth = _year(candidate.get("birth_year"))
        candidate_death = _year(candidate.get("death_year"))
        if (
            birth_year is not None
            and candidate_birth is not None
            and birth_year != candidate_birth
        ):
            return False
        if (
            death_year is not None
            and candidate_death is not None
            and death_year != candidate_death
        ):
            return False
        return True

    compatible_candidates = [
        candidate for candidate in candidates if compatible(candidate)
    ]
    if compatible_candidates:
        candidates = compatible_candidates
    return sorted(candidates, key=lambda row: str(row.get("person_id") or ""))[0]


def _person_name(person: Dict[str, Any] | None) -> str:
    if not person:
        return ""
    return str(person.get("name") or person.get("person_id") or "")


def _relationship_stable_key(
    relationship_type: str,
    source_person: Dict[str, Any],
    target_person: Dict[str, Any],
    claim_ids: Sequence[str],
) -> str:
    source_name = normalize_name(_person_name(source_person))
    target_name = normalize_name(_person_name(target_person))
    claim_key = "|".join(sorted(claim_ids))
    return f"{relationship_type}|{source_name}|{target_name}|{claim_key}"


def _relationship_confidence(existing: float, incoming: Any) -> float:
    try:
        return max(existing, float(incoming))
    except (TypeError, ValueError):
        return existing


def _add_relationship(
    relationships: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    relationship_type: str,
    source_person: Dict[str, Any],
    target_person: Dict[str, Any],
    claim: Dict[str, Any],
    symmetric: bool,
    derived: bool = False,
    inference_rule: str | None = None,
) -> None:
    source_person_id = str(source_person.get("person_id") or "")
    target_person_id = str(target_person.get("person_id") or "")
    if symmetric and source_person_id > target_person_id:
        source_person, target_person = target_person, source_person
        source_person_id, target_person_id = target_person_id, source_person_id

    claim_id = str(claim.get("claim_id") or "")
    evidence_ids = [
        str(row.get("evidence_id") or "")
        for row in claim.get("evidence") or []
        if isinstance(row, dict) and str(row.get("evidence_id") or "")
    ]
    key = (source_person_id, relationship_type, target_person_id)
    existing = relationships.get(key)
    if existing is None:
        relationships[key] = {
            "relationship_id": stable_id(
                "rel",
                _relationship_stable_key(
                    relationship_type,
                    source_person,
                    target_person,
                    [claim_id],
                ),
            ),
            "relationship_type": relationship_type,
            "source_person_id": source_person_id,
            "target_person_id": target_person_id,
            "claim_ids": [claim_id] if claim_id else [],
            "evidence_ids": sorted(set(evidence_ids)),
            "confidence": float(claim.get("confidence") or 0.0),
            "support_count": 1,
            "status": "accepted" if claim.get("applied", True) else "pending",
            "derived": derived,
            "symmetric": symmetric,
            "inference_rule": inference_rule,
        }
        return

    if claim_id and claim_id not in existing["claim_ids"]:
        existing["claim_ids"].append(claim_id)
    existing["evidence_ids"] = sorted(
        set(existing.get("evidence_ids") or []) | set(evidence_ids)
    )
    existing["confidence"] = _relationship_confidence(
        float(existing.get("confidence") or 0.0),
        claim.get("confidence"),
    )
    existing["support_count"] = len(existing["claim_ids"]) or int(
        existing.get("support_count") or 1
    )
    if existing["status"] == "pending" and claim.get("applied", True):
        existing["status"] = "accepted"


def _conflict(
    conflict_type: str,
    *,
    relationship_ids: Sequence[str] = (),
    claim_ids: Sequence[str] = (),
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "conflict_id": stable_id(
            "conflict",
            conflict_type,
            "|".join(sorted(relationship_ids)),
            "|".join(sorted(claim_ids)),
            json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
        ),
        "conflict_type": conflict_type,
        "relationship_ids": sorted(relationship_ids),
        "claim_ids": sorted(claim_ids),
        "status": "open",
        "details": details or {},
    }


def _resolution_mentions_by_normalized_name(
    resolution: Dict[str, Any] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not resolution:
        return {}

    index: Dict[str, List[Dict[str, Any]]] = {}
    for section, status in (
        ("unresolved_mentions", "unresolved"),
        ("ambiguous_mentions", "ambiguous"),
    ):
        for row in resolution.get(section) or []:
            if not isinstance(row, dict):
                continue
            normalized = normalize_name(str(row.get("normalized_name") or ""))
            if not normalized:
                continue
            index.setdefault(normalized, []).append(
                {
                    "mention_id": row.get("mention_id"),
                    "surface": row.get("surface"),
                    "normalized_name": normalized,
                    "candidate_person_ids": row.get("candidate_person_ids") or [],
                    "resolution_status": status,
                    "reason": row.get("reason"),
                }
            )

    for row in resolution.get("resolved") or []:
        if not isinstance(row, dict):
            continue
        normalized = normalize_name(str(row.get("normalized_name") or ""))
        if not normalized:
            continue
        for mention_id in row.get("mention_ids") or []:
            index.setdefault(normalized, []).append(
                {
                    "mention_id": mention_id,
                    "surface": None,
                    "normalized_name": normalized,
                    "candidate_person_ids": [row.get("person_id")],
                    "resolution_status": "resolved",
                    "reason": "normalized_name_match",
                }
            )
    return index


def _mention_candidates_for_payload(
    payload: Dict[str, Any],
    mentions_by_name: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    normalized = normalize_name(str(payload.get("name") or ""))
    if not normalized:
        return []
    return sorted(
        mentions_by_name.get(normalized) or [],
        key=lambda row: str(row.get("mention_id") or ""),
    )


def _relationship_id(row: Dict[str, Any]) -> str:
    return str(row.get("relationship_id") or "")


def _find_parent_path(
    adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    *,
    start_id: str,
    goal_id: str,
    skip_relationship_id: str,
) -> List[Dict[str, Any]]:
    stack: List[Tuple[str, List[Dict[str, Any]]]] = [(start_id, [])]
    visited = {start_id}
    while stack:
        person_id, path = stack.pop()
        for child_id, relationship in adjacency.get(person_id) or []:
            if _relationship_id(relationship) == skip_relationship_id:
                continue
            next_path = [*path, relationship]
            if child_id == goal_id:
                return next_path
            if child_id in visited:
                continue
            visited.add(child_id)
            stack.append((child_id, next_path))
    return []


def _validate_relationships(
    relationships: List[Dict[str, Any]],
    person_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []

    for relationship in relationships:
        source_id = str(relationship.get("source_person_id") or "")
        target_id = str(relationship.get("target_person_id") or "")
        if source_id == target_id:
            relationship["status"] = "conflict"
            conflicts.append(
                _conflict(
                    "self_relationship",
                    relationship_ids=[str(relationship.get("relationship_id") or "")],
                    claim_ids=relationship.get("claim_ids") or [],
                    details={"person_id": source_id},
                )
            )
            continue

        if relationship.get("relationship_type") != "parent_of":
            continue

        source = person_by_id.get(source_id) or {}
        target = person_by_id.get(target_id) or {}
        parent_birth = _year(source.get("birth_year"))
        child_birth = _year(target.get("birth_year"))
        if parent_birth is None or child_birth is None:
            continue
        age_gap = child_birth - parent_birth
        if age_gap < PARENT_MIN_AGE_GAP or age_gap > PARENT_MAX_AGE_GAP:
            relationship["status"] = "conflict"
            conflicts.append(
                _conflict(
                    "implausible_parent_age_gap",
                    relationship_ids=[str(relationship.get("relationship_id") or "")],
                    claim_ids=relationship.get("claim_ids") or [],
                    details={
                        "parent_person_id": source_id,
                        "child_person_id": target_id,
                        "parent_birth_year": parent_birth,
                        "child_birth_year": child_birth,
                        "age_gap": age_gap,
                    },
                )
            )

    relationship_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for relationship in relationships:
        key = (
            str(relationship.get("source_person_id") or ""),
            str(relationship.get("relationship_type") or ""),
            str(relationship.get("target_person_id") or ""),
        )
        relationship_groups.setdefault(key, []).append(relationship)

    for key, duplicate_relationships in relationship_groups.items():
        if len(duplicate_relationships) <= 1:
            continue
        for relationship in duplicate_relationships:
            if relationship.get("status") == "accepted":
                relationship["status"] = "conflict"
        conflicts.append(
            _conflict(
                "duplicate_relationship_conflict",
                relationship_ids=[
                    _relationship_id(relationship)
                    for relationship in duplicate_relationships
                ],
                claim_ids=[
                    claim_id
                    for relationship in duplicate_relationships
                    for claim_id in (relationship.get("claim_ids") or [])
                ],
                details={
                    "source_person_id": key[0],
                    "relationship_type": key[1],
                    "target_person_id": key[2],
                },
            )
        )

    parent_rels_by_child: Dict[str, List[Dict[str, Any]]] = {}
    for relationship in relationships:
        if relationship.get("relationship_type") != "parent_of":
            continue
        if relationship.get("status") not in {"accepted", "conflict"}:
            continue
        parent_rels_by_child.setdefault(
            str(relationship.get("target_person_id") or ""),
            [],
        ).append(relationship)

    for child_id, parent_relationships in parent_rels_by_child.items():
        if len(parent_relationships) <= 2:
            continue
        for relationship in parent_relationships:
            if relationship.get("status") == "accepted":
                relationship["status"] = "conflict"
        conflicts.append(
            _conflict(
                "too_many_parent_relationships",
                relationship_ids=[
                    str(relationship.get("relationship_id") or "")
                    for relationship in parent_relationships
                ],
                claim_ids=[
                    claim_id
                    for relationship in parent_relationships
                    for claim_id in (relationship.get("claim_ids") or [])
                ],
                details={
                    "child_person_id": child_id,
                    "parent_person_ids": sorted(
                        str(relationship.get("source_person_id") or "")
                        for relationship in parent_relationships
                    ),
                    "limit": 2,
                },
            )
        )

    accepted_parent_relationships = [
        relationship
        for relationship in relationships
        if relationship.get("relationship_type") == "parent_of"
        and relationship.get("status") == "accepted"
    ]
    parent_adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for relationship in accepted_parent_relationships:
        parent_adjacency.setdefault(
            str(relationship.get("source_person_id") or ""),
            [],
        ).append((str(relationship.get("target_person_id") or ""), relationship))

    seen_cycle_relationship_ids: set[Tuple[str, ...]] = set()
    for relationship in accepted_parent_relationships:
        source_id = str(relationship.get("source_person_id") or "")
        target_id = str(relationship.get("target_person_id") or "")
        path = _find_parent_path(
            parent_adjacency,
            start_id=target_id,
            goal_id=source_id,
            skip_relationship_id=_relationship_id(relationship),
        )
        if not path:
            continue

        cycle_relationships = [relationship, *path]
        relationship_ids = sorted(
            {
                _relationship_id(cycle_relationship)
                for cycle_relationship in cycle_relationships
                if _relationship_id(cycle_relationship)
            }
        )
        cycle_key = tuple(relationship_ids)
        if cycle_key in seen_cycle_relationship_ids:
            continue
        seen_cycle_relationship_ids.add(cycle_key)

        for cycle_relationship in cycle_relationships:
            if cycle_relationship.get("status") == "accepted":
                cycle_relationship["status"] = "conflict"
        conflicts.append(
            _conflict(
                "parent_cycle",
                relationship_ids=relationship_ids,
                claim_ids=[
                    claim_id
                    for cycle_relationship in cycle_relationships
                    for claim_id in (cycle_relationship.get("claim_ids") or [])
                ],
                details={
                    "person_ids": sorted(
                        {
                            str(cycle_relationship.get("source_person_id") or "")
                            for cycle_relationship in cycle_relationships
                        }
                        | {
                            str(cycle_relationship.get("target_person_id") or "")
                            for cycle_relationship in cycle_relationships
                        }
                    )
                },
            )
        )

    return sorted(conflicts, key=lambda row: str(row.get("conflict_id") or ""))


def build_knowledge_graph_artifact(
    *,
    people: Sequence[Dict[str, Any]],
    claim_rows: Sequence[Dict[str, Any]],
    resolution: Dict[str, Any] | None = None,
) -> KnowledgeGraphArtifact:
    person_index = _index_people(people)
    mentions_by_name = _resolution_mentions_by_normalized_name(resolution)
    relationships: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []

    for claim in claim_rows:
        claim_type = str(claim.get("claim_type") or "")
        data = claim.get("data") or {}
        if claim_type == "parent_child":
            child_payload = data.get("child") or {}
            child = _resolve_person(child_payload, person_index)
            parent_payloads = data.get("parents") or []
            if child is None:
                conflicts.append(
                    _conflict(
                        "unresolved_claim_person",
                        claim_ids=[str(claim.get("claim_id") or "")],
                        details={
                            "role": "child",
                            "payload": child_payload,
                            "mention_candidates": _mention_candidates_for_payload(
                                child_payload,
                                mentions_by_name,
                            ),
                        },
                    )
                )
                continue
            for parent_payload in parent_payloads:
                parent = _resolve_person(parent_payload, person_index)
                if parent is None:
                    conflicts.append(
                        _conflict(
                            "unresolved_claim_person",
                            claim_ids=[str(claim.get("claim_id") or "")],
                            details={
                                "role": "parent",
                                "payload": parent_payload,
                                "mention_candidates": _mention_candidates_for_payload(
                                    parent_payload,
                                    mentions_by_name,
                                ),
                            },
                        )
                    )
                    continue
                _add_relationship(
                    relationships,
                    relationship_type="parent_of",
                    source_person=parent,
                    target_person=child,
                    claim=claim,
                    symmetric=False,
                )
            continue

        if claim_type == "spouse":
            person1_payload = data.get("person1") or {}
            person2_payload = data.get("person2") or {}
            person1 = _resolve_person(person1_payload, person_index)
            person2 = _resolve_person(person2_payload, person_index)
            if person1 is None or person2 is None:
                conflicts.append(
                    _conflict(
                        "unresolved_claim_person",
                        claim_ids=[str(claim.get("claim_id") or "")],
                        details={
                            "role": "spouse",
                            "person1": person1_payload,
                            "person2": person2_payload,
                            "person1_mention_candidates": _mention_candidates_for_payload(
                                person1_payload,
                                mentions_by_name,
                            ),
                            "person2_mention_candidates": _mention_candidates_for_payload(
                                person2_payload,
                                mentions_by_name,
                            ),
                        },
                    )
                )
                continue
            _add_relationship(
                relationships,
                relationship_type="spouse_of",
                source_person=person1,
                target_person=person2,
                claim=claim,
                symmetric=True,
            )

    relationship_rows = [relationships[key] for key in sorted(relationships)]
    person_by_id = {str(person.get("person_id") or ""): person for person in people}
    conflicts.extend(_validate_relationships(relationship_rows, person_by_id))
    conflicts = sorted(
        {str(row["conflict_id"]): row for row in conflicts}.values(),
        key=lambda row: str(row.get("conflict_id") or ""),
    )
    return KnowledgeGraphArtifact(
        relationships=relationship_rows,
        conflicts=conflicts,
    )


def write_knowledge_graph_artifacts(
    output_dir: Path,
    *,
    people: Sequence[Dict[str, Any]],
    claim_rows: Sequence[Dict[str, Any]],
    resolution: Dict[str, Any] | None = None,
) -> KnowledgeGraphArtifact:
    artifact = build_knowledge_graph_artifact(
        people=people,
        claim_rows=claim_rows,
        resolution=resolution,
    )
    relationships_path = output_dir / "relationships.json"
    relationships_path.write_text(
        json.dumps(artifact.relationships, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    conflicts_path = output_dir / "conflicts.json"
    conflicts_path.write_text(
        json.dumps(artifact.conflicts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact
