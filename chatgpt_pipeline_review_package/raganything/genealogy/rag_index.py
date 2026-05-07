from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    source_id: str
    path: str
    title: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    chunk_id: str
    source_id: str
    text: str
    ordinal: int
    page_idx: int | None = None
    content_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenealogyRAGDocument:
    document_id: str
    kind: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def source_document_for_path(path: str | Path) -> SourceDocument:
    raw_path = Path(path).expanduser()
    resolved = str(raw_path.resolve(strict=False))
    return SourceDocument(
        source_id=_stable_id("source", resolved),
        path=resolved,
        title=raw_path.name or resolved,
        metadata={},
    )


def source_chunk_from_content_item(
    source: SourceDocument,
    item: Dict[str, Any],
    ordinal: int,
) -> SourceChunk | None:
    if not isinstance(item, dict):
        return None
    if item.get("type") != "text":
        return None

    text = str(item.get("text") or "").strip()
    if not text:
        return None

    page_raw = item.get("page_idx")
    page_idx = int(page_raw) if isinstance(page_raw, int) else None
    chunk_id = _stable_id("chunk", source.source_id, page_idx, ordinal, text)
    return SourceChunk(
        chunk_id=chunk_id,
        source_id=source.source_id,
        text=text,
        ordinal=ordinal,
        page_idx=page_idx,
        content_type="text",
        metadata={
            "source_path": source.path,
            "source_title": source.title,
        },
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                yield row


def write_sources(output_dir: Path, sources: Sequence[SourceDocument]) -> Path:
    path = output_dir / "sources.json"
    path.write_text(
        json.dumps([asdict(source) for source in sources], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def write_source_chunks(output_dir: Path, chunks: Sequence[SourceChunk]) -> Path:
    path = output_dir / "source_chunks.jsonl"
    write_jsonl(path, (asdict(chunk) for chunk in chunks))
    return path


def _person_lookup(people: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("person_id") or ""): row for row in people}


def _person_name(person_by_id: Dict[str, Dict[str, Any]], person_id: str) -> str:
    person = person_by_id.get(person_id) or {}
    return str(person.get("name") or person_id)


def _format_years(row: Dict[str, Any]) -> str:
    birth_year = row.get("birth_year")
    death_year = row.get("death_year")
    if birth_year or death_year:
        return f" ({birth_year or '?'}-{death_year or '?'})"
    return ""


def _rag_doc(kind: str, stable_key: str, text: str, metadata: Dict[str, Any]) -> GenealogyRAGDocument:
    return GenealogyRAGDocument(
        document_id=_stable_id("rag", kind, stable_key),
        kind=kind,
        text=text,
        metadata=metadata,
    )


def _claim_text(row: Dict[str, Any]) -> str:
    claim_type = str(row.get("claim_type") or "claim")
    confidence = row.get("confidence")
    data = row.get("data") or {}
    evidence = row.get("evidence") or []

    if claim_type == "parent_child":
        child = data.get("child") or {}
        parents = data.get("parents") or []
        child_name = str(child.get("name") or "unknown child")
        parent_names = ", ".join(str(parent.get("name") or "") for parent in parents)
        base = f"Genealogy claim: {child_name} is child of {parent_names}."
    elif claim_type == "spouse":
        person1 = data.get("person1") or {}
        person2 = data.get("person2") or {}
        base = (
            "Genealogy claim: "
            f"{person1.get('name') or 'unknown person'} is spouse of "
            f"{person2.get('name') or 'unknown person'}."
        )
    else:
        base = f"Genealogy claim ({claim_type}): {json.dumps(data, ensure_ascii=False)}."

    quotes = [
        str(item.get("quote") or "").strip()
        for item in evidence
        if isinstance(item, dict) and str(item.get("quote") or "").strip()
    ]
    quote_text = f" Evidence quote: {' '.join(quotes)}" if quotes else ""
    confidence_text = f" Confidence: {confidence}." if confidence is not None else ""
    return f"{base}{confidence_text}{quote_text}".strip()


def _claim_metadata(row: Dict[str, Any], ordinal: int) -> Dict[str, Any]:
    evidence = [item for item in row.get("evidence") or [] if isinstance(item, dict)]
    return {
        "claim_index": ordinal,
        "claim_id": row.get("claim_id"),
        "claim_type": row.get("claim_type"),
        "confidence": row.get("confidence"),
        "source_ids": sorted(
            {
                str(item.get("doc_id") or "")
                for item in evidence
                if str(item.get("doc_id") or "")
            }
        ),
        "chunk_ids": sorted(
            {
                str(item.get("chunk_id") or "")
                for item in evidence
                if str(item.get("chunk_id") or "")
            }
        ),
        "page_indices": sorted(
            {
                int(item["page_idx"])
                for item in evidence
                if isinstance(item.get("page_idx"), int)
            }
        ),
    }


def _accepted_relationships(
    relationships: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        row
        for row in relationships
        if str(row.get("status") or "").lower() == "accepted"
    ]


def _person_documents(
    people: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
) -> Iterator[GenealogyRAGDocument]:
    person_by_id = _person_lookup(people)
    accepted_relationships = _accepted_relationships(relationships)
    for person in people:
        person_id = str(person.get("person_id") or "")
        if not person_id:
            continue

        parents: set[str] = set()
        spouses: set[str] = set()
        children: set[str] = set()
        for relationship in accepted_relationships:
            relationship_type = str(relationship.get("relationship_type") or "")
            source_id = str(relationship.get("source_person_id") or "")
            target_id = str(relationship.get("target_person_id") or "")
            if relationship_type == "parent_of":
                if target_id == person_id:
                    parents.add(source_id)
                if source_id == person_id:
                    children.add(target_id)
            elif relationship_type == "spouse_of":
                if source_id == person_id:
                    spouses.add(target_id)
                if target_id == person_id:
                    spouses.add(source_id)

        lines = [f"Person: {person.get('name') or person_id}{_format_years(person)}."]
        if person.get("aliases"):
            lines.append(f"Aliases: {', '.join(str(alias) for alias in person['aliases'])}.")
        if parents:
            lines.append(
                "Parents: "
                + ", ".join(_person_name(person_by_id, parent_id) for parent_id in sorted(parents))
                + "."
            )
        if spouses:
            lines.append(
                "Spouses: "
                + ", ".join(_person_name(person_by_id, spouse_id) for spouse_id in sorted(spouses))
                + "."
            )
        if children:
            lines.append(
                "Children: "
                + ", ".join(_person_name(person_by_id, child_id) for child_id in sorted(children))
                + "."
            )

        yield _rag_doc(
            "person",
            person_id,
            " ".join(lines),
            {
                "person_id": person_id,
                "name": person.get("name"),
                "normalized_name": person.get("normalized_name"),
                "birth_year": person.get("birth_year"),
                "death_year": person.get("death_year"),
            },
        )


def _family_documents(
    people: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
) -> Iterator[GenealogyRAGDocument]:
    person_by_id = _person_lookup(people)
    accepted_relationships = _accepted_relationships(relationships)
    child_to_parents: Dict[str, set[str]] = {}
    spouse_pairs: set[tuple[str, ...]] = set()
    for relationship in accepted_relationships:
        relationship_type = str(relationship.get("relationship_type") or "")
        source_id = str(relationship.get("source_person_id") or "")
        target_id = str(relationship.get("target_person_id") or "")
        if not source_id or not target_id:
            continue
        if relationship_type == "parent_of":
            child_to_parents.setdefault(target_id, set()).add(source_id)
        elif relationship_type == "spouse_of":
            spouse_pairs.add(tuple(sorted((source_id, target_id))))

    children_by_parent_set: Dict[tuple[str, ...], set[str]] = {}
    for child_id, parent_ids in child_to_parents.items():
        if not parent_ids:
            continue
        children_by_parent_set.setdefault(tuple(sorted(parent_ids)), set()).add(child_id)

    family_keys = set(children_by_parent_set) | spouse_pairs
    for parent_key in sorted(family_keys):
        if not parent_key:
            continue
        family_id = _stable_id("family", *parent_key)
        parent_ids = list(parent_key)
        child_ids = sorted(children_by_parent_set.get(parent_key, set()))
        parent_names = [_person_name(person_by_id, person_id) for person_id in parent_ids]
        child_names = [_person_name(person_by_id, person_id) for person_id in child_ids]
        text = (
            f"Family: {family_id}. "
            f"Parents or spouses: {', '.join(parent_names) or 'unknown'}. "
            f"Children: {', '.join(child_names) or 'none recorded'}."
        )
        yield _rag_doc(
            "family",
            family_id,
            text,
            {
                "family_id": family_id,
                "family_type": "relationship_group",
                "parent_ids": parent_ids,
                "child_ids": child_ids,
                "source": "relationships",
            },
        )


def _relationship_documents(
    people: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
) -> Iterator[GenealogyRAGDocument]:
    person_by_id = _person_lookup(people)
    for relationship in relationships:
        relationship_id = str(relationship.get("relationship_id") or "")
        if not relationship_id:
            continue
        source_id = str(relationship.get("source_person_id") or "")
        target_id = str(relationship.get("target_person_id") or "")
        relationship_type = str(relationship.get("relationship_type") or "related_to")
        source_name = _person_name(person_by_id, source_id)
        target_name = _person_name(person_by_id, target_id)
        status = str(relationship.get("status") or "unknown")
        if status == "accepted":
            text = (
                f"Relationship: {source_name} {relationship_type} {target_name}. "
                f"Status: accepted. Confidence: {relationship.get('confidence') or 0}."
            )
        else:
            text = (
                "Conflict relationship, not accepted as fact: "
                f"{source_name} {relationship_type} {target_name}. "
                f"Status: {status}. Confidence: {relationship.get('confidence') or 0}."
            )
        claim_ids = [str(item) for item in relationship.get("claim_ids") or []]
        evidence_ids = [str(item) for item in relationship.get("evidence_ids") or []]
        yield _rag_doc(
            "relationship",
            relationship_id,
            text,
            {
                "relationship_id": relationship_id,
                "relationship_type": relationship_type,
                "source_person_id": source_id,
                "target_person_id": target_id,
                "claim_ids": claim_ids,
                "evidence_ids": evidence_ids,
                "status": relationship.get("status"),
                "derived": bool(relationship.get("derived")),
            },
        )


def _conflict_documents(
    people: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
    conflicts: Sequence[Dict[str, Any]],
) -> Iterator[GenealogyRAGDocument]:
    person_by_id = _person_lookup(people)
    relationship_by_id = {
        str(row.get("relationship_id") or ""): row for row in relationships
    }
    for conflict in conflicts:
        conflict_id = str(conflict.get("conflict_id") or "")
        if not conflict_id:
            continue
        conflict_type = str(conflict.get("conflict_type") or "conflict")
        relationship_ids = [
            str(item) for item in conflict.get("relationship_ids") or []
        ]
        claim_ids = [str(item) for item in conflict.get("claim_ids") or []]
        relation_texts: List[str] = []
        for relationship_id in relationship_ids:
            relationship = relationship_by_id.get(relationship_id)
            if not relationship:
                continue
            source_name = _person_name(
                person_by_id,
                str(relationship.get("source_person_id") or ""),
            )
            target_name = _person_name(
                person_by_id,
                str(relationship.get("target_person_id") or ""),
            )
            relation_texts.append(
                f"{source_name} {relationship.get('relationship_type')} {target_name}"
            )
        details = conflict.get("details") or {}
        text = (
            f"Graph conflict: {conflict_type}. "
            f"Affected relationships: {', '.join(relation_texts) or 'unknown'}. "
            f"Details: {json.dumps(details, ensure_ascii=False, sort_keys=True)}."
        )
        yield _rag_doc(
            "conflict",
            conflict_id,
            text,
            {
                "conflict_id": conflict_id,
                "conflict_type": conflict_type,
                "relationship_ids": relationship_ids,
                "claim_ids": claim_ids,
                "status": conflict.get("status"),
                "details": details,
            },
        )


def _mention_documents(
    mentions: Sequence[Dict[str, Any]],
) -> Iterator[GenealogyRAGDocument]:
    for mention in mentions:
        mention_id = str(mention.get("mention_id") or "")
        if not mention_id:
            continue
        surface = str(mention.get("surface") or "")
        normalized_name = str(mention.get("normalized_name") or "")
        candidate_person_ids = [
            str(item) for item in mention.get("candidate_person_ids") or []
        ]
        if candidate_person_ids:
            candidate_text = " Candidate people: " + ", ".join(candidate_person_ids) + "."
        else:
            candidate_text = " Candidate people: none resolved."
        text = (
            f"Mention: {surface}. Normalized name: {normalized_name}."
            f"{candidate_text}"
        )
        yield _rag_doc(
            "mention",
            mention_id,
            text,
            {
                "mention_id": mention_id,
                "source_id": mention.get("source_id"),
                "chunk_id": mention.get("chunk_id"),
                "surface": surface,
                "normalized_name": normalized_name,
                "page_idx": mention.get("page_idx"),
                "candidate_person_ids": candidate_person_ids,
            },
        )


def _resolution_documents(
    resolution: Dict[str, Any],
) -> Iterator[GenealogyRAGDocument]:
    for row in resolution.get("resolved") or []:
        if not isinstance(row, dict):
            continue
        person_id = str(row.get("person_id") or "")
        if not person_id:
            continue
        mention_ids = [str(item) for item in row.get("mention_ids") or []]
        claim_ids = [str(item) for item in row.get("claim_ids") or []]
        reasons = [str(item) for item in row.get("reasons") or []]
        text = (
            f"Resolution: mentions {', '.join(mention_ids) or 'none'} "
            f"resolve to person {row.get('name') or person_id}. "
            f"Reasons: {', '.join(reasons) or 'unknown'}. "
            f"Related claims: {', '.join(claim_ids) or 'none'}."
        )
        yield _rag_doc(
            "resolution",
            f"resolved:{person_id}",
            text,
            {
                "resolution_status": "resolved",
                "person_id": person_id,
                "mention_ids": mention_ids,
                "claim_ids": claim_ids,
                "reasons": reasons,
            },
        )

    for section, status in (
        ("unresolved_mentions", "unresolved"),
        ("ambiguous_mentions", "ambiguous"),
    ):
        for row in resolution.get(section) or []:
            if not isinstance(row, dict):
                continue
            mention_id = str(row.get("mention_id") or "")
            if not mention_id:
                continue
            candidate_person_ids = [
                str(item) for item in row.get("candidate_person_ids") or []
            ]
            text = (
                f"Resolution: mention {row.get('surface') or mention_id} is {status}. "
                f"Reason: {row.get('reason') or 'unknown'}. "
                f"Candidate people: {', '.join(candidate_person_ids) or 'none'}."
            )
            yield _rag_doc(
                "resolution",
                f"{status}:{mention_id}",
                text,
                {
                    "resolution_status": status,
                    "mention_id": mention_id,
                    "surface": row.get("surface"),
                    "normalized_name": row.get("normalized_name"),
                    "candidate_person_ids": candidate_person_ids,
                    "reason": row.get("reason"),
                },
            )


def build_rag_documents_from_artifacts(input_dir: Path) -> List[GenealogyRAGDocument]:
    input_dir = Path(input_dir).expanduser().resolve()
    people_path = input_dir / "people.json"
    relationships_path = input_dir / "relationships.json"
    conflicts_path = input_dir / "conflicts.json"
    claims_path = input_dir / "claims.jsonl"
    chunks_path = input_dir / "source_chunks.jsonl"
    mentions_path = input_dir / "mentions.jsonl"
    resolution_path = input_dir / "person_resolution.json"

    people = json.loads(people_path.read_text(encoding="utf-8")) if people_path.exists() else []
    relationships = (
        json.loads(relationships_path.read_text(encoding="utf-8"))
        if relationships_path.exists()
        else []
    )
    conflicts = (
        json.loads(conflicts_path.read_text(encoding="utf-8"))
        if conflicts_path.exists()
        else []
    )

    documents: List[GenealogyRAGDocument] = []
    documents.extend(_person_documents(people, relationships))
    documents.extend(_family_documents(people, relationships))
    documents.extend(_relationship_documents(people, relationships))
    documents.extend(_conflict_documents(people, relationships, conflicts))
    documents.extend(_mention_documents(list(read_jsonl(mentions_path))))

    if resolution_path.exists():
        resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
        if isinstance(resolution, dict):
            documents.extend(_resolution_documents(resolution))

    for ordinal, claim in enumerate(read_jsonl(claims_path), start=1):
        documents.append(
            _rag_doc(
                "claim",
                str(ordinal),
                _claim_text(claim),
                _claim_metadata(claim, ordinal),
            )
        )

    for chunk in read_jsonl(chunks_path):
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        documents.append(
            _rag_doc(
                "source_chunk",
                chunk_id,
                str(chunk.get("text") or ""),
                {
                    "source_id": chunk.get("source_id"),
                    "chunk_id": chunk_id,
                    "page_idx": chunk.get("page_idx"),
                    "source_path": (chunk.get("metadata") or {}).get("source_path"),
                    "source_title": (chunk.get("metadata") or {}).get("source_title"),
                },
            )
        )

    return documents


def write_rag_documents(
    input_dir: Path,
    output_path: Path | None = None,
) -> Path:
    input_dir = Path(input_dir).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else input_dir / "rag_documents.jsonl"
    )
    documents = build_rag_documents_from_artifacts(input_dir)
    write_jsonl(output_path, (asdict(document) for document in documents))
    return output_path
