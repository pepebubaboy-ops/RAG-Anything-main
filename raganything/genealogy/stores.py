from __future__ import annotations

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import Evidence, FamilyRecord, MediaSpec, PersonRecord, PersonSpec
from .normalize import coerce_year, normalize_name, normalize_place


class GenealogyStore(ABC):
    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def upsert_person(self, spec: PersonSpec) -> PersonRecord: ...

    @abstractmethod
    def upsert_family(self, parent_ids: List[str], family_type: str = "couple") -> FamilyRecord: ...

    @abstractmethod
    def link_parents_to_family(self, family_id: str, parent_ids: List[str]) -> None: ...

    @abstractmethod
    def link_child_to_family(self, family_id: str, child_id: str, props: Optional[Dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def link_spouses(self, family_id: str, person1_id: str, person2_id: str, props: Optional[Dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def create_claim(self, claim_type: str, confidence: float, data: Dict[str, Any], notes: Optional[str] = None) -> str: ...

    @abstractmethod
    def attach_evidence(self, claim_id: str, evidence: Evidence) -> str: ...

    @abstractmethod
    def update_person(self, person_id: str, spec: PersonSpec) -> None: ...

    @abstractmethod
    def upsert_media(
        self,
        media: MediaSpec,
    ) -> str: ...

    @abstractmethod
    def link_person_media(
        self, person_id: str, media_id: str, props: Optional[Dict[str, Any]] = None
    ) -> None: ...

    @abstractmethod
    def link_claim_to_person(self, claim_id: str, person_id: str, role: str) -> None: ...

    @abstractmethod
    def link_claim_to_family(self, claim_id: str, family_id: str, role: str) -> None: ...

    @abstractmethod
    def link_claim_to_media(self, claim_id: str, media_id: str, role: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def _hash_key(*parts: str) -> str:
    s = "\n".join(p for p in parts if p)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _as_neo4j_property_value(value: Any) -> Any:
    """Coerce Python values into Neo4j-compatible property values.

    Neo4j properties are limited to primitives or arrays of primitives.
    For richer structures we store JSON strings.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
            return [v for v in value if v is not None]
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


@dataclass
class InMemoryGenealogyStore(GenealogyStore):
    """A store implementation for tests and dry-runs."""

    people: Dict[str, PersonRecord]
    families: Dict[str, FamilyRecord]
    claims: Dict[str, Dict[str, Any]]
    evidences: Dict[str, Evidence]
    rel_parent_in: List[Tuple[str, str]]
    rel_has_child: List[Tuple[str, str]]
    rel_spouse_in: List[Tuple[str, str, str]]
    media: Dict[str, MediaSpec]
    rel_person_media: List[Tuple[str, str]]
    rel_claim_person: List[Tuple[str, str, str]]
    rel_claim_family: List[Tuple[str, str, str]]
    rel_claim_media: List[Tuple[str, str, str]]

    def __init__(self) -> None:
        self.people = {}
        self.families = {}
        self.claims = {}
        self.evidences = {}
        self.rel_parent_in = []
        self.rel_has_child = []
        self.rel_spouse_in = []
        self.media = {}
        self.rel_person_media = []
        self.rel_claim_person = []
        self.rel_claim_family = []
        self.rel_claim_media = []

    def ensure_schema(self) -> None:
        return None

    def _find_person(self, normalized_name: str, birth_year: Optional[int], birth_place: Optional[str]) -> Optional[PersonRecord]:
        for p in self.people.values():
            if p.normalized_name != normalized_name:
                continue
            if birth_year is not None and p.spec.birth_year is not None and p.spec.birth_year != birth_year:
                continue
            if birth_place is not None and p.spec.birth_place is not None and p.spec.birth_place != birth_place:
                continue
            return p
        return None

    def upsert_person(self, spec: PersonSpec) -> PersonRecord:
        normalized_name = normalize_name(spec.name)
        birth_year = coerce_year(spec.birth_year) or coerce_year(spec.birth_date)
        birth_place = normalize_place(spec.birth_place)

        existing = self._find_person(normalized_name, birth_year, birth_place)
        if existing:
            # Best-effort enrichment.
            merged = PersonSpec(
                name=existing.spec.name,
                birth_date=existing.spec.birth_date or spec.birth_date,
                death_date=existing.spec.death_date or spec.death_date,
                birth_year=existing.spec.birth_year or birth_year,
                death_year=existing.spec.death_year or coerce_year(spec.death_year),
                birth_place=existing.spec.birth_place or birth_place,
                death_place=existing.spec.death_place or normalize_place(spec.death_place),
                gender=existing.spec.gender or spec.gender,
                occupation=existing.spec.occupation or spec.occupation,
                biography=existing.spec.biography or spec.biography,
                aliases=sorted(set((existing.spec.aliases or []) + (spec.aliases or []))),
                media=list(existing.spec.media or []) + [m for m in (spec.media or []) if m not in (existing.spec.media or [])],
                extra={**spec.extra, **existing.spec.extra},
            )
            rec = PersonRecord(person_id=existing.person_id, spec=merged, normalized_name=normalized_name)
            self.people[existing.person_id] = rec
            return rec

        person_id = f"person-{uuid.uuid4()}"
        rec = PersonRecord(
            person_id=person_id,
            spec=PersonSpec(
                name=spec.name,
                birth_date=spec.birth_date,
                death_date=spec.death_date,
                birth_year=birth_year,
                death_year=coerce_year(spec.death_year),
                birth_place=birth_place,
                death_place=normalize_place(spec.death_place),
                gender=spec.gender,
                occupation=spec.occupation,
                biography=spec.biography,
                aliases=list(spec.aliases or []),
                media=list(spec.media or []),
                extra=dict(spec.extra or {}),
            ),
            normalized_name=normalized_name,
        )
        self.people[person_id] = rec
        return rec

    def update_person(self, person_id: str, spec: PersonSpec) -> None:
        existing = self.people.get(person_id)
        if not existing:
            return
        merged = PersonSpec(
            name=existing.spec.name,
            birth_date=existing.spec.birth_date or spec.birth_date,
            death_date=existing.spec.death_date or spec.death_date,
            birth_year=existing.spec.birth_year or coerce_year(spec.birth_year) or coerce_year(spec.birth_date),
            death_year=existing.spec.death_year or coerce_year(spec.death_year) or coerce_year(spec.death_date),
            birth_place=existing.spec.birth_place or normalize_place(spec.birth_place),
            death_place=existing.spec.death_place or normalize_place(spec.death_place),
            gender=existing.spec.gender or spec.gender,
            occupation=existing.spec.occupation or spec.occupation,
            biography=existing.spec.biography or spec.biography,
            aliases=sorted(set((existing.spec.aliases or []) + (spec.aliases or []))),
            media=list(existing.spec.media or []) + [m for m in (spec.media or []) if m not in (existing.spec.media or [])],
            extra={**spec.extra, **existing.spec.extra},
        )
        self.people[person_id] = PersonRecord(person_id=person_id, spec=merged, normalized_name=existing.normalized_name)

    def upsert_media(self, media: MediaSpec) -> str:
        key = _hash_key(media.kind, media.path)
        media_id = f"media-{key}"
        self.media[media_id] = media
        return media_id

    def link_person_media(
        self, person_id: str, media_id: str, props: Optional[Dict[str, Any]] = None
    ) -> None:
        edge = (person_id, media_id)
        if edge not in self.rel_person_media:
            self.rel_person_media.append(edge)

    def link_claim_to_person(self, claim_id: str, person_id: str, role: str) -> None:
        edge = (claim_id, person_id, role)
        if edge not in self.rel_claim_person:
            self.rel_claim_person.append(edge)

    def link_claim_to_family(self, claim_id: str, family_id: str, role: str) -> None:
        edge = (claim_id, family_id, role)
        if edge not in self.rel_claim_family:
            self.rel_claim_family.append(edge)

    def link_claim_to_media(self, claim_id: str, media_id: str, role: str) -> None:
        edge = (claim_id, media_id, role)
        if edge not in self.rel_claim_media:
            self.rel_claim_media.append(edge)

    def upsert_family(self, parent_ids: List[str], family_type: str = "couple") -> FamilyRecord:
        if len(parent_ids) >= 2:
            a, b = sorted(parent_ids[:2])
            family_key = f"parents:{a}:{b}"
            for f in self.families.values():
                if f.family_key == family_key:
                    return f
            family_id = f"family-{uuid.uuid4()}"
            rec = FamilyRecord(family_id=family_id, parent_ids=[a, b], family_type=family_type, family_key=family_key)
            self.families[family_id] = rec
            return rec

        family_id = f"family-{uuid.uuid4()}"
        rec = FamilyRecord(family_id=family_id, parent_ids=list(parent_ids), family_type=family_type, family_key=None)
        self.families[family_id] = rec
        return rec

    def link_parents_to_family(self, family_id: str, parent_ids: List[str]) -> None:
        for pid in parent_ids:
            edge = (pid, family_id)
            if edge not in self.rel_parent_in:
                self.rel_parent_in.append(edge)

    def link_child_to_family(self, family_id: str, child_id: str, props: Optional[Dict[str, Any]] = None) -> None:
        edge = (family_id, child_id)
        if edge not in self.rel_has_child:
            self.rel_has_child.append(edge)

    def link_spouses(self, family_id: str, person1_id: str, person2_id: str, props: Optional[Dict[str, Any]] = None) -> None:
        a, b = sorted([person1_id, person2_id])
        edge = (family_id, a, b)
        if edge not in self.rel_spouse_in:
            self.rel_spouse_in.append(edge)

    def create_claim(self, claim_type: str, confidence: float, data: Dict[str, Any], notes: Optional[str] = None) -> str:
        claim_id = f"claim-{uuid.uuid4()}"
        self.claims[claim_id] = {
            "claim_type": claim_type,
            "confidence": float(confidence),
            "data": dict(data or {}),
            "notes": notes,
            "created_at": int(time.time()),
        }
        return claim_id

    def attach_evidence(self, claim_id: str, evidence: Evidence) -> str:
        ev_key = _hash_key(
            str(evidence.file_path or ""),
            str(evidence.doc_id or ""),
            str(evidence.chunk_id or ""),
            str(evidence.page_idx or ""),
            str(evidence.quote or ""),
            str(evidence.image_path or ""),
        )
        evidence_id = f"evidence-{ev_key}"
        self.evidences[evidence_id] = evidence
        # Relationship is implied by claim payload in memory store.
        return evidence_id

    def close(self) -> None:
        return None


class Neo4jGenealogyStore(GenealogyStore):
    """Neo4j-backed canonical store.

    This uses the official neo4j Python driver (sync). Install with:
      uv add --dev neo4j
    """

    def __init__(self, uri: str, username: str, password: str, database: Optional[str] = None) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "neo4j Python driver is not installed. Install with: uv add --dev neo4j"
            ) from e

        self._GraphDatabase = GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        stmts = [
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE",
            "CREATE CONSTRAINT family_id IF NOT EXISTS FOR (f:Family) REQUIRE f.family_id IS UNIQUE",
            "CREATE CONSTRAINT family_key IF NOT EXISTS FOR (f:Family) REQUIRE f.family_key IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
            "CREATE CONSTRAINT media_id IF NOT EXISTS FOR (m:Media) REQUIRE m.media_id IS UNIQUE",
        ]
        with self._driver.session(database=self._database) as session:
            for s in stmts:
                session.run(s)

    def _run(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            res = session.run(cypher, params)
            return [r.data() for r in res]

    def upsert_person(self, spec: PersonSpec) -> PersonRecord:
        normalized_name = normalize_name(spec.name)
        birth_year = coerce_year(spec.birth_year) or coerce_year(spec.birth_date)
        birth_place = normalize_place(spec.birth_place)

        rows = self._run(
            """
            MATCH (p:Person)
            WHERE p.normalized_name = $normalized_name
              AND ($birth_year IS NULL OR p.birth_year IS NULL OR p.birth_year = $birth_year)
              AND ($birth_place IS NULL OR p.birth_place IS NULL OR p.birth_place = $birth_place)
            RETURN p.person_id AS person_id
            LIMIT 1
            """,
            {"normalized_name": normalized_name, "birth_year": birth_year, "birth_place": birth_place},
        )
        if rows:
            person_id = str(rows[0]["person_id"])
        else:
            person_id = f"person-{uuid.uuid4()}"

        props: Dict[str, Any] = {
            "primary_name": spec.name,
            "normalized_name": normalized_name or None,
            "birth_date": spec.birth_date,
            "death_date": spec.death_date,
            "birth_year": birth_year,
            "death_year": coerce_year(spec.death_year) or coerce_year(spec.death_date),
            "birth_place": birth_place,
            "death_place": normalize_place(spec.death_place),
            "gender": spec.gender,
            "occupation": spec.occupation,
            "biography": spec.biography,
            "extra_json": _as_neo4j_property_value(spec.extra) if spec.extra else None,
        }
        # Drop None to avoid overwriting with null.
        props = {k: v for k, v in props.items() if v is not None}

        self._run(
            """
            MERGE (p:Person {person_id: $person_id})
            ON CREATE SET p.created_at = timestamp()
            SET p += $props
            SET p.updated_at = timestamp()
            """,
            {"person_id": person_id, "props": props},
        )
        # Merge aliases without duplicates (avoid APOC dependency).
        if spec.aliases:
            self._run(
                """
                MATCH (p:Person {person_id: $person_id})
                SET p.aliases = coalesce(p.aliases, [])
                WITH p
                UNWIND $aliases AS a
                WITH p, a
                WHERE a IS NOT NULL AND trim(a) <> ""
                SET p.aliases = CASE WHEN a IN p.aliases THEN p.aliases ELSE p.aliases + a END
                """,
                {"person_id": person_id, "aliases": list(spec.aliases)},
            )
        return PersonRecord(person_id=person_id, spec=spec, normalized_name=normalized_name)

    def update_person(self, person_id: str, spec: PersonSpec) -> None:
        props: Dict[str, Any] = {
            "primary_name": spec.name,
            "birth_date": spec.birth_date,
            "death_date": spec.death_date,
            "birth_year": coerce_year(spec.birth_year) or coerce_year(spec.birth_date),
            "death_year": coerce_year(spec.death_year) or coerce_year(spec.death_date),
            "birth_place": normalize_place(spec.birth_place),
            "death_place": normalize_place(spec.death_place),
            "gender": spec.gender,
            "occupation": spec.occupation,
            "biography": spec.biography,
            "extra_json": _as_neo4j_property_value(spec.extra) if spec.extra else None,
        }
        props = {k: v for k, v in props.items() if v is not None}
        if props:
            self._run(
                """
                MATCH (p:Person {person_id: $person_id})
                SET p += $props
                SET p.updated_at = timestamp()
                """,
                {"person_id": person_id, "props": props},
            )
        if spec.aliases:
            self._run(
                """
                MATCH (p:Person {person_id: $person_id})
                SET p.aliases = coalesce(p.aliases, [])
                WITH p
                UNWIND $aliases AS a
                WITH p, a
                WHERE a IS NOT NULL AND trim(a) <> ""
                SET p.aliases = CASE WHEN a IN p.aliases THEN p.aliases ELSE p.aliases + a END
                """,
                {"person_id": person_id, "aliases": list(spec.aliases)},
            )

    def upsert_media(self, media: MediaSpec) -> str:
        key = _hash_key(media.kind, media.path)
        media_id = f"media-{key}"
        props: Dict[str, Any] = {
            "kind": media.kind,
            "path": media.path,
            "caption": media.caption,
        }
        props.update(
            {
                f"extra_{k}": _as_neo4j_property_value(v)
                for k, v in (media.extra or {}).items()
                if v is not None
            }
        )
        props = {k: v for k, v in props.items() if v is not None}
        self._run(
            """
            MERGE (m:Media {media_id: $media_id})
            ON CREATE SET m.created_at = timestamp()
            SET m += $props
            SET m.updated_at = timestamp()
            """,
            {"media_id": media_id, "props": props},
        )
        return media_id

    def link_person_media(
        self, person_id: str, media_id: str, props: Optional[Dict[str, Any]] = None
    ) -> None:
        rel_props = dict(props or {})
        self._run(
            """
            MATCH (p:Person {person_id: $person_id})
            MATCH (m:Media {media_id: $media_id})
            MERGE (p)-[r:HAS_MEDIA]->(m)
            SET r += $props
            """,
            {"person_id": person_id, "media_id": media_id, "props": rel_props},
        )

    def link_claim_to_person(self, claim_id: str, person_id: str, role: str) -> None:
        self._run(
            """
            MATCH (c:Claim {claim_id: $claim_id})
            MATCH (p:Person {person_id: $person_id})
            MERGE (c)-[:ABOUT_PERSON {role: $role}]->(p)
            """,
            {"claim_id": claim_id, "person_id": person_id, "role": role},
        )

    def link_claim_to_family(self, claim_id: str, family_id: str, role: str) -> None:
        self._run(
            """
            MATCH (c:Claim {claim_id: $claim_id})
            MATCH (f:Family {family_id: $family_id})
            MERGE (c)-[:ABOUT_FAMILY {role: $role}]->(f)
            """,
            {"claim_id": claim_id, "family_id": family_id, "role": role},
        )

    def link_claim_to_media(self, claim_id: str, media_id: str, role: str) -> None:
        self._run(
            """
            MATCH (c:Claim {claim_id: $claim_id})
            MATCH (m:Media {media_id: $media_id})
            MERGE (c)-[:ABOUT_MEDIA {role: $role}]->(m)
            """,
            {"claim_id": claim_id, "media_id": media_id, "role": role},
        )

    def upsert_family(self, parent_ids: List[str], family_type: str = "couple") -> FamilyRecord:
        family_id = f"family-{uuid.uuid4()}"
        family_key: Optional[str] = None
        parent_ids_norm = list(parent_ids)

        if len(parent_ids_norm) >= 2:
            a, b = sorted(parent_ids_norm[:2])
            parent_ids_norm = [a, b]
            family_key = f"parents:{a}:{b}"
            rows = self._run(
                "MATCH (f:Family {family_key: $family_key}) RETURN f.family_id AS family_id LIMIT 1",
                {"family_key": family_key},
            )
            if rows:
                return FamilyRecord(
                    family_id=str(rows[0]["family_id"]),
                    parent_ids=parent_ids_norm,
                    family_type=family_type,
                    family_key=family_key,
                )

            self._run(
                """
                CREATE (f:Family {family_id: $family_id})
                SET f.family_key = $family_key
                SET f.family_type = $family_type
                SET f.created_at = timestamp()
                SET f.updated_at = timestamp()
                """,
                {"family_id": family_id, "family_key": family_key, "family_type": family_type},
            )
            return FamilyRecord(family_id=family_id, parent_ids=parent_ids_norm, family_type=family_type, family_key=family_key)

        # Single-parent or unknown: create unique family without key.
        self._run(
            """
            CREATE (f:Family {family_id: $family_id})
            SET f.family_type = $family_type
            SET f.created_at = timestamp()
            SET f.updated_at = timestamp()
            """,
            {"family_id": family_id, "family_type": family_type},
        )
        return FamilyRecord(family_id=family_id, parent_ids=parent_ids_norm, family_type=family_type, family_key=None)

    def link_parents_to_family(self, family_id: str, parent_ids: List[str]) -> None:
        for pid in parent_ids:
            self._run(
                """
                MATCH (p:Person {person_id: $person_id})
                MATCH (f:Family {family_id: $family_id})
                MERGE (p)-[:PARENT_IN]->(f)
                """,
                {"person_id": pid, "family_id": family_id},
            )

    def link_child_to_family(self, family_id: str, child_id: str, props: Optional[Dict[str, Any]] = None) -> None:
        rel_props = dict(props or {})
        self._run(
            """
            MATCH (c:Person {person_id: $child_id})
            MATCH (f:Family {family_id: $family_id})
            MERGE (f)-[r:HAS_CHILD]->(c)
            SET r += $props
            """,
            {"child_id": child_id, "family_id": family_id, "props": rel_props},
        )

    def link_spouses(self, family_id: str, person1_id: str, person2_id: str, props: Optional[Dict[str, Any]] = None) -> None:
        rel_props = dict(props or {})
        self._run(
            """
            MATCH (a:Person {person_id: $a})
            MATCH (b:Person {person_id: $b})
            MATCH (f:Family {family_id: $family_id})
            MERGE (a)-[:PARTNER_IN]->(f)
            MERGE (b)-[:PARTNER_IN]->(f)
            MERGE (a)-[r:SPOUSE_OF]->(b)
            SET r += $props
            """,
            {"a": person1_id, "b": person2_id, "family_id": family_id, "props": rel_props},
        )

    def create_claim(self, claim_type: str, confidence: float, data: Dict[str, Any], notes: Optional[str] = None) -> str:
        claim_id = f"claim-{uuid.uuid4()}"
        data_json = json.dumps(data or {}, ensure_ascii=False)
        props = {
            "claim_type": claim_type,
            "confidence": float(confidence),
            "notes": notes,
            "created_at": timestamp_ms(),
            "data_json": data_json,
        }
        self._run(
            """
            CREATE (c:Claim {claim_id: $claim_id})
            SET c += $props
            """,
            {"claim_id": claim_id, "props": {k: v for k, v in props.items() if v is not None}},
        )
        return claim_id

    def attach_evidence(self, claim_id: str, evidence: Evidence) -> str:
        key = _hash_key(
            str(evidence.file_path or ""),
            str(evidence.doc_id or ""),
            str(evidence.chunk_id or ""),
            str(evidence.page_idx or ""),
            str(evidence.quote or ""),
            str(evidence.image_path or ""),
        )
        evidence_id = f"evidence-{key}"
        props = {
            "file_path": evidence.file_path,
            "doc_id": evidence.doc_id,
            "chunk_id": evidence.chunk_id,
            "page_idx": evidence.page_idx,
            "quote": evidence.quote,
            "image_path": evidence.image_path,
        }
        props = {k: v for k, v in props.items() if v is not None}
        self._run(
            """
            MERGE (e:Evidence {evidence_id: $evidence_id})
            ON CREATE SET e.created_at = timestamp()
            SET e += $props
            WITH e
            MATCH (c:Claim {claim_id: $claim_id})
            MERGE (c)-[:SUPPORTED_BY]->(e)
            """,
            {"evidence_id": evidence_id, "props": props, "claim_id": claim_id},
        )
        return evidence_id


def timestamp_ms() -> int:
    return int(time.time() * 1000)
