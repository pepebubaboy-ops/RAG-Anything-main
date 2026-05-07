from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class Evidence:
    """Provenance pointer for a claim."""

    file_path: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page_idx: Optional[int] = None
    quote: Optional[str] = None
    image_path: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MediaSpec:
    """Media (photo, document scan, etc.) linked to a person, if available."""

    kind: str  # e.g., "photo"
    path: str
    caption: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PersonSpec:
    """A possibly-partial description of a person extracted from evidence."""

    name: str
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    birth_place: Optional[str] = None
    death_place: Optional[str] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    biography: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    media: List[MediaSpec] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Claim:
    """A structured assertion extracted from sources."""

    claim_type: str
    confidence: float = 0.5
    data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    notes: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class Task:
    """A unit of work for expanding the family graph."""

    task_type: str
    subject_kind: str  # "person" or "family"
    subject_id: str
    depth: int = 0


@dataclass(frozen=True, slots=True)
class PersonRecord:
    person_id: str
    spec: PersonSpec
    normalized_name: str


@dataclass(frozen=True, slots=True)
class FamilyRecord:
    family_id: str
    parent_ids: List[str]
    family_type: str = "couple"
    family_key: Optional[str] = None
