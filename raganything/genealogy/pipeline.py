from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from .extractors import ClaimExtractor
from .models import Claim, Evidence, FamilyRecord, MediaSpec, PersonRecord, PersonSpec, Task
from .normalize import coerce_year, normalize_name
from .stores import GenealogyStore


TASK_FIND_PARENTS = "find_parents"
TASK_FIND_CHILDREN = "find_children"
TASK_FIND_SPOUSES = "find_spouses"
TASK_FIND_PROFILE = "find_profile"


@dataclass
class GenealogyPipelineConfig:
    max_depth: int = 3
    max_tasks: int = 100
    enable_spouse_search: bool = True
    enable_profile_search: bool = True
    # If True, children discovered from a parents-family also get spouse/children expansion.
    enable_descendant_expansion: bool = True


def _person_spec_from_dict(d: Dict[str, Any]) -> PersonSpec:
    name = str(d.get("name") or "").strip()
    if not name:
        raise ValueError("Person name is required in claim payload")
    aliases_val = d.get("aliases") or d.get("aka") or d.get("also_known_as") or []
    if isinstance(aliases_val, str):
        aliases = [a.strip() for a in aliases_val.split(",") if a.strip()]
    elif isinstance(aliases_val, list):
        aliases = [str(a).strip() for a in aliases_val if str(a).strip()]
    else:
        aliases = []

    occupation = d.get("occupation") or d.get("profession") or d.get("job")

    return PersonSpec(
        name=name,
        birth_date=d.get("birth_date") or d.get("birthDate"),
        death_date=d.get("death_date") or d.get("deathDate"),
        birth_year=coerce_year(d.get("birth_year") or d.get("birthYear") or d.get("birth")),
        death_year=coerce_year(d.get("death_year") or d.get("deathYear") or d.get("death")),
        birth_place=d.get("birth_place") or d.get("birthPlace"),
        death_place=d.get("death_place") or d.get("deathPlace"),
        gender=d.get("gender") or d.get("sex"),
        occupation=str(occupation).strip() if occupation is not None and str(occupation).strip() else None,
        biography=d.get("biography") or d.get("bio") or d.get("about"),
        aliases=aliases,
        extra={
            k: v
            for k, v in d.items()
            if k
            not in {
                "name",
                "birth_date",
                "birthDate",
                "death_date",
                "deathDate",
                "birth_year",
                "birthYear",
                "birth",
                "death_year",
                "deathYear",
                "death",
                "birth_place",
                "birthPlace",
                "death_place",
                "deathPlace",
                "gender",
                "sex",
                "occupation",
                "profession",
                "job",
                "biography",
                "bio",
                "about",
                "aliases",
                "aka",
                "also_known_as",
                "media",
                "photos",
            }
        },
    )


class GenealogyPipeline:
    def __init__(
        self,
        store: GenealogyStore,
        extractor: ClaimExtractor,
        config: Optional[GenealogyPipelineConfig] = None,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.config = config or GenealogyPipelineConfig()

        # In-run caches (avoid requerying Neo4j during a single run).
        self.people: Dict[str, PersonRecord] = {}
        self.families: Dict[str, FamilyRecord] = {}

    def seed_person(self, spec: PersonSpec) -> PersonRecord:
        rec = self.store.upsert_person(spec)
        self.people[rec.person_id] = rec
        return rec

    async def expand(self, seed_person_id: str) -> Dict[str, Any]:
        q: Deque[Task] = collections.deque()
        seen_tasks: Set[str] = set()

        def enqueue(t: Task) -> None:
            key = f"{t.task_type}:{t.subject_kind}:{t.subject_id}"
            if key in seen_tasks:
                return
            if t.depth > self.config.max_depth:
                return
            seen_tasks.add(key)
            q.append(t)

        enqueue(Task(task_type=TASK_FIND_PARENTS, subject_kind="person", subject_id=seed_person_id, depth=0))
        if self.config.enable_profile_search:
            enqueue(Task(task_type=TASK_FIND_PROFILE, subject_kind="person", subject_id=seed_person_id, depth=0))

        tasks_run = 0
        claims_written = 0

        while q and tasks_run < self.config.max_tasks:
            task = q.popleft()
            tasks_run += 1

            if task.subject_kind == "person" and task.subject_id not in self.people:
                # Not seeded in current run; skip.
                continue
            if task.subject_kind == "family" and task.subject_id not in self.families:
                continue

            subject: Dict[str, Any] = {"subject_id": task.subject_id, "task_depth": task.depth}
            if task.subject_kind == "person":
                subject["person_id"] = task.subject_id
                subject["person"] = self.people[task.subject_id].spec
            else:
                fam = self.families[task.subject_id]
                subject["family_id"] = fam.family_id
                subject["family"] = fam
                subject["parents"] = [self.people[pid].spec for pid in fam.parent_ids if pid in self.people]

            claims = await self.extractor.extract(task.task_type, subject)

            for claim in claims:
                wrote, newly_discovered = self._apply_claim(claim)
                if wrote:
                    claims_written += 1

                # Enqueue follow-ups.
                if claim.claim_type == "parent_child":
                    for parent_id, child_id, family_id in newly_discovered.get("parent_child", []):
                        if family_id:
                            enqueue(Task(task_type=TASK_FIND_CHILDREN, subject_kind="family", subject_id=family_id, depth=task.depth))
                        # Expand ancestry.
                        enqueue(Task(task_type=TASK_FIND_PARENTS, subject_kind="person", subject_id=parent_id, depth=task.depth + 1))
                        if self.config.enable_profile_search:
                            enqueue(Task(task_type=TASK_FIND_PROFILE, subject_kind="person", subject_id=parent_id, depth=task.depth + 1))
                            enqueue(Task(task_type=TASK_FIND_PROFILE, subject_kind="person", subject_id=child_id, depth=task.depth + 1))
                        # Expand descendants from siblings/children.
                        if self.config.enable_descendant_expansion and self.config.enable_spouse_search:
                            enqueue(Task(task_type=TASK_FIND_SPOUSES, subject_kind="person", subject_id=child_id, depth=task.depth + 1))

                if claim.claim_type == "spouse":
                    for (p1, p2, fam_id) in newly_discovered.get("spouse", []):
                        enqueue(Task(task_type=TASK_FIND_CHILDREN, subject_kind="family", subject_id=fam_id, depth=task.depth + 1))
                        enqueue(Task(task_type=TASK_FIND_PARENTS, subject_kind="person", subject_id=p1, depth=task.depth + 1))
                        enqueue(Task(task_type=TASK_FIND_PARENTS, subject_kind="person", subject_id=p2, depth=task.depth + 1))
                        if self.config.enable_profile_search:
                            enqueue(Task(task_type=TASK_FIND_PROFILE, subject_kind="person", subject_id=p1, depth=task.depth + 1))
                            enqueue(Task(task_type=TASK_FIND_PROFILE, subject_kind="person", subject_id=p2, depth=task.depth + 1))

        return {
            "tasks_run": tasks_run,
            "claims_written": claims_written,
            "people": len(self.people),
            "families": len(self.families),
        }

    def _apply_claim(self, claim: Claim) -> Tuple[bool, Dict[str, Any]]:
        """Apply claim to store, returning (wrote_anything, newly_discovered_ids)."""
        newly: Dict[str, Any] = {"parent_child": [], "spouse": []}
        wrote = False

        if claim.claim_type == "parent_child":
            parents_payload = claim.data.get("parents") or []
            child_payload = claim.data.get("child")
            if not child_payload:
                return False, newly

            child_spec = _person_spec_from_dict(child_payload)
            child_rec = self.store.upsert_person(child_spec)
            self.people[child_rec.person_id] = child_rec

            parent_ids: List[str] = []
            for p in parents_payload[:2]:
                try:
                    parent_spec = _person_spec_from_dict(p)
                except Exception:
                    continue
                parent_rec = self.store.upsert_person(parent_spec)
                self.people[parent_rec.person_id] = parent_rec
                parent_ids.append(parent_rec.person_id)

            family_id: Optional[str] = None
            if len(parent_ids) == 2:
                fam = self.store.upsert_family(parent_ids, family_type="parents")
                self.families[fam.family_id] = fam
                family_id = fam.family_id
                self.store.link_parents_to_family(fam.family_id, parent_ids)
                self.store.link_child_to_family(fam.family_id, child_rec.person_id, props={"confidence": claim.confidence})

            claim_id = self.store.create_claim(claim.claim_type, claim.confidence, claim.data, notes=claim.notes)
            for ev in claim.evidence:
                self.store.attach_evidence(claim_id, ev)
            # Link claim for provenance navigation.
            self.store.link_claim_to_person(claim_id, child_rec.person_id, role="child")
            for pid in parent_ids:
                self.store.link_claim_to_person(claim_id, pid, role="parent")
            if family_id:
                self.store.link_claim_to_family(claim_id, family_id, role="family")

            wrote = True
            for pid in parent_ids:
                newly["parent_child"].append((pid, child_rec.person_id, family_id))
            return wrote, newly

        if claim.claim_type == "spouse":
            p1 = claim.data.get("person1")
            p2 = claim.data.get("person2")
            if not p1 or not p2:
                return False, newly

            p1_rec = self.store.upsert_person(_person_spec_from_dict(p1))
            p2_rec = self.store.upsert_person(_person_spec_from_dict(p2))
            self.people[p1_rec.person_id] = p1_rec
            self.people[p2_rec.person_id] = p2_rec

            fam = self.store.upsert_family([p1_rec.person_id, p2_rec.person_id], family_type="couple")
            self.families[fam.family_id] = fam
            self.store.link_spouses(fam.family_id, p1_rec.person_id, p2_rec.person_id, props={"confidence": claim.confidence})

            claim_id = self.store.create_claim(claim.claim_type, claim.confidence, claim.data, notes=claim.notes)
            for ev in claim.evidence:
                self.store.attach_evidence(claim_id, ev)
            self.store.link_claim_to_person(claim_id, p1_rec.person_id, role="spouse")
            self.store.link_claim_to_person(claim_id, p2_rec.person_id, role="spouse")
            self.store.link_claim_to_family(claim_id, fam.family_id, role="family")

            wrote = True
            newly["spouse"].append((p1_rec.person_id, p2_rec.person_id, fam.family_id))
            return wrote, newly

        if claim.claim_type == "person_profile":
            person_id = str(claim.data.get("person_id") or "").strip()
            attrs = claim.data.get("attributes") or {}
            if not person_id or not isinstance(attrs, dict):
                return False, newly

            # Build PersonSpec from attributes. Keep name stable from existing record.
            existing = self.people.get(person_id)
            base_name = existing.spec.name if existing else (attrs.get("name") or "")
            # Force name to the canonical subject to avoid accidental cross-person merges.
            spec = _person_spec_from_dict({**attrs, "name": base_name})

            # Media support.
            media_specs: List[MediaSpec] = []
            raw_media = attrs.get("media") or attrs.get("photos") or []
            if isinstance(raw_media, list):
                for m in raw_media:
                    if not isinstance(m, dict):
                        continue
                    kind = str(m.get("kind") or "photo").strip() or "photo"
                    path = str(m.get("path") or m.get("img_path") or "").strip()
                    if not path:
                        continue
                    media_specs.append(
                        MediaSpec(
                            kind=kind,
                            path=path,
                            caption=m.get("caption"),
                            extra={k: v for k, v in m.items() if k not in {"kind", "path", "img_path", "caption"}},
                        )
                    )

            # Update canonical store.
            self.store.update_person(person_id, spec)
            if existing:
                self.people[person_id] = PersonRecord(person_id=person_id, spec=spec, normalized_name=existing.normalized_name)

            claim_id = self.store.create_claim(claim.claim_type, claim.confidence, claim.data, notes=claim.notes)
            for ev in claim.evidence:
                self.store.attach_evidence(claim_id, ev)
            self.store.link_claim_to_person(claim_id, person_id, role="profile")

            for ms in media_specs:
                media_id = self.store.upsert_media(ms)
                self.store.link_person_media(person_id, media_id, props={"confidence": claim.confidence})
                self.store.link_claim_to_media(claim_id, media_id, role="mentions")

            wrote = True
            return wrote, newly

        # Ignore unsupported claim types for now.
        return False, newly
