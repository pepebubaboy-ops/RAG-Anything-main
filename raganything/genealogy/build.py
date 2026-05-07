from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from ..offline import ensure_offline_allowed
from .claim_extraction import (
    _clean_name,
    _coerce_claim_year,
    _extract_claims_from_text,
)
from .export import (
    _build_dot_from_people_and_families,
    _build_html_from_people_and_families,
    _store_to_people_and_families,
)
from .knowledge_graph import (
    claim_id_for_row,
    evidence_id_for_row,
    write_evidences,
    write_knowledge_graph_artifacts,
)
from .mentions import MentionRecord, extract_mentions_from_text, write_mentions
from .models import Claim, PersonSpec, normalize_claim_status
from .normalize import normalize_name
from .rag_index import (
    SourceChunk,
    SourceDocument,
    source_chunk_from_content_item,
    source_document_for_path,
    write_rag_documents,
    write_source_chunks,
    write_sources,
)
from .resolution import resolve_mentions_to_people, write_person_resolution
from .results import BuildResult
from .stores import InMemoryGenealogyStore


def _iter_json_array_objects(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream top-level JSON array objects from file."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    in_array = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(65536)
            if chunk:
                buffer += chunk
            elif not buffer:
                return

            if not in_array:
                stripped = buffer.lstrip()
                if not stripped:
                    if not chunk:
                        return
                    continue
                if stripped[0] != "[":
                    data = json.loads(buffer + handle.read())
                    if not isinstance(data, list):
                        raise ValueError(f"Expected JSON array in {path}")
                    for item in data:
                        if isinstance(item, dict):
                            yield item
                    return
                leading = len(buffer) - len(stripped)
                pos = leading + 1
                in_array = True

            while True:
                while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                    pos += 1

                if pos >= len(buffer):
                    break
                if buffer[pos] == "]":
                    return

                try:
                    obj, next_pos = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break

                if isinstance(obj, dict):
                    yield obj
                pos = next_pos

            if pos:
                buffer = buffer[pos:]
                pos = 0

            if not chunk:
                stripped = buffer.strip()
                if stripped and stripped != "]":
                    raise ValueError(f"Malformed JSON array in {path}")
                return


def _find_content_list_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file, got: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.rglob("*_content_list.json"))
        if files:
            return files

        fallback_json = sorted(input_path.glob("*.json"))
        if fallback_json:
            return fallback_json

        raise ValueError(
            f"No *_content_list.json (or .json) files found in directory: {input_path}"
        )

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _parse_pdf_to_content_list(
    input_pdf: Path,
    parse_method: str,
    output_dir: Path,
) -> Iterable[Dict[str, Any]]:
    if parse_method == "none":
        raise RuntimeError(
            "PDF input requires parsing. Use --parse-method mineru|docling, "
            "or provide pre-parsed *_content_list.json."
        )

    ensure_offline_allowed(
        action="PDF parsing",
        hint="Use pre-parsed *_content_list.json for strict offline workflows.",
    )

    if parse_method == "mineru" and find_spec("mineru") is None:
        raise RuntimeError(
            "PDF parsing with MinerU requires optional dependency. "
            "Install extras: pip install 'raganything[mineru]'"
        )

    if parse_method == "docling" and shutil.which("docling") is None:
        raise RuntimeError(
            "docling command was not found in PATH. Install docling, "
            "or use --parse-method mineru."
        )

    from ..parser import DoclingParser, MineruParser

    parser = MineruParser() if parse_method == "mineru" else DoclingParser()
    parse_lang: str | None = None
    parse_engine_method = "auto"
    parser_kwargs: Dict[str, Any] = {}
    if parse_method == "mineru":
        backend_env = str(os.getenv("MINERU_BACKEND") or "").strip()
        if backend_env:
            parser_kwargs["backend"] = backend_env
        source_env = str(os.getenv("MINERU_MODEL_SOURCE") or "").strip().lower()
        if source_env in {"huggingface", "modelscope", "local"}:
            parser_kwargs["source"] = source_env
        lang_env = str(os.getenv("MINERU_LANG") or "").strip()
        if lang_env:
            parse_lang = lang_env
        method_env = str(os.getenv("MINERU_PARSE_METHOD") or "").strip().lower()
        if method_env in {"auto", "txt", "ocr"}:
            parse_engine_method = method_env

    parse_output_dir = output_dir / "_parsed_pdf"
    parse_output_dir.mkdir(parents=True, exist_ok=True)
    return parser.parse_document(
        file_path=str(input_pdf),
        output_dir=str(parse_output_dir),
        method=parse_engine_method,
        lang=parse_lang,
        **parser_kwargs,
    )


def _apply_claim_to_store(store: InMemoryGenealogyStore, claim: Claim) -> bool:
    if claim.claim_type == "parent_child":
        parents_payload = claim.data.get("parents") or []
        child_payload = claim.data.get("child") or {}
        child_name = _clean_name(child_payload.get("name") or "")
        if not child_name:
            return False

        child_rec = store.upsert_person(
            PersonSpec(
                name=child_name,
                birth_year=_coerce_claim_year(child_payload.get("birth_year")),
                death_year=_coerce_claim_year(child_payload.get("death_year")),
            )
        )

        parent_ids: List[str] = []
        for parent in parents_payload[:2]:
            parent_name = _clean_name(parent.get("name") or "")
            if not parent_name:
                continue
            parent_rec = store.upsert_person(
                PersonSpec(
                    name=parent_name,
                    birth_year=_coerce_claim_year(parent.get("birth_year")),
                    death_year=_coerce_claim_year(parent.get("death_year")),
                )
            )
            parent_ids.append(parent_rec.person_id)

        family_id = None
        if len(parent_ids) == 2:
            family = store.upsert_family(parent_ids, family_type="parents")
            store.link_parents_to_family(family.family_id, parent_ids)
            store.link_child_to_family(
                family.family_id,
                child_rec.person_id,
                props={"confidence": claim.confidence},
            )
            family_id = family.family_id

        claim_id = store.create_claim(
            claim.claim_type,
            claim.confidence,
            dict(claim.data),
            notes=claim.notes,
        )
        for evidence in claim.evidence:
            store.attach_evidence(claim_id, evidence)
        store.link_claim_to_person(claim_id, child_rec.person_id, role="child")
        for parent_id in parent_ids:
            store.link_claim_to_person(claim_id, parent_id, role="parent")
        if family_id:
            store.link_claim_to_family(claim_id, family_id, role="family")
        return True

    if claim.claim_type == "spouse":
        p1_payload = claim.data.get("person1") or {}
        p2_payload = claim.data.get("person2") or {}
        p1_name = _clean_name(p1_payload.get("name") or "")
        p2_name = _clean_name(p2_payload.get("name") or "")
        if not p1_name or not p2_name:
            return False

        p1 = store.upsert_person(
            PersonSpec(
                name=p1_name,
                birth_year=_coerce_claim_year(p1_payload.get("birth_year")),
                death_year=_coerce_claim_year(p1_payload.get("death_year")),
            )
        )
        p2 = store.upsert_person(
            PersonSpec(
                name=p2_name,
                birth_year=_coerce_claim_year(p2_payload.get("birth_year")),
                death_year=_coerce_claim_year(p2_payload.get("death_year")),
            )
        )

        family = store.upsert_family([p1.person_id, p2.person_id], family_type="couple")
        store.link_spouses(
            family.family_id,
            p1.person_id,
            p2.person_id,
            props={"confidence": claim.confidence},
        )

        claim_id = store.create_claim(
            claim.claim_type,
            claim.confidence,
            dict(claim.data),
            notes=claim.notes,
        )
        for evidence in claim.evidence:
            store.attach_evidence(claim_id, evidence)
        store.link_claim_to_person(claim_id, p1.person_id, role="spouse")
        store.link_claim_to_person(claim_id, p2.person_id, role="spouse")
        store.link_claim_to_family(claim_id, family.family_id, role="family")
        return True

    return False


def _claims_to_jsonl_row(claim: Claim, applied: bool) -> Dict[str, Any]:
    row = {
        "claim_type": claim.claim_type,
        "confidence": claim.confidence,
        "data": claim.data,
        "evidence": [asdict(evidence) for evidence in claim.evidence],
        "notes": claim.notes,
        "applied": applied,
        "status": normalize_claim_status(claim.status),
        "reason": claim.reason,
    }
    row["claim_id"] = claim_id_for_row(row)
    for evidence in row["evidence"]:
        if isinstance(evidence, dict):
            evidence["evidence_id"] = evidence_id_for_row(evidence)
    return row


def _claim_with_source_context(
    claim: Claim,
    *,
    source_id: str,
    chunk_id: str,
) -> Claim:
    return replace(
        claim,
        evidence=[
            replace(
                evidence,
                doc_id=evidence.doc_id or source_id,
                chunk_id=evidence.chunk_id or chunk_id,
            )
            for evidence in claim.evidence
        ],
    )


def _iter_claims_and_chunks_from_source_items(
    items: Iterable[Dict[str, Any]],
    *,
    source_name: str,
    source: SourceDocument,
    chunks: List[SourceChunk],
    mentions: List[MentionRecord],
) -> Iterator[Claim]:
    for ordinal, item in enumerate(items):
        chunk = source_chunk_from_content_item(source, item, ordinal)
        if chunk is None:
            continue
        chunks.append(chunk)
        mentions.extend(
            extract_mentions_from_text(
                chunk.text,
                source_id=source.source_id,
                chunk_id=chunk.chunk_id,
                page_idx=chunk.page_idx,
            )
        )
        for claim in _extract_claims_from_text(
            chunk.text,
            source=source_name,
            page_idx=chunk.page_idx,
        ):
            yield _claim_with_source_context(
                claim,
                source_id=source.source_id,
                chunk_id=chunk.chunk_id,
            )


def _name_similarity_score(left: str, right: str) -> float:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0

    seq_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    union = left_tokens | right_tokens
    token_score = len(left_tokens & right_tokens) / len(union) if union else 0.0

    subset_bonus = 0.0
    if len(left_tokens & right_tokens) >= 2 and (
        left_tokens <= right_tokens or right_tokens <= left_tokens
    ):
        subset_bonus = 0.08

    return min(1.0, (0.65 * seq_score) + (0.35 * token_score) + subset_bonus)


def _name_variants_for_matching(name: str) -> set[str]:
    normalized = normalize_name(name)
    if not normalized:
        return set()
    return {normalized}


def _reconcile_people_with_reference(
    people: Sequence[Dict[str, Any]],
    reference_people: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    prepared_reference: List[Dict[str, Any]] = []
    for row in reference_people:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        variants = _name_variants_for_matching(name)
        normalized_name = str(row.get("normalized_name") or "").strip()
        if normalized_name:
            variants.add(normalized_name)
        if not variants:
            continue
        prepared_reference.append({"row": row, "variants": variants})

    if not prepared_reference:
        return list(people), {"matched": 0, "renamed": 0, "enriched_fields": 0}

    used_reference_indexes: set[int] = set()
    matched = 0
    renamed = 0
    enriched_fields = 0
    reconciled_people: List[Dict[str, Any]] = []

    for person in people:
        person_name = str(person.get("name") or "").strip()
        person_normalized = str(person.get("normalized_name") or "").strip()
        person_variants = _name_variants_for_matching(person_name)
        if person_normalized:
            person_variants.add(person_normalized)
        if not person_variants:
            reconciled_people.append(dict(person))
            continue

        best_index: int | None = None
        best_score = 0.0
        second_best_score = 0.0
        for index, reference in enumerate(prepared_reference):
            if index in used_reference_indexes:
                continue
            reference_variants = reference["variants"]
            score = max(
                _name_similarity_score(left, right)
                for left in person_variants
                for right in reference_variants
            )
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_index = index
            elif score > second_best_score:
                second_best_score = score

        if best_index is None or best_score < 0.86:
            reconciled_people.append(dict(person))
            continue
        if second_best_score >= 0.86 and (best_score - second_best_score) < 0.04:
            # Ambiguous mapping between multiple close reference candidates.
            # Keep extracted identity unchanged to avoid relation corruption.
            reconciled_people.append(dict(person))
            continue

        used_reference_indexes.add(best_index)
        matched += 1
        reference_row = prepared_reference[best_index]["row"]
        row = dict(person)

        reference_name = str(reference_row.get("name") or "").strip()
        if reference_name and row.get("name") != reference_name:
            aliases = list(row.get("aliases") or [])
            existing_name = str(row.get("name") or "").strip()
            if existing_name and existing_name not in aliases:
                aliases.append(existing_name)
            row["aliases"] = aliases
            row["name"] = reference_name
            renamed += 1

        reference_normalized = str(
            reference_row.get("normalized_name") or normalize_name(reference_name)
        ).strip()
        if reference_normalized:
            row["normalized_name"] = reference_normalized

        for field in ("birth_year", "death_year", "birth_place", "death_place"):
            current = row.get(field)
            incoming = reference_row.get(field)
            if current in (None, "", 0) and incoming not in (None, "", 0):
                row[field] = incoming
                enriched_fields += 1

        if "aliases" in reference_row:
            aliases = list(row.get("aliases") or [])
            for alias in reference_row.get("aliases") or []:
                alias_str = str(alias).strip()
                if alias_str and alias_str not in aliases:
                    aliases.append(alias_str)
            row["aliases"] = aliases

        reconciled_people.append(row)

    return reconciled_people, {
        "matched": matched,
        "renamed": renamed,
        "enriched_fields": enriched_fields,
    }


def build_genealogy_tree(
    input_path: Path,
    output_dir: Path,
    *,
    parse_method: str = "none",
    reference_people: Path | None = None,
) -> BuildResult:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    store = InMemoryGenealogyStore()
    claims_path = output_dir / "claims.jsonl"
    source_docs: List[SourceDocument] = []
    source_chunks: List[SourceChunk] = []
    mentions: List[MentionRecord] = []
    claim_rows: List[Dict[str, Any]] = []

    total_claims = 0
    if input_path.suffix.lower() == ".pdf":
        content_items = _parse_pdf_to_content_list(
            input_pdf=input_path,
            parse_method=parse_method,
            output_dir=output_dir,
        )
        sources = [(str(input_path), content_items)]
    else:
        files = _find_content_list_files(input_path)
        sources = [
            (str(file_path), _iter_json_array_objects(file_path)) for file_path in files
        ]

    with claims_path.open("w", encoding="utf-8") as claims_file:
        for source_name, items in sources:
            source_doc = source_document_for_path(source_name)
            source_docs.append(source_doc)
            for claim in _iter_claims_and_chunks_from_source_items(
                items,
                source_name=source_name,
                source=source_doc,
                chunks=source_chunks,
                mentions=mentions,
            ):
                applied = _apply_claim_to_store(store, claim)
                claim_row = _claims_to_jsonl_row(claim, applied)
                claim_rows.append(claim_row)
                claims_file.write(
                    json.dumps(
                        claim_row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_claims += 1

    write_sources(output_dir, source_docs)
    write_source_chunks(output_dir, source_chunks)
    write_evidences(output_dir, claim_rows)

    people, families = _store_to_people_and_families(store)
    reconciliation_stats: Dict[str, int] | None = None
    if reference_people:
        reference_path = Path(reference_people).expanduser().resolve()
        if not reference_path.exists() or not reference_path.is_file():
            raise FileNotFoundError(
                f"Reference people file does not exist: {reference_path}"
            )
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
        if not isinstance(reference_payload, list):
            raise ValueError(
                f"Reference people file must contain JSON array: {reference_path}"
            )
        people, reconciliation_stats = _reconcile_people_with_reference(
            people,
            reference_payload,
        )
    person_resolution = resolve_mentions_to_people(mentions, people, claim_rows)
    write_mentions(output_dir, mentions)
    write_person_resolution(output_dir, person_resolution)

    graph_artifact = write_knowledge_graph_artifacts(
        output_dir,
        people=people,
        claim_rows=claim_rows,
        resolution=person_resolution,
    )

    people_path = output_dir / "people.json"
    families_path = output_dir / "families.json"
    dot_path = output_dir / "tree.dot"
    html_path = output_dir / "tree.html"

    people_path.write_text(
        json.dumps(people, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    families_path.write_text(
        json.dumps(families, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dot_path.write_text(
        _build_dot_from_people_and_families(people, families),
        encoding="utf-8",
    )
    html_path.write_text(
        _build_html_from_people_and_families(
            people,
            families,
            dot_path.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    rag_documents_path = write_rag_documents(output_dir)
    rag_documents_count = sum(
        1
        for line in rag_documents_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    resolution_summary = person_resolution.get("summary") or {}

    return BuildResult(
        output_dir=output_dir,
        people_count=len(people),
        families_count=len(families),
        claims_count=total_claims,
        reconciliation_stats=reconciliation_stats,
        details={
            "sources_count": len(source_docs),
            "source_chunks_count": len(source_chunks),
            "mentions_count": len(mentions),
            "resolved_mentions_count": int(
                resolution_summary.get("resolved_mentions_count") or 0
            ),
            "unresolved_mentions_count": int(
                resolution_summary.get("unresolved_mentions_count") or 0
            ),
            "ambiguous_mentions_count": int(
                resolution_summary.get("ambiguous_mentions_count") or 0
            ),
            "relationships_count": len(graph_artifact.relationships),
            "conflicts_count": len(graph_artifact.conflicts),
            "rag_documents_count": rag_documents_count,
            "rag_documents_path": str(rag_documents_path),
        },
    )
