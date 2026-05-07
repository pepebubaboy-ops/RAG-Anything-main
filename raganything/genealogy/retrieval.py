from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .query_resolution import ResolvedGenealogyQuery, resolve_genealogy_query
from .rag_index import GenealogyRAGDocument, read_jsonl


@dataclass(frozen=True, slots=True)
class RetrievedGenealogyContext:
    document_id: str
    kind: str
    score: float
    text: str
    metadata: Dict[str, Any]


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[0-9a-zа-яё]+", str(value or "").lower())


def _resolve_rag_documents_path(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path / "rag_documents.jsonl"
    return path


def _resolve_artifact_dir(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path
    return path.parent


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload is not None else default


def load_rag_documents(path: Path) -> List[GenealogyRAGDocument]:
    docs: List[GenealogyRAGDocument] = []
    for row in read_jsonl(_resolve_rag_documents_path(path)):
        docs.append(
            GenealogyRAGDocument(
                document_id=str(row.get("document_id") or ""),
                kind=str(row.get("kind") or ""),
                text=str(row.get("text") or ""),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return docs


def _context_from_document(
    document: GenealogyRAGDocument,
    *,
    score: float,
) -> RetrievedGenealogyContext:
    return RetrievedGenealogyContext(
        document_id=document.document_id,
        kind=document.kind,
        score=score,
        text=document.text,
        metadata=document.metadata,
    )


def _document_indexes(
    documents: Sequence[GenealogyRAGDocument],
) -> Dict[str, Dict[str, List[GenealogyRAGDocument]]]:
    indexes: Dict[str, Dict[str, List[GenealogyRAGDocument]]] = {
        "person_id": {},
        "relationship_id": {},
        "claim_id": {},
        "chunk_id": {},
    }
    for document in documents:
        metadata = document.metadata
        for field in indexes:
            value = metadata.get(field)
            if value:
                indexes[field].setdefault(str(value), []).append(document)
        for claim_id in metadata.get("claim_ids") or []:
            indexes["claim_id"].setdefault(str(claim_id), []).append(document)
        for chunk_id in metadata.get("chunk_ids") or []:
            indexes["chunk_id"].setdefault(str(chunk_id), []).append(document)
    return indexes


def _score_document(
    query_tokens: Sequence[str], document: GenealogyRAGDocument
) -> float:
    if not query_tokens or not document.text:
        return 0.0

    text_tokens = _tokenize(document.text)
    if not text_tokens:
        return 0.0
    text_token_set = set(text_tokens)
    query_token_set = set(query_tokens)
    overlap = query_token_set & text_token_set
    if not overlap:
        return 0.0

    score = len(overlap) / max(1, len(query_token_set))
    score += 0.05 * min(5, sum(text_tokens.count(token) for token in overlap))
    if document.kind in {"person", "claim"}:
        score += 0.1
    return score


def _is_allowed(kind: str, allowed_kinds: set[str] | None) -> bool:
    return allowed_kinds is None or kind in allowed_kinds


def _add_documents(
    output: List[RetrievedGenealogyContext],
    seen: set[tuple[str, str]],
    documents: Sequence[GenealogyRAGDocument],
    *,
    score: float,
    allowed_kinds: set[str] | None,
    document_kinds: set[str] | None = None,
) -> None:
    for document in documents:
        if document_kinds is not None and document.kind not in document_kinds:
            continue
        if not _is_allowed(document.kind, allowed_kinds):
            continue
        key = (document.kind, document.document_id)
        if key in seen:
            continue
        seen.add(key)
        output.append(_context_from_document(document, score=score))


def _accepted_relationships_for_query(
    relationships: Sequence[Dict[str, Any]],
    resolved_query: ResolvedGenealogyQuery,
) -> List[Dict[str, Any]]:
    person_id = resolved_query.person_id
    intent = resolved_query.intent
    matches: List[Dict[str, Any]] = []
    for relationship in relationships:
        if str(relationship.get("status") or "").lower() != "accepted":
            continue
        relationship_type = str(relationship.get("relationship_type") or "")
        source_id = str(relationship.get("source_person_id") or "")
        target_id = str(relationship.get("target_person_id") or "")
        if intent == "parents":
            if relationship_type == "parent_of" and target_id == person_id:
                matches.append(relationship)
            continue
        if intent == "children":
            if relationship_type == "parent_of" and source_id == person_id:
                matches.append(relationship)
            continue
        if intent == "spouses":
            if relationship_type == "spouse_of" and person_id in {source_id, target_id}:
                matches.append(relationship)
            continue
        if person_id in {source_id, target_id}:
            matches.append(relationship)

    return sorted(
        matches,
        key=lambda row: (
            str(row.get("relationship_type") or ""),
            str(row.get("source_person_id") or ""),
            str(row.get("target_person_id") or ""),
            str(row.get("relationship_id") or ""),
        ),
    )


def _graph_ranked_contexts(
    query: str,
    corpus_path: Path,
    documents: Sequence[GenealogyRAGDocument],
    allowed_kinds: set[str] | None,
) -> List[RetrievedGenealogyContext]:
    artifact_dir = _resolve_artifact_dir(corpus_path)
    if not (artifact_dir / "people.json").exists():
        return []
    resolved_query = resolve_genealogy_query(query, artifact_dir)
    if resolved_query is None:
        return []

    indexes = _document_indexes(documents)
    relationships = _load_json(artifact_dir / "relationships.json", [])
    if not isinstance(relationships, list):
        relationships = []
    selected_relationships = _accepted_relationships_for_query(
        relationships,
        resolved_query,
    )

    ranked: List[RetrievedGenealogyContext] = []
    seen: set[tuple[str, str]] = set()

    _add_documents(
        ranked,
        seen,
        indexes["person_id"].get(resolved_query.person_id, []),
        score=5.0,
        allowed_kinds=allowed_kinds,
        document_kinds={"person"},
    )

    related_person_ids: set[str] = set()
    claim_ids: set[str] = set()
    for relationship in selected_relationships:
        relationship_id = str(relationship.get("relationship_id") or "")
        if relationship_id:
            _add_documents(
                ranked,
                seen,
                indexes["relationship_id"].get(relationship_id, []),
                score=4.5,
                allowed_kinds=allowed_kinds,
                document_kinds={"relationship"},
            )
        for person_field in ("source_person_id", "target_person_id"):
            person_id = str(relationship.get(person_field) or "")
            if person_id and person_id != resolved_query.person_id:
                related_person_ids.add(person_id)
        claim_ids.update(
            str(item) for item in relationship.get("claim_ids") or [] if item
        )

    for person_id in sorted(related_person_ids):
        _add_documents(
            ranked,
            seen,
            indexes["person_id"].get(person_id, []),
            score=3.5,
            allowed_kinds=allowed_kinds,
            document_kinds={"person"},
        )

    chunk_ids: set[str] = set()
    for claim_id in sorted(claim_ids):
        claim_documents = indexes["claim_id"].get(claim_id, [])
        _add_documents(
            ranked,
            seen,
            claim_documents,
            score=3.0,
            allowed_kinds=allowed_kinds,
            document_kinds={"claim"},
        )
        for document in claim_documents:
            if document.kind == "claim":
                chunk_ids.update(
                    str(item) for item in document.metadata.get("chunk_ids") or []
                )

    _add_documents(
        ranked,
        seen,
        indexes["person_id"].get(resolved_query.person_id, []),
        score=2.2,
        allowed_kinds=allowed_kinds,
        document_kinds={"resolution"},
    )

    for chunk_id in sorted(chunk_ids):
        _add_documents(
            ranked,
            seen,
            indexes["chunk_id"].get(chunk_id, []),
            score=2.5,
            allowed_kinds=allowed_kinds,
            document_kinds={"source_chunk"},
        )

    return ranked


def retrieve_genealogy_context(
    query: str,
    corpus_path: Path,
    *,
    top_k: int = 8,
    kinds: Iterable[str] | None = None,
) -> List[RetrievedGenealogyContext]:
    query_tokens = _tokenize(query)
    allowed_kinds = {str(kind) for kind in kinds} if kinds is not None else None
    documents = load_rag_documents(corpus_path)
    ranked: List[RetrievedGenealogyContext] = _graph_ranked_contexts(
        query,
        corpus_path,
        documents,
        allowed_kinds,
    )
    seen = {(row.kind, row.document_id) for row in ranked}

    for document in documents:
        if allowed_kinds is not None and document.kind not in allowed_kinds:
            continue
        if (document.kind, document.document_id) in seen:
            continue
        score = _score_document(query_tokens, document)
        if score <= 0:
            continue
        ranked.append(_context_from_document(document, score=score))

    ranked.sort(
        key=lambda row: (
            row.score,
            1 if row.kind in {"person", "claim"} else 0,
            row.document_id,
        ),
        reverse=True,
    )
    return ranked[: max(0, int(top_k))]


def build_genealogy_answer_prompt(
    question: str,
    contexts: Sequence[RetrievedGenealogyContext],
) -> str:
    context_blocks: List[str] = []
    for index, context in enumerate(contexts, start=1):
        metadata = context.metadata
        source_bits = []
        for key in (
            "source_path",
            "source_title",
            "person_id",
            "family_id",
            "relationship_id",
            "claim_id",
            "claim_type",
        ):
            value = metadata.get(key)
            if value:
                source_bits.append(f"{key}={value}")
        if metadata.get("page_idx") is not None:
            source_bits.append(f"page_idx={metadata['page_idx']}")
        source_text = "; ".join(source_bits) if source_bits else "metadata=none"
        context_blocks.append(
            f"[{index}] kind={context.kind}; id={context.document_id}; {source_text}\n"
            f"{context.text}"
        )

    joined_context = "\n\n".join(context_blocks) or "No genealogy context retrieved."
    return (
        "You answer genealogy questions using only the provided context. "
        "If the context is insufficient, say what is missing. "
        "Cite context numbers like [1] when making factual claims.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{joined_context}\n\n"
        "Answer:"
    )
