from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .build import _apply_claim_to_store
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
from .mentions import (
    MentionRecord,
    _is_person_like_mention,
    extract_mentions_from_text,
    write_mentions,
)
from .models import Claim, Evidence
from .rag_index import (
    read_jsonl,
    sanitize_source_path,
    write_jsonl,
    write_rag_documents,
)
from .resolution import resolve_mentions_to_people, write_person_resolution
from .results import BuildResult
from .stores import InMemoryGenealogyStore

GENEALOGY_TRIGGER_TERMS = {
    "сын",
    "сына",
    "сынов",
    "дочь",
    "дочери",
    "дети",
    "детей",
    "женат",
    "жена",
    "супруга",
    "супруг",
    "муж",
    "брак",
    "браке",
    "родила",
    "родились",
    "родилось",
    "наследник",
    "наследником",
    "отец",
    "мать",
    "внук",
    "внучка",
    "племянник",
    "племянница",
    "son",
    "daughter",
    "child",
    "children",
    "spouse",
    "wife",
    "husband",
    "married",
    "born",
    "heir",
}

SUPPORTED_CLAIM_TYPES = {"parent_child", "spouse", "person_profile"}
MIN_ACCEPTED_CLAIM_CONFIDENCE = 0.55
CLAIM_STATUS_ACCEPTED = "accepted"
CLAIM_STATUS_PENDING = "pending"
CLAIM_STATUS_REJECTED = "rejected"
CLAIM_STATUS_NEEDS_REVIEW = "needs_review"
CLAIM_STATUS_CONFLICT = "conflict"

GENERIC_PERSON_NAMES = {
    "император",
    "императрица",
    "царь",
    "царица",
    "царевич",
    "царевна",
    "князь",
    "княгиня",
    "первая",
    "вторая",
    "первый",
    "второй",
    "он",
    "она",
    "ему",
    "ей",
    "они",
    "у них",
    "spouse",
    "wife",
    "husband",
    "emperor",
    "empress",
    "king",
    "queen",
}


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class LLMCandidateChunk:
    candidate_id: str
    chunk_id: str
    source_id: str
    text: str
    page_idx: int | None = None
    previous_text: str = ""
    next_text: str = ""
    subject_hint: str | None = None
    trigger_terms: list[str] = field(default_factory=list)
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class LLMRawExtraction:
    candidate_id: str
    chunk_id: str
    source_id: str
    page_idx: int | None
    model: str
    prompt: str
    raw_output: str
    parsed: Any | None
    parse_status: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedClaimRow:
    status: str
    reason: str | None
    candidate_id: str
    chunk_id: str
    claim: dict[str, Any] | None
    raw_claim: dict[str, Any] | None
    confidence: float | None = None
    evidence_quote: str | None = None
    audit_claim: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ClaimValidationResult:
    status: str
    reason: str | None
    claim: dict[str, Any] | None = None
    audit_claim: dict[str, Any] | None = None
    confidence: float | None = None
    evidence_quote: str | None = None


def _normalize_for_match(value: str) -> str:
    text = str(value or "").lower().replace("ё", "е")
    return " ".join(re.findall(r"[0-9a-zа-я]+", text))


def _coerce_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year <= 0 or year > 2500:
        return None
    return year


def _claim_confidence(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def _context_text(candidate: LLMCandidateChunk) -> str:
    return "\n\n".join(
        part
        for part in (
            f"PREVIOUS:\n{candidate.previous_text}" if candidate.previous_text else "",
            f"CURRENT:\n{candidate.text}",
            f"NEXT:\n{candidate.next_text}" if candidate.next_text else "",
        )
        if part
    )


def _trigger_terms_for_text(text: str) -> list[str]:
    normalized = _normalize_for_match(text)
    padded = f" {normalized} "
    return sorted(term for term in GENEALOGY_TRIGGER_TERMS if f" {term} " in padded)


_HEADING_PATTERNS = [
    re.compile(
        r"^(?P<name>[А-ЯЁA-Z][А-Яа-яЁёA-Za-z]+(?:\s+[IVXLCDM]+|\s+[А-ЯЁA-Z][А-Яа-яЁёA-Za-z]+){0,3})\s*\(",
    ),
    re.compile(
        r"^(?:Царствование|Правление|При)\s+(?P<name>[А-ЯЁA-Z][А-Яа-яЁёA-Za-z]+(?:\s+[IVXLCDM]+|\s+[А-ЯЁA-Z][А-Яа-яЁёA-Za-z]+){0,3})",
    ),
]


def _subject_hint_from_text(text: str) -> str | None:
    compact = " ".join(str(text or "").split())
    for pattern in _HEADING_PATTERNS:
        match = pattern.search(compact)
        if match:
            return match.group("name").strip(" .,:;-")
    return None


def _subject_hint_for_index(
    chunks: Sequence[dict[str, Any]],
    index: int,
) -> str | None:
    for cursor in range(index, max(-1, index - 5), -1):
        text = str(chunks[cursor].get("text") or "")
        hint = _subject_hint_from_text(text)
        if hint:
            return hint
    return None


def find_candidate_chunks(
    source_chunks: Sequence[dict[str, Any]],
    *,
    context_window: int = 1,
    max_candidates: int | None = None,
) -> list[LLMCandidateChunk]:
    candidates: list[LLMCandidateChunk] = []
    for index, chunk in enumerate(source_chunks):
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        triggers = _trigger_terms_for_text(text)
        if not triggers:
            continue

        previous_parts: list[str] = []
        next_parts: list[str] = []
        for offset in range(max(0, index - context_window), index):
            previous_text = str(source_chunks[offset].get("text") or "").strip()
            if previous_text:
                previous_parts.append(previous_text)
        for offset in range(
            index + 1, min(len(source_chunks), index + 1 + context_window)
        ):
            next_text = str(source_chunks[offset].get("text") or "").strip()
            if next_text:
                next_parts.append(next_text)

        chunk_id = str(chunk.get("chunk_id") or "")
        source_id = str(chunk.get("source_id") or "")
        candidate = LLMCandidateChunk(
            candidate_id=_stable_id("llm-candidate", source_id, chunk_id, text),
            chunk_id=chunk_id,
            source_id=source_id,
            text=text,
            page_idx=chunk.get("page_idx")
            if isinstance(chunk.get("page_idx"), int)
            else None,
            previous_text="\n\n".join(previous_parts),
            next_text="\n\n".join(next_parts),
            subject_hint=_subject_hint_for_index(source_chunks, index),
            trigger_terms=triggers,
            ordinal=len(candidates),
        )
        candidates.append(candidate)
        if max_candidates is not None and len(candidates) >= max_candidates:
            break
    return candidates


def write_candidate_chunks(
    output_dir: Path,
    candidates: Sequence[LLMCandidateChunk],
) -> Path:
    path = output_dir / "candidate_chunks.jsonl"
    write_jsonl(path, (asdict(candidate) for candidate in candidates))
    return path


def _build_llm_prompt(candidate: LLMCandidateChunk) -> str:
    subject_hint = candidate.subject_hint or "unknown"
    return f"""
You extract genealogy assertions from Russian/English historical text.
Return JSON only. Do not add facts from external knowledge.

Use this schema:
{{
  "claims": [
    {{
      "claim_type": "parent_child",
      "child": {{"name": "...", "birth_year": null, "death_year": null}},
      "parents": [{{"name": "...", "birth_year": null, "death_year": null}}],
      "evidence_quote": "exact substring from CURRENT/PREVIOUS/NEXT",
      "confidence": 0.0,
      "notes": "short reason"
    }},
    {{
      "claim_type": "spouse",
      "person1": {{"name": "..."}},
      "person2": {{"name": "..."}},
      "evidence_quote": "exact substring from CURRENT/PREVIOUS/NEXT",
      "confidence": 0.0,
      "notes": "short reason"
    }}
  ]
}}

Rules:
- Extract only explicit genealogy assertions.
- Use exact names as written in the text.
- If pronouns like "ему", "его", "у них" refer to the subject, use subject_hint.
- If subject_hint is needed, mention it in notes.
- evidence_quote must be a short exact substring from the provided text.
- If no genealogy claims are explicit, return {{"claims": []}}.
- Do not output reasoning or markdown. For reasoning models: /no_think.

subject_hint: {subject_hint}
page_idx: {candidate.page_idx}
trigger_terms: {", ".join(candidate.trigger_terms)}

Text:
{_context_text(candidate)}
""".strip()


def robust_json_loads(text: str) -> Any | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    candidates = [raw]
    first_object = raw.find("{")
    last_object = raw.rfind("}")
    if first_object != -1 and last_object > first_object:
        candidates.append(raw[first_object : last_object + 1])
    first_array = raw.find("[")
    last_array = raw.rfind("]")
    if first_array != -1 and last_array > first_array:
        candidates.append(raw[first_array : last_array + 1])

    repaired_candidates: list[str] = []
    for candidate in candidates:
        stripped = str(candidate or "").strip()
        if not stripped:
            continue
        repaired_candidates.append(stripped)
        repaired_candidates.append(re.sub(r",\s*([}\]])", r"\1", stripped))

    seen: set[str] = set()
    for candidate in repaired_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    try:
        from json_repair import repair_json

        repaired = repair_json(raw)
        if isinstance(repaired, str):
            return json.loads(repaired)
        return repaired
    except Exception:
        return None


def _ollama_extra_body(base_url: str, api_key: str) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if "11434" in str(base_url) or str(api_key).lower() == "ollama":
        body["format"] = "json"
    options: dict[str, Any] = {}
    for env_name, option_name in (
        ("OLLAMA_NUM_CTX", "num_ctx"),
        ("OLLAMA_NUM_PREDICT", "num_predict"),
        ("OLLAMA_SEED", "seed"),
    ):
        raw = os.getenv(env_name)
        if raw:
            try:
                options[option_name] = int(raw)
            except ValueError:
                pass
    if options:
        body["options"] = options
    return body


def _openai_compatible_completion(
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict genealogy extraction engine. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        extra_body=_ollama_extra_body(base_url, api_key) or None,
    )
    content = response.choices[0].message.content
    return str(content or "")


CompletionFunc = Callable[[str, LLMCandidateChunk], str]


def run_llm_on_candidates(
    candidates: Sequence[LLMCandidateChunk],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 300,
    max_tokens: int = 900,
    completion_func: CompletionFunc | None = None,
    raw_output_path: Path | None = None,
) -> list[LLMRawExtraction]:
    rows: list[LLMRawExtraction] = []
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text("", encoding="utf-8")
    for candidate in candidates:
        prompt = _build_llm_prompt(candidate)
        try:
            raw_output = (
                completion_func(prompt, candidate)
                if completion_func is not None
                else _openai_compatible_completion(
                    prompt,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            )
            if str(raw_output or "").strip():
                parsed = robust_json_loads(raw_output)
                parse_status = "parsed" if parsed is not None else "invalid_json"
            else:
                parsed = None
                parse_status = "empty_response"
            rows.append(
                LLMRawExtraction(
                    candidate_id=candidate.candidate_id,
                    chunk_id=candidate.chunk_id,
                    source_id=candidate.source_id,
                    page_idx=candidate.page_idx,
                    model=model,
                    prompt=prompt,
                    raw_output=raw_output,
                    parsed=parsed,
                    parse_status=parse_status,
                )
            )
        except Exception as exc:
            rows.append(
                LLMRawExtraction(
                    candidate_id=candidate.candidate_id,
                    chunk_id=candidate.chunk_id,
                    source_id=candidate.source_id,
                    page_idx=candidate.page_idx,
                    model=model,
                    prompt=prompt,
                    raw_output="",
                    parsed=None,
                    parse_status="error",
                    error=str(exc),
                )
            )
        if raw_output_path is not None:
            with raw_output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(rows[-1]), ensure_ascii=False) + "\n")
    return rows


def write_raw_extractions(
    output_dir: Path,
    raw_rows: Sequence[LLMRawExtraction],
) -> Path:
    path = output_dir / "llm_extraction_raw.jsonl"
    write_jsonl(path, (asdict(row) for row in raw_rows))
    return path


def _claims_payload(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        payload = parsed.get("claims") or []
    elif isinstance(parsed, list):
        payload = parsed
    else:
        return []
    return [row for row in payload if isinstance(row, dict)]


def _person_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        name = str(value.get("name") or "").strip()
        payload: dict[str, Any] = {"name": name}
        birth_year = _coerce_year(value.get("birth_year"))
        death_year = _coerce_year(value.get("death_year"))
        if birth_year is not None:
            payload["birth_year"] = birth_year
        if death_year is not None:
            payload["death_year"] = death_year
        return payload
    return {"name": str(value or "").strip()}


def _person_payload_issue(payload: dict[str, Any]) -> str | None:
    name = str(payload.get("name") or "").strip()
    if not name:
        return "missing_required_person_role"
    normalized = _normalize_for_match(name)
    if normalized in GENERIC_PERSON_NAMES:
        return "role_only_person"
    return None


def _evidence_quote(raw_claim: dict[str, Any]) -> str:
    for key in ("evidence_quote", "quote", "evidence"):
        value = raw_claim.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict) and str(value.get("quote") or "").strip():
            return str(value.get("quote")).strip()
    return ""


def _quote_is_supported(quote: str, candidate: LLMCandidateChunk) -> bool:
    if not quote:
        return False
    normalized_quote = _normalize_for_match(quote)
    if not normalized_quote:
        return False
    return normalized_quote in _normalize_for_match(_context_text(candidate))


def _source_path_by_id(
    sources: Sequence[dict[str, Any]],
    input_root: Path | str | None = None,
) -> dict[str, str]:
    return {
        str(row.get("source_id") or ""): sanitize_source_path(
            str(row.get("path") or ""),
            input_root,
        )
        or ""
        for row in sources
        if str(row.get("source_id") or "")
    }


def _candidate_by_id(
    candidates: Sequence[LLMCandidateChunk],
) -> dict[str, LLMCandidateChunk]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _normalize_claim_candidates(raw_claim: dict[str, Any]) -> list[dict[str, Any]]:
    claim_type = str(raw_claim.get("claim_type") or raw_claim.get("type") or "").strip()
    claim_type = claim_type.lower().replace("-", "_")
    if claim_type in {"parent", "parents", "parent_child_relation"}:
        claim_type = "parent_child"
    if claim_type in {"marriage", "married", "spouses"}:
        claim_type = "spouse"

    if claim_type == "child_list":
        parents = raw_claim.get("parents") or []
        children = raw_claim.get("children") or []
        child_items = children if isinstance(children, list) else []
        normalized: list[dict[str, Any]] = []
        for child in child_items:
            row = dict(raw_claim)
            row["claim_type"] = "parent_child"
            row["child"] = child
            row["parents"] = parents
            normalized.append(row)
        return normalized

    row = dict(raw_claim)
    row["claim_type"] = claim_type
    return [row]


def _claim_row_from_raw(
    raw_claim: dict[str, Any],
    *,
    candidate: LLMCandidateChunk,
    source_path: str,
) -> ClaimValidationResult:
    claim_type = str(raw_claim.get("claim_type") or "")
    confidence = _claim_confidence(raw_claim.get("confidence"))
    quote = _evidence_quote(raw_claim)

    if claim_type not in SUPPORTED_CLAIM_TYPES:
        return ClaimValidationResult(
            status=CLAIM_STATUS_REJECTED,
            reason="malformed_claim",
            confidence=confidence,
            evidence_quote=quote,
        )

    if not _quote_is_supported(quote, candidate):
        return ClaimValidationResult(
            status=CLAIM_STATUS_REJECTED,
            reason="unsupported_evidence_quote",
            confidence=confidence,
            evidence_quote=quote,
        )

    if claim_type == "parent_child":
        child = _person_payload(raw_claim.get("child") or raw_claim.get("person"))
        parents_payload = raw_claim.get("parents") or raw_claim.get("parent") or []
        if isinstance(parents_payload, dict):
            parents_payload = [parents_payload]
        if not isinstance(parents_payload, list):
            return ClaimValidationResult(
                status=CLAIM_STATUS_REJECTED,
                reason="malformed_claim",
                confidence=confidence,
                evidence_quote=quote,
            )
        parents = [_person_payload(parent) for parent in parents_payload]
        parents = [parent for parent in parents if parent.get("name")]
        child_issue = _person_payload_issue(child)
        if child_issue or not parents:
            return ClaimValidationResult(
                status=CLAIM_STATUS_NEEDS_REVIEW,
                reason=child_issue or "missing_required_person_role",
                confidence=confidence,
                evidence_quote=quote,
            )
        parent_issues = [_person_payload_issue(parent) for parent in parents]
        parent_issue = next((issue for issue in parent_issues if issue), None)
        if parent_issue:
            return ClaimValidationResult(
                status=CLAIM_STATUS_NEEDS_REVIEW,
                reason=parent_issue,
                confidence=confidence,
                evidence_quote=quote,
            )
        data = {"parents": parents[:2], "child": child}
    elif claim_type == "spouse":
        person1 = _person_payload(raw_claim.get("person1") or raw_claim.get("spouse1"))
        person2 = _person_payload(raw_claim.get("person2") or raw_claim.get("spouse2"))
        issue = _person_payload_issue(person1) or _person_payload_issue(person2)
        if issue:
            return ClaimValidationResult(
                status=CLAIM_STATUS_NEEDS_REVIEW,
                reason=issue,
                confidence=confidence,
                evidence_quote=quote,
            )
        data = {"person1": person1, "person2": person2}
    else:
        person = _person_payload(raw_claim.get("person") or raw_claim.get("subject"))
        issue = _person_payload_issue(person)
        if issue:
            return ClaimValidationResult(
                status=CLAIM_STATUS_NEEDS_REVIEW,
                reason=issue,
                confidence=confidence,
                evidence_quote=quote,
            )
        data = {"person": person, "attributes": raw_claim.get("attributes") or {}}

    evidence = {
        "file_path": sanitize_source_path(source_path),
        "doc_id": candidate.source_id,
        "chunk_id": candidate.chunk_id,
        "page_idx": candidate.page_idx,
        "quote": quote,
        "image_path": None,
    }
    status = CLAIM_STATUS_ACCEPTED
    reason = None
    if confidence is None or confidence < MIN_ACCEPTED_CLAIM_CONFIDENCE:
        status = CLAIM_STATUS_PENDING
        reason = "low_confidence"

    row = {
        "claim_type": claim_type,
        "confidence": confidence,
        "status": status,
        "data": data,
        "evidence": [evidence],
        "notes": raw_claim.get("notes"),
        "applied": status == CLAIM_STATUS_ACCEPTED
        and claim_type in {"parent_child", "spouse"},
        "raw_llm": raw_claim,
    }
    row["claim_id"] = claim_id_for_row(row)
    row["evidence"][0]["evidence_id"] = evidence_id_for_row(row["evidence"][0])
    if status == CLAIM_STATUS_ACCEPTED:
        return ClaimValidationResult(
            status=status,
            reason=None,
            claim=row,
            confidence=confidence,
            evidence_quote=quote,
        )
    return ClaimValidationResult(
        status=status,
        reason=reason,
        audit_claim=row,
        confidence=confidence,
        evidence_quote=quote,
    )


def validate_llm_extractions(
    raw_rows: Sequence[LLMRawExtraction],
    candidates: Sequence[LLMCandidateChunk],
    sources: Sequence[dict[str, Any]],
    input_root: Path | str | None = None,
) -> list[ValidatedClaimRow]:
    candidates_by_id = _candidate_by_id(candidates)
    paths_by_source = _source_path_by_id(sources, input_root)
    rows: list[ValidatedClaimRow] = []

    for raw_row in raw_rows:
        candidate = candidates_by_id.get(raw_row.candidate_id)
        if candidate is None:
            rows.append(
                ValidatedClaimRow(
                    status=CLAIM_STATUS_REJECTED,
                    reason="missing_candidate",
                    candidate_id=raw_row.candidate_id,
                    chunk_id=raw_row.chunk_id,
                    claim=None,
                    raw_claim=None,
                    confidence=None,
                    evidence_quote=None,
                )
            )
            continue
        if raw_row.parsed is None:
            rows.append(
                ValidatedClaimRow(
                    status=CLAIM_STATUS_REJECTED,
                    reason=raw_row.parse_status,
                    candidate_id=raw_row.candidate_id,
                    chunk_id=raw_row.chunk_id,
                    claim=None,
                    raw_claim=None,
                    confidence=None,
                    evidence_quote=None,
                )
            )
            continue

        raw_claims = _claims_payload(raw_row.parsed)
        if not raw_claims:
            rows.append(
                ValidatedClaimRow(
                    status=CLAIM_STATUS_REJECTED,
                    reason="no_claims",
                    candidate_id=raw_row.candidate_id,
                    chunk_id=raw_row.chunk_id,
                    claim=None,
                    raw_claim=None,
                    confidence=None,
                    evidence_quote=None,
                )
            )
            continue

        for raw_claim in raw_claims:
            for normalized_claim in _normalize_claim_candidates(raw_claim):
                validation = _claim_row_from_raw(
                    normalized_claim,
                    candidate=candidate,
                    source_path=paths_by_source.get(candidate.source_id, ""),
                )
                rows.append(
                    ValidatedClaimRow(
                        status=validation.status,
                        reason=validation.reason,
                        candidate_id=raw_row.candidate_id,
                        chunk_id=raw_row.chunk_id,
                        claim=validation.claim,
                        raw_claim=normalized_claim,
                        confidence=validation.confidence,
                        evidence_quote=validation.evidence_quote,
                        audit_claim=validation.audit_claim,
                    )
                )

    return rows


def write_validated_claims(
    output_dir: Path,
    validated_rows: Sequence[ValidatedClaimRow],
) -> tuple[Path, Path, Path]:
    accepted_claims = [
        row.claim
        for row in validated_rows
        if row.status == CLAIM_STATUS_ACCEPTED and row.claim is not None
    ]
    claims_path = output_dir / "claims.jsonl"
    write_jsonl(claims_path, accepted_claims)

    candidates_path = output_dir / "llm_claim_candidates.jsonl"
    write_jsonl(candidates_path, (asdict(row) for row in validated_rows))

    rejected_path = output_dir / "llm_rejected_claims.jsonl"
    write_jsonl(
        rejected_path,
        (asdict(row) for row in validated_rows if row.status != CLAIM_STATUS_ACCEPTED),
    )
    return claims_path, candidates_path, rejected_path


def _claim_from_row(row: dict[str, Any]) -> Claim:
    evidence = [
        Evidence(
            file_path=item.get("file_path"),
            doc_id=item.get("doc_id"),
            chunk_id=item.get("chunk_id"),
            page_idx=item.get("page_idx"),
            quote=item.get("quote"),
            image_path=item.get("image_path"),
        )
        for item in row.get("evidence") or []
        if isinstance(item, dict)
    ]
    return Claim(
        claim_type=str(row.get("claim_type") or ""),
        confidence=float(row.get("confidence") or 0.0),
        data=dict(row.get("data") or {}),
        evidence=evidence,
        notes=row.get("notes"),
        raw=row.get("raw_llm"),
    )


def _person_names_from_claim_rows(claim_rows: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in claim_rows:
        data = row.get("data") or {}
        for key in ("child", "person", "person1", "person2"):
            payload = data.get(key)
            if isinstance(payload, dict) and str(payload.get("name") or "").strip():
                names.append(str(payload.get("name")).strip())
        for parent in data.get("parents") or []:
            if isinstance(parent, dict) and str(parent.get("name") or "").strip():
                names.append(str(parent.get("name")).strip())
    return names


def _mention_allowed_for_claim_names(
    mention: MentionRecord,
    accepted_person_names: Sequence[str],
) -> bool:
    return _is_person_like_mention(
        mention.surface,
        mention.normalized_name,
        accepted_person_names,
    )


def _load_mentions(
    input_dir: Path,
    chunks: Sequence[dict[str, Any]],
    claim_rows: Sequence[dict[str, Any]],
) -> list[MentionRecord]:
    accepted_person_names = _person_names_from_claim_rows(claim_rows)
    mentions_path = input_dir / "mentions.jsonl"
    mentions: list[MentionRecord] = []
    if mentions_path.exists():
        for row in read_jsonl(mentions_path):
            mention = MentionRecord(
                mention_id=str(row.get("mention_id") or ""),
                source_id=str(row.get("source_id") or ""),
                chunk_id=str(row.get("chunk_id") or ""),
                surface=str(row.get("surface") or ""),
                normalized_name=str(row.get("normalized_name") or ""),
                page_idx=row.get("page_idx")
                if isinstance(row.get("page_idx"), int)
                else None,
                span_start=(
                    row.get("span_start")
                    if isinstance(row.get("span_start"), int)
                    else None
                ),
                span_end=(
                    row.get("span_end")
                    if isinstance(row.get("span_end"), int)
                    else None
                ),
                mention_type=str(row.get("mention_type") or "person"),
                attributes=dict(row.get("attributes") or {}),
                candidate_person_ids=[
                    str(item) for item in row.get("candidate_person_ids") or []
                ],
            )
            if _mention_allowed_for_claim_names(mention, accepted_person_names):
                mentions.append(mention)
        return mentions

    for chunk in chunks:
        mentions.extend(
            extract_mentions_from_text(
                str(chunk.get("text") or ""),
                source_id=str(chunk.get("source_id") or ""),
                chunk_id=str(chunk.get("chunk_id") or ""),
                page_idx=chunk.get("page_idx")
                if isinstance(chunk.get("page_idx"), int)
                else None,
                accepted_person_names=accepted_person_names,
            )
        )
    return mentions


def _copy_artifact_if_exists(input_dir: Path, output_dir: Path, name: str) -> None:
    source = input_dir / name
    if source.exists() and source.is_file():
        destination = output_dir / name
        if name == "sources.json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        row["path"] = sanitize_source_path(row.get("path"), input_dir)
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if name == "source_chunks.jsonl":
            rows = []
            for row in read_jsonl(source):
                metadata = row.get("metadata")
                if isinstance(metadata, dict):
                    metadata["source_path"] = sanitize_source_path(
                        metadata.get("source_path"),
                        input_dir,
                    )
                rows.append(row)
            write_jsonl(destination, rows)
            return
        shutil.copy2(source, destination)


def build_graph_from_validated_claims(
    *,
    input_dir: Path,
    output_dir: Path,
    claim_rows: Sequence[dict[str, Any]],
) -> BuildResult:
    store = InMemoryGenealogyStore()
    for row in claim_rows:
        _apply_claim_to_store(store, _claim_from_row(row))

    people, families = _store_to_people_and_families(store)
    chunks = _read_jsonl(output_dir / "source_chunks.jsonl")
    mentions = _load_mentions(input_dir, chunks, claim_rows)
    person_resolution = resolve_mentions_to_people(mentions, people, claim_rows)
    write_mentions(output_dir, mentions)
    write_person_resolution(output_dir, person_resolution)
    write_evidences(output_dir, claim_rows)
    graph_artifact = write_knowledge_graph_artifacts(
        output_dir,
        people=people,
        claim_rows=claim_rows,
        resolution=person_resolution,
    )

    (output_dir / "people.json").write_text(
        json.dumps(people, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "families.json").write_text(
        json.dumps(families, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dot = _build_dot_from_people_and_families(people, families)
    (output_dir / "tree.dot").write_text(dot, encoding="utf-8")
    (output_dir / "tree.html").write_text(
        _build_html_from_people_and_families(people, families, dot),
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
        claims_count=len(claim_rows),
        details={
            "source_chunks_count": len(chunks),
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


def run_llm_claim_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    model: str,
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    timeout: int = 300,
    max_tokens: int = 900,
    max_candidates: int | None = None,
    context_window: int = 1,
    completion_func: CompletionFunc | None = None,
) -> BuildResult:
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_chunks_path = input_dir / "source_chunks.jsonl"
    if not source_chunks_path.exists():
        raise FileNotFoundError(f"source_chunks.jsonl not found: {source_chunks_path}")

    for name in ("sources.json", "source_chunks.jsonl"):
        _copy_artifact_if_exists(input_dir, output_dir, name)

    source_chunks = _read_jsonl(source_chunks_path)
    sources = []
    sources_path = input_dir / "sources.json"
    if sources_path.exists():
        sources_payload = json.loads(sources_path.read_text(encoding="utf-8"))
        if isinstance(sources_payload, list):
            sources = [row for row in sources_payload if isinstance(row, dict)]

    candidates = find_candidate_chunks(
        source_chunks,
        context_window=context_window,
        max_candidates=max_candidates,
    )
    write_candidate_chunks(output_dir, candidates)

    raw_rows = run_llm_on_candidates(
        candidates,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        completion_func=completion_func,
        raw_output_path=output_dir / "llm_extraction_raw.jsonl",
    )
    write_raw_extractions(output_dir, raw_rows)

    validated_rows = validate_llm_extractions(
        raw_rows,
        candidates,
        sources,
        input_root=input_dir,
    )
    write_validated_claims(output_dir, validated_rows)
    accepted_claims = [
        row.claim
        for row in validated_rows
        if row.status == CLAIM_STATUS_ACCEPTED and row.claim is not None
    ]
    result = build_graph_from_validated_claims(
        input_dir=input_dir,
        output_dir=output_dir,
        claim_rows=accepted_claims,
    )
    result.details.update(
        {
            "candidate_chunks_count": len(candidates),
            "llm_raw_extractions_count": len(raw_rows),
            "accepted_claim_candidates_count": len(accepted_claims),
            "rejected_claim_candidates_count": sum(
                1 for row in validated_rows if row.status == CLAIM_STATUS_REJECTED
            ),
            "pending_claim_candidates_count": sum(
                1 for row in validated_rows if row.status == CLAIM_STATUS_PENDING
            ),
            "needs_review_claim_candidates_count": sum(
                1 for row in validated_rows if row.status == CLAIM_STATUS_NEEDS_REVIEW
            ),
            "empty_candidate_chunks_count": sum(
                1 for row in validated_rows if row.reason == "no_claims"
            ),
        }
    )
    return result
