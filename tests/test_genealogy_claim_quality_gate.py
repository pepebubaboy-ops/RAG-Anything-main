from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from raganything.genealogy.llm_claim_extraction import run_llm_claim_pipeline
from raganything.genealogy.mentions import extract_mentions_from_text
from raganything.genealogy.rag_index import read_jsonl


def _write_llm_input(
    tmp_path: Path,
    text: str,
    *,
    source_path: str = "/Users/test/private/book.pdf",
) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_id = "source-test"
    (input_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": source_id,
                    "path": source_path,
                    "title": "book.pdf",
                    "metadata": {},
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (input_dir / "source_chunks.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "chunk_id": "chunk-test",
                    "source_id": source_id,
                    "text": text,
                    "ordinal": 0,
                    "page_idx": 1,
                    "content_type": "text",
                    "metadata": {"source_path": source_path},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return input_dir


def _run_llm_case(
    tmp_path: Path,
    text: str,
    response: str | dict[str, Any],
    *,
    source_path: str = "/Users/test/private/book.pdf",
) -> Path:
    input_dir = _write_llm_input(tmp_path, text, source_path=source_path)
    output_dir = tmp_path / "out"
    raw_response = (
        json.dumps(response, ensure_ascii=False)
        if isinstance(response, dict)
        else response
    )

    def fake_completion(_prompt: str, _candidate: object) -> str:
        return raw_response

    run_llm_claim_pipeline(
        input_dir,
        output_dir,
        model="fake-model",
        completion_func=fake_completion,
    )
    return output_dir


def _parent_claim(confidence: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "claim_type": "parent_child",
        "child": {"name": "Ivan Petrov"},
        "parents": [{"name": "Peter Petrov"}, {"name": "Anna Petrova"}],
        "evidence_quote": "Ivan Petrov was child of Peter Petrov and Anna Petrova",
    }
    if confidence is not None:
        row["confidence"] = confidence
    return row


def _spouse_claim(confidence: float) -> dict[str, Any]:
    return {
        "claim_type": "spouse",
        "person1": {"name": "Ivan Petrov"},
        "person2": {"name": "Maria Petrova"},
        "evidence_quote": "Ivan Petrov spouse of Maria Petrova",
        "confidence": confidence,
    }


def test_low_confidence_parent_claim_is_pending_audit_only(tmp_path: Path) -> None:
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov was child of Peter Petrov and Anna Petrova.",
        {"claims": [_parent_claim(0.0)]},
    )

    assert list(read_jsonl(output_dir / "claims.jsonl")) == []
    audit_rows = list(read_jsonl(output_dir / "llm_rejected_claims.jsonl"))
    assert audit_rows[0]["status"] == "pending"
    assert audit_rows[0]["reason"] == "low_confidence"
    assert audit_rows[0]["confidence"] == 0.0
    assert json.loads((output_dir / "relationships.json").read_text()) == []


def test_low_confidence_spouse_claim_does_not_enter_graph(tmp_path: Path) -> None:
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov spouse of Maria Petrova.",
        {"claims": [_spouse_claim(0.0)]},
    )

    assert list(read_jsonl(output_dir / "claims.jsonl")) == []
    relationships = json.loads((output_dir / "relationships.json").read_text())
    assert [row["relationship_type"] for row in relationships] == []


def test_supported_high_confidence_claim_is_accepted(tmp_path: Path) -> None:
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov was child of Peter Petrov and Anna Petrova.",
        {"claims": [_parent_claim(0.8)]},
    )

    claims = list(read_jsonl(output_dir / "claims.jsonl"))
    assert len(claims) == 1
    assert claims[0]["status"] == "accepted"
    relationships = json.loads((output_dir / "relationships.json").read_text())
    assert {row["relationship_type"] for row in relationships} == {"parent_of"}
    assert {row["status"] for row in relationships} == {"accepted"}


def test_unsupported_quote_is_rejected_without_relationship(tmp_path: Path) -> None:
    claim = _parent_claim(0.9)
    claim["evidence_quote"] = "not present in source text"
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov was child of Peter Petrov and Anna Petrova.",
        {"claims": [claim]},
    )

    audit_rows = list(read_jsonl(output_dir / "llm_rejected_claims.jsonl"))
    assert audit_rows[0]["status"] == "rejected"
    assert audit_rows[0]["reason"] == "unsupported_evidence_quote"
    assert list(read_jsonl(output_dir / "claims.jsonl")) == []
    assert json.loads((output_dir / "relationships.json").read_text()) == []


def test_empty_llm_output_is_empty_response_not_invalid_json(tmp_path: Path) -> None:
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov was child of Peter Petrov and Anna Petrova.",
        "",
    )

    raw_rows = list(read_jsonl(output_dir / "llm_extraction_raw.jsonl"))
    assert raw_rows[0]["parse_status"] == "empty_response"
    audit_rows = list(read_jsonl(output_dir / "llm_rejected_claims.jsonl"))
    assert audit_rows[0]["status"] == "rejected"
    assert audit_rows[0]["reason"] == "empty_response"
    assert "invalid_json" not in json.dumps(audit_rows, ensure_ascii=False)


def test_co_parents_are_not_inferred_spouses_in_graph_or_dot(tmp_path: Path) -> None:
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov was child of Peter Petrov and Anna Petrova.",
        {"claims": [_parent_claim(0.8)]},
    )

    relationships = json.loads((output_dir / "relationships.json").read_text())
    assert [row["relationship_type"] for row in relationships] == [
        "parent_of",
        "parent_of",
    ]
    assert "spouse_of" not in {row["relationship_type"] for row in relationships}
    assert 'label="spouse"' not in (output_dir / "tree.dot").read_text(encoding="utf-8")


def test_common_heading_word_is_not_person_mention() -> None:
    mentions = extract_mentions_from_text(
        "История родословной семьи",
        source_id="source-test",
        chunk_id="chunk-test",
    )

    assert "История" not in {mention.surface for mention in mentions}


def test_generated_artifacts_sanitize_absolute_source_paths(tmp_path: Path) -> None:
    output_dir = _run_llm_case(
        tmp_path,
        "Ivan Petrov was child of Peter Petrov and Anna Petrova.",
        {"claims": [_parent_claim(0.8)]},
        source_path="/Users/test/private/book.pdf",
    )

    claims = list(read_jsonl(output_dir / "claims.jsonl"))
    assert claims[0]["evidence"][0]["file_path"] == "book.pdf"
    sources = json.loads((output_dir / "sources.json").read_text(encoding="utf-8"))
    assert sources[0]["path"] == "book.pdf"
    for artifact in (
        "claims.jsonl",
        "evidences.jsonl",
        "sources.json",
        "source_chunks.jsonl",
        "rag_documents.jsonl",
    ):
        assert "/Users/" not in (output_dir / artifact).read_text(encoding="utf-8")
