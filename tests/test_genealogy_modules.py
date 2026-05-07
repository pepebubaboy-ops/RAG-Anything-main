from __future__ import annotations

import json
from pathlib import Path

from raganything.genealogy.build import build_genealogy_tree
from raganything.genealogy.claim_extraction import extract_claims_from_text
from raganything.genealogy.export import export_genealogy
from raganything.genealogy.living_graph import build_living_graph
from raganything.genealogy.llm_claim_extraction import (
    find_candidate_chunks,
    robust_json_loads,
    run_llm_claim_pipeline,
)
from raganything.genealogy.mentions import extract_mentions_from_text
from raganything.genealogy.rag_index import read_jsonl
from raganything.genealogy.query_resolution import resolve_genealogy_query
from raganything.genealogy.retrieval import (
    build_genealogy_answer_prompt,
    retrieve_genealogy_context,
)


def test_claim_extraction_cleans_name_noise_and_years() -> None:
    claims = list(
        extract_claims_from_text(
            (
                "Alice Smith Line (1900-1980) was child of "
                "Bob Smith Branch (1870-1930) and Carol Jones Dynasty (1875-1940). "
                "Bob Smith Branch spouse of Carol Jones Dynasty."
            ),
            source="fixture.json",
            page_idx=7,
        )
    )

    parent_claim = next(row for row in claims if row.claim_type == "parent_child")
    assert parent_claim.data["child"] == {
        "name": "Alice Smith",
        "birth_year": 1900,
        "death_year": 1980,
    }
    assert parent_claim.data["parents"][0] == {
        "name": "Bob Smith",
        "birth_year": 1870,
        "death_year": 1930,
    }
    assert parent_claim.evidence[0].page_idx == 7


def test_mention_extraction_supports_english_russian_and_alias_text() -> None:
    mentions = extract_mentions_from_text(
        (
            "Alice Smith (1900-1980) was child of Bob Smith. "
            "Петр — сын Алексей и Мария. "
            "Иван IV, известный как Иван Грозный."
        ),
        source_id="source-test",
        chunk_id="chunk-test",
        page_idx=3,
    )

    by_surface = {mention.surface: mention for mention in mentions}
    assert by_surface["Alice Smith"].attributes == {
        "birth_year": 1900,
        "death_year": 1980,
    }
    assert {"Bob Smith", "Петр", "Алексей", "Мария", "Иван IV", "Иван Грозный"} <= set(
        by_surface
    )
    assert by_surface["Иван IV"].normalized_name == "иван iv"
    assert all(mention.mention_id.startswith("mention-") for mention in mentions)


def test_build_genealogy_tree_and_export_public_api(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/sample_content_list.json").resolve()
    output_dir = tmp_path / "tree"

    result = build_genealogy_tree(fixture_path, output_dir)

    assert result.people_count == 3
    assert result.families_count == 1
    assert result.claims_count == 2
    assert result.details["sources_count"] == 1
    assert result.details["source_chunks_count"] == 2
    assert result.details["mentions_count"] == 5
    assert result.details["resolved_mentions_count"] == 5
    assert result.details["unresolved_mentions_count"] == 0
    assert result.details["ambiguous_mentions_count"] == 0
    assert result.details["relationships_count"] == 3
    assert result.details["conflicts_count"] == 0
    assert (output_dir / "people.json").exists()
    assert (output_dir / "families.json").exists()
    assert (output_dir / "sources.json").exists()
    assert (output_dir / "source_chunks.jsonl").exists()
    assert (output_dir / "mentions.jsonl").exists()
    assert (output_dir / "person_resolution.json").exists()
    assert (output_dir / "evidences.jsonl").exists()
    assert (output_dir / "relationships.json").exists()
    assert (output_dir / "conflicts.json").exists()
    assert (output_dir / "rag_documents.jsonl").exists()

    claims = list(read_jsonl(output_dir / "claims.jsonl"))
    assert claims[0]["claim_id"].startswith("claim-")
    assert claims[0]["evidence"][0]["evidence_id"].startswith("evidence-")
    assert claims[0]["evidence"][0]["doc_id"].startswith("source-")
    assert claims[0]["evidence"][0]["chunk_id"].startswith("chunk-")

    relationships = json.loads((output_dir / "relationships.json").read_text(encoding="utf-8"))
    relationship_types = {row["relationship_type"] for row in relationships}
    assert relationship_types == {"parent_of", "spouse_of"}
    assert all(row["claim_ids"] for row in relationships)
    assert all(row["evidence_ids"] for row in relationships)
    assert all(row["status"] == "accepted" for row in relationships)

    mentions = list(read_jsonl(output_dir / "mentions.jsonl"))
    assert len(mentions) == 5
    assert all(row["candidate_person_ids"] for row in mentions)
    resolution = json.loads(
        (output_dir / "person_resolution.json").read_text(encoding="utf-8")
    )
    assert len(resolution["resolved"]) == 3
    assert resolution["unresolved_mentions"] == []
    assert resolution["ambiguous_mentions"] == []
    assert all(row["claim_ids"] for row in resolution["resolved"])

    rag_docs = list(read_jsonl(output_dir / "rag_documents.jsonl"))
    rag_kinds = {row["kind"] for row in rag_docs}
    assert {
        "person",
        "family",
        "relationship",
        "claim",
        "source_chunk",
        "mention",
        "resolution",
    } <= rag_kinds
    assert any("Alice Doe" in row["text"] for row in rag_docs if row["kind"] == "person")

    contexts = retrieve_genealogy_context("Who are Alice Doe parents?", output_dir, top_k=3)
    assert contexts
    assert any("Bob Doe" in context.text for context in contexts)
    prompt = build_genealogy_answer_prompt("Who are Alice Doe parents?", contexts)
    assert "Cite context numbers" in prompt
    assert "Question: Who are Alice Doe parents?" in prompt

    gedcom_path = export_genealogy(output_dir, "gedcom")
    assert gedcom_path == output_dir / "tree.ged"
    assert "0 HEAD" in gedcom_path.read_text(encoding="utf-8")


def test_unresolved_mentions_do_not_create_people_or_relationships(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "text": "Иван IV, известный как Иван Грозный.",
            "page_idx": 2,
        }
    ]
    fixture_path = tmp_path / "content_list_mentions_only.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = build_genealogy_tree(fixture_path, output_dir)

    assert result.people_count == 0
    assert result.details["mentions_count"] == 2
    assert result.details["resolved_mentions_count"] == 0
    assert result.details["unresolved_mentions_count"] == 2
    assert json.loads((output_dir / "relationships.json").read_text(encoding="utf-8")) == []
    resolution = json.loads(
        (output_dir / "person_resolution.json").read_text(encoding="utf-8")
    )
    assert {row["surface"] for row in resolution["unresolved_mentions"]} == {
        "Иван IV",
        "Иван Грозный",
    }


def test_build_genealogy_tree_reports_graph_conflicts(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "text": "Child One (1900-1980) was child of Parent Too Young (1895-1970) and Parent Two (1870-1940).",
            "page_idx": 1,
        }
    ]
    fixture_path = tmp_path / "content_list_conflict.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = build_genealogy_tree(fixture_path, output_dir)

    assert result.details["relationships_count"] == 2
    assert result.details["conflicts_count"] == 1
    conflicts = json.loads((output_dir / "conflicts.json").read_text(encoding="utf-8"))
    assert conflicts[0]["conflict_type"] == "implausible_parent_age_gap"
    relationships = json.loads((output_dir / "relationships.json").read_text(encoding="utf-8"))
    assert {row["status"] for row in relationships} == {"accepted", "conflict"}

    rag_docs = list(read_jsonl(output_dir / "rag_documents.jsonl"))
    parent_too_young_person_docs = [
        row
        for row in rag_docs
        if row["kind"] == "person" and row["metadata"].get("name") == "Parent Too Young"
    ]
    assert parent_too_young_person_docs
    assert "Children: Child One" not in parent_too_young_person_docs[0]["text"]
    assert any(row["kind"] == "conflict" for row in rag_docs)
    assert any(
        "not accepted as fact" in row["text"]
        for row in rag_docs
        if row["kind"] == "relationship"
    )


def test_build_genealogy_tree_reports_parent_cycles(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "text": "Alice Doe was child of Bob Doe. Bob Doe was child of Alice Doe.",
            "page_idx": 1,
        }
    ]
    fixture_path = tmp_path / "content_list_cycle.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = build_genealogy_tree(fixture_path, output_dir)

    assert result.details["relationships_count"] == 2
    conflicts = json.loads((output_dir / "conflicts.json").read_text(encoding="utf-8"))
    assert "parent_cycle" in {row["conflict_type"] for row in conflicts}
    relationships = json.loads((output_dir / "relationships.json").read_text(encoding="utf-8"))
    assert {row["status"] for row in relationships} == {"conflict"}


def test_retrieval_prioritizes_exact_person_and_graph_relationships(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "type": "text",
            "text": "Paul I Romanov Line was son of Peter III Romanov Line and Catherine II Romanov Line.",
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Nicholas I Romanov Line was son of Paul I Romanov Line and Maria Feodorovna Wurttemberg Line.",
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Alexander III Romanov Line was married to Maria Feodorovna Dagmar Line.",
            "page_idx": 2,
        },
        {
            "type": "text",
            "text": "Nicholas II Romanov Line was son of Alexander III Romanov Line and Maria Feodorovna Dagmar Line.",
            "page_idx": 2,
        },
    ]
    fixture_path = tmp_path / "content_list_romanov_retrieval.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    build_genealogy_tree(fixture_path, output_dir)

    resolved = resolve_genealogy_query("Who are Nicholas II Romanov parents?", output_dir)
    assert resolved is not None
    assert resolved.name == "Nicholas II Romanov"
    assert resolved.intent == "parents"

    contexts = retrieve_genealogy_context(
        "Who are Nicholas II Romanov parents?",
        output_dir,
        top_k=3,
    )
    joined_context = " ".join(context.text for context in contexts)

    assert contexts[0].kind == "person"
    assert contexts[0].metadata["name"] == "Nicholas II Romanov"
    assert "Alexander III Romanov" in joined_context
    assert "Maria Feodorovna Dagmar" in joined_context


def test_retrieval_keeps_similar_romanov_names_separate(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "text": "Paul I Romanov Line was married to Maria Feodorovna Wurttemberg Line.",
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Nicholas I Romanov Line was son of Paul I Romanov Line and Maria Feodorovna Wurttemberg Line.",
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Alexander III Romanov Line was married to Maria Feodorovna Dagmar Line.",
            "page_idx": 2,
        },
        {
            "type": "text",
            "text": "Nicholas II Romanov Line was son of Alexander III Romanov Line and Maria Feodorovna Dagmar Line.",
            "page_idx": 2,
        },
    ]
    fixture_path = tmp_path / "content_list_similar_romanov_names.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    build_genealogy_tree(fixture_path, output_dir)

    contexts = retrieve_genealogy_context(
        "Who is Maria Feodorovna Dagmar connected to?",
        output_dir,
        top_k=3,
    )
    joined_context = " ".join(context.text for context in contexts)

    assert contexts[0].kind == "person"
    assert contexts[0].metadata["name"] == "Maria Feodorovna Dagmar"
    assert "Alexander III Romanov" in joined_context
    assert "Nicholas II Romanov" in joined_context
    assert "Maria Feodorovna Wurttemberg" not in joined_context


def test_llm_candidate_search_finds_genealogy_chunks() -> None:
    chunks = [
        {
            "chunk_id": "chunk-heading",
            "source_id": "source-test",
            "text": "Алексей Михайлович (1629-1676)",
            "page_idx": 1,
        },
        {
            "chunk_id": "chunk-family",
            "source_id": "source-test",
            "text": "Был дважды женат. Первая жена Мария Милославская родила ему 13 детей.",
            "page_idx": 1,
        },
    ]

    candidates = find_candidate_chunks(chunks, context_window=1)

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chunk-family"
    assert candidates[0].subject_hint == "Алексей Михайлович"
    assert {"женат", "жена", "детей", "родила"} <= set(candidates[0].trigger_terms)


def test_llm_claim_pipeline_repairs_validates_and_builds_graph(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_id = "source-test"
    (input_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": source_id,
                    "path": str(input_dir / "fixture.pdf"),
                    "title": "fixture.pdf",
                    "metadata": {},
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_chunks = [
        {
            "chunk_id": "chunk-heading",
            "source_id": source_id,
            "text": "Алексей Михайлович (1629-1676)",
            "ordinal": 0,
            "page_idx": 1,
            "content_type": "text",
            "metadata": {},
        },
        {
            "chunk_id": "chunk-family",
            "source_id": source_id,
            "text": "Был дважды женат. Первая жена Мария Милославская родила ему 13 детей.",
            "ordinal": 1,
            "page_idx": 1,
            "content_type": "text",
            "metadata": {},
        },
    ]
    with (input_dir / "source_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for row in source_chunks:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert robust_json_loads('```json\n{"claims": [{"claim_type": "spouse",}],}\n```')

    def fake_completion(_prompt: str, _candidate: object) -> str:
        return """
```json
{
  "claims": [
    {
      "claim_type": "spouse",
      "person1": {"name": "Алексей Михайлович"},
      "person2": {"name": "Мария Милославская"},
      "evidence_quote": "Первая жена Мария Милославская",
      "confidence": 0.82,
    }
  ],
}
```
"""

    output_dir = tmp_path / "out"
    result = run_llm_claim_pipeline(
        input_dir,
        output_dir,
        model="fake-model",
        completion_func=fake_completion,
    )

    assert result.claims_count == 1
    assert result.people_count == 2
    assert result.details["candidate_chunks_count"] == 1
    assert result.details["accepted_claim_candidates_count"] == 1
    assert (output_dir / "llm_extraction_raw.jsonl").exists()
    assert (output_dir / "llm_claim_candidates.jsonl").exists()

    relationships = json.loads((output_dir / "relationships.json").read_text(encoding="utf-8"))
    assert len(relationships) == 1
    assert relationships[0]["relationship_type"] == "spouse_of"
    assert relationships[0]["status"] == "accepted"


def test_build_living_graph_public_api_filters_relations(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/sample_living_graph.json").resolve()
    output_dir = tmp_path / "living"

    result = build_living_graph(
        fixture_path,
        output_dir,
        exclude_relation_types={"spouse"},
    )

    assert result.entities_count == 3
    assert result.relations_count == 1
    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    assert [row["relation_type"] for row in relations] == ["father_of"]
