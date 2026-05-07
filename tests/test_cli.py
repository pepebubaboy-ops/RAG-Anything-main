from __future__ import annotations

import json
from pathlib import Path

import pytest

from raganything.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["genealogy", "build", "--help"],
    ],
)
def test_cli_help(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def test_cli_genealogy_build_and_export_smoke(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/sample_content_list.json").resolve()
    output_dir = tmp_path / "out"

    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
        ]
    )
    assert code == 0

    tree_dot = output_dir / "tree.dot"
    tree_html = output_dir / "tree.html"
    claims_jsonl = output_dir / "claims.jsonl"
    people_json = output_dir / "people.json"
    families_json = output_dir / "families.json"

    assert tree_dot.exists()
    assert tree_html.exists()
    assert claims_jsonl.exists()
    assert people_json.exists()
    assert families_json.exists()

    people = json.loads(people_json.read_text(encoding="utf-8"))
    names = {row["name"] for row in people}
    assert {"Alice Doe", "Bob Doe", "Carol Smith"}.issubset(names)

    export_code = main(
        [
            "genealogy",
            "export",
            "--input",
            str(output_dir),
            "--format",
            "json",
        ]
    )
    assert export_code == 0
    assert (output_dir / "tree.json").exists()

    export_html_code = main(
        [
            "genealogy",
            "export",
            "--input",
            str(output_dir),
            "--format",
            "html",
        ]
    )
    assert export_html_code == 0
    assert (output_dir / "tree.html").exists()


def test_cli_genealogy_build_normalizes_name_noise_and_years(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "text": (
                "Alice Smith Line (1900-1980) was child of "
                "Bob Smith Branch (1870-1930) and Carol Jones Dynasty (1875-1940)."
            ),
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Bob Smith Branch spouse of Carol Jones Dynasty.",
            "page_idx": 1,
        },
    ]
    fixture_path = tmp_path / "content_list_noise.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
        ]
    )
    assert code == 0

    people = json.loads((output_dir / "people.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in people}
    assert set(by_name) == {"Alice Smith", "Bob Smith", "Carol Jones"}
    assert by_name["Alice Smith"]["birth_year"] == 1900
    assert by_name["Alice Smith"]["death_year"] == 1980
    assert by_name["Bob Smith"]["birth_year"] == 1870
    assert by_name["Bob Smith"]["death_year"] == 1930
    assert by_name["Carol Jones"]["birth_year"] == 1875
    assert by_name["Carol Jones"]["death_year"] == 1940


def test_cli_genealogy_build_reconciles_with_reference_people(tmp_path: Path) -> None:
    payload = [
        {
            "type": "text",
            "text": "Alice Smith Line was child of Bob Smith Line and Carol Jones Line.",
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Bob Smith Line spouse of Carol Jones Line.",
            "page_idx": 1,
        },
    ]
    fixture_path = tmp_path / "content_list.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reference_people_path = tmp_path / "reference_people.json"
    reference_people_path.write_text(
        json.dumps(
            [
                {
                    "person_id": "r1",
                    "name": "Alice Smith",
                    "normalized_name": "alice smith",
                    "birth_year": 1900,
                    "death_year": 1980,
                },
                {
                    "person_id": "r2",
                    "name": "Bob Smith",
                    "normalized_name": "bob smith",
                    "birth_year": 1870,
                    "death_year": 1930,
                },
                {
                    "person_id": "r3",
                    "name": "Carol Jones",
                    "normalized_name": "carol jones",
                    "birth_year": 1875,
                    "death_year": 1940,
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--reference-people",
            str(reference_people_path),
        ]
    )
    assert code == 0

    people = json.loads((output_dir / "people.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in people}
    assert set(by_name) == {"Alice Smith", "Bob Smith", "Carol Jones"}
    assert by_name["Alice Smith"]["birth_year"] == 1900
    assert by_name["Alice Smith"]["death_year"] == 1980
    assert by_name["Bob Smith"]["birth_year"] == 1870
    assert by_name["Bob Smith"]["death_year"] == 1930
    assert by_name["Carol Jones"]["birth_year"] == 1875
    assert by_name["Carol Jones"]["death_year"] == 1940


def test_cli_reference_reconciliation_keeps_distinct_similar_names(
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
            "page_idx": 1,
        },
        {
            "type": "text",
            "text": "Nicholas II Romanov Line was son of Alexander III Romanov Line and Maria Feodorovna Dagmar Line.",
            "page_idx": 1,
        },
    ]
    fixture_path = tmp_path / "content_list_similar_names.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    reference_people_path = tmp_path / "reference_people_similar.json"
    reference_people_path.write_text(
        json.dumps(
            [
                {"name": "Peter III Romanov", "normalized_name": "peter iii romanov"},
                {"name": "Catherine II", "normalized_name": "catherine ii"},
                {"name": "Paul I Romanov", "normalized_name": "paul i romanov"},
                {
                    "name": "Maria Feodorovna Wurttemberg",
                    "normalized_name": "maria feodorovna wurttemberg",
                },
                {"name": "Nicholas I Romanov", "normalized_name": "nicholas i romanov"},
                {
                    "name": "Alexander III Romanov",
                    "normalized_name": "alexander iii romanov",
                },
                {
                    "name": "Maria Feodorovna Dagmar",
                    "normalized_name": "maria feodorovna dagmar",
                },
                {
                    "name": "Nicholas II Romanov",
                    "normalized_name": "nicholas ii romanov",
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--reference-people",
            str(reference_people_path),
        ]
    )
    assert code == 0

    people = json.loads((output_dir / "people.json").read_text(encoding="utf-8"))
    families = json.loads((output_dir / "families.json").read_text(encoding="utf-8"))
    id_to_name = {row["person_id"]: row["name"] for row in people}

    parent_child_edges = {
        (id_to_name[parent_id], id_to_name[child_id])
        for family in families
        for parent_id in (family.get("parent_ids") or [])
        for child_id in (family.get("child_ids") or [])
    }

    assert ("Maria Feodorovna Wurttemberg", "Nicholas I Romanov") in parent_child_edges
    assert ("Maria Feodorovna Dagmar", "Nicholas II Romanov") in parent_child_edges


def test_cli_living_graph_build_from_json(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/sample_living_graph.json").resolve()
    output_dir = tmp_path / "living"

    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    assert (output_dir / "living_graph.dot").exists()
    assert (output_dir / "living_graph.html").exists()
    assert (output_dir / "entities.json").exists()
    assert (output_dir / "relations.json").exists()
    assert (output_dir / "conflicts.json").exists()

    dot_text = (output_dir / "living_graph.dot").read_text(encoding="utf-8")
    html_text = (output_dir / "living_graph.html").read_text(encoding="utf-8")
    assert 'class="rel-type-spouse"' in dot_text
    assert 'class="rel-type-father-of"' in dot_text
    assert "Show all" in html_text
    assert "rel-toggle" in html_text

    conflicts = json.loads((output_dir / "conflicts.json").read_text(encoding="utf-8"))
    assert "entity_merge_groups" in conflicts
    assert "parent_timeline_repairs" in conflicts
    assert "parent_role_conflicts" in conflicts


def test_cli_living_graph_relation_type_filters(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/sample_living_graph.json").resolve()
    output_dir = tmp_path / "living_filtered"

    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
            "--exclude-relation-types",
            "spouse",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    assert [row["relation_type"] for row in relations] == ["father_of"]

    dot_text = (output_dir / "living_graph.dot").read_text(encoding="utf-8")
    assert "spouse" not in dot_text
    assert "father_of" in dot_text

    output_dir_include = tmp_path / "living_included"
    include_code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir_include),
            "--graph-mode",
            "living",
            "--include-relation-types",
            "spouse",
        ]
    )
    assert include_code == 0

    include_relations = json.loads(
        (output_dir_include / "relations.json").read_text(encoding="utf-8")
    )
    assert [row["relation_type"] for row in include_relations] == ["spouse"]

    output_dir_parent_alias = tmp_path / "living_parent_alias"
    alias_code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir_parent_alias),
            "--graph-mode",
            "living",
            "--include-relation-types",
            "parent_child",
        ]
    )
    assert alias_code == 0

    alias_relations = json.loads(
        (output_dir_parent_alias / "relations.json").read_text(encoding="utf-8")
    )
    assert [row["relation_type"] for row in alias_relations] == ["father_of"]


def test_cli_living_graph_derive_kinship(tmp_path: Path) -> None:
    payload = {
        "entities": [
            {"entity_id": "gp1", "canonical_name": "Grandpa", "gender": "male"},
            {"entity_id": "gp2", "canonical_name": "Grandma", "gender": "female"},
            {"entity_id": "p1", "canonical_name": "Father", "gender": "male"},
            {"entity_id": "p2", "canonical_name": "Uncle", "gender": "male"},
            {"entity_id": "c1", "canonical_name": "Child One", "gender": "male"},
            {"entity_id": "c2", "canonical_name": "Child Two", "gender": "female"},
        ],
        "relations": [
            {
                "source_entity_id": "gp1",
                "target_entity_id": "p1",
                "relation_type": "parent_child",
            },
            {
                "source_entity_id": "gp2",
                "target_entity_id": "p1",
                "relation_type": "parent_child",
            },
            {
                "source_entity_id": "gp1",
                "target_entity_id": "p2",
                "relation_type": "parent_child",
            },
            {
                "source_entity_id": "gp2",
                "target_entity_id": "p2",
                "relation_type": "parent_child",
            },
            {
                "source_entity_id": "p1",
                "target_entity_id": "c1",
                "relation_type": "parent_child",
            },
            {
                "source_entity_id": "p2",
                "target_entity_id": "c2",
                "relation_type": "parent_child",
            },
        ],
    }
    fixture_path = tmp_path / "living_with_kinship.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_kinship"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
            "--derive-kinship",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    relation_keys = {
        (
            row.get("source_entity_id"),
            row.get("relation_type"),
            row.get("target_entity_id"),
        )
        for row in relations
    }

    assert ("p1", "brother_of", "p2") in relation_keys
    assert ("p2", "uncle_of", "c1") in relation_keys
    assert ("c1", "nephew_of", "p2") in relation_keys
    assert ("gp1", "grandfather_of", "c1") in relation_keys
    assert ("c1", "grandson_of", "gp1") in relation_keys
    assert ("c1", "cousin_of", "c2") in relation_keys


def test_cli_living_graph_promotes_parent_from_weak_relation(tmp_path: Path) -> None:
    payload = {
        "entities": [
            {
                "entity_id": "ent-parent",
                "canonical_name": "John King",
                "gender": "male",
                "birth_year": 1950,
            },
            {
                "entity_id": "ent-child",
                "canonical_name": "Mark King",
                "gender": "male",
                "birth_year": 1980,
            },
        ],
        "relations": [
            {
                "source_entity_id": "ent-child",
                "target_entity_id": "ent-parent",
                "relation_type": "relative",
                "directed": False,
                "symmetric": True,
                "confidence": 0.86,
                "evidence": [
                    {
                        "quote": "Mark King is the son of John King.",
                        "page_idx": 3,
                    }
                ],
            }
        ],
    }
    fixture_path = tmp_path / "living_parent_promote.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_parent_promote"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "father_of"
    assert relations[0]["source_entity_id"] == "ent-parent"
    assert relations[0]["target_entity_id"] == "ent-child"


def test_cli_living_graph_does_not_promote_without_name_anchors(tmp_path: Path) -> None:
    payload = {
        "entities": [
            {
                "entity_id": "ent-parent",
                "canonical_name": "John King",
                "gender": "male",
                "birth_year": 1950,
            },
            {
                "entity_id": "ent-child",
                "canonical_name": "Mark King",
                "gender": "male",
                "birth_year": 1980,
            },
        ],
        "relations": [
            {
                "source_entity_id": "ent-child",
                "target_entity_id": "ent-parent",
                "relation_type": "relative",
                "directed": False,
                "symmetric": True,
                "confidence": 0.95,
                "evidence": [
                    {
                        "quote": "He was the son of the emperor.",
                        "page_idx": 3,
                    }
                ],
            }
        ],
    }
    fixture_path = tmp_path / "living_parent_no_anchors.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_parent_no_anchors"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "relative"


def test_cli_living_graph_resolves_parent_role_conflicts(tmp_path: Path) -> None:
    payload = {
        "entities": [
            {
                "entity_id": "father-1",
                "canonical_name": "Father One",
                "gender": "male",
            },
            {
                "entity_id": "father-2",
                "canonical_name": "Father Two",
                "gender": "male",
            },
            {
                "entity_id": "father-3",
                "canonical_name": "Father Three",
                "gender": "male",
            },
            {"entity_id": "child-1", "canonical_name": "Child One", "gender": "male"},
        ],
        "relations": [
            {
                "source_entity_id": "father-1",
                "target_entity_id": "child-1",
                "relation_type": "parent_child",
                "confidence": 0.95,
                "support_count": 2,
            },
            {
                "source_entity_id": "father-2",
                "target_entity_id": "child-1",
                "relation_type": "parent_child",
                "confidence": 0.8,
                "support_count": 1,
            },
            {
                "source_entity_id": "father-3",
                "target_entity_id": "child-1",
                "relation_type": "father_of",
                "confidence": 0.85,
                "support_count": 1,
            },
        ],
    }
    fixture_path = tmp_path / "living_parent_conflicts.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_parent_conflicts"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    father_relations = [
        row for row in relations if row.get("relation_type") == "father_of"
    ]
    assert len(father_relations) == 1
    assert father_relations[0]["source_entity_id"] == "father-1"
    assert father_relations[0]["target_entity_id"] == "child-1"

    downgraded_relatives = [
        row
        for row in relations
        if row.get("relation_type") == "relative"
        and row.get("target_entity_id") == "child-1"
    ]
    assert len(downgraded_relatives) == 2

    conflicts = json.loads((output_dir / "conflicts.json").read_text(encoding="utf-8"))
    parent_conflicts = conflicts["parent_role_conflicts"]
    assert len(parent_conflicts) == 1
    conflict = parent_conflicts[0]
    assert conflict["child_entity_id"] == "child-1"
    assert conflict["role"] == "father"
    assert conflict["limit"] == 1
    assert len(conflict["kept_relations"]) == 1
    assert len(conflict["downgraded_relations"]) == 2


def test_cli_living_graph_timeline_guardrail_reverses_parent_direction(
    tmp_path: Path,
) -> None:
    payload = {
        "entities": [
            {
                "entity_id": "older",
                "canonical_name": "Older Parent",
                "gender": "male",
                "birth_year": 1940,
                "death_year": 2000,
            },
            {
                "entity_id": "younger",
                "canonical_name": "Younger Child",
                "gender": "male",
                "birth_year": 1970,
            },
        ],
        "relations": [
            {
                "source_entity_id": "younger",
                "target_entity_id": "older",
                "relation_type": "parent_child",
                "confidence": 0.9,
                "support_count": 1,
            }
        ],
    }
    fixture_path = tmp_path / "living_timeline_reverse.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_timeline_reverse"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "father_of"
    assert relations[0]["source_entity_id"] == "older"
    assert relations[0]["target_entity_id"] == "younger"

    conflicts = json.loads((output_dir / "conflicts.json").read_text(encoding="utf-8"))
    repairs = conflicts["parent_timeline_repairs"]
    assert len(repairs) == 1
    assert repairs[0]["action"] == "reversed"


def test_cli_living_graph_parent_conflict_prefers_quote_parent_anchor(
    tmp_path: Path,
) -> None:
    payload = {
        "entities": [
            {
                "entity_id": "paul",
                "canonical_name": "Павел I",
                "gender": "male",
                "birth_year": 1754,
                "death_year": 1801,
            },
            {
                "entity_id": "alexander",
                "canonical_name": "Александр I",
                "gender": "male",
                "birth_year": 1777,
                "death_year": 1825,
            },
            {
                "entity_id": "nicholas",
                "canonical_name": "Николай I",
                "gender": "male",
                "birth_year": 1796,
                "death_year": 1855,
            },
        ],
        "relations": [
            {
                "source_entity_id": "paul",
                "target_entity_id": "nicholas",
                "relation_type": "parent_child",
                "confidence": 0.9,
                "support_count": 2,
                "evidence": [
                    {"quote": "Сын императора Павла I, брат императора Александра I."}
                ],
            },
            {
                "source_entity_id": "alexander",
                "target_entity_id": "nicholas",
                "relation_type": "parent_child",
                "confidence": 0.98,
                "support_count": 2,
                "evidence": [
                    {"quote": "Сын императора Павла I, брат императора Александра I."}
                ],
            },
        ],
    }
    fixture_path = tmp_path / "living_parent_anchor.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_parent_anchor"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    father_relations = [
        row for row in relations if row.get("relation_type") == "father_of"
    ]
    assert len(father_relations) == 1
    assert father_relations[0]["source_entity_id"] == "paul"
    assert father_relations[0]["target_entity_id"] == "nicholas"

    downgraded = [
        row
        for row in relations
        if row.get("relation_type") == "relative"
        and row.get("source_entity_id") == "alexander"
        and row.get("target_entity_id") == "nicholas"
    ]
    assert len(downgraded) == 1


def test_cli_living_graph_merges_duplicate_entities(tmp_path: Path) -> None:
    payload = {
        "entities": [
            {
                "entity_id": "romanov_1",
                "canonical_name": "Николай II",
                "gender": "male",
                "birth_year": 1868,
                "confidence": 0.9,
            },
            {
                "entity_id": "romanov_2",
                "canonical_name": "Император Николай II",
                "gender": "male",
                "birth_year": 1868,
                "confidence": 0.6,
            },
            {
                "entity_id": "child_1",
                "canonical_name": "Алексей Николаевич",
                "gender": "male",
                "birth_year": 1904,
            },
        ],
        "relations": [
            {
                "source_entity_id": "romanov_1",
                "target_entity_id": "child_1",
                "relation_type": "parent_child",
                "confidence": 0.91,
            },
            {
                "source_entity_id": "romanov_2",
                "target_entity_id": "child_1",
                "relation_type": "father_of",
                "confidence": 0.92,
            },
        ],
    }
    fixture_path = tmp_path / "living_merge_entities.json"
    fixture_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_dir = tmp_path / "living_merge_entities"
    code = main(
        [
            "genealogy",
            "build",
            "--input",
            str(fixture_path),
            "--output",
            str(output_dir),
            "--graph-mode",
            "living",
        ]
    )
    assert code == 0

    entities = json.loads((output_dir / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((output_dir / "relations.json").read_text(encoding="utf-8"))
    conflicts = json.loads((output_dir / "conflicts.json").read_text(encoding="utf-8"))

    entity_ids = {row["entity_id"] for row in entities}
    assert "romanov_2" not in entity_ids
    assert {"romanov_1", "child_1"} == entity_ids

    father_relations = [
        row for row in relations if row.get("relation_type") == "father_of"
    ]
    assert len(father_relations) == 1
    assert father_relations[0]["source_entity_id"] == "romanov_1"
    assert father_relations[0]["target_entity_id"] == "child_1"

    merge_groups = conflicts["entity_merge_groups"]
    assert len(merge_groups) == 1
    assert merge_groups[0]["representative_entity_id"] == "romanov_1"
    assert sorted(merge_groups[0]["merged_entity_ids"]) == ["romanov_1", "romanov_2"]
