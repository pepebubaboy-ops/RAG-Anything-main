import pytest

from raganything.genealogy.extractors import ClaimExtractor
from raganything.genealogy.models import Claim, Evidence
from raganything.genealogy.pipeline import GenealogyPipeline, GenealogyPipelineConfig
from raganything.genealogy.stores import InMemoryGenealogyStore
from raganything.genealogy.models import PersonSpec


class ToyExtractor(ClaimExtractor):
    async def extract(self, task_type: str, subject: dict) -> list[Claim]:
        # Seed: Alice -> parents Bob+Carol
        if task_type == "find_parents" and subject["person"].name == "Alice Doe":
            return [
                Claim(
                    claim_type="parent_child",
                    confidence=0.9,
                    data={
                        "parents": [{"name": "Bob Doe"}, {"name": "Carol Smith"}],
                        "child": {"name": "Alice Doe", "birth_year": 1980},
                    },
                    evidence=[
                        Evidence(
                            file_path="toy.txt",
                            quote="Alice, daughter of Bob and Carol",
                        )
                    ],
                )
            ]

        # Bob+Carol -> children: David (Alice already known)
        if task_type == "find_children":
            parents = [p.name for p in (subject.get("parents") or [])]
            if set(parents) == {"Bob Doe", "Carol Smith"}:
                return [
                    Claim(
                        claim_type="parent_child",
                        confidence=0.8,
                        data={
                            "parents": [{"name": "Bob Doe"}, {"name": "Carol Smith"}],
                            "child": {"name": "David Doe", "birth_year": 1982},
                        },
                        evidence=[
                            Evidence(
                                file_path="toy.txt",
                                quote="Bob and Carol had a son David",
                            )
                        ],
                    )
                ]

            if set(parents) == {"David Doe", "Emma Roe"}:
                return [
                    Claim(
                        claim_type="parent_child",
                        confidence=0.85,
                        data={
                            "parents": [{"name": "David Doe"}, {"name": "Emma Roe"}],
                            "child": {"name": "Frank Doe", "birth_year": 2010},
                        },
                        evidence=[
                            Evidence(
                                file_path="toy.txt",
                                quote="Frank, child of David and Emma",
                            )
                        ],
                    )
                ]

        # David -> spouse Emma
        if task_type == "find_spouses" and subject["person"].name == "David Doe":
            return [
                Claim(
                    claim_type="spouse",
                    confidence=0.7,
                    data={
                        "person1": {"name": "David Doe"},
                        "person2": {"name": "Emma Roe"},
                    },
                    evidence=[
                        Evidence(file_path="toy.txt", quote="David married Emma")
                    ],
                )
            ]

        # Profile enrichment (occupation + alias + photo reference)
        if task_type == "find_profile" and subject["person"].name == "David Doe":
            return [
                Claim(
                    claim_type="person_profile",
                    confidence=0.6,
                    data={
                        "person_id": subject.get("person_id"),
                        "attributes": {
                            "occupation": "Engineer",
                            "aliases": ["Dave Doe"],
                            "media": [
                                {
                                    "kind": "photo",
                                    "path": "/tmp/david_doe_portrait.jpg",
                                    "caption": "Portrait photo",
                                }
                            ],
                        },
                    },
                    evidence=[
                        Evidence(
                            file_path="toy.txt",
                            quote="David Doe (also known as Dave Doe) worked as an engineer.",
                            image_path="/tmp/david_doe_portrait.jpg",
                        )
                    ],
                )
            ]

        return []


@pytest.mark.asyncio
async def test_genealogy_pipeline_expands_graph():
    store = InMemoryGenealogyStore()
    pipeline = GenealogyPipeline(
        store=store,
        extractor=ToyExtractor(),
        config=GenealogyPipelineConfig(
            max_depth=4,
            max_tasks=50,
            enable_spouse_search=True,
            enable_descendant_expansion=True,
        ),
    )

    seed = pipeline.seed_person(
        PersonSpec(name="Alice Doe", birth_year=1980, birth_place="Springfield")
    )
    stats = await pipeline.expand(seed.person_id)

    assert (
        stats["people"] >= 5
    )  # Alice, Bob, Carol, David, Emma, Frank (Emma may be created)

    names = sorted({p.spec.name for p in store.people.values()})
    assert "Alice Doe" in names
    assert "Bob Doe" in names
    assert "Carol Smith" in names
    assert "David Doe" in names
    assert "Emma Roe" in names
    assert "Frank Doe" in names

    # At least two families: Bob+Carol and David+Emma
    assert stats["families"] >= 2

    # Ensure claim/evidence were recorded in memory store.
    assert store.claims
    assert store.evidences

    # Ensure profile enrichment applied to David.
    david = next(p for p in store.people.values() if p.spec.name == "David Doe")
    assert david.spec.occupation == "Engineer"
    assert "Dave Doe" in (david.spec.aliases or [])

    # Media linked.
    assert store.media
    assert any(pid == david.person_id for (pid, mid) in store.rel_person_media)
