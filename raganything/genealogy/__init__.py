"""Genealogy expansion helpers for building a canonical family tree.

This package is intentionally lightweight:
- RAGAnything/LightRAG remains the evidence retrieval layer.
- A canonical store (e.g., Neo4j) holds the tree, claims, and provenance.
"""

from .models import (
    Claim,
    Evidence,
    FamilyRecord,
    MediaSpec,
    PersonRecord,
    PersonSpec,
    Task,
)
from .pipeline import GenealogyPipeline, GenealogyPipelineConfig
from .stores import InMemoryGenealogyStore, Neo4jGenealogyStore
from .extractors import ClaimExtractor, MockClaimExtractor

__all__ = [
    "Claim",
    "Evidence",
    "FamilyRecord",
    "MediaSpec",
    "PersonRecord",
    "PersonSpec",
    "Task",
    "GenealogyPipeline",
    "GenealogyPipelineConfig",
    "InMemoryGenealogyStore",
    "Neo4jGenealogyStore",
    "ClaimExtractor",
    "MockClaimExtractor",
]
