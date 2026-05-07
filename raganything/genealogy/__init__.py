"""Local-first genealogy extraction helpers.

This package contains source chunk handling, claim extraction, evidence validation,
graph construction, retrieval context helpers, and export utilities for private
family-history books.
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
from .build import build_genealogy_tree
from .export import export_genealogy
from .living_graph import build_living_graph
from .llm_claim_extraction import (
    LLMCandidateChunk,
    LLMRawExtraction,
    ValidatedClaimRow,
    find_candidate_chunks,
    robust_json_loads,
    run_llm_claim_pipeline,
    validate_llm_extractions,
)
from .mentions import MentionRecord, extract_mentions_from_text, write_mentions
from .name_normalization import (
    PersonNameParts,
    generate_name_variants,
    normalize_name_text,
    normalize_person_name,
    split_person_name,
)
from .knowledge_graph import (
    KnowledgeGraphArtifact,
    build_knowledge_graph_artifact,
    write_knowledge_graph_artifacts,
)
from .rag_index import (
    GenealogyRAGDocument,
    SourceChunk,
    SourceDocument,
    build_rag_documents_from_artifacts,
    write_rag_documents,
)
from .query_resolution import (
    ResolvedGenealogyQuery,
    detect_genealogy_intent,
    resolve_genealogy_query,
    resolve_query_person,
)
from .retrieval import (
    RetrievedGenealogyContext,
    build_genealogy_answer_prompt,
    load_rag_documents,
    retrieve_genealogy_context,
)
from .resolution import resolve_mentions_to_people, write_person_resolution
from .results import BuildResult
from .stores import InMemoryGenealogyStore, Neo4jGenealogyStore
from .extractors import ClaimExtractor, MockClaimExtractor

__all__ = [
    "BuildResult",
    "Claim",
    "Evidence",
    "FamilyRecord",
    "MediaSpec",
    "PersonRecord",
    "PersonSpec",
    "PersonNameParts",
    "Task",
    "GenealogyRAGDocument",
    "GenealogyPipeline",
    "GenealogyPipelineConfig",
    "KnowledgeGraphArtifact",
    "LLMCandidateChunk",
    "LLMRawExtraction",
    "MentionRecord",
    "ResolvedGenealogyQuery",
    "ValidatedClaimRow",
    "InMemoryGenealogyStore",
    "Neo4jGenealogyStore",
    "SourceChunk",
    "SourceDocument",
    "RetrievedGenealogyContext",
    "ClaimExtractor",
    "MockClaimExtractor",
    "build_genealogy_answer_prompt",
    "build_genealogy_tree",
    "build_knowledge_graph_artifact",
    "build_living_graph",
    "build_rag_documents_from_artifacts",
    "detect_genealogy_intent",
    "extract_mentions_from_text",
    "export_genealogy",
    "find_candidate_chunks",
    "generate_name_variants",
    "load_rag_documents",
    "normalize_name_text",
    "normalize_person_name",
    "robust_json_loads",
    "resolve_genealogy_query",
    "retrieve_genealogy_context",
    "resolve_query_person",
    "resolve_mentions_to_people",
    "run_llm_claim_pipeline",
    "split_person_name",
    "validate_llm_extractions",
    "write_knowledge_graph_artifacts",
    "write_mentions",
    "write_person_resolution",
    "write_rag_documents",
]
