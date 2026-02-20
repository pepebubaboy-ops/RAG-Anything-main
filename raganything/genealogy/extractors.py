from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .models import Claim


class ClaimExtractor(ABC):
    """Extract structured claims for a task.

    Production extractors typically:
    - query RAG (LightRAG) for relevant context
    - run an LLM pass to output strict JSON claims
    """

    @abstractmethod
    async def extract(self, task_type: str, subject: Dict[str, Any]) -> List[Claim]:
        raise NotImplementedError


@dataclass
class MockClaimExtractor(ClaimExtractor):
    """Deterministic extractor for tests and offline smoke runs."""

    responses: Mapping[str, List[Claim]]

    def _key(self, task_type: str, subject: Dict[str, Any]) -> str:
        # Prefer explicit key if provided.
        if "task_key" in subject:
            return str(subject["task_key"])
        sid = subject.get("subject_id") or subject.get("person_id") or subject.get("family_id")
        return f"{task_type}:{sid}"

    async def extract(self, task_type: str, subject: Dict[str, Any]) -> List[Claim]:
        return list(self.responses.get(self._key(task_type, subject), []))

