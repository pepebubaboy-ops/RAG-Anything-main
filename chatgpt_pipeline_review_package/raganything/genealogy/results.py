from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class BuildResult:
    output_dir: Path
    people_count: int = 0
    families_count: int = 0
    entities_count: int = 0
    relations_count: int = 0
    claims_count: int = 0
    reconciliation_stats: Dict[str, int] | None = None
    message_suffix: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
