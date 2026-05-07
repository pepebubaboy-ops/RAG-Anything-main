from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.2.9"
__author__ = "Zirui Guo"
__url__ = "https://github.com/HKUDS/RAG-Anything"

__all__ = ["RAGAnything", "RAGAnythingConfig"]


def __getattr__(name: str) -> Any:
    if name == "RAGAnything":
        try:
            module = import_module(".raganything", __name__)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "")
            if missing in {
                "lightrag",
                "huggingface_hub",
                "dotenv",
                "tqdm",
                "mineru",
            }:
                raise RuntimeError(
                    "RAGAnything requires optional dependencies. "
                    "Install extras: pip install 'raganything[rag]' and optionally "
                    "'raganything[mineru]' / 'raganything[dotenv]'."
                ) from exc
            raise
        return module.RAGAnything

    if name == "RAGAnythingConfig":
        module = import_module(".config", __name__)
        return module.RAGAnythingConfig

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
