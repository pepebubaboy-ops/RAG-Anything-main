from __future__ import annotations

import os


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_offline_mode() -> bool:
    return _is_truthy(os.getenv("GENEALOGY_RAG_OFFLINE", "0")) or _is_truthy(
        os.getenv("RAGANYTHING_OFFLINE", "0")
    )


def configure_offline_environment() -> None:
    """Configure common offline flags for libraries that may download artifacts."""
    if not is_offline_mode():
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_offline_allowed(action: str, hint: str | None = None) -> None:
    if not is_offline_mode():
        return

    message = (
        f"Offline mode blocks action: {action}. "
        "Provide local artifacts or disable offline mode."
    )
    if hint:
        message = f"{message} {hint}"
    raise RuntimeError(message)
