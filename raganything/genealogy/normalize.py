from __future__ import annotations

from typing import Optional


def normalize_name(name: str) -> str:
    """Normalize person names for coarse matching.

    Keeps Unicode letters/digits, drops punctuation, lowercases, and collapses spaces.
    """
    if not name:
        return ""
    cleaned = []
    for ch in str(name).strip():
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch.lower())
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def normalize_place(place: Optional[str]) -> Optional[str]:
    if place is None:
        return None
    p = str(place).strip()
    return p or None


def coerce_year(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        s = str(value).strip()
        if not s:
            return None
        # Extract leading year-ish token.
        token = ""
        for ch in s:
            if ch.isdigit():
                token += ch
            elif token:
                break
        if len(token) >= 4:
            return int(token[:4])
        return int(token) if token else None
    except Exception:
        return None

