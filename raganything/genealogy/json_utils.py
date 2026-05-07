from __future__ import annotations

import json
import re
from typing import Any


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_tags(text: str) -> str:
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    return text


def _balanced_slices(text: str, open_char: str, close_char: str) -> list[str]:
    candidates: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == open_char:
            if depth == 0:
                start = index
            depth += 1
        elif char == close_char and depth:
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(text[start : index + 1])
                start = -1

    return candidates


def extract_json_candidates(text: str) -> list[str]:
    """Return plausible JSON candidates from model output."""
    if not text:
        return []

    cleaned = _strip_think_tags(str(text))
    candidates: list[str] = []
    candidates.extend(match.strip() for match in _FENCED_JSON_RE.findall(cleaned))
    candidates.extend(_balanced_slices(cleaned, "{", "}"))
    candidates.extend(_balanced_slices(cleaned, "[", "]"))
    candidates.append(cleaned.strip())

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _basic_cleanup(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text.strip())


def robust_json_loads(text: str) -> Any | None:
    """Best-effort JSON parse for local LLM outputs.

    The optional json-repair package is used when installed through the llm/dev
    extras. Without it, strict JSON and simple trailing-comma cleanup still work.
    """
    for candidate in extract_json_candidates(text):
        for attempt in (candidate, _basic_cleanup(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass

        try:
            from json_repair import repair_json

            repaired = repair_json(candidate)
            if isinstance(repaired, str):
                return json.loads(repaired)
            return repaired
        except Exception:
            continue

    return None
