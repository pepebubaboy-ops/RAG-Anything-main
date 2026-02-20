from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    text = _THINK_TAG_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    return text


def extract_json_candidates(text: str) -> List[str]:
    """Return a list of plausible JSON object candidates from a blob of text."""
    if not text:
        return []
    cleaned = _strip_think_tags(text)
    candidates: List[str] = []

    # Code blocks first.
    candidates.extend(
        re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    )

    # Balanced braces scan.
    brace = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if brace == 0:
                start = i
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0 and start != -1:
                candidates.append(cleaned[start : i + 1])
                start = -1

    # Simple greedy fallback.
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _basic_cleanup(s: str) -> str:
    s = s.strip()
    # Remove trailing commas.
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def robust_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse for model outputs that may include extra text."""
    for cand in extract_json_candidates(text):
        for attempt in (cand, _basic_cleanup(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None

