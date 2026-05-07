from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersonNameParts:
    surname: str | None = None
    given_name: str | None = None
    patronymic: str | None = None
    suffix: str | None = None
    notes: str | None = None


_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "`": "'",
    }
)
_LEADING_PUNCTUATION = set(" \t\r\n,;:!?()[]{}<>\"'")
_TRAILING_PUNCTUATION = set(" \t\r\n,;:!?()[]{}<>\"'")
_INITIAL_RE = re.compile(r"^[A-Za-zА-ЯЁа-яё]\.?$")
_INITIAL_WITH_PERIOD_RE = re.compile(r"(?:^|\s)[A-Za-zА-ЯЁ]\.$")
_WORD_BOUNDED_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])\s*-\s*(?=[^\W\d_])")
_TRAILING_NOTE_RE = re.compile(r"^(?P<name>.+?)\s+\((?P<notes>[^()]*)\)$")
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
_PATRONYMIC_SUFFIXES = (
    "ович",
    "евич",
    "ич",
    "овна",
    "евна",
    "ична",
    "инична",
)
_GIVEN_NAME_CASE_MAP = {
    "алексей": "Алексей",
    "анна": "Анна",
    "анна-мария": "Анна-Мария",
    "иван": "Иван",
    "мария": "Мария",
    "михаил": "Михаил",
    "михаила": "Михаил",
}
_PATRONYMIC_CASE_MAP = {
    "петрович": "Петрович",
    "петровича": "Петрович",
    "федорович": "Федорович",
    "федоровича": "Федорович",
    "ивановна": "Ивановна",
}


def normalize_name_text(value: str) -> str:
    text = str(value or "").translate(_DASH_TRANSLATION).translate(_QUOTE_TRANSLATION)
    text = _WORD_BOUNDED_HYPHEN_RE.sub("-", text)
    text = " ".join(text.split())

    while text and text[0] in _LEADING_PUNCTUATION:
        text = text[1:].lstrip()
    while text and text[-1] in _TRAILING_PUNCTUATION:
        if text[-1] == ")" and "(" in text[:-1] and not text.startswith("("):
            break
        text = text[:-1].rstrip()
    while text.endswith(".") and not _INITIAL_WITH_PERIOD_RE.search(text):
        text = text[:-1].rstrip()
    return text


def split_person_name(value: str) -> PersonNameParts:
    cleaned = normalize_name_text(value)
    if not cleaned:
        return PersonNameParts()

    notes = None
    note_match = _TRAILING_NOTE_RE.match(cleaned)
    if note_match:
        cleaned = normalize_name_text(note_match.group("name"))
        notes = normalize_name_text(note_match.group("notes")) or None

    tokens = cleaned.split()
    suffix = None
    if tokens and tokens[-1].lower().rstrip(".") in _SUFFIXES:
        suffix = tokens.pop()
    if not tokens:
        return PersonNameParts(suffix=suffix, notes=notes)

    parsed = _split_tokens(tokens)
    return PersonNameParts(
        surname=parsed.get("surname"),
        given_name=parsed.get("given_name"),
        patronymic=parsed.get("patronymic"),
        suffix=suffix,
        notes=notes,
    )


def normalize_person_name(value: str) -> str:
    cleaned = normalize_name_text(value)
    parts = split_person_name(cleaned)
    if not any((parts.surname, parts.given_name, parts.patronymic, parts.suffix)):
        return cleaned
    if _contains_initials(parts):
        return cleaned

    ordered = [parts.given_name, parts.patronymic, parts.surname, parts.suffix]
    normalized = " ".join(part for part in ordered if part)
    return normalized or cleaned


def generate_name_variants(value: str) -> list[str]:
    cleaned = normalize_name_text(value)
    if not cleaned:
        return []

    parts = split_person_name(cleaned)
    variants: list[str] = [cleaned]

    given = parts.given_name
    patronymic = parts.patronymic
    surname = parts.surname
    suffix = parts.suffix

    _append_variant(variants, given, patronymic, surname, suffix)
    _append_variant(variants, surname, given, patronymic, suffix)

    if given and surname:
        _append_variant(variants, given, surname, suffix)
        _append_variant(variants, surname, given, suffix)

    given_initial = _initial_for(given)
    patronymic_initial = _initial_for(patronymic)
    if surname and given_initial and patronymic_initial:
        _append_variant(variants, given_initial, patronymic_initial, surname, suffix)
        _append_variant(variants, surname, given_initial, patronymic_initial, suffix)
    elif surname and given_initial:
        _append_variant(variants, given_initial, surname, suffix)
        _append_variant(variants, surname, given_initial, suffix)
    elif given_initial and patronymic_initial:
        _append_variant(variants, given_initial, patronymic_initial, suffix)

    return _dedupe(variants)


def _split_tokens(tokens: list[str]) -> dict[str, str | None]:
    if len(tokens) == 1:
        token = _normalize_given_name(tokens[0])
        if _is_known_given(tokens[0]):
            return {"given_name": token}
        return {}

    if len(tokens) == 2:
        first, second = tokens
        if _is_known_given(first) and _is_patronymic(second):
            return {
                "given_name": _normalize_given_name(first),
                "patronymic": _normalize_patronymic(second),
            }
        if _is_known_given(first) or _is_initial(first):
            return {
                "given_name": _normalize_given_name(first),
                "surname": _normalize_name_token(second),
            }
        if _is_known_given(second) or _is_initial(second):
            return {
                "surname": _normalize_name_token(first),
                "given_name": _normalize_given_name(second),
            }
        return {}

    first, second, third = tokens[:3]
    if (_is_known_given(first) or _is_initial(first)) and (
        _is_patronymic(second) or _is_initial(second)
    ):
        return {
            "given_name": _normalize_given_name(first),
            "patronymic": _normalize_patronymic(second),
            "surname": _normalize_name_token(third),
        }
    if (_is_known_given(second) or _is_initial(second)) and (
        _is_patronymic(third) or _is_initial(third)
    ):
        return {
            "surname": _normalize_name_token(first),
            "given_name": _normalize_given_name(second),
            "patronymic": _normalize_patronymic(third),
        }
    return {}


def _is_initial(value: str | None) -> bool:
    return bool(value and _INITIAL_RE.fullmatch(value))


def _normalize_initial(value: str) -> str:
    return f"{value[0].upper()}."


def _case_key(value: str) -> str:
    return value.strip().lower().replace("ё", "е")


def _is_known_given(value: str | None) -> bool:
    return bool(value and _case_key(value) in _GIVEN_NAME_CASE_MAP)


def _is_patronymic(value: str | None) -> bool:
    if not value:
        return False
    key = _case_key(value)
    return key in _PATRONYMIC_CASE_MAP or key.endswith(_PATRONYMIC_SUFFIXES)


def _normalize_given_name(value: str) -> str:
    if _is_initial(value):
        return _normalize_initial(value)
    return _GIVEN_NAME_CASE_MAP.get(_case_key(value), value)


def _normalize_patronymic(value: str | None) -> str | None:
    if value is None:
        return None
    if _is_initial(value):
        return _normalize_initial(value)
    return _PATRONYMIC_CASE_MAP.get(_case_key(value), value)


def _normalize_name_token(value: str) -> str:
    if _is_initial(value):
        return _normalize_initial(value)
    return value


def _contains_initials(parts: PersonNameParts) -> bool:
    return _is_initial(parts.given_name) or _is_initial(parts.patronymic)


def _initial_for(value: str | None) -> str | None:
    if not value:
        return None
    if _is_initial(value):
        return _normalize_initial(value)
    pieces = [piece for piece in value.split("-") if piece]
    if not pieces:
        return None
    return "-".join(f"{piece[0].upper()}." for piece in pieces)


def _append_variant(variants: list[str], *parts: str | None) -> None:
    text = " ".join(part for part in parts if part)
    text = normalize_name_text(text)
    if text:
        variants.append(text)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "PersonNameParts",
    "generate_name_variants",
    "normalize_name_text",
    "normalize_person_name",
    "split_person_name",
]
