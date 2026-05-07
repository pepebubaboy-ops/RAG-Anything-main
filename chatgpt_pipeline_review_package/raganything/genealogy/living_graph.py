from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .export import escape_dot as _escape_dot
from .export import try_generate_svg_from_dot as _try_generate_svg_from_dot
from .results import BuildResult


_PARENT_RELATION_TYPES = {"parent_child", "parent_of", "father_of", "mother_of"}
_WEAK_PARENT_CANDIDATE_TYPES = {"relative", "associated_with", "knows"}
_PARENT_PROMOTION_MIN_CONFIDENCE = 0.75
_PARENT_HINT_TERMS = (
    " сын ",
    " сына ",
    " сыном ",
    " сыновей ",
    " дочь ",
    " дочери ",
    " дочерью ",
    " детей ",
    " ребенок ",
    " ребёнок ",
    " child ",
    " children ",
    " son ",
    " daughter ",
)
_PARENT_EXPLICIT_CUE_TERMS = (
    " son of ",
    " daughter of ",
    " child of ",
    " сын ",
    " сына ",
    " дочь ",
    " дочери ",
)
_PARENT_DIRECTION_HINT_TERMS = (
    " сын ",
    " сына ",
    " дочь ",
    " дочери ",
    " child of ",
    " son of ",
    " daughter of ",
)
_PARENT_FORWARD_HINT_TERMS = (
    " родил ",
    " родила ",
    " родились ",
    " имел ",
    " имела ",
    " had ",
    " children ",
    " child ",
)
_PARENT_TIMELINE_MIN_AGE_GAP = 12
_PARENT_TIMELINE_MAX_AGE_GAP = 80
_SPOUSE_HINT_TERMS = (
    " wife ",
    " husband ",
    " spouse ",
    " married ",
    " супруга ",
    " супруг ",
    " муж ",
    " жена ",
    " брак ",
)
_COLLATERAL_HINT_TERMS = (
    " brother ",
    " sister ",
    " uncle ",
    " aunt ",
    " cousin ",
    " nephew ",
    " niece ",
    " брат ",
    " сестра ",
    " дядя ",
    " тётя ",
    " тетя ",
    " кузен ",
    " кузина ",
    " племянник ",
    " племянница ",
)
_AVUNCULAR_HINT_SUBSTRINGS = (
    "племянник",
    "племянниц",
    " niece ",
    " nephew ",
)
_AVUNCULAR_RELATION_TYPES = {"uncle_of", "aunt_of", "aunt_uncle_of"}
_DIRECT_GENEALOGICAL_RELATION_TYPES = {
    "parent_child",
    "parent_of",
    "father_of",
    "mother_of",
    "child_of",
    "spouse",
    "sibling",
    "sibling_of",
    "brother_of",
    "sister_of",
    "grandparent_of",
    "grandfather_of",
    "grandmother_of",
    "grandchild_of",
    "grandson_of",
    "granddaughter_of",
    "aunt_uncle_of",
    "aunt_of",
    "uncle_of",
    "nibling_of",
    "nephew_of",
    "niece_of",
    "cousin_of",
}
_MERGE_ANCHOR_RELATION_TYPES = {
    "relative",
    "associated_with",
    "knows",
    "works_with",
    "colleague_of",
    "ally_of",
}
_MERGE_DECORATOR_TOKENS = {
    "царь",
    "царица",
    "царевич",
    "царевна",
    "император",
    "императрица",
    "король",
    "королева",
    "князь",
    "княгиня",
    "государь",
    "государыня",
    "великий",
    "великая",
    "the",
    "king",
    "queen",
    "emperor",
    "empress",
    "tsar",
}
_MERGE_KEEP_ROLES = {"father", "mother", "parent"}



def _looks_like_living_graph_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("entities"), list)
        and isinstance(obj.get("relations"), list)
    )


def _pick_relation_color(relation_type: str) -> str:
    palette = {
        "parent_child": "#2563eb",
        "parent_of": "#2563eb",
        "father_of": "#1d4ed8",
        "mother_of": "#9333ea",
        "spouse": "#db2777",
        "friend": "#16a34a",
        "enemy": "#b91c1c",
        "associated_with": "#7c3aed",
        "related_to": "#0f766e",
    }
    return palette.get((relation_type or "").lower(), "#475569")


def _normalize_relation_type(value: Any) -> str:
    relation_type = str(value or "").strip()
    return relation_type or "related_to"


def _parse_relation_types_arg(raw_value: str | None) -> set[str]:
    if not raw_value:
        return set()
    return {
        item.strip().lower()
        for item in str(raw_value).split(",")
        if item and item.strip()
    }


def _relation_type_class(relation_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", relation_type.lower()).strip("-")
    if not slug:
        slug = "related-to"
    return f"rel-type-{slug}"


def _normalize_gender(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None

    male_markers = {
        "m",
        "male",
        "man",
        "boy",
        "masculine",
        "masc",
        "м",
        "муж",
        "мужчина",
        "мужской",
    }
    female_markers = {
        "f",
        "female",
        "woman",
        "girl",
        "feminine",
        "fem",
        "ж",
        "жен",
        "женщина",
        "женский",
    }

    if raw in male_markers:
        return "male"
    if raw in female_markers:
        return "female"
    return None


def _normalize_for_cue_search(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return f" {text} "


def _coerce_year(value: Any) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    if year <= 0 or year > 2500:
        return None
    return year


def _extract_roman_numeral(name: str) -> str | None:
    matches = re.findall(r"\b[ivxlcdm]+\b", str(name or "").lower())
    if not matches:
        return None
    return matches[-1]


def _extract_patronymic(name: str) -> str | None:
    normalized = re.sub(r"[^\w\s]", " ", str(name or "").lower())
    non_patronymic_tokens = {
        "цесаревич",
        "царевич",
        "королевич",
        "княжич",
    }
    for token in normalized.split():
        if token in non_patronymic_tokens:
            continue
        if token.endswith(("ович", "евич", "овна", "евна", "ична", "оглы", "кызы")):
            return token
    return None


def _name_tokens_for_merge(name: str) -> List[str]:
    normalized = re.sub(r"[^\w\s]", " ", str(name or "").lower())
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return []

    cleaned: List[str] = []
    for token in tokens:
        if token in _MERGE_DECORATOR_TOKENS:
            continue
        cleaned.append(token)

    return cleaned or tokens


def _name_features_for_merge(name: str) -> Dict[str, Any]:
    tokens = _name_tokens_for_merge(name)
    roman = _extract_roman_numeral(" ".join(tokens))
    core_tokens = [
        token
        for token in tokens
        if not re.fullmatch(r"[ivxlcdm]+", token, flags=re.IGNORECASE)
    ]
    core = " ".join(core_tokens).strip()
    first_token = core_tokens[0] if core_tokens else (tokens[0] if tokens else "")
    return {
        "tokens": tokens,
        "core_tokens": core_tokens,
        "core": core,
        "token_count": len(core_tokens) if core_tokens else len(tokens),
        "first_token": first_token,
        "roman": roman,
    }


def _is_roman_token(token: str) -> bool:
    return bool(re.fullmatch(r"[ivxlcdm]+", token, flags=re.IGNORECASE))


def _first_person_name_token(name: str) -> str:
    for token in _name_tokens_for_merge(name):
        if _is_roman_token(token):
            continue
        return token
    return ""


def _patronymic_root(patronymic: str | None) -> str:
    value = str(patronymic or "").strip().lower()
    if not value:
        return ""
    suffixes = (
        "ович",
        "евич",
        "ич",
        "овна",
        "евна",
        "ична",
        "инична",
        "оглы",
        "кызы",
    )
    for suffix in suffixes:
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            value = value[: -len(suffix)]
            break
    return value.rstrip("аеиоуыэюяьй")


def _name_token_root(token: str) -> str:
    value = str(token or "").strip().lower()
    if not value:
        return ""
    return value.rstrip("аеиоуыэюяьй")


def _looks_like_patronymic_parent(
    parent_name: str,
    child_name: str,
) -> bool:
    child_patronymic = _extract_patronymic(child_name)
    patronymic_root = _patronymic_root(child_patronymic)
    if not patronymic_root:
        return False
    parent_first_name = _first_person_name_token(parent_name)
    parent_root = _name_token_root(parent_first_name)
    if not parent_root:
        return False
    if patronymic_root == parent_root:
        return True
    probe = parent_root[: max(3, min(6, len(parent_root)))]
    return patronymic_root.startswith(probe)


def _quote_parent_segment(quote_norm: str) -> str:
    if not quote_norm:
        return ""
    cue_terms = (" сын ", " дочь ", " child ", " son ", " daughter ")
    cue_pos = _first_term_position(quote_norm, cue_terms)
    if cue_pos is None:
        return ""
    segment = quote_norm[cue_pos:]
    stop_positions = [segment.find(marker) for marker in (",", ".", ";", "!", "?")]
    stop_positions = [pos for pos in stop_positions if pos >= 0]
    if stop_positions:
        segment = segment[: min(stop_positions)]
    return segment.strip()


def _quote_segment_tokens(segment: str) -> List[str]:
    if not segment:
        return []
    return re.findall(r"[a-zа-яё0-9]+", segment.lower())


def _token_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _segment_mentions_name_token(segment_tokens: Sequence[str], name_token: str) -> bool:
    if not segment_tokens or not name_token:
        return False
    normalized_name = _name_token_root(name_token)
    for token in segment_tokens:
        normalized_token = _name_token_root(token)
        if not normalized_token:
            continue
        if normalized_token == normalized_name:
            return True
        if _token_similarity(normalized_token, normalized_name) >= 0.78:
            return True
    return False


def _relation_parent_anchor_score(
    relation: Dict[str, Any],
    *,
    source_name: str,
    target_name: str,
) -> int:
    quote_norm = _normalize_for_cue_search(_extract_relation_quote(relation))
    if not quote_norm:
        return 0
    if not any(term in quote_norm for term in _PARENT_EXPLICIT_CUE_TERMS):
        return 0
    segment = _quote_parent_segment(quote_norm)
    tokens = _quote_segment_tokens(segment)
    if not tokens:
        return -1
    source_token = _first_person_name_token(source_name)
    target_token = _first_person_name_token(target_name)
    source_match = _segment_mentions_name_token(tokens, source_token)
    target_match = _segment_mentions_name_token(tokens, target_token)
    if source_match and not target_match:
        return 2
    if target_match and not source_match:
        return -2
    if source_match:
        return 1
    return -1


def _types_compatible_for_merge(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_type = str(left.get("entity_type") or "").strip().lower()
    right_type = str(right.get("entity_type") or "").strip().lower()
    if not left_type or not right_type:
        return True
    if left_type == right_type:
        return True
    return {left_type, right_type} == {"human", "other_living"}


def _genders_compatible_for_merge(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_gender = _normalize_gender(left.get("gender") or left.get("sex"))
    right_gender = _normalize_gender(right.get("gender") or right.get("sex"))
    if not left_gender or not right_gender:
        return True
    return left_gender == right_gender


def _years_compatible_for_merge(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_birth = _coerce_year(left.get("birth_year"))
    right_birth = _coerce_year(right.get("birth_year"))
    if left_birth is not None and right_birth is not None and abs(left_birth - right_birth) > 15:
        return False

    left_death = _coerce_year(left.get("death_year"))
    right_death = _coerce_year(right.get("death_year"))
    if left_death is not None and right_death is not None and abs(left_death - right_death) > 15:
        return False
    return True


def _build_entity_pair_link_index(
    relations: Sequence[Dict[str, Any]],
) -> Dict[tuple[str, str], Dict[str, Any]]:
    index: Dict[tuple[str, str], Dict[str, Any]] = {}
    for relation in relations:
        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if not source_id or not target_id or source_id == target_id:
            continue
        key = tuple(sorted((source_id, target_id)))
        row = index.setdefault(
            key,
            {"max_confidence": 0.0, "support_count": 0, "relation_types": set()},
        )
        row["max_confidence"] = max(
            float(row["max_confidence"]),
            _safe_float(relation.get("confidence"), 0.0),
        )
        row["support_count"] = int(row["support_count"]) + max(
            1,
            int(relation.get("support_count") or 1),
        )
        row["relation_types"].add(
            _normalize_relation_type(relation.get("relation_type")).lower()
        )
    return index


def _build_entity_neighbor_index(
    relations: Sequence[Dict[str, Any]],
) -> Dict[str, set[str]]:
    neighbors: Dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if not source_id or not target_id or source_id == target_id:
            continue
        neighbors[source_id].add(target_id)
        neighbors[target_id].add(source_id)
    return neighbors


def _pair_has_merge_anchor(pair_link: Dict[str, Any] | None) -> bool:
    if not pair_link:
        return False
    relation_types = set(pair_link.get("relation_types") or set())
    if relation_types & _DIRECT_GENEALOGICAL_RELATION_TYPES:
        return False
    if (
        pair_link.get("max_confidence", 0.0) >= 0.92
        and relation_types & _MERGE_ANCHOR_RELATION_TYPES
    ):
        return True
    if (
        pair_link.get("support_count", 0) >= 2
        and relation_types & _MERGE_ANCHOR_RELATION_TYPES
    ):
        return True
    return False


def _entities_should_merge(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    pair_links: Dict[tuple[str, str], Dict[str, Any]],
    neighbor_index: Dict[str, set[str]],
) -> bool:
    if not _types_compatible_for_merge(left, right):
        return False
    if not _genders_compatible_for_merge(left, right):
        return False
    if not _years_compatible_for_merge(left, right):
        return False

    left_name = str(left.get("canonical_name") or "")
    right_name = str(right.get("canonical_name") or "")
    left_features = _name_features_for_merge(left_name)
    right_features = _name_features_for_merge(right_name)

    if not left_features["core"] or not right_features["core"]:
        return False

    left_roman = left_features["roman"]
    right_roman = right_features["roman"]
    if left_roman and right_roman and left_roman != right_roman:
        return False

    left_patronymic = _extract_patronymic(left_name)
    right_patronymic = _extract_patronymic(right_name)
    if left_patronymic and right_patronymic and left_patronymic != right_patronymic:
        return False

    if left_features["core"] == right_features["core"]:
        left_birth = _coerce_year(left.get("birth_year"))
        right_birth = _coerce_year(right.get("birth_year"))
        birth_anchor = (
            left_birth is not None
            and right_birth is not None
            and abs(left_birth - right_birth) <= 8
        )
        if bool(left_roman) != bool(right_roman) and not birth_anchor:
            return False
        if max(left_features["token_count"], right_features["token_count"]) <= 1:
            return birth_anchor
        return True

    left_id = str(left.get("entity_id") or "")
    right_id = str(right.get("entity_id") or "")
    pair_link = pair_links.get(tuple(sorted((left_id, right_id))))
    pair_relation_types = set(pair_link.get("relation_types") or set()) if pair_link else set()
    if pair_relation_types & _DIRECT_GENEALOGICAL_RELATION_TYPES:
        return False
    merge_anchor = _pair_has_merge_anchor(pair_link)
    left_neighbors = neighbor_index.get(left_id, set())
    right_neighbors = neighbor_index.get(right_id, set())
    has_neighbor_overlap = bool(left_neighbors & right_neighbors)

    if left_features["first_token"] and left_features["first_token"] == right_features["first_token"]:
        left_birth = _coerce_year(left.get("birth_year"))
        right_birth = _coerce_year(right.get("birth_year"))
        year_anchor = (
            left_birth is not None
            and right_birth is not None
            and abs(left_birth - right_birth) <= 12
        )
        same_roman = bool(left_roman and right_roman and left_roman == right_roman)
        if same_roman:
            left_tokens = left_features["tokens"]
            right_tokens = right_features["tokens"]
            shorter, longer = (
                (left_tokens, right_tokens)
                if len(left_tokens) <= len(right_tokens)
                else (right_tokens, left_tokens)
            )
            if len(shorter) >= 2 and shorter == longer[: len(shorter)]:
                return True
        if max(left_features["token_count"], right_features["token_count"]) <= 1:
            return year_anchor
        if min(left_features["token_count"], right_features["token_count"]) <= 1:
            if year_anchor:
                return True
            if same_roman and merge_anchor:
                return True
            return False

    if (
        left_features["token_count"] <= 1
        and right_features["token_count"] <= 1
        and left_features["first_token"]
        and right_features["first_token"]
    ):
        name_similarity = max(
            _token_similarity(
                _name_token_root(left_features["first_token"]),
                _name_token_root(right_features["first_token"]),
            ),
            _token_similarity(
                left_features["first_token"],
                right_features["first_token"],
            ),
        )
        if name_similarity >= 0.9 and (merge_anchor or has_neighbor_overlap):
            return True

    left_tokens = left_features["core_tokens"]
    right_tokens = right_features["core_tokens"]
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        shorter, longer = (
            (left_tokens, right_tokens)
            if len(left_tokens) <= len(right_tokens)
            else (right_tokens, left_tokens)
        )
        if shorter == longer[: len(shorter)]:
            return True
    return False


def _entity_rank_for_merge(entity: Dict[str, Any]) -> tuple[int, int, int, float, int]:
    features = _name_features_for_merge(str(entity.get("canonical_name") or ""))
    known_years = int(_coerce_year(entity.get("birth_year")) is not None) + int(
        _coerce_year(entity.get("death_year")) is not None
    )
    alias_count = len(entity.get("aliases") or [])
    confidence = _safe_float(entity.get("confidence"), 0.0)
    return (
        known_years,
        int(features["roman"] is not None),
        int(features["token_count"]),
        confidence,
        alias_count,
    )


def _merge_entity_records(
    representative: Dict[str, Any],
    entity: Dict[str, Any],
) -> None:
    aliases: set[str] = {
        str(alias).strip()
        for alias in representative.get("aliases") or []
        if str(alias).strip()
    }
    aliases.update(
        str(alias).strip()
        for alias in entity.get("aliases") or []
        if str(alias).strip()
    )
    representative_name = str(representative.get("canonical_name") or "").strip()
    entity_name = str(entity.get("canonical_name") or "").strip()
    if representative_name and entity_name and representative_name != entity_name:
        aliases.add(entity_name)

    for key in ("birth_year", "death_year", "birth_place", "death_place", "gender", "species", "occupation", "description"):
        if representative.get(key) is None and entity.get(key) is not None:
            representative[key] = entity.get(key)

    representative["confidence"] = max(
        _safe_float(representative.get("confidence"), 0.0),
        _safe_float(entity.get("confidence"), 0.0),
    )
    representative["normalized_name"] = str(
        representative.get("normalized_name")
        or " ".join(_name_tokens_for_merge(representative_name))
    ).strip()
    representative["aliases"] = sorted(
        alias
        for alias in aliases
        if alias and alias != representative_name
    )


def _remap_and_dedup_relations_after_entity_merge(
    relations: Sequence[Dict[str, Any]],
    *,
    id_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    deduped: Dict[tuple[str, str, str, bool, bool], Dict[str, Any]] = {}

    for relation in relations:
        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if not source_id or not target_id:
            continue
        source_id = id_map.get(source_id, source_id)
        target_id = id_map.get(target_id, target_id)
        if source_id == target_id:
            continue

        relation_type = _normalize_relation_type(relation.get("relation_type"))
        directed = bool(relation.get("directed", True))
        symmetric = bool(relation.get("symmetric", False)) or not directed
        if symmetric and source_id > target_id:
            source_id, target_id = target_id, source_id
        key = (source_id, target_id, relation_type.lower(), directed, symmetric)

        confidence = _safe_float(relation.get("confidence"), 0.0)
        support_count = max(1, int(relation.get("support_count") or 1))

        row = deduped.get(key)
        if row is None:
            row = dict(relation)
            row["source_entity_id"] = source_id
            row["target_entity_id"] = target_id
            row["relation_type"] = relation_type
            row["confidence"] = confidence
            row["support_count"] = support_count
            row["directed"] = directed
            row["symmetric"] = symmetric
            row["relation_id"] = (
                f"rel-{hashlib.md5('|'.join(map(str, key)).encode('utf-8')).hexdigest()}"
            )
            deduped[key] = row
            continue

        row["confidence"] = max(_safe_float(row.get("confidence"), 0.0), confidence)
        row["support_count"] = int(row.get("support_count") or 0) + support_count

        existing_quote = _extract_relation_quote(row)
        incoming_quote = _extract_relation_quote(relation)
        if incoming_quote and incoming_quote not in existing_quote:
            evidence = row.get("evidence")
            if isinstance(evidence, list):
                evidence.append({"quote": incoming_quote})
            elif isinstance(evidence, dict):
                row["evidence"] = [evidence, {"quote": incoming_quote}]
            else:
                row["evidence"] = [{"quote": incoming_quote}]

    return [deduped[key] for key in sorted(deduped)]


def _merge_entities_for_living_graph(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    entity_by_id: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_by_id[entity_id] = dict(entity)

    entity_ids = sorted(entity_by_id)
    if len(entity_ids) < 2:
        return list(entity_by_id.values()), list(relations), []

    pair_links = _build_entity_pair_link_index(relations)
    neighbor_index = _build_entity_neighbor_index(relations)
    parent: Dict[str, str] = {entity_id: entity_id for entity_id in entity_ids}

    def find(entity_id: str) -> str:
        root = entity_id
        while parent[root] != root:
            root = parent[root]
        while parent[entity_id] != entity_id:
            next_id = parent[entity_id]
            parent[entity_id] = root
            entity_id = next_id
        return root

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left_index in range(len(entity_ids)):
        left_id = entity_ids[left_index]
        left_entity = entity_by_id[left_id]
        for right_index in range(left_index + 1, len(entity_ids)):
            right_id = entity_ids[right_index]
            right_entity = entity_by_id[right_id]
            if _entities_should_merge(
                left_entity,
                right_entity,
                pair_links=pair_links,
                neighbor_index=neighbor_index,
            ):
                union(left_id, right_id)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for entity_id in entity_ids:
        clusters[find(entity_id)].append(entity_id)

    id_map: Dict[str, str] = {}
    merged_entities: List[Dict[str, Any]] = []
    merge_groups: List[Dict[str, Any]] = []

    for cluster_ids in sorted(clusters.values(), key=lambda row: (len(row), row), reverse=True):
        ranked_ids = sorted(
            cluster_ids,
            key=lambda entity_id: _entity_rank_for_merge(entity_by_id[entity_id]),
            reverse=True,
        )
        representative_id = ranked_ids[0]
        representative = dict(entity_by_id[representative_id])
        for entity_id in ranked_ids[1:]:
            _merge_entity_records(representative, entity_by_id[entity_id])
        representative["entity_id"] = representative_id
        merged_entities.append(representative)

        for entity_id in cluster_ids:
            id_map[entity_id] = representative_id

        if len(cluster_ids) > 1:
            merge_groups.append(
                {
                    "representative_entity_id": representative_id,
                    "representative_name": representative.get("canonical_name"),
                    "merged_entity_ids": sorted(cluster_ids),
                    "merged_entity_names": [
                        entity_by_id[entity_id].get("canonical_name")
                        for entity_id in sorted(cluster_ids)
                    ],
                }
            )

    merged_entities.sort(key=lambda row: str(row.get("entity_id") or ""))
    merged_relations = _remap_and_dedup_relations_after_entity_merge(
        relations,
        id_map=id_map,
    )
    return merged_entities, merged_relations, merge_groups


def _extract_relation_quote(relation: Dict[str, Any]) -> str:
    quotes: List[str] = []
    evidence = relation.get("evidence")
    if isinstance(evidence, dict):
        quote = str(evidence.get("quote") or "").strip()
        if quote:
            quotes.append(quote)
    elif isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            quote = str(row.get("quote") or "").strip()
            if quote:
                quotes.append(quote)
    direct_quote = str(relation.get("quote") or "").strip()
    if direct_quote:
        quotes.append(direct_quote)
    return " ".join(quotes).strip()


def _name_search_forms(name: str) -> List[str]:
    normalized = re.sub(r"[^\w\s]", " ", str(name or "").lower())
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return []

    forms: List[str] = []
    forms.append(" ".join(tokens))
    if len(tokens) >= 2:
        forms.append(f"{tokens[0]} {tokens[1]}")
    if len(tokens[0]) >= 4:
        forms.append(tokens[0])
    if len(tokens) >= 2 and re.fullmatch(r"[ivxlcdm]+", tokens[1], re.IGNORECASE):
        forms.append(f"{tokens[0]} {tokens[1]}")

    seen: set[str] = set()
    unique_forms: List[str] = []
    for form in forms:
        if form and form not in seen:
            seen.add(form)
            unique_forms.append(form)
    return unique_forms


def _name_position_in_quote(name: str, quote_norm: str) -> int | None:
    if not quote_norm:
        return None

    best_pos: int | None = None
    for form in _name_search_forms(name):
        match = re.search(rf"(?<!\w){re.escape(form)}(?!\w)", quote_norm)
        if not match:
            continue
        pos = int(match.start())
        if best_pos is None or pos < best_pos:
            best_pos = pos
    return best_pos


def _first_term_position(text: str, terms: Sequence[str]) -> int | None:
    best_pos: int | None = None
    for term in terms:
        pos = text.find(term)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
    return best_pos


def _promote_parent_relations_from_evidence(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    entity_info: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_info[entity_id] = {
            "name": str(entity.get("canonical_name") or entity_id),
            "birth_year": _coerce_year(entity.get("birth_year")),
        }

    promoted_count = 0
    promoted_relations: List[Dict[str, Any]] = []
    for relation in relations:
        relation_type = _normalize_relation_type(relation.get("relation_type")).lower()
        if relation_type not in _WEAK_PARENT_CANDIDATE_TYPES:
            promoted_relations.append(relation)
            continue
        if _safe_float(relation.get("confidence"), 0.0) < _PARENT_PROMOTION_MIN_CONFIDENCE:
            promoted_relations.append(relation)
            continue

        quote = _extract_relation_quote(relation)
        quote_norm = _normalize_for_cue_search(quote)
        if not quote_norm:
            promoted_relations.append(relation)
            continue
        if any(term in quote_norm for term in _SPOUSE_HINT_TERMS):
            promoted_relations.append(relation)
            continue

        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if (
            not source_id
            or not target_id
            or source_id == target_id
            or source_id not in entity_info
            or target_id not in entity_info
        ):
            promoted_relations.append(relation)
            continue

        source_info = entity_info[source_id]
        target_info = entity_info[target_id]
        source_name = str(source_info.get("name") or "")
        target_name = str(target_info.get("name") or "")
        source_name_pos = _name_position_in_quote(source_name, quote_norm)
        target_name_pos = _name_position_in_quote(target_name, quote_norm)
        source_patronymic_match = _looks_like_patronymic_parent(source_name, target_name)
        target_patronymic_match = _looks_like_patronymic_parent(target_name, source_name)
        has_parent_hint = any(term in quote_norm for term in _PARENT_HINT_TERMS)
        if (
            not has_parent_hint
            and not source_patronymic_match
            and not target_patronymic_match
        ):
            promoted_relations.append(relation)
            continue

        has_explicit_parent_cue = any(
            term in quote_norm for term in _PARENT_EXPLICIT_CUE_TERMS
        )
        if not has_explicit_parent_cue:
            if source_patronymic_match and not target_patronymic_match:
                source_is_parent = True
            elif target_patronymic_match and not source_patronymic_match:
                source_is_parent = False
            else:
                promoted_relations.append(relation)
                continue
        else:
            has_collateral_terms = any(
                term in quote_norm for term in _COLLATERAL_HINT_TERMS
            )
            if (
                source_name_pos is None
                and target_name_pos is None
                and not source_patronymic_match
                and not target_patronymic_match
            ):
                promoted_relations.append(relation)
                continue
            parent_marker_pos = _first_term_position(
                quote_norm,
                _PARENT_DIRECTION_HINT_TERMS,
            )
            forward_marker_pos = _first_term_position(
                quote_norm,
                _PARENT_FORWARD_HINT_TERMS,
            )

            source_parent_votes = 0
            target_parent_votes = 0
            if parent_marker_pos is not None:
                if source_name_pos is not None and source_name_pos > parent_marker_pos:
                    source_parent_votes += 1
                if target_name_pos is not None and target_name_pos > parent_marker_pos:
                    target_parent_votes += 1
            if forward_marker_pos is not None:
                if source_name_pos is not None and source_name_pos < forward_marker_pos:
                    source_parent_votes += 1
                if target_name_pos is not None and target_name_pos < forward_marker_pos:
                    target_parent_votes += 1

            source_birth_year = source_info.get("birth_year")
            target_birth_year = target_info.get("birth_year")
            if source_birth_year is not None and target_birth_year is not None:
                if source_birth_year <= target_birth_year - _PARENT_TIMELINE_MIN_AGE_GAP:
                    source_parent_votes += 1
                elif target_birth_year <= source_birth_year - _PARENT_TIMELINE_MIN_AGE_GAP:
                    target_parent_votes += 1

            if source_patronymic_match:
                source_parent_votes += 1
            if target_patronymic_match:
                target_parent_votes += 1

            if source_parent_votes == 0 and target_parent_votes == 0:
                promoted_relations.append(relation)
                continue
            if source_parent_votes == target_parent_votes:
                promoted_relations.append(relation)
                continue
            if has_collateral_terms and abs(source_parent_votes - target_parent_votes) < 2:
                promoted_relations.append(relation)
                continue

            source_is_parent = source_parent_votes > target_parent_votes

        promoted_relation = dict(relation)
        if not source_is_parent:
            promoted_relation["source_entity_id"] = target_id
            promoted_relation["target_entity_id"] = source_id
        promoted_relation["relation_type"] = "parent_child"
        promoted_relation["directed"] = True
        promoted_relation["symmetric"] = False
        promoted_relation.setdefault("inference_rule", "evidence_parent_cue")
        promoted_relations.append(promoted_relation)
        promoted_count += 1

    return promoted_relations, promoted_count


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parent_role_key(relation_type: str) -> str | None:
    relation_key = _normalize_relation_type(relation_type).lower()
    if relation_key == "father_of":
        return "father"
    if relation_key == "mother_of":
        return "mother"
    if relation_key == "parent_of":
        return "parent"
    return None


def _parent_relation_plausible(
    source_info: Dict[str, Any],
    target_info: Dict[str, Any],
    relation_type: str,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    source_birth = _coerce_year(source_info.get("birth_year"))
    source_death = _coerce_year(source_info.get("death_year"))
    source_gender = _normalize_gender(source_info.get("gender") or source_info.get("sex"))
    target_birth = _coerce_year(target_info.get("birth_year"))
    target_death = _coerce_year(target_info.get("death_year"))

    relation_key = _normalize_relation_type(relation_type).lower()
    if relation_key == "father_of" and source_gender and source_gender != "male":
        reasons.append("father_of_requires_male_source")
    if relation_key == "mother_of" and source_gender and source_gender != "female":
        reasons.append("mother_of_requires_female_source")

    if source_birth is not None and target_birth is not None:
        age_gap = target_birth - source_birth
        if age_gap < _PARENT_TIMELINE_MIN_AGE_GAP:
            reasons.append("parent_too_young")
        if age_gap > _PARENT_TIMELINE_MAX_AGE_GAP:
            reasons.append("parent_too_old")
    if source_death is not None and target_birth is not None:
        if source_death + 1 < target_birth:
            reasons.append("parent_died_before_child_birth")
    if source_birth is not None and target_death is not None:
        if source_birth > target_death:
            reasons.append("parent_born_after_child_death")
    return (len(reasons) == 0), reasons


def _append_inference_rule(relation: Dict[str, Any], rule: str) -> None:
    prior_rule = str(relation.get("inference_rule") or "").strip()
    relation["inference_rule"] = f"{prior_rule};{rule}" if prior_rule else rule


def _promote_avuncular_relations_from_evidence(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    entity_info: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_info[entity_id] = {
            "name": str(entity.get("canonical_name") or entity_id),
            "birth_year": _coerce_year(entity.get("birth_year")),
            "gender": _normalize_gender(entity.get("gender") or entity.get("sex")),
        }

    promoted_count = 0
    promoted_relations: List[Dict[str, Any]] = []
    for relation in relations:
        relation_type_key = _normalize_relation_type(relation.get("relation_type")).lower()
        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if (
            not source_id
            or not target_id
            or source_id == target_id
            or source_id not in entity_info
            or target_id not in entity_info
        ):
            promoted_relations.append(relation)
            continue

        quote_norm = _normalize_for_cue_search(_extract_relation_quote(relation))
        if not quote_norm or not any(
            marker in quote_norm for marker in _AVUNCULAR_HINT_SUBSTRINGS
        ):
            promoted_relations.append(relation)
            continue

        source_name = str(entity_info[source_id].get("name") or "")
        target_name = str(entity_info[target_id].get("name") or "")
        if relation_type_key in _PARENT_RELATION_TYPES:
            parent_anchor_score = _relation_parent_anchor_score(
                relation,
                source_name=source_name,
                target_name=target_name,
            )
            if parent_anchor_score >= 1:
                promoted_relations.append(relation)
                continue

        source_name_pos = _name_position_in_quote(source_name, quote_norm)
        target_name_pos = _name_position_in_quote(target_name, quote_norm)
        cue_pos = _first_term_position(
            quote_norm,
            (" племянник ", " племянница ", " niece ", " nephew "),
        )

        source_is_avuncular: bool | None = None
        if cue_pos is not None:
            if (
                source_name_pos is not None
                and source_name_pos > cue_pos
                and (target_name_pos is None or target_name_pos < cue_pos)
            ):
                source_is_avuncular = True
            elif (
                target_name_pos is not None
                and target_name_pos > cue_pos
                and (source_name_pos is None or source_name_pos < cue_pos)
            ):
                source_is_avuncular = False

        if source_is_avuncular is None:
            source_birth = entity_info[source_id].get("birth_year")
            target_birth = entity_info[target_id].get("birth_year")
            if source_birth is not None and target_birth is not None:
                if source_birth <= target_birth - _PARENT_TIMELINE_MIN_AGE_GAP:
                    source_is_avuncular = True
                elif target_birth <= source_birth - _PARENT_TIMELINE_MIN_AGE_GAP:
                    source_is_avuncular = False

        if source_is_avuncular is None:
            promoted_relations.append(relation)
            continue

        promoted_relation = dict(relation)
        if not source_is_avuncular:
            promoted_relation["source_entity_id"] = target_id
            promoted_relation["target_entity_id"] = source_id
            source_id, target_id = target_id, source_id

        source_gender = entity_info[source_id].get("gender")
        promoted_relation["relation_type"] = _aunt_uncle_relation_type(source_gender)
        promoted_relation["directed"] = True
        promoted_relation["symmetric"] = False
        _append_inference_rule(promoted_relation, "evidence_avuncular_cue")
        promoted_relations.append(promoted_relation)
        promoted_count += 1

    return promoted_relations, promoted_count


def _apply_parent_timeline_guardrails(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entity_info: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_info[entity_id] = dict(entity)

    repaired_relations: List[Dict[str, Any]] = []
    timeline_repairs: List[Dict[str, Any]] = []

    for relation in relations:
        relation_type = _normalize_relation_type(relation.get("relation_type")).lower()
        if relation_type not in _PARENT_RELATION_TYPES:
            repaired_relations.append(relation)
            continue

        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if (
            not source_id
            or not target_id
            or source_id == target_id
            or source_id not in entity_info
            or target_id not in entity_info
        ):
            repaired_relations.append(relation)
            continue

        source_info = entity_info[source_id]
        target_info = entity_info[target_id]
        relation_key = _normalize_relation_type(relation.get("relation_type"))
        current_ok, current_reasons = _parent_relation_plausible(
            source_info,
            target_info,
            relation_key,
        )
        if current_ok:
            repaired_relations.append(relation)
            continue

        reverse_relation_type = _parent_relation_type_for_gender(
            _normalize_gender(target_info.get("gender") or target_info.get("sex"))
        )
        reverse_ok, reverse_reasons = _parent_relation_plausible(
            target_info,
            source_info,
            reverse_relation_type,
        )

        if reverse_ok:
            repaired_relation = dict(relation)
            repaired_relation["source_entity_id"] = target_id
            repaired_relation["target_entity_id"] = source_id
            repaired_relation["relation_type"] = reverse_relation_type
            repaired_relation["directed"] = True
            repaired_relation["symmetric"] = False
            _append_inference_rule(repaired_relation, "parent_timeline_reverse")
            repaired_relations.append(repaired_relation)
            timeline_repairs.append(
                {
                    "action": "reversed",
                    "relation_id": str(relation.get("relation_id") or ""),
                    "from_source_entity_id": source_id,
                    "from_target_entity_id": target_id,
                    "to_source_entity_id": target_id,
                    "to_target_entity_id": source_id,
                    "from_relation_type": relation_key,
                    "to_relation_type": reverse_relation_type,
                    "reasons": current_reasons,
                }
            )
            continue

        downgraded_relation = dict(relation)
        downgraded_relation["relation_type"] = "relative"
        downgraded_relation["directed"] = False
        downgraded_relation["symmetric"] = True
        _append_inference_rule(downgraded_relation, "parent_timeline_downgrade")
        repaired_relations.append(downgraded_relation)
        timeline_repairs.append(
            {
                "action": "downgraded",
                "relation_id": str(relation.get("relation_id") or ""),
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "from_relation_type": relation_key,
                "to_relation_type": "relative",
                "reasons": current_reasons,
                "reverse_reasons": reverse_reasons,
            }
        )

    timeline_repairs.sort(
        key=lambda row: (
            str(row.get("action") or ""),
            str(row.get("source_entity_id") or row.get("from_source_entity_id") or ""),
            str(row.get("target_entity_id") or row.get("from_target_entity_id") or ""),
        )
    )
    return repaired_relations, timeline_repairs


def _relation_rank_for_parent_conflict(
    relation: Dict[str, Any],
    *,
    entity_info: Dict[str, Dict[str, Any]],
) -> tuple[int, int, float, int, int, int]:
    source_id = str(relation.get("source_entity_id") or "").strip()
    target_id = str(relation.get("target_entity_id") or "").strip()
    source_name = str(entity_info.get(source_id, {}).get("canonical_name") or source_id)
    target_name = str(entity_info.get(target_id, {}).get("canonical_name") or target_id)
    parent_anchor_score = _relation_parent_anchor_score(
        relation,
        source_name=source_name,
        target_name=target_name,
    )
    polarity = str(relation.get("polarity") or "").strip().lower()
    polarity_score = 0 if polarity == "negative" else 1
    confidence = _safe_float(relation.get("confidence"), 0.0)
    support_count = int(relation.get("support_count") or 0)
    evidence_len = len(_extract_relation_quote(relation))
    is_promoted = (
        str(relation.get("inference_rule") or "").strip() == "evidence_parent_cue"
    )
    return (
        parent_anchor_score,
        polarity_score,
        confidence,
        support_count,
        evidence_len,
        0 if is_promoted else 1,
    )


def _resolve_parent_role_conflicts(
    relations: Sequence[Dict[str, Any]],
    entities: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    indexed_relations = list(relations)
    role_limits = {"father": 1, "mother": 1, "parent": 2}
    grouped_indices: Dict[tuple[str, str], List[int]] = defaultdict(list)
    entity_info: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_info[entity_id] = dict(entity)

    for index, relation in enumerate(indexed_relations):
        relation_type = _normalize_relation_type(relation.get("relation_type"))
        role = _parent_role_key(relation_type)
        if role is None:
            continue
        child_id = str(relation.get("target_entity_id") or "").strip()
        if not child_id:
            continue
        grouped_indices[(child_id, role)].append(index)

    keep_indices: set[int] = set(range(len(indexed_relations)))
    downgraded_indices: set[int] = set()
    conflict_rows: List[Dict[str, Any]] = []
    for (child_id, role), indices in grouped_indices.items():
        limit = role_limits[role]
        if len(indices) <= limit:
            continue
        ranked = sorted(
            indices,
            key=lambda idx: _relation_rank_for_parent_conflict(
                indexed_relations[idx],
                entity_info=entity_info,
            ),
            reverse=True,
        )
        conflict_rows.append(
            {
                "child_entity_id": child_id,
                "role": role,
                "limit": limit,
                "kept_relation_ids": [
                    str(indexed_relations[idx].get("relation_id") or "")
                    for idx in ranked[:limit]
                ],
                "downgraded_relation_ids": [
                    str(indexed_relations[idx].get("relation_id") or "")
                    for idx in ranked[limit:]
                ],
                "kept_relations": [
                    {
                        "source_entity_id": str(
                            indexed_relations[idx].get("source_entity_id") or ""
                        ),
                        "target_entity_id": str(
                            indexed_relations[idx].get("target_entity_id") or ""
                        ),
                        "relation_type": _normalize_relation_type(
                            indexed_relations[idx].get("relation_type")
                        ),
                        "confidence": _safe_float(
                            indexed_relations[idx].get("confidence"), 0.0
                        ),
                        "support_count": int(
                            indexed_relations[idx].get("support_count") or 1
                        ),
                    }
                    for idx in ranked[:limit]
                ],
                "downgraded_relations": [
                    {
                        "source_entity_id": str(
                            indexed_relations[idx].get("source_entity_id") or ""
                        ),
                        "target_entity_id": str(
                            indexed_relations[idx].get("target_entity_id") or ""
                        ),
                        "relation_type": _normalize_relation_type(
                            indexed_relations[idx].get("relation_type")
                        ),
                        "confidence": _safe_float(
                            indexed_relations[idx].get("confidence"), 0.0
                        ),
                        "support_count": int(
                            indexed_relations[idx].get("support_count") or 1
                        ),
                    }
                    for idx in ranked[limit:]
                ],
            }
        )
        for idx in ranked[limit:]:
            keep_indices.discard(idx)
            downgraded_indices.add(idx)

    resolved_relations: List[Dict[str, Any]] = [
        indexed_relations[index]
        for index in range(len(indexed_relations))
        if index in keep_indices
    ]
    existing_relative_keys = {
        (
            str(relation.get("source_entity_id") or "").strip(),
            str(relation.get("target_entity_id") or "").strip(),
            _normalize_relation_type(relation.get("relation_type")).lower(),
        )
        for relation in resolved_relations
    }

    downgraded_count = 0
    for index in sorted(downgraded_indices):
        relation = dict(indexed_relations[index])
        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        relative_key = (source_id, target_id, "relative")
        if relative_key in existing_relative_keys:
            downgraded_count += 1
            continue
        relation["relation_type"] = "relative"
        relation["directed"] = False
        relation["symmetric"] = True
        prior_rule = str(relation.get("inference_rule") or "").strip()
        conflict_rule = "parent_role_conflict_downgrade"
        relation["inference_rule"] = (
            f"{prior_rule};{conflict_rule}" if prior_rule else conflict_rule
        )
        resolved_relations.append(relation)
        existing_relative_keys.add(relative_key)
        downgraded_count += 1

    conflict_rows.sort(
        key=lambda row: (
            str(row.get("child_entity_id") or ""),
            str(row.get("role") or ""),
        )
    )
    return resolved_relations, downgraded_count, conflict_rows


def _sibling_relation_type(gender: str | None) -> str:
    if gender == "male":
        return "brother_of"
    if gender == "female":
        return "sister_of"
    return "sibling_of"


def _aunt_uncle_relation_type(gender: str | None) -> str:
    if gender == "male":
        return "uncle_of"
    if gender == "female":
        return "aunt_of"
    return "aunt_uncle_of"


def _niece_nephew_relation_type(gender: str | None) -> str:
    if gender == "male":
        return "nephew_of"
    if gender == "female":
        return "niece_of"
    return "nibling_of"


def _grandparent_relation_type(gender: str | None) -> str:
    if gender == "male":
        return "grandfather_of"
    if gender == "female":
        return "grandmother_of"
    return "grandparent_of"


def _grandchild_relation_type(gender: str | None) -> str:
    if gender == "male":
        return "grandson_of"
    if gender == "female":
        return "granddaughter_of"
    return "grandchild_of"


def _derived_relation_id(source_id: str, relation_type: str, target_id: str) -> str:
    digest = hashlib.md5(
        f"{source_id}|{relation_type}|{target_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"derived-{digest}"


def _parent_relation_type_for_gender(gender: str | None) -> str:
    if gender == "male":
        return "father_of"
    if gender == "female":
        return "mother_of"
    return "parent_of"


def _normalize_parent_relations(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gender_by_id: Dict[str, str | None] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        gender_by_id[entity_id] = _normalize_gender(
            entity.get("gender") or entity.get("sex")
        )

    normalized_relations: List[Dict[str, Any]] = []
    for relation in relations:
        relation_type = _normalize_relation_type(relation.get("relation_type")).lower()
        if relation_type != "parent_child":
            normalized_relations.append(relation)
            continue

        source_id = str(relation.get("source_entity_id") or "").strip()
        specialized_type = _parent_relation_type_for_gender(gender_by_id.get(source_id))
        normalized_relation = dict(relation)
        normalized_relation["relation_type"] = specialized_type
        normalized_relations.append(normalized_relation)

    return normalized_relations


def _augment_relations_with_kinship(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entity_ids: set[str] = set()
    gender_by_id: Dict[str, str | None] = {}
    entity_info: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_ids.add(entity_id)
        entity_info[entity_id] = dict(entity)
        gender_by_id[entity_id] = _normalize_gender(
            entity.get("gender") or entity.get("sex")
        )

    parent_to_children: Dict[str, set[str]] = defaultdict(set)
    child_to_parents: Dict[str, set[str]] = defaultdict(set)
    avuncular_to_children: Dict[str, set[str]] = defaultdict(set)
    avuncular_sibling_pairs: set[tuple[str, str]] = set()
    for relation in relations:
        rel_type = _normalize_relation_type(relation.get("relation_type")).lower()
        if rel_type not in _PARENT_RELATION_TYPES:
            source_id = str(relation.get("source_entity_id") or "").strip()
            target_id = str(relation.get("target_entity_id") or "").strip()
            if (
                rel_type in _AVUNCULAR_RELATION_TYPES
                and source_id
                and target_id
                and source_id in entity_ids
                and target_id in entity_ids
                and source_id != target_id
            ):
                avuncular_to_children[source_id].add(target_id)
            elif rel_type in {"nephew_of", "niece_of", "nibling_of"}:
                if (
                    source_id
                    and target_id
                    and source_id in entity_ids
                    and target_id in entity_ids
                    and source_id != target_id
                ):
                    avuncular_to_children[target_id].add(source_id)
            continue
        parent_id = str(relation.get("source_entity_id") or "").strip()
        child_id = str(relation.get("target_entity_id") or "").strip()
        if (
            not parent_id
            or not child_id
            or parent_id == child_id
            or parent_id not in entity_ids
            or child_id not in entity_ids
        ):
            continue
        parent_to_children[parent_id].add(child_id)
        child_to_parents[child_id].add(parent_id)

    existing_keys: set[tuple[str, str, str]] = set()
    for relation in relations:
        source_id = str(relation.get("source_entity_id") or "").strip()
        target_id = str(relation.get("target_entity_id") or "").strip()
        if not source_id or not target_id:
            continue
        relation_type = _normalize_relation_type(relation.get("relation_type")).lower()
        existing_keys.add((source_id, relation_type, target_id))

    sibling_map: Dict[str, set[str]] = defaultdict(set)
    for children in parent_to_children.values():
        child_ids = sorted(children)
        for left_idx in range(len(child_ids)):
            for right_idx in range(left_idx + 1, len(child_ids)):
                left = child_ids[left_idx]
                right = child_ids[right_idx]
                sibling_map[left].add(right)
                sibling_map[right].add(left)
    for avuncular_id, child_ids in avuncular_to_children.items():
        for child_id in child_ids:
            for parent_id in child_to_parents.get(child_id, set()):
                if parent_id == avuncular_id:
                    continue
                sibling_map[avuncular_id].add(parent_id)
                sibling_map[parent_id].add(avuncular_id)
                pair = tuple(sorted((avuncular_id, parent_id)))
                avuncular_sibling_pairs.add(pair)

    derived_relations: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    def add_derived_relation(
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        inference_rule: str,
        confidence: float = 0.7,
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        normalized_type = _normalize_relation_type(relation_type)
        key = (source_id, normalized_type.lower(), target_id)
        if key in existing_keys or key in derived_relations:
            return
        derived_relations[key] = {
            "relation_id": _derived_relation_id(source_id, normalized_type, target_id),
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "relation_type": normalized_type,
            "confidence": confidence,
            "support_count": 1,
            "directed": True,
            "symmetric": False,
            "derived": True,
            "inference_rule": inference_rule,
        }

    for source_id, sibling_ids in sibling_map.items():
        for sibling_id in sorted(sibling_ids):
            add_derived_relation(
                source_id,
                sibling_id,
                _sibling_relation_type(gender_by_id.get(source_id)),
                inference_rule="shared_parent",
                confidence=0.75,
            )
    for left_id, right_id in sorted(avuncular_sibling_pairs):
        for person_id, sibling_id in ((left_id, right_id), (right_id, left_id)):
            for parent_id in sorted(child_to_parents.get(person_id, set())):
                parent_type = _parent_relation_type_for_gender(gender_by_id.get(parent_id))
                parent_row = entity_info.get(parent_id, {})
                sibling_row = entity_info.get(sibling_id, {})
                is_plausible, _ = _parent_relation_plausible(
                    parent_row,
                    sibling_row,
                    parent_type,
                )
                if not is_plausible:
                    continue
                add_derived_relation(
                    parent_id,
                    sibling_id,
                    parent_type,
                    inference_rule="avuncular_parent_transfer",
                    confidence=0.68,
                )

    for child_id, parent_ids in child_to_parents.items():
        for parent_id in parent_ids:
            for relative_id in sibling_map.get(parent_id, set()):
                add_derived_relation(
                    relative_id,
                    child_id,
                    _aunt_uncle_relation_type(gender_by_id.get(relative_id)),
                    inference_rule="parent_sibling",
                    confidence=0.72,
                )
                add_derived_relation(
                    child_id,
                    relative_id,
                    _niece_nephew_relation_type(gender_by_id.get(child_id)),
                    inference_rule="parent_sibling_inverse",
                    confidence=0.72,
                )

    for grandparent_id, parent_ids in parent_to_children.items():
        for parent_id in parent_ids:
            for grandchild_id in parent_to_children.get(parent_id, set()):
                add_derived_relation(
                    grandparent_id,
                    grandchild_id,
                    _grandparent_relation_type(gender_by_id.get(grandparent_id)),
                    inference_rule="two_hop_parent_child",
                    confidence=0.73,
                )
                add_derived_relation(
                    grandchild_id,
                    grandparent_id,
                    _grandchild_relation_type(gender_by_id.get(grandchild_id)),
                    inference_rule="two_hop_parent_child_inverse",
                    confidence=0.73,
                )

    child_ids = sorted(child_to_parents)
    for left_idx in range(len(child_ids)):
        for right_idx in range(left_idx + 1, len(child_ids)):
            left_child = child_ids[left_idx]
            right_child = child_ids[right_idx]
            left_parents = child_to_parents[left_child]
            right_parents = child_to_parents[right_child]

            is_cousin = any(
                right_parent in sibling_map.get(left_parent, set())
                for left_parent in left_parents
                for right_parent in right_parents
            )
            if not is_cousin:
                continue
            add_derived_relation(
                left_child,
                right_child,
                "cousin_of",
                inference_rule="parents_are_siblings",
                confidence=0.7,
            )
            add_derived_relation(
                right_child,
                left_child,
                "cousin_of",
                inference_rule="parents_are_siblings",
                confidence=0.7,
            )

    if not derived_relations:
        return list(relations)

    merged_relations = list(relations)
    for relation_key in sorted(derived_relations):
        merged_relations.append(derived_relations[relation_key])
    return merged_relations


def _filter_living_graph_relations(
    relations: Sequence[Dict[str, Any]],
    *,
    include_relation_types: set[str],
    exclude_relation_types: set[str],
) -> List[Dict[str, Any]]:
    expanded_include = set(include_relation_types)
    expanded_exclude = set(exclude_relation_types)
    if "parent_child" in expanded_include:
        expanded_include.update(_PARENT_RELATION_TYPES)
    if "parent_child" in expanded_exclude:
        expanded_exclude.update(_PARENT_RELATION_TYPES)

    filtered: List[Dict[str, Any]] = []
    for relation in relations:
        relation_type = _normalize_relation_type(relation.get("relation_type"))
        relation_key = relation_type.lower()
        if expanded_include and relation_key not in expanded_include:
            continue
        if relation_key in expanded_exclude:
            continue
        filtered.append(relation)
    return filtered


def _build_living_graph_dot(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    max_relations: int = 400,
) -> str:
    lines = [
        "digraph living_graph {",
        '  graph [overlap=false splines=true ranksep=1.0 nodesep=0.35 rankdir=TB fontsize=10 fontname="Helvetica"];',
        '  node [shape=ellipse style=filled fillcolor="#f8fafc" color="#94a3b8" fontsize=10 fontname="Helvetica"];',
        '  edge [fontsize=9 fontname="Helvetica" color="#475569"];',
    ]

    entity_id_to_name: Dict[str, str] = {}
    entity_birth_year: Dict[str, int | None] = {}
    birth_decade_buckets: Dict[int, List[str]] = defaultdict(list)
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "").strip()
        if not entity_id:
            continue
        canonical_name = (
            str(entity.get("canonical_name") or entity_id).strip() or entity_id
        )
        birth_year = _coerce_year(entity.get("birth_year"))
        death_year = entity.get("death_year")
        years = ""
        if birth_year or death_year:
            years = f"\\n({birth_year or '?'}-{death_year or '?'})"
        label = _escape_dot(canonical_name + years)
        node_id = _escape_dot(entity_id)
        lines.append(f'  "{node_id}" [label="{label}"];')
        entity_id_to_name[entity_id] = canonical_name
        entity_birth_year[entity_id] = birth_year
        if birth_year is not None:
            birth_decade_buckets[(birth_year // 10) * 10].append(entity_id)

    # Time layout: group same decade on one horizontal rank and force decade flow top->bottom.
    if birth_decade_buckets:
        lines.append("  // Time layering (older generations above younger)")
        previous_anchor_id: str | None = None
        for decade in sorted(birth_decade_buckets):
            ranked_ids = sorted(
                birth_decade_buckets[decade],
                key=lambda entity_id: (
                    entity_birth_year.get(entity_id) or 0,
                    entity_id_to_name.get(entity_id, "").lower(),
                    entity_id,
                ),
            )
            if not ranked_ids:
                continue
            lines.append(f"  subgraph cluster_birth_{decade} {{")
            lines.append('    style="invis";')
            lines.append('    label="";')
            lines.append("    rank=same;")
            for entity_id in ranked_ids:
                lines.append(f'    "{_escape_dot(entity_id)}";')
            lines.append("  }")
            anchor_id = ranked_ids[0]
            if previous_anchor_id is not None:
                lines.append(
                    f'  "{_escape_dot(previous_anchor_id)}" -> "{_escape_dot(anchor_id)}" '
                    '[style="invis" weight=100 minlen=2];'
                )
            previous_anchor_id = anchor_id

    for relation in relations[: max(0, int(max_relations))]:
        src = str(relation.get("source_entity_id") or "").strip()
        tgt = str(relation.get("target_entity_id") or "").strip()
        if not src or not tgt:
            continue
        if src not in entity_id_to_name or tgt not in entity_id_to_name:
            continue

        rel_type = _normalize_relation_type(relation.get("relation_type"))
        confidence = relation.get("confidence")
        label = rel_type
        if isinstance(confidence, (int, float)):
            label = f"{rel_type} ({float(confidence):.2f})"

        color = _pick_relation_color(rel_type)
        relation_class = _relation_type_class(rel_type)
        directed = bool(relation.get("directed", True))
        symmetric = bool(relation.get("symmetric", False))
        arrow = "none" if (symmetric or not directed) else "normal"
        is_parent_edge = rel_type.lower() in _PARENT_RELATION_TYPES
        constraint = "true" if is_parent_edge else "false"
        weight = 6 if is_parent_edge else 1

        src_id = _escape_dot(src)
        tgt_id = _escape_dot(tgt)
        lines.append(
            f'  "{src_id}" -> "{tgt_id}" [label="{_escape_dot(label)}" color="{color}" '
            f'fontcolor="{color}" arrowhead="{arrow}" class="{_escape_dot(relation_class)}" '
            f'constraint="{constraint}" weight={weight}];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _build_living_graph_html(
    entities: Sequence[Dict[str, Any]],
    relations: Sequence[Dict[str, Any]],
    dot_graph: str,
    svg_graph: str | None = None,
) -> str:
    relation_rows = []
    relation_type_counts: Counter[str] = Counter()
    relation_type_classes: Dict[str, str] = {}
    for relation in relations:
        relation_type = _normalize_relation_type(relation.get("relation_type"))
        relation_type_key = relation_type.lower()
        relation_class = _relation_type_class(relation_type)
        relation_type_counts[relation_type] += 1
        relation_type_classes.setdefault(relation_type, relation_class)
        relation_rows.append(
            (
                f"<tr data-relation-type='{html.escape(relation_type_key)}' data-relation-class='{html.escape(relation_class)}'>"
                f"<td>{html.escape(str(relation.get('source_entity_id', '')))}</td>"
                f"<td>{html.escape(str(relation_type))}</td>"
                f"<td>{html.escape(str(relation.get('target_entity_id', '')))}</td>"
                f"<td>{html.escape(str(relation.get('confidence', '')))}</td>"
                f"<td>{html.escape(str(relation.get('support_count', '')))}</td></tr>"
            )
        )

    entity_rows = []
    for entity in entities:
        entity_rows.append(
            (
                f"<tr><td>{html.escape(str(entity.get('entity_id', '')))}</td>"
                f"<td>{html.escape(str(entity.get('canonical_name', '')))}</td>"
                f"<td>{html.escape(str(entity.get('entity_type', '')))}</td></tr>"
            )
        )

    relation_filters: List[str] = []
    for relation_type, count in sorted(
        relation_type_counts.items(),
        key=lambda item: (-item[1], item[0].lower()),
    ):
        relation_filters.append(
            (
                "<label class='filter-chip'>"
                f"<input class='rel-toggle' type='checkbox' checked "
                f"data-rel-type='{html.escape(relation_type.lower())}' "
                f"data-rel-class='{html.escape(relation_type_classes[relation_type])}'>"
                f"{html.escape(relation_type)} ({count})"
                "</label>"
            )
        )
    if not relation_filters:
        relation_filters.append("<span class='meta'>No relation types found</span>")

    graph_block = (
        f"<div class='svg-wrap'>{svg_graph}</div>"
        if svg_graph
        else f"<pre>{html.escape(dot_graph)}</pre>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Living Graph</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ color: #475569; font-size: 14px; margin-bottom: 12px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    .svg-wrap {{ overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; background: #fff; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .filter-chip {{ display: inline-flex; align-items: center; gap: 6px; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 999px; padding: 4px 10px; font-size: 12px; }}
    .filter-actions {{ margin-bottom: 12px; display: flex; gap: 8px; }}
    .filter-actions button {{ border: 1px solid #94a3b8; border-radius: 6px; background: #fff; color: #0f172a; padding: 4px 10px; cursor: pointer; }}
    .filter-actions button:hover {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Living Graph Report</h1>
  <p class="meta">Entities: {len(entities)} | Relations: {len(relations)} | Visible relations: <span id="visible-relations">{len(relations)}</span></p>

  <section class="card">
    <h2>Graph</h2>
    <div class="filter-actions">
      <button id="select-all-rel-types" type="button">Show all</button>
      <button id="clear-all-rel-types" type="button">Hide all</button>
    </div>
    <div class="filters">{''.join(relation_filters)}</div>
    {graph_block}
  </section>

  <section class="card">
    <h2>Entities</h2>
    <table>
      <thead><tr><th>entity_id</th><th>canonical_name</th><th>entity_type</th></tr></thead>
      <tbody>{''.join(entity_rows)}</tbody>
    </table>
  </section>

  <section class="card">
    <h2>Relations</h2>
    <table>
      <thead><tr><th>source</th><th>type</th><th>target</th><th>confidence</th><th>support_count</th></tr></thead>
      <tbody>{''.join(relation_rows)}</tbody>
    </table>
  </section>

  <script>
    (() => {{
      const toggles = Array.from(document.querySelectorAll(".rel-toggle"));
      const relationRows = Array.from(document.querySelectorAll("tr[data-relation-type]"));
      const edgeGroups = Array.from(document.querySelectorAll("g.edge"));
      const visibleRelations = document.getElementById("visible-relations");
      const showAll = document.getElementById("select-all-rel-types");
      const hideAll = document.getElementById("clear-all-rel-types");

      const getActiveTypes = () =>
        new Set(
          toggles
            .filter((toggle) => toggle.checked)
            .map((toggle) => String(toggle.dataset.relType || ""))
        );
      const getActiveClasses = () =>
        new Set(
          toggles
            .filter((toggle) => toggle.checked)
            .map((toggle) => String(toggle.dataset.relClass || ""))
        );

      const updateVisibility = () => {{
        const activeTypes = getActiveTypes();
        const activeClasses = getActiveClasses();
        let visibleCount = 0;

        for (const row of relationRows) {{
          const relationType = String(row.dataset.relationType || "");
          const isVisible = activeTypes.has(relationType);
          row.style.display = isVisible ? "" : "none";
          if (isVisible) {{
            visibleCount += 1;
          }}
        }}

        for (const edgeGroup of edgeGroups) {{
          const relationClass = Array.from(edgeGroup.classList).find((name) =>
            name.startsWith("rel-type-")
          );
          if (!relationClass) {{
            continue;
          }}
          edgeGroup.style.display = activeClasses.has(relationClass) ? "" : "none";
        }}

        if (visibleRelations) {{
          visibleRelations.textContent = String(visibleCount);
        }}
      }};

      if (showAll) {{
        showAll.addEventListener("click", () => {{
          toggles.forEach((toggle) => {{
            toggle.checked = true;
          }});
          updateVisibility();
        }});
      }}

      if (hideAll) {{
        hideAll.addEventListener("click", () => {{
          toggles.forEach((toggle) => {{
            toggle.checked = false;
          }});
          updateVisibility();
        }});
      }}

      toggles.forEach((toggle) => {{
        toggle.addEventListener("change", updateVisibility);
      }});
      updateVisibility();
    }})();
  </script>
</body>
</html>
"""



def parse_relation_types_arg(raw_value: str | None) -> set[str]:
    return _parse_relation_types_arg(raw_value)


def looks_like_living_graph_json(path: Path) -> bool:
    return _looks_like_living_graph_json(path)


def build_living_graph(
    input_path: Path,
    output_dir: Path,
    *,
    include_relation_types: set[str] | None = None,
    exclude_relation_types: set[str] | None = None,
    derive_kinship: bool = False,
    max_relations: int = 400,
) -> BuildResult:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_entities = payload.get("entities") or []
    source_relations = payload.get("relations") or []
    entities, merged_source_relations, entity_merge_groups = _merge_entities_for_living_graph(
        source_entities,
        source_relations,
    )
    promoted_relations, promoted_count = _promote_parent_relations_from_evidence(
        entities,
        merged_source_relations,
    )
    normalized_relations = _normalize_parent_relations(entities, promoted_relations)
    avuncular_promoted_relations, avuncular_promoted_count = (
        _promote_avuncular_relations_from_evidence(entities, normalized_relations)
    )
    timeline_guarded_relations, parent_timeline_repairs = _apply_parent_timeline_guardrails(
        entities,
        avuncular_promoted_relations,
    )
    (
        conflict_resolved_relations,
        parent_conflict_downgraded_count,
        parent_role_conflicts,
    ) = _resolve_parent_role_conflicts(timeline_guarded_relations, entities)

    raw_relations = conflict_resolved_relations
    if derive_kinship:
        raw_relations = _augment_relations_with_kinship(
            entities,
            conflict_resolved_relations,
        )
    derived_count = max(0, len(raw_relations) - len(conflict_resolved_relations))

    include_relation_types = include_relation_types or set()
    exclude_relation_types = exclude_relation_types or set()
    relations = _filter_living_graph_relations(
        raw_relations,
        include_relation_types=include_relation_types,
        exclude_relation_types=exclude_relation_types,
    )

    entities_path = output_dir / "entities.json"
    relations_path = output_dir / "relations.json"
    conflicts_path = output_dir / "conflicts.json"
    dot_path = output_dir / "living_graph.dot"
    svg_path = output_dir / "living_graph.svg"
    html_path = output_dir / "living_graph.html"

    entities_path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    relations_path.write_text(
        json.dumps(relations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    conflicts_payload = {
        "entity_merge_groups": entity_merge_groups,
        "avuncular_promoted": avuncular_promoted_count,
        "parent_timeline_repairs": parent_timeline_repairs,
        "parent_role_conflicts": parent_role_conflicts,
    }
    conflicts_path.write_text(
        json.dumps(conflicts_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    dot_graph = _build_living_graph_dot(
        entities,
        relations,
        max_relations=max_relations,
    )
    dot_path.write_text(dot_graph, encoding="utf-8")

    svg_graph = None
    if _try_generate_svg_from_dot(dot_path, svg_path):
        svg_graph = svg_path.read_text(encoding="utf-8", errors="ignore")

    html_path.write_text(
        _build_living_graph_html(entities, relations, dot_graph, svg_graph=svg_graph),
        encoding="utf-8",
    )

    filtered_out = len(raw_relations) - len(relations)
    filter_suffix = ""
    if (
        include_relation_types
        or exclude_relation_types
        or derived_count
        or promoted_count
        or avuncular_promoted_count
        or entity_merge_groups
        or parent_timeline_repairs
        or parent_conflict_downgraded_count
    ):
        include_text = (
            ",".join(sorted(include_relation_types)) if include_relation_types else "-"
        )
        exclude_text = (
            ",".join(sorted(exclude_relation_types)) if exclude_relation_types else "-"
        )
        filter_suffix = (
            f" Filters: include=[{include_text}] exclude=[{exclude_text}]"
            f", removed={filtered_out}, parent_promoted={promoted_count}, "
            f"avuncular_promoted={avuncular_promoted_count}, "
            f"entity_merged_groups={len(entity_merge_groups)}, "
            f"parent_timeline_repairs={len(parent_timeline_repairs)}, "
            f"parent_conflict_downgraded={parent_conflict_downgraded_count}, "
            f"derived_added={derived_count}."
        )

    return BuildResult(
        output_dir=output_dir,
        entities_count=len(entities),
        relations_count=len(relations),
        message_suffix=filter_suffix,
        details={
            "filtered_out": filtered_out,
            "parent_promoted": promoted_count,
            "avuncular_promoted": avuncular_promoted_count,
            "entity_merge_groups": entity_merge_groups,
            "parent_timeline_repairs": parent_timeline_repairs,
            "parent_conflict_downgraded": parent_conflict_downgraded_count,
            "derived_added": derived_count,
        },
    )
