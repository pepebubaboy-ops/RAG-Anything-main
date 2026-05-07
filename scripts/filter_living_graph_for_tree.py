#!/usr/bin/env python3
"""Filter a general living graph into a stricter genealogy-focused source graph.

The script is generic for any dynasty/family:
- keeps only selected base relation types from the raw living graph
- optionally keeps top connected components by size
- prunes entities not used by remaining relations
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def _parse_types(raw: str) -> Set[str]:
    items = [x.strip().lower() for x in str(raw or "").split(",")]
    return {x for x in items if x}


def _iter_relation_endpoints(rel: Dict[str, object]) -> Tuple[str, str]:
    src = str(rel.get("source_entity_id") or "").strip()
    dst = str(rel.get("target_entity_id") or "").strip()
    return src, dst


def _components_from_edges(edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    nodes: Set[str] = set()
    for a, b in edges:
        if not a or not b or a == b:
            continue
        nodes.add(a)
        nodes.add(b)
        adj[a].add(b)
        adj[b].add(a)
    visited: Set[str] = set()
    comps: List[Set[str]] = []
    for n in sorted(nodes):
        if n in visited:
            continue
        q = deque([n])
        visited.add(n)
        comp: Set[str] = set()
        while q:
            cur = q.popleft()
            comp.add(cur)
            for nxt in adj.get(cur, set()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                q.append(nxt)
        comps.append(comp)
    comps.sort(key=lambda c: len(c), reverse=True)
    return comps


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter living_graph.json to genealogy-focused source graph")
    p.add_argument("--input", required=True, help="Path to living_graph.json")
    p.add_argument("--output", required=True, help="Output filtered JSON path")
    p.add_argument(
        "--keep-relation-types",
        default="parent_child,spouse,sibling",
        help="Comma-separated raw relation_type values to keep",
    )
    p.add_argument(
        "--min-component-size",
        type=int,
        default=3,
        help="Minimum node count for a connected component to keep",
    )
    p.add_argument(
        "--max-components",
        type=int,
        default=2,
        help="Keep at most N largest connected components (0 means keep all)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    entities = payload.get("entities") or []
    relations = payload.get("relations") or []

    keep_types = _parse_types(args.keep_relation_types)

    kept_relations: List[Dict[str, object]] = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rel_type = str(rel.get("relation_type") or "").strip().lower()
        if rel_type not in keep_types:
            continue
        src, dst = _iter_relation_endpoints(rel)
        if not src or not dst or src == dst:
            continue
        kept_relations.append(rel)

    comps = _components_from_edges(_iter_relation_endpoints(r) for r in kept_relations)
    min_size = max(1, int(args.min_component_size))
    max_components = max(0, int(args.max_components))

    kept_components: List[Set[str]] = []
    for comp in comps:
        if len(comp) < min_size:
            continue
        kept_components.append(comp)
        if max_components > 0 and len(kept_components) >= max_components:
            break

    kept_ids: Set[str] = set()
    for comp in kept_components:
        kept_ids.update(comp)

    if kept_ids:
        kept_relations = [
            r
            for r in kept_relations
            if str(r.get("source_entity_id") or "") in kept_ids
            and str(r.get("target_entity_id") or "") in kept_ids
        ]
    else:
        kept_relations = []

    entity_by_id = {}
    for e in entities:
        if not isinstance(e, dict):
            continue
        entity_id = str(e.get("entity_id") or "").strip()
        if entity_id:
            entity_by_id[entity_id] = e
    kept_entities = [entity_by_id[eid] for eid in sorted(kept_ids) if eid in entity_by_id]

    out = {
        **payload,
        "entities": kept_entities,
        "relations": kept_relations,
        "_genealogy_filter": {
            "keep_relation_types": sorted(keep_types),
            "min_component_size": min_size,
            "max_components": max_components,
            "input_entities": len(entities),
            "input_relations": len(relations),
            "output_entities": len(kept_entities),
            "output_relations": len(kept_relations),
            "components_total": len(comps),
            "components_kept": len(kept_components),
        },
    }

    output_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Filtered living graph:"
        f" entities {len(entities)} -> {len(kept_entities)},"
        f" relations {len(relations)} -> {len(kept_relations)},"
        f" components_kept={len(kept_components)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
