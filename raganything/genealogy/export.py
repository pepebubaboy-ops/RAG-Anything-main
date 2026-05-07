from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .stores import InMemoryGenealogyStore


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _try_generate_svg_from_dot(dot_path: Path, svg_path: Path) -> bool:
    if shutil.which("dot") is None:
        return False
    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def _store_to_people_and_families(
    store: InMemoryGenealogyStore,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    child_map: Dict[str, List[str]] = {}
    for family_id, child_id in store.rel_has_child:
        child_map.setdefault(family_id, []).append(child_id)

    people: List[Dict[str, Any]] = []
    for person_id, person in sorted(store.people.items()):
        people.append(
            {
                "person_id": person_id,
                "name": person.spec.name,
                "normalized_name": person.normalized_name,
                "birth_year": person.spec.birth_year,
                "death_year": person.spec.death_year,
                "birth_place": person.spec.birth_place,
                "death_place": person.spec.death_place,
                "aliases": list(person.spec.aliases or []),
            }
        )

    families: List[Dict[str, Any]] = []
    for family_id, family in sorted(store.families.items()):
        families.append(
            {
                "family_id": family_id,
                "family_type": family.family_type,
                "family_key": family.family_key,
                "parent_ids": list(family.parent_ids),
                "child_ids": sorted(set(child_map.get(family_id, []))),
            }
        )

    return people, families


def _build_dot_from_people_and_families(
    people: Sequence[Dict[str, Any]],
    families: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "digraph family_tree {",
        "  rankdir=LR;",
        '  graph [fontsize=10 fontname="Helvetica"];',
        '  node [fontsize=10 fontname="Helvetica"];',
        '  edge [fontsize=9 fontname="Helvetica"];',
    ]

    for person in people:
        name = _escape_dot(str(person.get("name") or person.get("person_id") or ""))
        person_id = _escape_dot(str(person.get("person_id") or ""))
        lines.append(f'  "{person_id}" [shape=box label="{name}"];')

    for family in families:
        family_id = _escape_dot(str(family.get("family_id") or ""))
        lines.append(f'  "{family_id}" [shape=diamond label="family"];')

        for parent_id in family.get("parent_ids") or []:
            pid = _escape_dot(str(parent_id))
            lines.append(f'  "{pid}" -> "{family_id}" [label="parent"];')

        for child_id in family.get("child_ids") or []:
            cid = _escape_dot(str(child_id))
            lines.append(f'  "{family_id}" -> "{cid}" [label="child"];')

        parents = list(family.get("parent_ids") or [])
        if len(parents) >= 2:
            p1 = _escape_dot(str(parents[0]))
            p2 = _escape_dot(str(parents[1]))
            lines.append(f'  "{p1}" -> "{p2}" [dir=none style=dashed label="spouse"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def _build_gedcom_from_people_and_families(
    people: Sequence[Dict[str, Any]],
    families: Sequence[Dict[str, Any]],
) -> str:
    lines = ["0 HEAD", "1 SOUR RAGANYTHING", "1 CHAR UTF-8"]

    person_map: Dict[str, str] = {}
    for index, person in enumerate(people, start=1):
        person_xref = f"@I{index}@"
        person_id = str(person.get("person_id") or "")
        person_map[person_id] = person_xref
        lines.append(f"0 {person_xref} INDI")
        lines.append(f"1 NAME {person.get('name') or 'Unknown'}")
        if person.get("birth_year"):
            lines.append("1 BIRT")
            lines.append(f"2 DATE {person['birth_year']}")
        if person.get("death_year"):
            lines.append("1 DEAT")
            lines.append(f"2 DATE {person['death_year']}")

    for index, family in enumerate(families, start=1):
        family_xref = f"@F{index}@"
        lines.append(f"0 {family_xref} FAM")

        parents = list(family.get("parent_ids") or [])
        if len(parents) >= 1 and parents[0] in person_map:
            lines.append(f"1 HUSB {person_map[parents[0]]}")
        if len(parents) >= 2 and parents[1] in person_map:
            lines.append(f"1 WIFE {person_map[parents[1]]}")

        for child_id in family.get("child_ids") or []:
            if child_id in person_map:
                lines.append(f"1 CHIL {person_map[child_id]}")

    lines.append("0 TRLR")
    return "\n".join(lines) + "\n"


def _build_html_from_people_and_families(
    people: Sequence[Dict[str, Any]],
    families: Sequence[Dict[str, Any]],
    dot_graph: str,
) -> str:
    person_rows = []
    for person in people:
        person_rows.append(
            (
                f"<tr><td>{person.get('person_id', '')}</td>"
                f"<td>{person.get('name', '')}</td>"
                f"<td>{person.get('birth_year', '') or ''}</td>"
                f"<td>{person.get('death_year', '') or ''}</td></tr>"
            )
        )

    family_rows = []
    for family in families:
        family_rows.append(
            (
                f"<tr><td>{family.get('family_id', '')}</td>"
                f"<td>{', '.join(family.get('parent_ids') or [])}</td>"
                f"<td>{', '.join(family.get('child_ids') or [])}</td></tr>"
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Genealogy Tree</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f1f5f9; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    .meta {{ color: #475569; font-size: 14px; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>Genealogy Tree Report</h1>
  <p class="meta">People: {len(people)} | Families: {len(families)}</p>

  <section class="card">
    <h2>People</h2>
    <table>
      <thead><tr><th>person_id</th><th>name</th><th>birth_year</th><th>death_year</th></tr></thead>
      <tbody>
        {"".join(person_rows)}
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>Families</h2>
    <table>
      <thead><tr><th>family_id</th><th>parent_ids</th><th>child_ids</th></tr></thead>
      <tbody>
        {"".join(family_rows)}
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>DOT Source</h2>
    <pre>{dot_graph}</pre>
  </section>
</body>
</html>
"""
    return html


def export_genealogy(
    input_dir: Path,
    output_format: str,
    output_path: Path | None = None,
) -> Path:
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    people_path = input_dir / "people.json"
    families_path = input_dir / "families.json"
    if not people_path.exists() or not families_path.exists():
        raise FileNotFoundError(
            "Expected people.json and families.json in input directory. "
            f"Missing in: {input_dir}"
        )

    people = json.loads(people_path.read_text(encoding="utf-8"))
    families = json.loads(families_path.read_text(encoding="utf-8"))
    output_format = str(output_format).lower()

    if output_format == "dot":
        resolved_output = (
            Path(output_path).expanduser().resolve()
            if output_path
            else input_dir / "tree.dot"
        )
        resolved_output.write_text(
            _build_dot_from_people_and_families(people, families),
            encoding="utf-8",
        )
        return resolved_output

    if output_format == "json":
        resolved_output = (
            Path(output_path).expanduser().resolve()
            if output_path
            else input_dir / "tree.json"
        )
        payload = {"people": people, "families": families}
        resolved_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return resolved_output

    if output_format == "gedcom":
        resolved_output = (
            Path(output_path).expanduser().resolve()
            if output_path
            else input_dir / "tree.ged"
        )
        resolved_output.write_text(
            _build_gedcom_from_people_and_families(people, families),
            encoding="utf-8",
        )
        return resolved_output

    if output_format == "html":
        resolved_output = (
            Path(output_path).expanduser().resolve()
            if output_path
            else input_dir / "tree.html"
        )
        dot_path = input_dir / "tree.dot"
        if dot_path.exists():
            dot_graph = dot_path.read_text(encoding="utf-8")
        else:
            dot_graph = _build_dot_from_people_and_families(people, families)
        resolved_output.write_text(
            _build_html_from_people_and_families(people, families, dot_graph),
            encoding="utf-8",
        )
        return resolved_output

    raise ValueError(f"Unsupported export format: {output_format}")


escape_dot = _escape_dot
try_generate_svg_from_dot = _try_generate_svg_from_dot
build_dot_from_people_and_families = _build_dot_from_people_and_families
build_gedcom_from_people_and_families = _build_gedcom_from_people_and_families
build_html_from_people_and_families = _build_html_from_people_and_families
store_to_people_and_families = _store_to_people_and_families
