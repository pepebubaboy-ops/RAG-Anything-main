from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_import_raganything_baseline() -> None:
    module = importlib.import_module("raganything")
    assert module.__version__ == "0.1.0"


def test_pyproject_package_name() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "genealogy-rag-core"' in pyproject


def test_env_example_has_no_cloud_or_legacy_defaults() -> None:
    env_example = (ROOT / "env.example").read_text(encoding="utf-8")
    forbidden = (
        "api" + ".openai.com",
        "gpt" + "-4o",
        "neo4j+s" + "://",
        "LIGHT" + "RAG_",
    )
    for marker in forbidden:
        assert marker not in env_example


def test_legacy_cleanup_artifacts_are_absent() -> None:
    assert not (ROOT / ".github/workflows/linting.yaml").exists()
    assert not (ROOT / "chatgpt_pipeline_review_package.zip").exists()
    assert not (ROOT / "chatgpt_pipeline_review_package").exists()


def test_text_files_are_not_collapsed_to_one_line() -> None:
    paths = [
        "pyproject.toml",
        "README.md",
        "NOTICE.md",
        "docs/ARCHITECTURE.md",
        "docs/ROADMAP.md",
        ".gitignore",
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
        "env.example",
        "raganything/__init__.py",
        "raganything/cli.py",
        "raganything/genealogy/models.py",
        "tests/test_repo_hygiene.py",
    ]
    for relative in paths:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert text.count("\n") >= 3, f"{relative} appears collapsed"
