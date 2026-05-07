from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from raganything.batch_parser import BatchParser


@pytest.mark.asyncio
async def test_process_batch_async_accepts_kwargs_on_dry_run(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.7")

    parser = BatchParser(
        parser_type="mineru",
        max_workers=1,
        show_progress=False,
        skip_installation_check=True,
    )

    result = await parser.process_batch_async(
        file_paths=[str(file_path)],
        output_dir=str(tmp_path / "out"),
        dry_run=True,
        foo="bar",
    )

    assert result.dry_run is True
    assert result.total_files == 1
    assert result.failed_files == []


def test_process_batch_marks_timeout_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    file_a.write_bytes(b"%PDF-1.7")
    file_b.write_bytes(b"%PDF-1.7")

    parser = BatchParser(
        parser_type="mineru",
        max_workers=2,
        show_progress=False,
        timeout_per_file=42,
        skip_installation_check=True,
    )

    seen_kwargs: dict[str, dict[str, object]] = {}

    def fake_parse_document(
        file_path: str,
        output_dir: str,
        method: str = "auto",
        **kwargs,
    ) -> list[dict[str, str]]:
        seen_kwargs[file_path] = kwargs
        if Path(file_path).name == "a.pdf":
            raise TimeoutError("timed out after 42 seconds")
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(parser.parser, "parse_document", fake_parse_document)

    result = parser.process_batch(
        file_paths=[str(file_a), str(file_b)],
        output_dir=str(tmp_path / "out"),
    )

    assert str(file_b) in result.successful_files
    assert str(file_a) in result.failed_files
    assert "timed out" in result.errors[str(file_a)].lower()
    assert seen_kwargs[str(file_a)]["timeout_sec"] == 42
    assert seen_kwargs[str(file_b)]["timeout_sec"] == 42


def test_process_batch_uses_unique_deterministic_output_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_one = tmp_path / "dir1" / "same.pdf"
    file_two = tmp_path / "dir2" / "same.pdf"
    file_one.parent.mkdir(parents=True, exist_ok=True)
    file_two.parent.mkdir(parents=True, exist_ok=True)
    file_one.write_bytes(b"%PDF-1.7")
    file_two.write_bytes(b"%PDF-1.7")

    parser = BatchParser(
        parser_type="mineru",
        max_workers=2,
        show_progress=False,
        skip_installation_check=True,
    )

    output_dirs: dict[str, str] = {}

    def fake_parse_document(
        file_path: str,
        output_dir: str,
        method: str = "auto",
        **kwargs,
    ) -> list[dict[str, str]]:
        output_dirs[file_path] = output_dir
        return [{"type": "text", "text": "ok"}]

    monkeypatch.setattr(parser.parser, "parse_document", fake_parse_document)

    result = parser.process_batch(
        file_paths=[str(file_one), str(file_two)],
        output_dir=str(tmp_path / "out"),
    )

    assert result.failed_files == []

    actual_dir_names = {Path(path).name for path in output_dirs.values()}
    expected_dir_names = {
        f"{path.stem}-{hashlib.md5(str(path.resolve()).encode('utf-8')).hexdigest()[:8]}"
        for path in [file_one, file_two]
    }
    assert actual_dir_names == expected_dir_names
