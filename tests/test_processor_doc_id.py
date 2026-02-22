from __future__ import annotations

import hashlib
from typing import Any

from raganything.processor import ProcessorMixin


class DummyProcessor(ProcessorMixin):
    pass


def _legacy_generate_content_based_doc_id(content_list: list[dict[str, Any]]) -> str:
    content_hash_data = []

    for item in content_list:
        if isinstance(item, dict):
            if item.get("type") == "text" and item.get("text"):
                content_hash_data.append(item["text"].strip())
            elif item.get("type") == "image" and item.get("img_path"):
                content_hash_data.append(f"image:{item['img_path']}")
            elif item.get("type") == "table" and item.get("table_body"):
                content_hash_data.append(f"table:{item['table_body']}")
            elif item.get("type") == "equation" and item.get("text"):
                content_hash_data.append(f"equation:{item['text']}")
            else:
                content_hash_data.append(str(item))

    content_signature = "\n".join(content_hash_data)
    return f"doc-{hashlib.md5(content_signature.encode('utf-8')).hexdigest()}"


def test_generate_content_based_doc_id_matches_legacy_algorithm() -> None:
    content_list = [
        {"type": "text", "text": "  Hello world  "},
        {"type": "image", "img_path": "/tmp/image.png"},
        {"type": "table", "table_body": [["a", "b"], ["1", "2"]]},
        {"type": "equation", "text": "E=mc^2"},
        {"type": "text", "text": ""},
        {"type": "other", "value": 123},
    ]

    processor = DummyProcessor()

    expected = _legacy_generate_content_based_doc_id(content_list)
    actual = processor._generate_content_based_doc_id(content_list)

    assert actual == expected
