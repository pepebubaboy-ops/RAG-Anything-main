from __future__ import annotations

from raganything.modalprocessors import ContextConfig, ContextExtractor


def test_extract_context_reuses_page_text_index() -> None:
    extractor = ContextExtractor(
        config=ContextConfig(
            context_window=1,
            context_mode="page",
            include_headers=True,
            include_captions=True,
            filter_content_types=["text", "image", "table"],
        )
    )

    content_list = [
        {"type": "text", "text": "Intro", "text_level": 1, "page_idx": 0},
        {"type": "image", "image_caption": ["Cover"], "page_idx": 0},
        {"type": "text", "text": "Body", "page_idx": 1},
        {"type": "table", "table_caption": ["Metrics"], "page_idx": 2},
    ]

    context_first = extractor.extract_context(
        content_list,
        {"page_idx": 1},
        content_format="minerU",
    )
    context_second = extractor.extract_context(
        content_list,
        {"page_idx": 1},
        content_format="minerU",
    )

    assert context_first == context_second
    assert extractor._page_index_build_count == 1
    assert "[Page 0]" in context_first
    assert "Body" in context_first
    assert "[Page 2]" in context_first
