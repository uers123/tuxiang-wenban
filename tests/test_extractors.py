from pathlib import Path

from doc_textify.extractors import _text_to_blocks
from doc_textify.models import Block, Page


def test_text_to_blocks_keeps_paragraphs() -> None:
    blocks = _text_to_blocks("Title\n\nFirst line\nsecond line", engine="test")

    assert [block.type for block in blocks] == ["heading", "paragraph"]
    assert blocks[1].text == "First line second line"


def test_text_to_blocks_detects_simple_tables() -> None:
    blocks = _text_to_blocks("Name  Amount\nAlice  42\nBob  7", engine="test")

    assert len(blocks) == 1
    assert blocks[0].type == "table"
    assert "| Name | Amount |" in blocks[0].text
    assert "| Alice | 42 |" in blocks[0].text


def test_page_orders_blocks_by_bbox() -> None:
    page = Page(
        number=1,
        blocks=[
            Block(type="paragraph", text="bottom", bbox=(0, 100, 10, 110)),
            Block(type="paragraph", text="top", bbox=(0, 10, 10, 20)),
        ],
    )

    assert [block.text for block in page.ordered_blocks()] == ["top", "bottom"]


def test_unsupported_path_type_raises(tmp_path: Path) -> None:
    source = tmp_path / "input.xyz"
    source.write_text("nope", encoding="utf-8")

    from doc_textify.extractors import extract_document

    try:
        extract_document(source)
    except ValueError as exc:
        assert "Unsupported input type" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
