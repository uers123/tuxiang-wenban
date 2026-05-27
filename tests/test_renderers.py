from pathlib import Path

from doc_textify.models import Block, Document, Page
from doc_textify.renderers import render_json, render_llm_text, render_markdown, render_text


def test_markdown_contract_renders_page_blocks() -> None:
    document = Document(
        source=Path("sample.pdf"),
        pages=[
            Page(
                number=1,
                blocks=[
                    Block(type="heading", text="Invoice"),
                    Block(type="paragraph", text="Total: 42"),
                    Block(type="figure", text="", bbox=(1, 2, 3, 4)),
                    Block(type="uncertain", text="T0ta1"),
                ],
            )
        ],
    )

    markdown = render_markdown(document)

    assert "# Page 1" in markdown
    assert "## Invoice" in markdown
    assert "Total: 42" in markdown
    assert "[Figure: page=1, bbox=1,2,3,4, caption=not detected]" in markdown
    assert "[uncertain: T0ta1]" in markdown


def test_text_renderer_is_plain_text() -> None:
    document = Document(source=Path("sample.png"), pages=[Page(number=1, blocks=[Block(type="paragraph", text="Hello")])])

    text = render_text(document)

    assert "Page 1" in text
    assert "Hello" in text


def test_json_sidecar_contains_blocks() -> None:
    document = Document(
        source=Path("sample.png"),
        pages=[Page(number=1, blocks=[Block(type="paragraph", text="Hello", confidence=99.5)])],
    )

    data = render_json(document)

    assert '"source": "sample.png"' in data
    assert '"confidence": 99.5' in data


def test_llm_renderer_compacts_chart_data() -> None:
    document = Document(
        source=Path("chart.png"),
        pages=[
            Page(
                number=1,
                width=100,
                height=200,
                blocks=[
                    Block(
                        type="figure",
                        text="depth/class chart",
                        metadata={
                            "chart_data": [
                                {
                                    "type": "interval",
                                    "panel_id": "a",
                                    "class": 1,
                                    "start_depth": 2.0,
                                    "end_depth": 5.0,
                                },
                                {"type": "point", "panel_id": "a", "class": 2, "depth": 3.5},
                            ]
                        },
                    )
                ],
            )
        ],
    )

    text = render_llm_text(document)

    assert "DOC_TEXTIFY_LLM_PROTOCOL v1" in text
    assert "panel a:" in text
    assert "class 1 depth 2.0-5.0" in text
    assert "class 2 depth 3.5" in text
