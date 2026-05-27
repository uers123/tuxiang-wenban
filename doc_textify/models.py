from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


BlockType = Literal[
    "title",
    "heading",
    "paragraph",
    "list",
    "table",
    "figure",
    "formula",
    "footer",
    "header",
    "uncertain",
    "placeholder",
]


@dataclass(slots=True)
class Block:
    type: BlockType
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    engine: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data


@dataclass(slots=True)
class Page:
    number: int
    width: float | None = None
    height: float | None = None
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ordered_blocks(self) -> list[Block]:
        return sorted(self.blocks, key=_reading_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "warnings": self.warnings,
            "blocks": [block.to_dict() for block in self.ordered_blocks()],
        }


@dataclass(slots=True)
class Document:
    source: Path
    pages: list[Page] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "metadata": self.metadata,
            "warnings": self.warnings,
            "pages": [page.to_dict() for page in self.pages],
        }


def _reading_key(block: Block) -> tuple[float, float, int]:
    if not block.bbox:
        return (0.0, 0.0, _type_priority(block.type))
    x0, y0, _x1, _y1 = block.bbox
    return (round(y0 / 12) * 12, x0, _type_priority(block.type))


def _type_priority(block_type: BlockType) -> int:
    order = {
        "title": 0,
        "heading": 1,
        "paragraph": 2,
        "list": 2,
        "table": 3,
        "figure": 4,
        "formula": 4,
        "uncertain": 5,
        "placeholder": 6,
        "footer": 7,
        "header": 7,
    }
    return order.get(block_type, 9)
