"""AI-readable PDF/image textification without vision LLMs.

Phases:
  Phase 1 - OCR pipeline (extractors)
  Phase 2 - Layout analysis (layout)
  Phase 3 - Chart understanding (chart)
"""

from .models import Block, Document, Page

__version__ = "0.3.1"

__all__ = ["Block", "Document", "Page", "__version__"]
