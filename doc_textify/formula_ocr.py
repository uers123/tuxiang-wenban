"""Formula OCR: LaTeX recognition for detected formula regions.

Two backends, tried in order:
  1. pix2tex (LaTeX-OCR): ML-based LaTeX recognition (optional)
  2. SimpleOCR fallback: Tesseract with equation config

Neither backend is required; graceful degradation returns raw text.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image


# ── Backend availability checks ────────────────────────────────────────

_pix2tex_cache: bool | None = None
_pix2tex_model_cache: object | None = None


def _pix2tex_available() -> bool:
    """Check if pix2tex (LaTeX-OCR) is importable."""
    global _pix2tex_cache
    if _pix2tex_cache is not None:
        return _pix2tex_cache
    try:
        import pix2tex.cli  # noqa: F401
        _pix2tex_cache = True
    except ImportError:
        _pix2tex_cache = False
    return _pix2tex_cache


def _tesseract_available() -> bool:
    """Check if tesseract is on PATH."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_formula_ocr_available() -> bool:
    """Return True if any formula OCR backend (pix2tex or Tesseract) is usable."""
    return _pix2tex_available() or _tesseract_available()


# ── Main recognition entry point ───────────────────────────────────────

def recognize_formula(image_region: "PIL.Image.Image") -> str | None:
    """Try to recognize a formula image region as LaTeX or raw text.

    Backends are tried in order; returns *None* when all backends fail.

    Args:
        image_region: A PIL Image cropped to the formula region.

    Returns:
        Recognized LaTeX string, or raw text with ``[formula]`` prefix, or
        ``None`` when every backend fails.
    """
    # Backend 1: pix2tex (ML-based LaTeX OCR)
    if _pix2tex_available():
        latex = _recognize_with_pix2tex(image_region)
        if latex:
            return latex

    # Backend 2: Tesseract in equation-like mode
    if _tesseract_available():
        raw = _recognize_with_tesseract_equation(image_region)
        if raw and raw.strip():
            return f"[formula] {raw.strip()}"

    return None


# ── pix2tex (LaTeX-OCR) backend ────────────────────────────────────────

def _load_pix2tex_model():
    """Lazily load the pix2tex model (downloads on first use)."""
    global _pix2tex_model_cache
    if _pix2tex_model_cache is not None:
        return _pix2tex_model_cache
    try:
        from pix2tex.cli import LatexOCR
        _pix2tex_model_cache = LatexOCR()
    except Exception:
        _pix2tex_model_cache = False
    return _pix2tex_model_cache


def _recognize_with_pix2tex(image_region: "PIL.Image.Image") -> str | None:
    """Attempt LaTeX recognition via pix2tex (LaTeX-OCR)."""
    model = _load_pix2tex_model()
    if model is False:
        return None
    try:
        result = model(image_region)
        if result and isinstance(result, str):
            return result.strip()
    except Exception:
        pass
    return None


# ── Tesseract equation-mode fallback ───────────────────────────────────

def _recognize_with_tesseract_equation(image_region: "PIL.Image.Image") -> str | None:
    """Use Tesseract with --psm 6 to capture raw formula text.

    Tries ``eng+equ`` (equation traineddata) first; falls back to ``eng``.
    """
    import PIL.Image

    # Save the image region to a temp file for Tesseract CLI
    with tempfile.NamedTemporaryFile(
        prefix="doc_textify_formula_", suffix=".png", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Ensure RGB
        if image_region.mode not in ("RGB", "L"):
            image_region = image_region.convert("RGB")
        image_region.save(tmp_path)

        # Try equ traineddata first
        for lang in ("eng+equ", "eng"):
            result = subprocess.run(
                [
                    "tesseract", str(tmp_path), "stdout",
                    "-l", lang, "--psm", "6",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                text = result.stdout.strip()
                if text:
                    return text_sub(text)
    except Exception:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)

    return None


# ── Helpers ────────────────────────────────────────────────────────────

def text_sub(raw: str) -> str:
    """Remove common Tesseract formula noise."""
    return raw.strip()
