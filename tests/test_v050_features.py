"""Tests for v0.5.0 features: LaTeX formula reconstruction + chart point data."""

from __future__ import annotations

from doc_textify.formula_ocr import reconstruct_formula_text, _reconstruct_latex
from doc_textify.renderers import _render_block_deepseek, _formula_is_latex
from doc_textify.models import Block


# ── LaTeX reconstruction ────────────────────────────────────────────────

def test_reconstruct_subscripts_and_units() -> None:
    """OCR 'qt = qc + (1 - a) u2' → LaTeX subscripts."""
    out = reconstruct_formula_text("qt = qc + (1 - a) u2")
    assert "q_t" in out
    assert "q_c" in out
    assert "u_2" in out


def test_reconstruct_domain_vocabulary() -> None:
    """OCR 'By = Au / (qt - ovo)' → full domain-aware LaTeX."""
    out = reconstruct_formula_text("By = Au / (qt - ovo)")
    assert "B_q" in out
    assert r"\frac" in out
    assert r"\Delta u" in out
    assert r"\sigma_{v0}" in out


def test_reconstruct_greek_and_percent() -> None:
    """Greek letters and percent sign conversion."""
    out = _reconstruct_latex("sigma = 5%")
    assert r"\sigma" in out
    assert r"\%" in out


def test_reconstruct_keeps_plain_text() -> None:
    """Non-formula text should be returned cleaned but not mangled."""
    out = reconstruct_formula_text("FRICTION RATIO, R_f (%) .")
    # The '%' gets escaped but the rest stays readable
    assert "FRICTION RATIO" in out
    assert "R_f" in out


def test_formula_is_latex_detection() -> None:
    """_formula_is_latex distinguishes raw OCR from reconstructed LaTeX."""
    assert _formula_is_latex(r"B_q = \frac{\Delta u}{(q_t - \sigma_{v0})}")
    assert not _formula_is_latex("By = Au / (qt - ovo)")
    assert not _formula_is_latex("[formula] raw text")


# ── DeepSeek renderer formula path ─────────────────────────────────────

def test_deepseek_renderer_reconstructs_formula() -> None:
    """_render_block_deepseek should wrap reconstructed LaTeX in $$...$$."""
    block = Block(type="formula", text="qt = qc + (1 - a) u2")
    out = _render_block_deepseek(block, page_number=1)
    assert out.startswith("$$")
    assert out.endswith("$$")
    assert "q_t" in out
    assert "u_2" in out


def test_deepseek_renderer_keeps_existing_latex() -> None:
    """Already-LaTeX formula text should pass through untouched."""
    block = Block(type="formula", text=r"B_q = \frac{\Delta u}{(q_t - \sigma_{v0})}")
    out = _render_block_deepseek(block, page_number=1)
    assert out == r"$$B_q = \frac{\Delta u}{(q_t - \sigma_{v0})}$$"
