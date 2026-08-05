"""Formula OCR: LaTeX recognition for detected formula regions.

Two backends, tried in order:
  1. pix2tex (LaTeX-OCR): ML-based LaTeX recognition (optional)
  2. SimpleOCR fallback: Tesseract with equation config

Neither backend is required; graceful degradation returns raw text.
A rule-based reconstruction pass (``_reconstruct_latex``) converts common
OCR formula fragments into LaTeX (subscripts, Greek letters, fractions,
symbols) so scanned formulas degrade into readable ``$$...$$`` output
instead of raw OCR noise.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
#  Rule-based LaTeX reconstruction (robust path — no ML backend required)
# ---------------------------------------------------------------------------

#: Greek letter names → LaTeX macros (word-boundary replacements).
_GREEK_WORD_MAP = {
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
    "delta": r"\delta", "epsilon": r"\epsilon", "zeta": r"\zeta",
    "eta": r"\eta", "theta": r"\theta", "iota": r"\iota",
    "kappa": r"\kappa", "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "xi": r"\xi", "pi": r"\pi", "rho": r"\rho",
    "sigma": r"\sigma", "tau": r"\tau", "phi": r"\phi",
    "chi": r"\chi", "psi": r"\psi", "omega": r"\omega",
    "Gamma": r"\Gamma", "Delta": r"\Delta", "Theta": r"\Theta",
    "Lambda": r"\Lambda", "Xi": r"\Xi", "Pi": r"\Pi",
    "Sigma": r"\Sigma", "Phi": r"\Phi", "Psi": r"\Psi",
    "Omega": r"\Omega",
}

#: Unicode math glyphs → LaTeX tokens.
_SYMBOL_MAP = {
    "×": r"\times", "≈": r"\approx", "≠": r"\neq",
    "≤": r"\leq", "≥": r"\geq", "±": r"\pm", "÷": r"\div",
    "∞": r"\infty", "∑": r"\sum", "∫": r"\int", "√": r"\sqrt",
    "∂": r"\partial", "Δ": r"\Delta", "σ": r"\sigma",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma",
    "δ": r"\delta", "θ": r"\theta", "λ": r"\lambda",
    "μ": r"\mu", "π": r"\pi", "φ": r"\phi", "ρ": r"\rho",
    "τ": r"\tau", "η": r"\eta", "ω": r"\omega",
}

#: ASCII OCR tokens → LaTeX (domain vocabulary for geotechnical / CPT
#: formulas plus common Tesseract confusions observed on scanned pages).
_TOKEN_MAP = [
    # pore-pressure / cone-resistance vocabulary (OCR variants first)
    (r"\bqt\b", r"q_t"), (r"\bqc\b", r"q_c"), (r"\bq,(?=\s|$)", r"q_t"),
    (r"\bg,(?=\s|$)", r"q_t"), (r"\bQl\b", r"Q_t"), (r"\bQD:\b", r"Q_t ="),
    (r"\bfs\b", r"f_s"), (r"\bf,(?=\s|$)", r"f_s"), (r"\bRf\b", r"R_f"),
    (r"\bFr\b", r"F_r"), (r"\bBy\b", r"B_q"), (r"\bBq\b", r"B_q"),
    (r"\bB,(?=\s|$)", r"B_q"), (r"\bAu\b", r"\Delta u"),
    (r"\bCvo\b", r"\sigma_{v0}"), (r"\bovo\b", r"\sigma_{v0}"),
    (r"\bov0\b", r"\sigma_{v0}"), (r"\boy0\b", r"\sigma_{v0}"),
    (r"\bsv0\b", r"\sigma_{v0}"), (r"\bσv0\b", r"\sigma_{v0}"),
    (r"\bσ'v0\b", r"\sigma'_{v0}"), (r"\bOm\b", r"\sigma'_{v0}"),
    (r"\bve\b", r"\sigma_{v0}"), (r"\boy\.", r"\sigma_{v0}"),
    (r"\boy0", r"\sigma_{v0}"), (r"\bUM\b", r"f_s"),
    (r"\bu2\b", r"u_2"), (r"\bu0\b", r"u_0"), (r"\buo\b", r"u_0"),
    (r"\bup\b", r"u_0"), (r"\bQt\b", r"Q_t"),
]

#: punctuation noise that OCR sprinkles into scanned formulas.
_NOISE_RE = [
    re.compile(r"[\u2500-\u257f]+"),       # box-drawing junk
    re.compile(r"[~]{2,}"),
    re.compile(r"[─―━]+"),
]

_LATEX_MARKER_RE = re.compile(
    r"\\(?:frac|sigma|Delta|times|approx|alpha|beta|gamma|delta|theta|lambda|mu|pi|phi|rho|tau|eta|omega|sqrt|sum|int|partial|pm|div|neq|leq|geq|infty)"
    r"|_\\{\^|\^|_(?=[A-Za-z0-9{])"
)


def _reconstruct_latex(fragment_text: str) -> str:
    """Convert a common OCR formula fragment into LaTeX.

    Handles, in order:
      1. OCR noise removal (dashes, box-drawing, stray brackets).
      2. Greek letter names ("sigma", "Delta", …) → LaTeX macros.
      3. Clear fractions (a/b, X/(...) → \\frac).
      4. Domain vocabulary (qt→q_t, Bq→B_q, Au→Δu, ovo→σ_{v0}, …).
      5. Unicode math glyphs (× ≈ σ Δ …) → LaTeX macros.
      6. Generic single-letter/digit subscripts (u2→u_2).
      7. "x 100" → \\times 100, "%" → \\%.

    Returns the reconstructed string (possibly unchanged when the input
too noisy to interpret).
    """
    t = " ".join(fragment_text.split())
    if not t:
        return t

    # ── 1. noise removal ────────────────────────────────────────────────
    for pat in _NOISE_RE:
        t = pat.sub(" ", t)
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    t = re.sub(r"[\[\]{}](?=\s*[=,;]|$)", "", t)  # trailing stray brackets
    t = re.sub(r"\)\]", ")", t)                    # ")" + junk bracket
    t = re.sub(r"=\s*/\s*(?=\d)", "= ", t)         # line-wrap artifact "= / 5.6"
    t = t.replace("\u00b7", ".")

    # ── 2. greek letter names (word boundary; prime-aware) ──────────────
    def _greek_sub(m: re.Match) -> str:
        word = m.group(1)
        prime = m.group(2) or ""
        return _GREEK_WORD_MAP.get(word, word) + ("'" if prime else "")

    t = re.sub(
        r"\b(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega)(')?\b",
        _greek_sub, t,
    )

    # ── 3. clear fractions ──────────────────────────────────────────────
    #   A/B where A is a short plain token and B is a parenthesised group
    #   or a short plain token (fs/(qt-σv0) → \frac{fs}{(qt-σv0)}).
    #   Numeric ratios and expressions with spaces/parens in the numerator
    #   are left untouched.
    def _frac_sub(m: re.Match) -> str:
        num = m.group(1)
        den = m.group(2).strip()
        if re.fullmatch(r"\d+", num) and re.fullmatch(r"\d+", den):
            return f"{num}/{den}"   # plain numeric ratios stay as-is
        return r"\frac{%s}{%s}" % (num, den)

    t = re.sub(
        r"\b([A-Za-z0-9._]{1,24})\s*/(\s*\([^)]*\)|[A-Za-z0-9._]{1,24})",
        _frac_sub, t,
    )

    # ── 4. domain vocabulary (lambda avoids re-sub escape processing) ──
    # compound OCR-artifact first: "G - Cvo)" / "G - ve)" is "(qt - σv0)"
    t = re.sub(r"G\s*-\s*(?:Cvo|ve)\)", r"(q_t - \\sigma_{v0})", t)
    for pat, repl in _TOKEN_MAP:
        t = re.sub(pat, lambda m, r=repl: r, t)
    t = re.sub(r"=\s*=", "=", t)   # doubled equals from token fixes

    # ── 5. unicode math glyphs ──────────────────────────────────────────
    for ch, latex in _SYMBOL_MAP.items():
        t = t.replace(ch, latex)

    # ── 6. generic subscripts ───────────────────────────────────────────
    # single letter + digit (u2 → u_2); skip digits already inside LaTeX
    # braces from the vocabulary pass (\sigma_{v0} must not become
    # \sigma_{v_0}).  A generic Capital+lowercase rule is deliberately
    # avoided — it would corrupt ordinary words ("The" → "T_he").
    t = re.sub(r"(?<!\{)\b([A-Za-z])([0-9])\b", r"\1_{\2}", t)

    # ── 7. percent / multiplication artifacts ───────────────────────────
    t = re.sub(r"(?<![A-Za-z0-9])[xX]\s*100(?=\s*[%=]|$)", r"\\times 100", t)
    t = t.replace("%", r"\%")
    t = re.sub(r"\s+([=+\-×≈<>])\s+", r" \1 ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"(?<=\()\s+|\s+(?=\))", "", t)
    return t


def reconstruct_formula_text(raw: str) -> str:
    """Public helper: reconstruct LaTeX from a raw OCR formula fragment.

    Strips the ``[formula]`` prefix produced by :func:`recognize_formula`,
    runs :func:`_reconstruct_latex`, and returns the result when it contains
    recognizable LaTeX markers; otherwise returns the cleaned raw text so
    callers can degrade gracefully.
    """
    if not raw:
        return raw
    t = raw.strip()
    if t.startswith("[formula]"):
        t = t[len("[formula]"):].strip()
    latex = _reconstruct_latex(t)
    if _LATEX_MARKER_RE.search(latex):
        return latex
    return t


def looks_like_latex(text: str) -> bool:
    """Return True when *text* already contains recognizable LaTeX markers."""
    return bool(_LATEX_MARKER_RE.search(text or ""))


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
            latex = _reconstruct_latex(raw)
            if looks_like_latex(latex):
                return f"[formula] {latex}"
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
    """Use Tesseract to capture raw formula text.

    The region is upscaled (3×, minimum 240 px wide) before OCR because
    scanned formula glyphs are usually too small for stock Tesseract.
    Tries ``--psm 6`` first, then ``--psm 7`` when the first pass yields
    nothing.
    """
    import PIL.Image

    # Save the image region to a temp file for Tesseract CLI
    with tempfile.NamedTemporaryFile(
        prefix="doc_textify_formula_", suffix=".png", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Ensure RGB and upscale
        if image_region.mode not in ("RGB", "L"):
            image_region = image_region.convert("RGB")
        w, h = image_region.size
        scale = max(3, int(240 / max(w, 1)), int(80 / max(h, 1)))
        if scale > 1:
            image_region = image_region.resize((w * scale, h * scale), PIL.Image.LANCZOS)
        image_region.save(tmp_path)

        for psm in ("6", "7"):
            result = subprocess.run(
                [
                    "tesseract", str(tmp_path), "stdout",
                    "-l", "eng", "--psm", psm,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
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
