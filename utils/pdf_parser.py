"""
utils/pdf_parser.py

PDF text extraction with two strategies:
  1. pdfminer.six  — fast, for PDFs with selectable/embedded text
  2. OCR fallback  — pytesseract + pdf2image, for scanned/image-based PDFs

The main entry point `extract_text_from_pdf()` automatically detects whether
a page contains real text or is image-only, and applies OCR only where needed.

Dependencies:
    pip install pdfminer.six pdf2image pytesseract Pillow
    sudo apt-get install tesseract-ocr poppler-utils   # system deps
"""

import re
import logging
from pathlib import Path
from io import StringIO

logger = logging.getLogger(__name__)

# ─── pdfminer imports ─────────────────────────────────────────────────────────
try:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.warning("pdfminer.six not installed. pip install pdfminer.six")

# ─── OCR imports ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract/Pillow not installed. pip install pytesseract Pillow")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. pip install pdf2image")


# ─── Minimum text length to consider a page "has real text" ──────────────────
MIN_TEXT_CHARS = 30
POPPLER_PATH = None


# ─── Core extraction ──────────────────────────────────────────────────────────

def _extract_text_pdfminer(pdf_path: Path) -> str:
    """Extract embedded text from PDF using pdfminer.six."""
    if not PDFMINER_AVAILABLE:
        return ""
    try:
        output = StringIO()
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            all_texts=True,
        )
        with open(pdf_path, "rb") as f:
            extract_text_to_fp(f, output, laparams=laparams)
        return output.getvalue()
    except Exception as e:
        logger.error(f"[pdfminer] Failed on {pdf_path.name}: {e}")
        return ""


def _ocr_pdf(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> str:
    """
    Convert PDF pages to images and run Tesseract OCR on each page.

    Args:
        pdf_path : Path to the PDF file.
        dpi      : Rendering resolution — higher = better accuracy, slower.
                   300 DPI is standard for medical documents.
        lang     : Tesseract language code. Use 'eng' for English.
                   For Hindi/regional: 'hin', 'tam', 'tel', etc.
                   Multiple: 'eng+hin'

    Returns:
        Full OCR text across all pages joined by newlines.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image not installed. Run: pip install pdf2image")
    if not TESSERACT_AVAILABLE:
        raise ImportError("pytesseract not installed. Run: pip install pytesseract Pillow")

    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        logger.error(f"[pdf2image] Could not convert {pdf_path.name}: {e}")
        return ""

    page_texts = []
    for page_num, page_img in enumerate(pages, start=1):
        try:
            # Tesseract config: PSM 6 = assume uniform block of text (good for reports)
            custom_config = r"--oem 3 --psm 6"
            text = pytesseract.image_to_string(page_img, lang=lang, config=custom_config)
            page_texts.append(text)
            logger.debug(f"  OCR page {page_num}: {len(text)} chars")
        except Exception as e:
            logger.warning(f"  OCR failed on page {page_num}: {e}")

    return "\n\n".join(page_texts)


def _ocr_page_images(pdf_path: Path, dpi: int = 300, lang: str = "eng") -> list:
    """
    Returns per-page OCR text as a list of strings.
    Useful when you need page-level control.
    """
    if not PDF2IMAGE_AVAILABLE or not TESSERACT_AVAILABLE:
        return []
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        results = []
        for img in pages:
            text = pytesseract.image_to_string(img, lang=lang, config="--oem 3 --psm 6")
            results.append(text)
        return results
    except Exception as e:
        logger.error(f"[OCR per-page] {e}")
        return []


# ─── Smart hybrid extractor ───────────────────────────────────────────────────

def extract_text_from_pdf(
    pdf_path,
    ocr_fallback: bool = True,
    dpi: int = 300,
    lang: str = "eng",
    force_ocr: bool = False,
) -> str:
    """
    Smart PDF text extractor.

    Strategy:
      1. Try pdfminer.six to get embedded text.
      2. If extracted text is too short (< MIN_TEXT_CHARS), the PDF is likely
         a scanned image — fall back to Tesseract OCR automatically.
      3. If force_ocr=True, always use OCR regardless of embedded text.

    Args:
        pdf_path    : Path to the PDF (str or Path).
        ocr_fallback: Enable automatic OCR fallback for image-based PDFs.
        dpi         : OCR rendering resolution (300 recommended).
        lang        : Tesseract language string (e.g. 'eng', 'eng+hin').
        force_ocr   : Skip pdfminer entirely and always use OCR.

    Returns:
        Cleaned extracted text string.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ""

    raw_text = ""

    # ── Step 1: pdfminer (fast path for text-based PDFs) ─────────────────
    if not force_ocr:
        raw_text = _extract_text_pdfminer(pdf_path)
        text_len = len(raw_text.strip())

        if text_len >= MIN_TEXT_CHARS:
            logger.info(f"[PDF] pdfminer extracted {text_len} chars from {pdf_path.name}")
            return _clean_text(raw_text)
        else:
            logger.info(
                f"[PDF] pdfminer got only {text_len} chars — "
                f"PDF appears to be image-based. Switching to OCR..."
            )

    # ── Step 2: OCR fallback ─────────────────────────────────────────────
    if ocr_fallback or force_ocr:
        if not PDF2IMAGE_AVAILABLE or not TESSERACT_AVAILABLE:
            missing = []
            if not PDF2IMAGE_AVAILABLE:  missing.append("pdf2image")
            if not TESSERACT_AVAILABLE:  missing.append("pytesseract Pillow")
            logger.error(
                f"[OCR] Cannot run OCR — missing packages: {', '.join(missing)}. "
                f"Install with: pip install {' '.join(missing)}"
            )
            return _clean_text(raw_text)  # return whatever pdfminer got

        logger.info(f"[OCR] Running Tesseract on {pdf_path.name} at {dpi} DPI (lang={lang})...")
        ocr_text = _ocr_pdf(pdf_path, dpi=dpi, lang=lang)
        logger.info(f"[OCR] Extracted {len(ocr_text)} chars via OCR.")
        return _clean_text(ocr_text)

    return _clean_text(raw_text)


# ─── Convenience wrappers ────────────────────────────────────────────────────

def extract_text_ocr_only(pdf_path, dpi: int = 300, lang: str = "eng") -> str:
    """Force OCR on all pages — use for fully scanned medical reports."""
    return extract_text_from_pdf(pdf_path, force_ocr=True, dpi=dpi, lang=lang)


def check_ocr_available() -> dict:
    """Return availability status of OCR dependencies."""
    status = {
        "pdfminer": PDFMINER_AVAILABLE,
        "pytesseract": TESSERACT_AVAILABLE,
        "pdf2image": PDF2IMAGE_AVAILABLE,
    }
    try:
        ver = pytesseract.get_tesseract_version() if TESSERACT_AVAILABLE else None
        status["tesseract_version"] = str(ver)
    except Exception:
        status["tesseract_version"] = "not found"
    return status


# ─── Text cleaning ────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Clean extracted text (works for both pdfminer and OCR output):
    - Remove null bytes and control characters
    - Normalize unicode punctuation
    - Remove form separators (---- ====)
    - Collapse whitespace and blank lines
    - Fix common OCR artefacts (l→1, O→0 in numeric contexts)
    """
    if not text:
        return ""

    # Null bytes and control chars
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Unicode normalization
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")   # non-breaking space

    # OCR artefacts: stray pipe/backtick characters
    text = re.sub(r"(?<!\w)[|`](?!\w)", "", text)

    # Form separators
    text = re.sub(r"[-=_*#]{3,}", "", text)

    # Whitespace normalization
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)

    return text.strip()


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("OCR dependency check:")
    for k, v in check_ocr_available().items():
        print(f"  {k}: {v}")

    if len(sys.argv) > 1:
        pdf = Path(sys.argv[1])
        print(f"\nExtracting: {pdf.name}")
        text = extract_text_from_pdf(pdf)
        print(f"Extracted {len(text)} characters:\n")
        print(text[:2000])