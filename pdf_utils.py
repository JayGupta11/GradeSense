"""
pdf_utils.py
------------
Converts a multi-page PDF answer sheet (e.g. a scanned/photographed
answer booklet, all pages in one file) into a single block of OCR'd text,
ready to feed into paper_parser.parse_numbered_text().

Uses PyMuPDF (fitz) to rasterize each PDF page to an image — no external
poppler/ghostscript binary required (unlike pdf2image), which makes this
much easier to deploy alongside a Streamlit app.

Each page is:
    1. Rendered to an image
    2. Blur-checked (a soft warning in batch mode, not a hard reject —
        you don't want one badly-scanned page to void an entire student's
        submission; the warning is surfaced in the report instead)
    3. OCR'd (via ocr_module, same engine used for single-image uploads)

Pages are concatenated in order, so a student's numbered answers
("1.", "2.", "Ans 3:", etc.) can span across pages and still be parsed
correctly downstream.
"""

import io
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF

from blur_detection import check_image_quality
from ocr_module import extract_text


@dataclass
class PageOCRResult:
    page_number: int
    text: str
    blur_score: float
    quality_warning: bool


def extract_pdf_text(pdf_bytes: bytes, ocr_blur_threshold: float = 80.0, dpi: int = 200) -> str:
    """
    Extracts text from a PDF that is expected to be mostly TYPED/printed
    (e.g. a question paper or answer key), not handwritten.

    Tries the PDF's built-in text layer first (fast, exact, no OCR errors)
    for each page. If a page's text layer is empty or near-empty (e.g. the
    question paper was itself scanned as an image rather than exported as
    text), that page automatically falls back to OCR instead — so this one
    function safely handles both "real" digital PDFs and scanned ones.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    page_texts = []
    for page in doc:
        text_layer = page.get_text().strip()
        if len(text_layer) >= 20:  # real text layer present, use it directly
            page_texts.append(text_layer)
            continue

        # Near-empty text layer -> this page is likely a scanned image; OCR it
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            quality = check_image_quality(tmp_path, threshold=ocr_blur_threshold)
            page_texts.append(extract_text(tmp_path))
        finally:
            os.unlink(tmp_path)

    doc.close()
    return "\n".join(page_texts)


def pdf_to_page_images(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:
    """
    Renders every page of a PDF (given as raw bytes) to PNG image bytes.
    dpi=200 is a good balance of OCR accuracy vs. speed/memory for
    typical scanned answer sheets.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72  # PDF default is 72 dpi
    matrix = fitz.Matrix(zoom, zoom)

    page_images = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        page_images.append(pix.tobytes("png"))
    doc.close()
    return page_images


def ocr_pdf(pdf_bytes: bytes, blur_threshold: float = 80.0, dpi: int = 200) -> Tuple[str, List[PageOCRResult]]:
    """
    Full pipeline: PDF bytes -> per-page OCR -> concatenated text.

    Returns (full_text, per_page_results). blur_threshold is lower than
    the single-image-upload default (100) because batch mode shouldn't
    be overly strict — a borderline page still gets OCR'd, just flagged
    with quality_warning=True so it shows up in the class report instead
    of silently failing.
    """
    page_images = pdf_to_page_images(pdf_bytes, dpi=dpi)
    results: List[PageOCRResult] = []
    full_text_parts = []

    for i, img_bytes in enumerate(page_images, start=1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        try:
            quality = check_image_quality(tmp_path, threshold=blur_threshold)
            page_text = extract_text(tmp_path)
            results.append(
                PageOCRResult(
                    page_number=i,
                    text=page_text,
                    blur_score=quality["blur_score"],
                    quality_warning=not quality["accepted"],
                )
            )
            full_text_parts.append(page_text)
        finally:
            os.unlink(tmp_path)

    return "\n".join(full_text_parts), results


def roll_number_from_filename(filename: str) -> str:
    """
    Extracts a roll number/student ID from an uploaded filename, e.g.
    '12345.pdf' -> '12345', 'Roll_2201.pdf' -> 'Roll_2201'.
    Simply strips the extension — students are expected to name (or be
    given) files as <roll_number>.pdf.
    """
    return os.path.splitext(os.path.basename(filename))[0]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_utils.py <answer_sheet.pdf>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        pdf_bytes = f.read()

    text, page_results = ocr_pdf(pdf_bytes)
    for pr in page_results:
        flag = " [LOW QUALITY]" if pr.quality_warning else ""
        print(f"--- Page {pr.page_number} (sharpness={pr.blur_score}){flag} ---")
        print(pr.text)
        print()