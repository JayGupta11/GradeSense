import io
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import pymupdf as fitz

from blur_detection import check_image_quality
from ocr_module import (
    extract_text,
    extract_text_column_aware,
    extract_printed_text,
)

@dataclass
class PageOCRResult:
    page_number: int
    text: str
    blur_score: float
    quality_warning: bool

def extract_pdf_text(
    pdf_bytes: bytes,
    ocr_blur_threshold: float = 80.0,
    dpi: int = 200,
) -> str:
    """
    Extract text from a PDF expected to contain TYPED/PRINTED content
    such as a question paper or answer key.

    The built-in PDF text layer is preferred. If it is absent, the page is
    sent to Qwen's printed-document OCR prompt, NOT the handwritten-answer
    prompt.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    page_texts = []

    for page in doc:
        text_layer = page.get_text().strip()

        if len(text_layer) >= 20:
            page_texts.append(text_layer)
            continue

        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        try:
            check_image_quality(
                tmp_path,
                threshold=ocr_blur_threshold,
            )
            page_texts.append(extract_printed_text(tmp_path))
        finally:
            os.unlink(tmp_path)

    doc.close()
    return "\n".join(page_texts)

def pdf_to_page_images(
    pdf_bytes: bytes,
    dpi: int = 200,
) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    page_images = []

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        page_images.append(pix.tobytes("png"))

    doc.close()
    return page_images

def ocr_pdf(
    pdf_bytes: bytes,
    blur_threshold: float = 80.0,
    dpi: int = 200,
    use_layout_aware: bool = True,
) -> Tuple[str, List[PageOCRResult]]:
    """
    OCR a student answer-sheet PDF.

    These pages use the handwritten-answer Qwen prompt.
    """
    page_images = pdf_to_page_images(pdf_bytes, dpi=dpi)

    results: List[PageOCRResult] = []
    full_text_parts = []

    for i, img_bytes in enumerate(page_images, start=1):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png",
        ) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        try:
            quality = check_image_quality(
                tmp_path,
                threshold=blur_threshold,
            )

            if use_layout_aware:
                page_text = extract_text_column_aware(tmp_path)
            else:
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
    return os.path.splitext(os.path.basename(filename))[0]
