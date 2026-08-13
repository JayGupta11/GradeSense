import io
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

# PyMuPDF
import fitz  

from blur_detection import check_image_quality
from ocr_module import extract_text

@dataclass
class PageOCRResult:
    page_number: int
    text: str
    blur_score: float
    quality_warning: bool

def extract_pdf_text(pdf_bytes: bytes, ocr_blur_threshold: float = 80.0, dpi: int = 200) -> str:
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