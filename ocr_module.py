"""
ocr_module.py
-------------
Converts a (quality-checked, non-blurry) handwritten answer-sheet image
into digital text.

Primary engine : EasyOCR (deep-learning based, works reasonably well on
                handwriting compared to classic Tesseract, no extra
                system binary needed).
Fallback engine: pytesseract (classic Tesseract OCR) — used automatically
                if EasyOCR is not installed/available.

For production-grade handwriting OCR you would fine-tune a model such as
TrOCR (microsoft/trocr-base-handwritten) on the IAM Handwriting Database —
see README.md for dataset links and notes on that upgrade path.
"""

from typing import Optional
import cv2
import numpy as np

_easyocr_reader = None
_engine = None


def _preprocess_for_ocr(image_path: str) -> np.ndarray:
    """Light preprocessing: grayscale + adaptive threshold + denoise.
    Helps both EasyOCR and Tesseract read handwriting more reliably."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return gray


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # imported lazily so the project still runs w/o it

        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def extract_text(image_path: str, preprocess: bool = True) -> str:
    """
    Main entry point: returns the extracted text as a single string.
    Tries EasyOCR first, falls back to pytesseract if EasyOCR is missing.
    """
    global _engine

    source = _preprocess_for_ocr(image_path) if preprocess else image_path

    # --- Try EasyOCR ---
    try:
        reader = _get_easyocr_reader()
        _engine = "easyocr"
        results = reader.readtext(source, detail=0, paragraph=True)
        text = " ".join(results).strip()
        if text:
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[ocr_module] EasyOCR failed ({e}), falling back to Tesseract...")

    # --- Fallback: pytesseract ---
    try:
        import pytesseract
        from PIL import Image

        _engine = "tesseract"
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except ImportError:
        raise RuntimeError(
            "Neither easyocr nor pytesseract is installed. "
            "Install one of them: `pip install easyocr` or "
            "`pip install pytesseract` (plus the tesseract binary)."
        )


def get_active_engine() -> Optional[str]:
    return _engine


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_module.py <image_path>")
        sys.exit(1)

    print(f"Extracted text:\n{extract_text(sys.argv[1])}")
    print(f"\nEngine used: {get_active_engine()}")