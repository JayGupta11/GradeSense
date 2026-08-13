"""
blur_detection.py
------------------
Detects whether an uploaded answer-sheet image is too blurry to be OCR'd
reliably. Uses the variance of the Laplacian (a standard, fast, no-training
blur-detection technique): sharp images have high-frequency edges that
produce high variance after a Laplacian (2nd derivative) filter; blurry
images lose those edges and produce low variance.
"""

import cv2
import numpy as np


def compute_blur_score(image_path: str) -> float:
    """
    Returns the Laplacian variance of the image (higher = sharper).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def is_blurry(image_path: str, threshold: float = 100.0) -> tuple[bool, float]:
    """
    Returns (blurry: bool, score: float).

    threshold: tune this for your camera/scanner setup.
        - Scanned documents / good phone cameras: 100-150 works well.
        - Low-light or older phone cameras: consider lowering to 60-80.
    """
    score = compute_blur_score(image_path)
    return score < threshold, score


def check_image_quality(image_path: str, threshold: float = 100.0) -> dict:
    """
    High-level check used by the upload pipeline.
    Returns a dict the API/UI layer can use directly to accept/reject
    the uploaded image.
    """
    blurry, score = is_blurry(image_path, threshold)

    if blurry:
        return {
            "accepted": False,
            "blur_score": round(score, 2),
            "flag": "BLURRY_IMAGE",
            "message": (
                "The uploaded image appears too blurry to read reliably "
                f"(sharpness score: {round(score, 2)}, minimum required: {threshold}). "
                "Please retake the photo in good lighting, hold the camera "
                "steady, and ensure the full answer sheet is in focus, then "
                "upload again."
            ),
        }

    return {
        "accepted": True,
        "blur_score": round(score, 2),
        "flag": None,
        "message": "Image quality OK. Proceeding to OCR.",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blur_detection.py <image_path>")
        sys.exit(1)

    result = check_image_quality(sys.argv[1])
    print(result)