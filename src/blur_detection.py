import cv2
import numpy as np

def compute_blur_score(image_path: str) -> float:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)

def is_blurry(image_path: str, threshold: float = 100.0) -> tuple[bool, float]:
    score = compute_blur_score(image_path)
    return score < threshold, score

def check_image_quality(image_path: str, threshold: float = 100.0) -> dict:
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
