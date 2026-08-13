from dataclasses import dataclass, asdict
from typing import List, Optional

from blur_detection import check_image_quality
from ocr_module import extract_text, get_active_engine
from nlp_evaluator import (
    semantic_similarity,
    keyword_match,
    extract_keywords_from_model_answer,
    answer_length_ratio,
)
from scoring_engine import evaluate, EvaluationResult

@dataclass
class PipelineResult:
    status: str 
    message: str
    blur_score: Optional[float] = None
    ocr_text: Optional[str] = None
    ocr_engine: Optional[str] = None
    evaluation: Optional[dict] = None

def evaluate_answer_sheet(
    image_path: str,
    model_answer: str,
    keywords: Optional[List[str]] = None,
    max_marks: float = 10.0,
    blur_threshold: float = 100.0,
) -> PipelineResult:
    # Quality gate — block blurry uploads before wasting OCR/NLP compute
    quality = check_image_quality(image_path, threshold=blur_threshold)
    if not quality["accepted"]:
        return PipelineResult(
            status="REJECTED_BLURRY",
            message=quality["message"],
            blur_score=quality["blur_score"],
        )

    # OCR
    try:
        student_text = extract_text(image_path)
    except Exception as e:
        return PipelineResult(
            status="ERROR",
            message=f"OCR failed: {e}",
            blur_score=quality["blur_score"],
        )

    if not student_text.strip():
        return PipelineResult(
            status="ERROR",
            message=(
                "No readable text could be extracted from this image, even "
                "though it passed the blur check. Please ensure the answer "
                "is written clearly and re-upload."
            ),
            blur_score=quality["blur_score"],
            ocr_text="",
            ocr_engine=get_active_engine(),
        )

    # NLP comparison
    if not keywords:
        keywords = extract_keywords_from_model_answer(model_answer)

    similarity, sim_method = semantic_similarity(student_text, model_answer)
    kw_result = keyword_match(student_text, keywords)
    length_ratio = answer_length_ratio(student_text, model_answer)

    result: EvaluationResult = evaluate(
        similarity=similarity,
        similarity_method=sim_method,
        keyword_result=kw_result,
        length_ratio=length_ratio,
        max_marks=max_marks,
    )

    return PipelineResult(
        status="SUCCESS",
        message="Evaluation complete.",
        blur_score=quality["blur_score"],
        ocr_text=student_text,
        ocr_engine=get_active_engine(),
        evaluation=asdict(result),
    )

def _print_result(result: PipelineResult):
    print(f"\nStatus: {result.status}")
    print(f"Message: {result.message}")
    if result.blur_score is not None:
        print(f"Blur/sharpness score: {result.blur_score}")

    if result.status == "REJECTED_BLURRY":
        return

    if result.ocr_text is not None:
        print(f"\nOCR engine used: {result.ocr_engine}")
        print(f"Extracted answer text:\n{result.ocr_text}")

    if result.evaluation:
        ev = result.evaluation
        print(f"\n--- Evaluation ---")
        print(f"Similarity ({ev['similarity_method']}): {ev['similarity']}")
        print(f"Keyword coverage: {ev['keyword_coverage']}")
        print(f"Matched keywords: {ev['matched_keywords']}")
        print(f"Missing keywords: {ev['missing_keywords']}")
        print(f"Marks: {ev['marks']} / {ev['max_marks']}")
        print(f"Feedback: {ev['feedback']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a handwritten answer sheet image.")
    parser.add_argument("image_path", help="Path to the scanned/photographed answer image")
    parser.add_argument("model_answer_file", help="Path to a .txt file containing the model answer")
    parser.add_argument("--keywords", nargs="*", default=None, help="Optional explicit keyword list")
    parser.add_argument("--max-marks", type=float, default=10.0)
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    args = parser.parse_args()

    with open(args.model_answer_file, "r") as f:
        model_answer_text = f.read()

    outcome = evaluate_answer_sheet(
        image_path=args.image_path,
        model_answer=model_answer_text,
        keywords=args.keywords,
        max_marks=args.max_marks,
        blur_threshold=args.blur_threshold,
    )
    _print_result(outcome)