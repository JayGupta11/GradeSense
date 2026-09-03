from dataclasses import dataclass, field
from typing import List

@dataclass
class EvaluationResult:
    similarity: float
    similarity_method: str
    keyword_coverage: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    length_ratio: float
    marks: float
    max_marks: float
    feedback: str

def generate_score(
    similarity: float,
    keyword_coverage: float,
    length_ratio: float,
    max_marks: float = 10.0,
    weight_similarity: float = 0.55,
    weight_keywords: float = 0.35,
    weight_length: float = 0.10,
    min_length_ratio_for_full_credit: float = 0.4,
) -> float:
    length_factor = min(length_ratio / min_length_ratio_for_full_credit, 1.0)

    raw_score = (
        similarity * weight_similarity
        + keyword_coverage * weight_keywords
        + length_factor * weight_length
    )
    raw_score = max(0.0, min(1.0, raw_score))
    return round(raw_score * max_marks, 2)

def generate_feedback(
    similarity: float,
    keyword_coverage: float,
    matched_keywords: List[str],
    missing_keywords: List[str],
    length_ratio: float,
) -> str:
    parts = []

    if similarity >= 0.8:
        parts.append("Excellent alignment with the expected answer.")
    elif similarity >= 0.6:
        parts.append("Good understanding shown overall; some details could be sharper.")
    elif similarity >= 0.4:
        parts.append("Partial understanding — the answer touches the topic but lacks depth.")
    else:
        parts.append("The answer deviates significantly from what was expected.")

    if missing_keywords:
        shown = ", ".join(missing_keywords[:6])
        parts.append(f"Missing or unclear on key terms: {shown}.")
    elif matched_keywords:
        parts.append("All key terms/concepts were covered.")

    if length_ratio < 0.3:
        parts.append("The answer seems too short — consider elaborating further.")
    elif length_ratio > 1.3:
        parts.append("The answer is quite long relative to what was expected; try to be more concise.")

    return " ".join(parts)

def evaluate(
    similarity: float,
    similarity_method: str,
    keyword_result: dict,
    length_ratio: float,
    max_marks: float = 10.0,
) -> EvaluationResult:
    marks = generate_score(
        similarity=similarity,
        keyword_coverage=keyword_result["coverage"],
        length_ratio=length_ratio,
        max_marks=max_marks,
    )
    feedback = generate_feedback(
        similarity=similarity,
        keyword_coverage=keyword_result["coverage"],
        matched_keywords=keyword_result["matched"],
        missing_keywords=keyword_result["missing"],
        length_ratio=length_ratio,
    )
    return EvaluationResult(
        similarity=round(similarity, 3),
        similarity_method=similarity_method,
        keyword_coverage=keyword_result["coverage"],
        matched_keywords=keyword_result["matched"],
        missing_keywords=keyword_result["missing"],
        length_ratio=length_ratio,
        marks=marks,
        max_marks=max_marks,
        feedback=feedback,
    )
