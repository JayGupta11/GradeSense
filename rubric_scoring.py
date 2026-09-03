import re
from dataclasses import dataclass, field
from typing import List, Optional

from nlp_evaluator import clean_text, semantic_similarity

_trained_predictor = None


def load_trained_model(model_path: str = "../models/marks_predictor.joblib"):
    global _trained_predictor
    from marks_predictor import MarksPredictor
    _trained_predictor = MarksPredictor().load(model_path)
    return _trained_predictor


def using_trained_model() -> bool:
    return _trained_predictor is not None


def _token_overlap_ratio(point_text: str, sentence_text: str) -> float:
    point_tokens = set(clean_text(point_text).split())
    sentence_tokens = set(clean_text(sentence_text).split())
    if not point_tokens:
        return 0.0
    common = point_tokens & sentence_tokens
    return len(common) / len(point_tokens)


@dataclass
class RubricPoint:
    text: str
    marks: float


@dataclass
class Question:
    question_id: str
    question_text: str
    max_marks: float
    model_answer: str
    rubric_points: Optional[List[RubricPoint]] = None
    expected_length_words: Optional[int] = None
    choice_group: Optional[str] = None
    choice_required: Optional[int] = None
    question_type: str = "Short_Answer"  # "Short_Answer" or "MCQ"
    options: Optional[dict] = None  # for MCQ: {"A": "option text", "B": "...", ...}


@dataclass
class PointResult:
    point_text: str
    point_max_marks: float
    best_match_sentence: str
    similarity: float
    awarded_marks: float


@dataclass
class RubricEvaluationResult:
    question_id: str
    max_marks: float
    awarded_marks: float
    expected_length_words: int
    actual_length_words: int
    point_results: List[PointResult]
    feedback: str


def estimate_expected_length(max_marks: float) -> int:
    if max_marks <= 1:
        return 15
    words_per_mark = 20
    length = int(max_marks * words_per_mark)
    return max(20, min(length, 250))


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.\n;!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def auto_generate_rubric(model_answer: str, max_marks: float) -> List[RubricPoint]:
    sentences = _split_sentences(model_answer)
    if not sentences:
        return [RubricPoint(text=model_answer.strip(), marks=max_marks)]
    import math
    max_points = max(1, min(len(sentences), math.ceil(max_marks) if max_marks >= 1 else 1, 15))
    chosen = sentences[:max_points]
    marks_each = round(max_marks / len(chosen), 3)
    points = [RubricPoint(text=s, marks=marks_each) for s in chosen]
    drift = round(max_marks - sum(p.marks for p in points), 3)
    if drift != 0:
        points[-1].marks = round(points[-1].marks + drift, 3)
    return points


def _credit_fraction(similarity: float, high: float, low: float) -> float:
    if similarity >= high:
        return 1.0
    if similarity <= low:
        return 0.0
    return round((similarity - low) / (high - low), 3)


_THRESHOLDS = {
    "semantic": {"high": 0.62, "low": 0.30},
    "overlap": {"high": 0.55, "low": 0.20},
}


def _extract_mcq_letter(text: str) -> Optional[str]:
    """Pulls a single A/B/C/D choice out of free text, e.g. 'Ans: C' -> 'C',
    'B. Little or no autocorrelation' -> 'B' (letter+value combined)."""
    m = re.search(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
    return m.group(1).upper() if m else None


def _match_mcq_option_by_value(student_answer: str, options: dict, threshold: float = 0.5) -> Optional[str]:
    """
    Fallback for when the student wrote out an option's actual TEXT
    instead of its letter (e.g. wrote 'Little or no autocorrelation'
    instead of 'B'). Matches the student's answer against each option's
    value text and returns the best-matching option's letter, if the
    match is strong enough to be confident. Uses both token overlap and
    semantic similarity so it works whether or not sentence-transformers
    is available.
    """
    if not options:
        return None
    best_letter, best_score = None, 0.0
    for letter, option_text in options.items():
        overlap = _token_overlap_ratio(option_text, student_answer)
        sim, _method = semantic_similarity(student_answer, option_text)
        combined = max(overlap, sim)
        if combined > best_score:
            best_score, best_letter = combined, letter
    return best_letter if best_score >= threshold else None


def evaluate_question(question: Question, student_answer: str) -> RubricEvaluationResult:
    # MCQ questions: exact-match grading, not NLP similarity — but the
    # student may have written the option LETTER (e.g. "B"), the option
    # VALUE (e.g. "Little or no autocorrelation"), or BOTH together
    # (e.g. "B. Little or no autocorrelation"). All three are handled:
    # try letter extraction first (covers letter-only and letter+value),
    # and only fall back to value-matching against the paper's parsed
    # options when no letter was found at all (value-only case).
    if question.question_type == "MCQ":
        selected = _extract_mcq_letter(student_answer)
        matched_by_value = False
        if not selected and question.options:
            selected = _match_mcq_option_by_value(student_answer, question.options)
            matched_by_value = selected is not None

        correct = _extract_mcq_letter(question.model_answer) or question.model_answer.strip().upper()
        awarded = question.max_marks if (selected and selected == correct) else 0.0

        if awarded:
            feedback = f"Correct (selected {selected}" + (", matched by answer text)" if matched_by_value else ").")
        elif selected:
            detail = " (matched by answer text)" if matched_by_value else ""
            feedback = f"Incorrect (selected {selected}{detail}, correct answer is {correct})."
        else:
            feedback = f"Incorrect (selected nothing, correct answer is {correct})."

        return RubricEvaluationResult(
            question_id=question.question_id,
            max_marks=question.max_marks,
            awarded_marks=awarded,
            expected_length_words=1,
            actual_length_words=len(student_answer.split()),
            point_results=[PointResult(point_text=question.question_text, point_max_marks=question.max_marks,
                                        best_match_sentence=student_answer, similarity=1.0 if awarded else 0.0,
                                        awarded_marks=awarded)],
            feedback=feedback,
        )

    rubric_points = question.rubric_points or auto_generate_rubric(question.model_answer, question.max_marks)
    expected_length = question.expected_length_words or estimate_expected_length(question.max_marks)

    actual_length = len(student_answer.split())

    # Hard floor: a genuinely blank/unanswered question always gets 0
    # marks, full stop — never handed to the similarity model or the
    # trained ML model at all. Without this, an empty string can still
    # produce an arbitrary non-zero similarity/prediction (embedding
    # models don't guarantee "empty input -> zero score"), which is
    # exactly what caused random-looking marks on unanswered questions.
    if actual_length == 0:
        point_results = [
            PointResult(point_text=p.text, point_max_marks=p.marks, best_match_sentence="", similarity=0.0, awarded_marks=0.0)
            for p in rubric_points
        ]
        return RubricEvaluationResult(
            question_id=question.question_id, max_marks=question.max_marks, awarded_marks=0.0,
            expected_length_words=expected_length, actual_length_words=0,
            point_results=point_results,
            feedback=f"No answer detected for this question. Marks awarded: 0.0/{question.max_marks}.",
        )

    student_sentences = _split_sentences(student_answer) or [student_answer.strip()]

    point_results: List[PointResult] = []
    total_awarded = 0.0

    for point in rubric_points:
        best_sim = 0.0
        best_sentence = ""
        best_scoring_method = "overlap"

        for sentence in student_sentences:
            sim, embed_method = semantic_similarity(sentence, point.text)
            overlap = _token_overlap_ratio(point.text, sentence)

            if embed_method == "sentence-transformers":
                combined = max(sim, 0.6 * sim + 0.4 * overlap)
                scoring_method = "semantic"
            else:
                combined = overlap
                scoring_method = "overlap"

            if combined > best_sim:
                best_sim = combined
                best_sentence = sentence
                best_scoring_method = scoring_method

        thresholds = _THRESHOLDS[best_scoring_method]

        if not best_sentence.strip():
            # No matching content was found for this point at all (e.g. every
            # student sentence was empty/whitespace) — zero credit, no model call.
            fraction = 0.0
        elif _trained_predictor is not None:
            fraction = _trained_predictor.predict_fraction(point.text, best_sentence)
        else:
            fraction = _credit_fraction(best_sim, high=thresholds["high"], low=thresholds["low"])

        awarded = round(point.marks * fraction, 3)
        total_awarded += awarded

        point_results.append(PointResult(
            point_text=point.text, point_max_marks=point.marks,
            best_match_sentence=best_sentence, similarity=round(best_sim, 3), awarded_marks=awarded,
        ))

    total_awarded = round(min(total_awarded, question.max_marks), 2)
    feedback = _generate_rubric_feedback(point_results, question.max_marks, total_awarded, expected_length, actual_length)

    return RubricEvaluationResult(
        question_id=question.question_id, max_marks=question.max_marks, awarded_marks=total_awarded,
        expected_length_words=expected_length, actual_length_words=actual_length,
        point_results=point_results, feedback=feedback,
    )


def _generate_rubric_feedback(point_results, max_marks, awarded_marks, expected_length, actual_length) -> str:
    parts = []
    fully_covered = [p for p in point_results if p.awarded_marks >= 0.9 * p.point_max_marks]
    missed = [p for p in point_results if p.awarded_marks <= 0.1 * p.point_max_marks]
    partially_covered = [p for p in point_results if p not in fully_covered and p not in missed]

    if not missed and not partially_covered:
        parts.append("All expected points were covered clearly.")
    else:
        if fully_covered:
            parts.append(f"{len(fully_covered)}/{len(point_results)} point(s) covered well.")
        if partially_covered:
            parts.append("Partially addressed: " + "; ".join(f"'{p.point_text[:50]}'" for p in partially_covered) + ".")
        if missed:
            parts.append("Missing: " + "; ".join(f"'{p.point_text[:50]}'" for p in missed) + ".")

    ratio = actual_length / expected_length if expected_length else 1.0
    if ratio < 0.4:
        parts.append(f"Answer is quite short for a {max_marks}-mark question (~{actual_length} words vs. an expected ~{expected_length}); consider elaborating.")
    elif ratio > 2.0:
        parts.append(f"Answer is much longer than needed for a {max_marks}-mark question — try to be more concise.")

    parts.append(f"Marks awarded: {awarded_marks}/{max_marks}.")
    return " ".join(parts)