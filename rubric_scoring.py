"""
rubric_scoring.py
------------------
CBSE-style rubric-based evaluation for questions of different mark values
(1-mark, 2-mark, 3-mark, 5-mark, etc.), each expecting a different answer
length and a different number of key points.

Instead of comparing the whole student answer to the whole model answer
as one blob (which unfairly treats a 1-mark and a 5-mark question the
same way), this module:

    1. Breaks the model answer into rubric points (teacher-supplied or
        auto-generated from the model answer's sentences), each worth a
        fraction of the question's total marks.
    2. Breaks the student's OCR'd answer into sentences.
    3. For each rubric point, finds its best-matching student sentence and
        scores that match on a sliding scale (full / partial / no credit) —
        mirroring how a human examiner gives partial marks for a partially
        correct point.
    4. Sums point-marks (capped at max_marks) and estimates expected answer
        length from the mark value, purely for feedback purposes.
    """

import re
from dataclasses import dataclass, field
from typing import List, Optional

from nlp_evaluator import clean_text, semantic_similarity

# --------------------------------------------------------------------
# Optional trained model (see marks_predictor.py / train_marks_predictor.py)
# --------------------------------------------------------------------
# By default this system scores each rubric point using the hand-picked
# thresholds in _THRESHOLDS below. If you've trained marks_predictor.py
# on a real labeled dataset (e.g. ASAG2024), call load_trained_model()
# once at startup to swap in the trained model instead — everything else
# (paper_parser, batch_grade, choice_grouping, the Streamlit apps) needs
# NO changes to benefit from this, since they all call evaluate_question()
# the same way regardless of which scoring path is active underneath.

_trained_predictor = None


def load_trained_model(model_path: str = "../models/marks_predictor.joblib"):
    """
    Loads a trained MarksPredictor and makes evaluate_question() use it
    for every rubric point's credit-fraction calculation, instead of the
    hand-picked _THRESHOLDS rules. Call this once, e.g. at the top of
    batch_grade.py's __main__ or app_batch.py, before grading anything.
    Returns the loaded predictor (or raises if the file doesn't exist).
    """
    global _trained_predictor
    from marks_predictor import MarksPredictor

    _trained_predictor = MarksPredictor().load(model_path)
    return _trained_predictor


def using_trained_model() -> bool:
    return _trained_predictor is not None


def _token_overlap_ratio(point_text: str, sentence_text: str) -> float:
    """
    Recall-oriented fallback signal: what fraction of the rubric point's
    key content words are present in the student's sentence, regardless
    of extra words/paraphrasing the student added.

    This matters a lot for CBSE-style grading where exact terminology
    (e.g. 'mitochondria', 'chloroplast', a named law/date/formula) needs
    to be present, and is used as a floor alongside the semantic/TF-IDF
    similarity so a point isn't marked wrong just because the surrounding
    embedding/TF-IDF similarity is noisy on very short text.
    """
    point_tokens = set(clean_text(point_text).split())
    sentence_tokens = set(clean_text(sentence_text).split())
    if not point_tokens:
        return 0.0
    common = point_tokens & sentence_tokens
    return len(common) / len(point_tokens)


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass
class RubricPoint:
    text: str            # the key idea/fact/step expected
    marks: float          # marks allocated to this point


@dataclass
class Question:
    question_id: str
    question_text: str
    max_marks: float
    model_answer: str
    rubric_points: Optional[List[RubricPoint]] = None   # auto-generated if None
    expected_length_words: Optional[int] = None          # auto-estimated if None
    choice_group: Optional[str] = None    # questions sharing a group_id are alternatives/options
    choice_required: Optional[int] = None  # how many from this group count toward the total


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


# ---------------------------------------------------------------------
# CBSE-style length estimation (rule-of-thumb: ~15-22 words per mark,
# with a sensible floor/ceiling so 1-mark and 10-mark questions both
# stay realistic)
# ---------------------------------------------------------------------

def estimate_expected_length(max_marks: float) -> int:
    if max_marks <= 1:
        return 15                     # one-word/one-line factual answer
    words_per_mark = 20
    length = int(max_marks * words_per_mark)
    return max(20, min(length, 250))  # clamp: 20-250 words


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter (avoids an nltk punkt dependency here);
    # splits on '.', '!', '?', ';', and newlines.
    parts = re.split(r"[.\n;!?]+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------
# Rubric auto-generation (used when the teacher hasn't supplied explicit
# rubric points — splits the model answer into sentences and distributes
# marks evenly across them, which is a reasonable CBSE-style default:
# one key idea per sentence)
# ---------------------------------------------------------------------

def auto_generate_rubric(model_answer: str, max_marks: float) -> List[RubricPoint]:
    sentences = _split_sentences(model_answer)
    if not sentences:
        return [RubricPoint(text=model_answer.strip(), marks=max_marks)]

    # Cap the number of auto-generated points at whichever is smaller:
    # how many sentences the model answer actually has, or how many marks
    # the question is worth (rounded up) — scales correctly for ANY mark
    # value (1-mark, 5-mark, 7-mark, 10-mark, etc.), not just low-mark
    # CBSE-style questions. A hard ceiling of 15 avoids absurdly many tiny
    # fractional points on a long model answer for a huge-mark question.
    import math
    max_points = max(1, min(len(sentences), math.ceil(max_marks) if max_marks >= 1 else 1, 15))
    chosen = sentences[:max_points]

    marks_each = round(max_marks / len(chosen), 3)
    points = [RubricPoint(text=s, marks=marks_each) for s in chosen]

    # Fix rounding drift so points sum exactly to max_marks
    drift = round(max_marks - sum(p.marks for p in points), 3)
    if drift != 0:
        points[-1].marks = round(points[-1].marks + drift, 3)

    return points


# ---------------------------------------------------------------------
# Partial-credit mapping: similarity score -> fraction of that point's marks
# ---------------------------------------------------------------------

def _credit_fraction(similarity: float, high: float, low: float) -> float:
    """
    similarity >= high  -> full credit (1.0)
    similarity <= low   -> no credit   (0.0)
    in between          -> linear partial credit
    """
    if similarity >= high:
        return 1.0
    if similarity <= low:
        return 0.0
    return round((similarity - low) / (high - low), 3)


# Thresholds differ by matching method: sentence-embedding cosine similarity
# and token-overlap ratio live on different natural scales, so each gets
# its own calibration. (Tune these against a few real graded scripts for
# your subject — see README "Calibrating thresholds".)
_THRESHOLDS = {
    "semantic": {"high": 0.62, "low": 0.30},   # sentence-transformers embeddings
    "overlap": {"high": 0.55, "low": 0.20},     # token-overlap fallback (TF-IDF unavailable/weak)
}


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate_question(question: Question, student_answer: str) -> RubricEvaluationResult:
    rubric_points = question.rubric_points or auto_generate_rubric(
        question.model_answer, question.max_marks
    )
    expected_length = question.expected_length_words or estimate_expected_length(question.max_marks)

    student_sentences = _split_sentences(student_answer) or [student_answer.strip()]
    actual_length = len(student_answer.split())

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

        if _trained_predictor is not None:
            fraction = _trained_predictor.predict_fraction(point.text, best_sentence)
        else:
            fraction = _credit_fraction(best_sim, high=thresholds["high"], low=thresholds["low"])

        awarded = round(point.marks * fraction, 3)
        total_awarded += awarded

        point_results.append(
            PointResult(
                point_text=point.text,
                point_max_marks=point.marks,
                best_match_sentence=best_sentence,
                similarity=round(best_sim, 3),
                awarded_marks=awarded,
            )
        )

    total_awarded = round(min(total_awarded, question.max_marks), 2)

    feedback = _generate_rubric_feedback(
        point_results, question.max_marks, total_awarded, expected_length, actual_length
    )

    return RubricEvaluationResult(
        question_id=question.question_id,
        max_marks=question.max_marks,
        awarded_marks=total_awarded,
        expected_length_words=expected_length,
        actual_length_words=actual_length,
        point_results=point_results,
        feedback=feedback,
    )


def _generate_rubric_feedback(
    point_results: List[PointResult],
    max_marks: float,
    awarded_marks: float,
    expected_length: int,
    actual_length: int,
) -> str:
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
            parts.append(
                "Partially addressed: " + "; ".join(f"'{p.point_text[:50]}'" for p in partially_covered) + "."
            )
        if missed:
            parts.append(
                "Missing: " + "; ".join(f"'{p.point_text[:50]}'" for p in missed) + "."
            )

    ratio = actual_length / expected_length if expected_length else 1.0
    if ratio < 0.4:
        parts.append(
            f"Answer is quite short for a {max_marks}-mark question "
            f"(~{actual_length} words vs. an expected ~{expected_length}); "
            "consider elaborating with more explanation or examples."
        )
    elif ratio > 2.0:
        parts.append(
            f"Answer is much longer than needed for a {max_marks}-mark question "
            "— try to be more concise and focus on the key points."
        )

    parts.append(f"Marks awarded: {awarded_marks}/{max_marks}.")
    return " ".join(parts)