"""
marks_predictor.py
-------------------
A genuinely TRAINED model (not a pretrained/off-the-shelf one) that learns
to map answer-quality features -> a normalized score in [0, 1].

This replaces the hand-picked threshold logic in rubric_scoring.py's
_credit_fraction() with something learned from real human-graded data
(see train_marks_predictor.py).

Why normalized [0,1] and not raw marks: different datasets/questions use
different mark scales (0-3, 0-5, 0-10...). Training the model to predict
a FRACTION of full marks makes it reusable for ANY question regardless of
its actual max_marks — at inference time you just do:

    predicted_marks = model.predict(features) * question.max_marks

Features used (all already computed elsewhere in this project):
    - semantic/overlap similarity between student answer and reference point
    - token overlap ratio (keyword coverage)
    - length ratio (student answer length / expected length)
    - student answer sentence count (proxy for structure/completeness)

Model: RandomForestRegressor (scikit-learn) — chosen over a deep net
because ASAG feature sets are small/tabular (a handful of numeric
features), where tree ensembles reliably outperform small neural nets
and train in seconds on CPU with no GPU required. This is a legitimate,
defensible "trained from scratch" model for a project viva.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nlp_evaluator import clean_text, semantic_similarity


def _token_overlap_ratio(reference_text: str, candidate_text: str) -> float:
    ref_tokens = set(clean_text(reference_text).split())
    cand_tokens = set(clean_text(candidate_text).split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def extract_features(reference_answer: str, student_answer: str) -> np.ndarray:
    """
    Turns a (reference_answer, student_answer) pair into a fixed-size
    numeric feature vector for the model.
    """
    similarity, _method = semantic_similarity(student_answer, reference_answer)
    overlap = _token_overlap_ratio(reference_answer, student_answer)

    ref_len = max(len(reference_answer.split()), 1)
    stu_len = len(student_answer.split())
    length_ratio = min(stu_len / ref_len, 2.0)  # capped so extreme verbosity doesn't dominate

    sentence_count = len(re.split(r"[.!?]+", student_answer.strip())) if student_answer.strip() else 0

    return np.array([similarity, overlap, length_ratio, sentence_count], dtype=float)


FEATURE_NAMES = ["similarity", "token_overlap", "length_ratio", "sentence_count"]


@dataclass
class TrainingReport:
    train_mae: float
    test_mae: float
    test_rmse: float
    n_train: int
    n_test: int
    feature_importances: dict


class MarksPredictor:
    """Thin wrapper around a trained sklearn model with save/load + predict."""

    def __init__(self, model: RandomForestRegressor = None):
        self.model = model

    def predict_fraction(self, reference_answer: str, student_answer: str) -> float:
        """Returns predicted score as a fraction in [0, 1] of full marks."""
        if self.model is None:
            raise RuntimeError("Model not loaded/trained. Call train() or load().")
        features = extract_features(reference_answer, student_answer).reshape(1, -1)
        pred = float(self.model.predict(features)[0])
        return max(0.0, min(1.0, pred))

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> TrainingReport:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=random_state,
        )
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        report = TrainingReport(
            train_mae=round(mean_absolute_error(y_train, train_pred), 4),
            test_mae=round(mean_absolute_error(y_test, test_pred), 4),
            test_rmse=round(mean_squared_error(y_test, test_pred) ** 0.5, 4),
            n_train=len(X_train),
            n_test=len(X_test),
            feature_importances=dict(zip(FEATURE_NAMES, [round(float(i), 4) for i in self.model.feature_importances_])),
        )
        return report

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python marks_predictor.py <model_path> <reference_answer> <student_answer>")
        sys.exit(1)

    predictor = MarksPredictor().load(sys.argv[1])
    fraction = predictor.predict_fraction(sys.argv[2], sys.argv[3])
    print(f"Predicted score fraction: {fraction:.3f}")