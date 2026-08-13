"""
train_marks_predictor.py
-------------------------
Trains the MarksPredictor model (marks_predictor.py) on ASAG2024, using
the proper three-way split:

    train.parquet      -> the ONLY data the model ever learns from
    validation.parquet -> used purely for model/hyperparameter selection
                            (never used to fit the model itself)
    test.parquet       -> touched exactly once, at the very end, for the
                            final unbiased performance report

Confirmed real ASAG2024 schema:
    columns: question, provided_answer, reference_answer, grade,
                data_source, normalized_grade, weight

    - provided_answer   = the student's answer
    - reference_answer  = the correct/model answer
    - normalized_grade  = already on a [0,1] scale across all combined
                            source datasets -> used directly as the label
    - data_source       = SciEntsBank, Beetle, SAF, DigiKlausur, Mohler,
                            Stita, CU-NLP (7 real combined datasets)

Known data quality issues in the real files (handled below):
    - `question` has ~6% missing values -> harmless, not used as a feature
    - `provided_answer` has a small number of missing/blank values, all
        with grade 0.0 -> converted to an explicit empty string (NOT the
        literal text "nan", which pandas would otherwise produce and which
        would corrupt the similarity features)
    - a handful of rows have an empty/missing `reference_answer` -> these
        are dropped since there's nothing to compare the student's answer to

Usage:
    python train_marks_predictor.py
    python train_marks_predictor.py --train-data ../data/train.parquet --validation-data ../data/validation.parquet --test-data ../data/test.parquet --out ../models/marks_predictor.joblib
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from marks_predictor import MarksPredictor, extract_features, FEATURE_NAMES

COLUMN_MAP = {
    "reference_answer": "reference_answer",
    "student_answer": "provided_answer",
    "score": "normalized_grade",
}
MAX_SCORE = 1.0  # normalized_grade is already on [0,1] in ASAG2024

# A small hyperparameter grid to select from using the validation set.
CANDIDATE_HYPERPARAMS = [
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
]


def _load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .parquet.")


def build_training_data(data_path: str):
    """
    Loads a data file and builds (X, y) arrays, explicitly handling the
    real dataset's missing values rather than letting them silently
    corrupt features (see module docstring for details).
    """
    df = _load_any(data_path)

    missing_cols = set(COLUMN_MAP.values()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"File is missing expected columns: {missing_cols}. "
            f"Actual columns found: {list(df.columns)}."
        )

    ref_col = COLUMN_MAP["reference_answer"]
    stu_col = COLUMN_MAP["student_answer"]
    score_col = COLUMN_MAP["score"]

    # A missing/blank reference answer means there's nothing to score
    # against -> drop these rows entirely (only affects a handful of rows).
    before = len(df)
    df = df.dropna(subset=[ref_col, score_col])
    df = df[df[ref_col].astype(str).str.strip() != ""]
    dropped_no_reference = before - len(df)

    # A missing student answer -> a TRUE blank ("") not the literal text
    # "nan" that str(NaN) would otherwise produce.
    df[stu_col] = df[stu_col].fillna("")

    X, y = [], []
    for _, row in df.iterrows():
        reference = str(row[ref_col]).strip()
        student = str(row[stu_col]).strip()
        normalized_score = float(row[score_col]) / MAX_SCORE
        normalized_score = max(0.0, min(1.0, normalized_score))

        X.append(extract_features(reference, student))
        y.append(normalized_score)

    if dropped_no_reference:
        print(f"  (dropped {dropped_no_reference} rows with no reference answer to compare against)")

    return np.array(X), np.array(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the marks-prediction model with a proper train/validation/test split.")
    parser.add_argument("--train-data", default="../data/train.parquet")
    parser.add_argument("--validation-data", default="../data/validation.parquet")
    parser.add_argument("--test-data", default="../data/test.parquet")
    parser.add_argument("--out", default="../models/marks_predictor.joblib")
    args = parser.parse_args()

    print(f"Loading training data from {args.train_data} ...")
    X_train, y_train = build_training_data(args.train_data)
    print(f"  {len(X_train)} training examples")

    print(f"Loading validation data from {args.validation_data} ...")
    X_val, y_val = build_training_data(args.validation_data)
    print(f"  {len(X_val)} validation examples")

    print(f"Loading test data from {args.test_data} ...")
    X_test, y_test = build_training_data(args.test_data)
    print(f"  {len(X_test)} test examples")

    print("\n--- Model selection on validation set (test set NOT used here) ---")
    best_model, best_val_mae, best_params = None, float("inf"), None
    for params in CANDIDATE_HYPERPARAMS:
        model = RandomForestRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        print(f"  {params} -> validation MAE: {val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae, best_model, best_params = val_mae, model, params

    print(f"\nSelected config: {best_params} (validation MAE: {best_val_mae:.4f})")

    print("\n--- Final test set report (touched only once, never used for tuning) ---")
    test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Feature importances: {dict(zip(FEATURE_NAMES, [round(float(i), 4) for i in best_model.feature_importances_]))}")

    predictor = MarksPredictor(model=best_model)
    predictor.save(args.out)
    print(f"\nModel saved to {args.out}")