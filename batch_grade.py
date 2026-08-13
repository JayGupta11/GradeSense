import json
import argparse
import csv
import os
from typing import Dict, List

from rubric_scoring import Question, RubricPoint, evaluate_question, load_trained_model, using_trained_model
from paper_parser import build_question_bank, parse_numbered_text, match_student_answers
from choice_grouping import aggregate_with_choice_groups

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "marks_predictor.joblib")

def try_load_trained_model(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    if os.path.exists(model_path):
        load_trained_model(model_path)
        print(f"[batch_grade] Using trained model: {model_path}")
        return True
    print(f"[batch_grade] No trained model found at {model_path} — using rule-based scoring.")
    return False

def load_question_bank(path: str) -> Dict[str, Question]:
    with open(path, "r") as f:
        raw = json.load(f)

    questions = {}
    for q in raw:
        rubric = None
        if q.get("rubric_points"):
            rubric = [RubricPoint(text=p["text"], marks=p["marks"]) for p in q["rubric_points"]]
        questions[q["question_id"]] = Question(
            question_id=q["question_id"],
            question_text=q.get("question_text", ""),
            max_marks=q["max_marks"],
            model_answer=q["model_answer"],
            rubric_points=rubric,
            expected_length_words=q.get("expected_length_words"),
            choice_group=q.get("choice_group"),
            choice_required=q.get("choice_required"),
        )
    return questions

def _grade_bank(questions: Dict[str, Question], student_answers: Dict[str, str]):
    """Shared grading + choice-group-aware aggregation logic used by all input modes."""
    results = {}
    for qid, question in questions.items():
        student_text = student_answers.get(qid, "")
        results[qid] = evaluate_question(question, student_text)

    total_awarded, total_max, notes = aggregate_with_choice_groups(questions, results)
    return list(results.values()), total_awarded, total_max, notes

def grade_from_json(questions_path: str, answers_path: str):
    questions = load_question_bank(questions_path)
    with open(answers_path, "r") as f:
        student_answers = json.load(f)  # {question_id: answer_text}
    return _grade_bank(questions, student_answers)

def grade_from_csv(csv_path: str):
    questions, student_answers = {}, {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["question_id"]
            choice_required = row.get("choice_required")
            questions[qid] = Question(
                question_id=qid,
                question_text=row.get("question_text", ""),
                max_marks=float(row["max_marks"]),
                model_answer=row["model_answer"],
                choice_group=row.get("choice_group") or None,
                choice_required=int(choice_required) if choice_required else None,
            )
            student_answers[qid] = row["student_answer"]
    return _grade_bank(questions, student_answers)

def grade_paper_texts(paper_text: str, answer_key_text: str, student_sheet_text: str):
    questions, missing_answers = build_question_bank(paper_text, answer_key_text)

    student_answers = match_student_answers(student_sheet_text, list(questions.keys()))
    unanswered = [qid for qid in questions if qid not in student_answers]

    results_list, total_awarded, total_max, notes = _grade_bank(questions, student_answers)
    return results_list, total_awarded, total_max, notes, missing_answers, unanswered

def grade_from_paper(paper_path: str, answer_key_path: str, student_sheet_path: str):
    with open(paper_path) as f:
        paper_text = f.read()
    with open(answer_key_path) as f:
        answer_key_text = f.read()
    with open(student_sheet_path) as f:
        student_sheet_text = f.read()

    results, total_awarded, total_max, notes, missing_answers, unanswered = grade_paper_texts(
        paper_text, answer_key_text, student_sheet_text
    )
    if missing_answers:
        print(f"WARNING: no model answer found in answer key for question(s): {missing_answers}")
    for qid in unanswered:
        print(f"WARNING: no student answer found for question {qid} (left unanswered / not detected)")

    return results, total_awarded, total_max, notes

def print_report(results: List, total_awarded: float, total_max: float, notes: List[str] = None):
    for r in results:
        print(f"\n=== {r.question_id} ({r.max_marks} marks) ===")
        print(f"Expected length: ~{r.expected_length_words} words | Actual: {r.actual_length_words} words")
        for pr in r.point_results:
            print(f"  - [{pr.awarded_marks}/{pr.point_max_marks}] {pr.point_text[:70]}")
        print(f"Feedback: {r.feedback}")

    if notes:
        print("\n" + "-" * 50)
        print("Choice-group resolution:")
        for n in notes:
            print(f"  - {n}")

    print("\n" + "=" * 50)
    print(f"TOTAL: {round(total_awarded, 2)} / {total_max}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-grade a full answer sheet.")
    parser.add_argument("--csv", help="Path to a CSV file (question_id,max_marks,model_answer,student_answer[,question_text,choice_group,choice_required])")
    parser.add_argument("--paper", help="Path to raw question paper text (auto-detects sections/marks/question count/choice groups)")
    parser.add_argument("--answer-key", help="Path to answer key text (required with --paper)")
    parser.add_argument("--student-sheet", help="Path to student's (OCR'd) answer sheet text (required with --paper)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained marks_predictor.joblib (auto-used if it exists)")
    parser.add_argument("questions_json", nargs="?", help="Path to question bank JSON")
    parser.add_argument("answers_json", nargs="?", help="Path to student answers JSON ({question_id: answer_text})")
    args = parser.parse_args()

    try_load_trained_model(args.model)

    if args.paper:
        if not (args.answer_key and args.student_sheet):
            parser.error("--paper requires --answer-key and --student-sheet")
        res, awarded, max_total, notes = grade_from_paper(args.paper, args.answer_key, args.student_sheet)
    elif args.csv:
        res, awarded, max_total, notes = grade_from_csv(args.csv)
    elif args.questions_json and args.answers_json:
        res, awarded, max_total, notes = grade_from_json(args.questions_json, args.answers_json)
    else:
        parser.error("Provide --paper/--answer-key/--student-sheet, or --csv <file>, or <questions_json> <answers_json>")

    print_report(res, awarded, max_total, notes)