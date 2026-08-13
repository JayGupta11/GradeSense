"""
choice_grouping.py
-------------------
Handles optional/choice questions so the final total is ALWAYS correct:

    - Internal choice ("Q9 OR Q9-alternate"): only the better-scoring
        version counts; the other is simply not added to the total.
    - "Attempt any N of the following M": all M are graded (in case the
        student answered more than required), but only the best N scores
        count toward the total — extra attempts don't inflate the score.

In both cases this guarantees: final total marks <= paper's declared
max marks, no matter how many optional questions the student attempts.

Works directly on Question objects (rubric_scoring.py) and their
RubricEvaluationResult objects (the output of evaluate_question()).
"""

from typing import Dict, List, Tuple
from rubric_scoring import Question, RubricEvaluationResult


def aggregate_with_choice_groups(
    questions: Dict[str, Question],
    results: Dict[str, RubricEvaluationResult],
) -> Tuple[float, float, List[str]]:
    """
    Combines per-question results into a final total, correctly handling
    choice groups (OR-choices and "attempt N of M" sections).

    Returns (total_awarded, total_max, notes) where `notes` explains how
    each choice group was resolved (useful to print/log for transparency).
    """
    groups: Dict[str, dict] = {}
    total_awarded = 0.0
    total_max = 0.0

    # First pass: separate grouped (optional) questions from regular ones
    for qid, q in questions.items():
        r = results[qid]
        if q.choice_group:
            g = groups.setdefault(q.choice_group, {"required": q.choice_required or 1, "members": []})
            g["members"].append((qid, r))
        else:
            total_awarded += min(r.awarded_marks, r.max_marks)  # per-question safety cap
            total_max += r.max_marks

    notes: List[str] = []

    # Second pass: resolve each choice group — only the best `required`
    # attempted answers count; the rest are graded but excluded from the total.
    for group_id, info in groups.items():
        required = info["required"]
        members = info["members"]  # [(qid, RubricEvaluationResult), ...]

        attempted = [(qid, r) for qid, r in members if r.actual_length_words > 0]
        pool = attempted if attempted else members  # if nothing attempted, still report zero credit

        pool_sorted = sorted(pool, key=lambda pair: pair[1].awarded_marks, reverse=True)
        chosen = pool_sorted[:required]

        group_awarded = sum(r.awarded_marks for _, r in chosen)
        # Group's max marks = sum of the `required` highest max_marks among
        # ALL declared members (not just attempted ones) — this is the
        # paper's intended max for that group, regardless of what the
        # student actually attempted.
        member_max_sorted = sorted((r.max_marks for _, r in members), reverse=True)
        group_max = sum(member_max_sorted[:required])

        group_awarded = min(round(group_awarded, 3), group_max)  # safety cap

        total_awarded += group_awarded
        total_max += group_max

        chosen_ids = [qid for qid, _ in chosen]
        skipped_ids = [qid for qid, _ in members if qid not in chosen_ids]
        note = (
            f"Choice group '{group_id}': required {required} of {len(members)} question(s). "
            f"Counted: {chosen_ids} -> {round(group_awarded, 2)}/{group_max}."
        )
        if skipped_ids:
            note += f" Not counted (extra attempts or lower-scoring alternative): {skipped_ids}."
        notes.append(note)

    # Final safety cap — belt and suspenders, total should already respect
    # this, but guarantee it regardless of any upstream edge case.
    total_awarded = min(round(total_awarded, 2), total_max)

    return total_awarded, total_max, notes