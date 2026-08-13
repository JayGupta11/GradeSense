from typing import Dict, List, Tuple
from rubric_scoring import Question, RubricEvaluationResult

def aggregate_with_choice_groups(
    questions: Dict[str, Question],
    results: Dict[str, RubricEvaluationResult],
) -> Tuple[float, float, List[str]]:
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

    # Second pass: resolve each choice group — only the best required
    # attempted answers count, the rest are graded but excluded from the total.
    for group_id, info in groups.items():
        required = info["required"]
        members = info["members"]

        attempted = [(qid, r) for qid, r in members if r.actual_length_words > 0]
        pool = attempted if attempted else members  # if nothing attempted, still report zero credit

        pool_sorted = sorted(pool, key=lambda pair: pair[1].awarded_marks, reverse=True)
        chosen = pool_sorted[:required]

        group_awarded = sum(r.awarded_marks for _, r in chosen)
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

    total_awarded = min(round(total_awarded, 2), total_max)

    return total_awarded, total_max, notes