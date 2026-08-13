"""
paper_parser.py
---------------
Auto-detects exam structure from typed/OCR'd text of:
    1. The question paper  -> number of questions, sections, marks per question
    2. The answer key       -> model answer text per question number
    3. The student's answer sheet -> student's answer text per question number

This lets you build a full rubric-ready Question bank (see rubric_scoring.py)
WITHOUT manually typing question_id/max_marks for every question — you only
need the question paper + answer key text, which any teacher already has.

Supported conventions (typical CBSE-style formatting):
    SECTION A (1 mark each)
    1. What is the powerhouse of the cell?
    2. Name two greenhouse gases.

    SECTION B (3 marks each)
    3. Explain the process of photosynthesis.

    ...or with inline marks instead of a section default:
    7. Explain X. (3)
    8. Describe Y. [5 Marks]

Student answer sheets are expected to number their answers similarly:
    1. Mitochondria is the powerhouse of the cell.
    2. Carbon dioxide and methane.
    Ans 3: Photosynthesis happens in chloroplasts...

OPTIONAL / CHOICE QUESTIONS ("internal choice" / "attempt N of M"):

    1) Internal choice on a single question (repeats the same number, with
        "OR" as a standalone line between the two versions):
        9. Explain the theory of relativity.
                            OR
        9. Explain the concept of black holes.
        -> both are parsed, grouped together, and only the BETTER-scoring one
        counts toward the total (required = 1 out of that group).

    2) "Attempt any N of the following M" at section level:
        SECTION E (5 marks each) [Attempt any 3 of the following 5 questions]
        10. ...
        11. ...
        12. ...
        13. ...
        14. ...
    -> all 5 are parsed and grouped; only the top 3 scoring answers the
        student actually attempted count toward the total.

In both cases the grading step (see batch_grade.py / choice_grouping.py)
guarantees the final total never exceeds the paper's declared max marks,
even if the student answers every optional question.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SECTION_RE = re.compile(r"^\s*SECTION[\s\-]*([A-Z0-9]+)\b(.*)$", re.IGNORECASE)
SECTION_MARKS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*marks?\s*each", re.IGNORECASE)
ATTEMPT_ANY_RE = re.compile(
    r"attempt\s+any\s+(\d+)|answer\s+any\s+(\d+)|do\s+any\s+(\d+)",
    re.IGNORECASE,
)
OR_LINE_RE = re.compile(r"^\s*-{0,3}\s*OR\s*-{0,3}\s*$", re.IGNORECASE)

# Matches question/answer numbering like "1.", "1)", "Q1.", "Q.1", "Ans 1:", "Answer 1.",
# "(1).", "1).", and also "9_OR." (the ID format used for OR-choice alternates).
# The trailing punctuation class is "one or more" so reversed/doubled
# punctuation like ")." is consumed fully rather than leaking into the text.
NUMBERING_RE = re.compile(
    r"^\s*\(?\s*(?:Q\.?\s*|Ans(?:wer)?\.?\s*)?(\d+(?:_OR)?)\s*[\.\):]+\s*(.*)$",
    re.IGNORECASE,
)

# Fallback for lettered/roman-numeral sub-parts common in university exams:
# "1.a", "1(a)", "1a)", "1.i", "1(i)", "1(ii)", "(1.a)", "1.a)", etc.
# Only tried when NUMBERING_RE above doesn't match.
#
# Roman numerals are listed longest-first in the alternation (viii before
# iii before ii before i, etc.) so the regex engine doesn't grab a short
# prefix of a longer numeral.
#
# Guarded against a real false-positive risk: an ordinary answer that
# simply starts with the word "A" or "I" (e.g. "1 A deadlock occurs...",
# "1 I think that...") must NOT be mistaken for subpart "1a"/"1i". This is
# enforced by requiring EITHER punctuation immediately before the subpart
# marker (".", "(") OR punctuation immediately after it (".", ")", ":") —
# a bare single letter surrounded only by spaces is never treated as a
# subpart marker.
_ROMAN = r"(?:viii|iii|vii|ii|iv|vi|ix|i|v|x)"
_SUBPART_TOKEN = rf"(?:{_ROMAN}|[a-hA-H])"
SUBPART_NUMBERING_RE = re.compile(
    r"^\s*\(?\s*(?:Q\.?\s*|Ans(?:wer)?\.?\s*)?(\d+)\s*"
    r"(?:"
    rf"[\.\(]({_SUBPART_TOKEN})(?![a-zA-Z])\s*[\.\):]*"     # punctuation BEFORE the letter (trailing optional)
    r"|"
    rf"({_SUBPART_TOKEN})(?![a-zA-Z])\s*[\.\):]+"           # punctuation AFTER the letter (required, since none came before)
    r")"
    r"\s*\)?\s*(.*)$",
    re.IGNORECASE,
)


def _match_numbering(line: str):
    """
    Tries to match a line against a question/answer numbering pattern.
    Returns (question_id, rest_of_line_text) or None if no match.
    Tries the more SPECIFIC sub-part pattern first (e.g. "1(a)", "1.i",
    "(1.a)") — trying the general plain-number pattern first would
    incorrectly match just the "1." in "1.a" and leave "a" stuck in the
    answer text. Falls back to the general plain-number pattern for
    everything else (e.g. "1.", "1)", "Ans 1:").
    """
    m = SUBPART_NUMBERING_RE.match(line)
    if m:
        subpart = m.group(2) or m.group(3)
        rest = m.group(4)
        return f"{m.group(1)}{subpart.lower()}", rest

    m = NUMBERING_RE.match(line)
    if m:
        return m.group(1), m.group(2)

    return None

INLINE_MARKS_RE = re.compile(r"[\(\[]\s*(\d+(?:\.\d+)?)\s*(?:marks?)?\s*[\)\]]\s*$", re.IGNORECASE)


@dataclass
class ParsedQuestion:
    question_number: str
    section: Optional[str]
    question_text: str
    max_marks: float
    choice_group: Optional[str] = None    # questions sharing a group_id are alternatives/options
    choice_required: Optional[int] = None  # how many from this group count toward the total


def parse_question_paper(text: str, default_marks_if_unspecified: float = 1.0) -> List[ParsedQuestion]:
    """
    Detects sections, question count, marks-per-question, AND optional/
    choice question groups (internal "OR" choices and "attempt any N of M"
    sections) from raw question-paper text (typed or OCR'd).
    """
    lines = [l for l in text.split("\n")]
    questions: List[ParsedQuestion] = []

    current_section = None
    current_section_default_marks = None
    current_section_choice_required = None   # set by "attempt any N of..." in/near a SECTION header
    current_section_group_id = None

    buffer_qnum, buffer_lines = None, []
    last_flushed_qnum = None
    expecting_or_alt_for = None  # question_number we expect an "OR" alternate for

    def flush():
        nonlocal buffer_qnum, buffer_lines, last_flushed_qnum
        if buffer_qnum is not None:
            full_text = " ".join(l.strip() for l in buffer_lines).strip()
            marks = None
            m = INLINE_MARKS_RE.search(full_text)
            if m:
                marks = float(m.group(1))
                full_text = INLINE_MARKS_RE.sub("", full_text).strip()
            if marks is None:
                marks = current_section_default_marks or default_marks_if_unspecified

            is_or_alt = expecting_or_alt_for == buffer_qnum
            qid = f"{buffer_qnum}_OR" if is_or_alt else buffer_qnum

            choice_group = None
            choice_required = None
            if is_or_alt:
                choice_group = f"choice_{buffer_qnum}"
                choice_required = 1
                # retroactively tag the primary version of this question too
                for q in questions:
                    if q.question_number == buffer_qnum and q.choice_group is None:
                        q.choice_group = choice_group
                        q.choice_required = 1
            elif current_section_choice_required:
                choice_group = current_section_group_id
                choice_required = current_section_choice_required

            questions.append(
                ParsedQuestion(
                    question_number=qid,
                    section=current_section,
                    question_text=full_text,
                    max_marks=marks,
                    choice_group=choice_group,
                    choice_required=choice_required,
                )
            )
            last_flushed_qnum = buffer_qnum
        buffer_qnum, buffer_lines = None, []

    for line in lines:
        if not line.strip():
            continue

        sec_match = SECTION_RE.match(line)
        if sec_match:
            flush()
            current_section = f"Section {sec_match.group(1).upper()}"
            marks_match = SECTION_MARKS_RE.search(line)
            current_section_default_marks = float(marks_match.group(1)) if marks_match else None
            attempt_match = ATTEMPT_ANY_RE.search(line)
            if attempt_match:
                n = next(g for g in attempt_match.groups() if g is not None)
                current_section_choice_required = int(n)
                current_section_group_id = f"group_{current_section}"
            else:
                current_section_choice_required = None
                current_section_group_id = None
            continue

        if OR_LINE_RE.match(line):
            flush()
            expecting_or_alt_for = last_flushed_qnum
            continue

        # A standalone "attempt any N of M" instruction line (not on the SECTION line itself)
        if buffer_qnum is None:
            attempt_match = ATTEMPT_ANY_RE.search(line)
            if attempt_match and not _match_numbering(line):
                n = next(g for g in attempt_match.groups() if g is not None)
                current_section_choice_required = int(n)
                current_section_group_id = f"group_{current_section}"
                continue

        q_match = _match_numbering(line)
        if q_match:
            flush()
            buffer_qnum, rest_text = q_match
            buffer_lines = [rest_text]
            if expecting_or_alt_for != buffer_qnum:
                expecting_or_alt_for = None  # this question wasn't the awaited OR-alternate; reset
        elif buffer_qnum is not None:
            buffer_lines.append(line)

    flush()
    return questions


def parse_numbered_text(text: str) -> Dict[str, str]:
    """
    Generic parser for anything numbered the same way as the question
    paper: an answer key ("1. Mitochondria...") or a student's OCR'd
    answer sheet ("Ans 1: Mitochondria is..."). Returns {question_number: text}.

    Answers do NOT need to be in order — this builds a dict keyed by
    whatever number/sub-part id each line declares, so a student who
    writes question 5 first and question 1 last is matched correctly.
    """
    lines = text.split("\n")
    result: Dict[str, str] = {}
    current_num, buffer_lines = None, []

    def flush():
        nonlocal current_num, buffer_lines
        if current_num is not None:
            result[current_num] = " ".join(l.strip() for l in buffer_lines).strip()
        current_num, buffer_lines = None, []

    for line in lines:
        if not line.strip():
            continue
        m = _match_numbering(line)
        if m:
            flush()
            current_num, rest_text = m
            buffer_lines = [rest_text]
        elif current_num is not None:
            buffer_lines.append(line)

    flush()
    return result


def match_student_answers(student_text: str, ordered_question_ids: List[str]) -> Dict[str, str]:
    """
    Matches a student's answer text to question ids, with a fallback for
    students who write no question numbers at all.

        1. Primary: parse_numbered_text() — works for ANY order, as long as
            the student writes some form of question number/sub-part label.
        2. Fallback: if NO numbered answers were detected anywhere in the
            text (the student wrote continuous unlabeled prose), split the
            text into paragraphs (blank-line separated) and assign them to
            questions IN THE PAPER'S DECLARED ORDER. This only activates when
            there is truly no numbering to go on, and requires the student to
            have answered in order for that specific case (there is no other
            way to disambiguate unlabeled answers).
        """
    numbered = parse_numbered_text(student_text)
    if numbered:
        return numbered

    # Fallback: split on blank lines into paragraphs, assign positionally.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", student_text) if p.strip()]
    return dict(zip(ordered_question_ids, paragraphs))


def build_question_bank(paper_text: str, answer_key_text: str):
    """
    Combines the parsed question paper + answer key into ready-to-grade
    Question objects (from rubric_scoring.py), with max_marks and section
    auto-detected and model_answer pulled from the matching answer-key entry.
    """
    from rubric_scoring import Question  # local import to avoid a hard dependency for pure parsing use

    parsed_questions = parse_question_paper(paper_text)
    answer_key = parse_numbered_text(answer_key_text)

    bank = {}
    missing_answers = []
    for pq in parsed_questions:
        model_answer = answer_key.get(pq.question_number, "")
        if not model_answer:
            missing_answers.append(pq.question_number)
        bank[pq.question_number] = Question(
            question_id=pq.question_number,
            question_text=pq.question_text,
            max_marks=pq.max_marks,
            model_answer=model_answer,
            choice_group=pq.choice_group,
            choice_required=pq.choice_required,
        )

    return bank, missing_answers


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python paper_parser.py <question_paper.txt>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        paper = f.read()

    parsed = parse_question_paper(paper)
    print(f"Detected {len(parsed)} questions:\n")
    for q in parsed:
        print(f"  [{q.section or '—'}] Q{q.question_number} ({q.max_marks} marks): {q.question_text[:70]}")