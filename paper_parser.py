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
# "(1).", "1).", and also "9_OR.",Handles optional Questions.
NUMBERING_RE = re.compile(
    r"^\s*\(?\s*(?:Q\.?\s*|Ans(?:wer)?\.?\s*)?(\d+(?:_OR)?)\s*[\.\):]+\s*(.*)$",
    re.IGNORECASE,
)

# Handle Roman numerals and lettered sub-parts. 
_ROMAN = r"(?:viii|iii|vii|ii|iv|vi|ix|i|v|x)"
_SUBPART_TOKEN = rf"(?:{_ROMAN}|[a-hA-H])"
SUBPART_NUMBERING_RE = re.compile(
    r"^\s*\(?\s*(?:Q\.?\s*|Ans(?:wer)?\.?\s*)?(\d+)\s*"
    r"(?:"
    rf"[\.\(]({_SUBPART_TOKEN})(?![a-zA-Z])\s*[\.\):]*"     
    r"|"
    rf"({_SUBPART_TOKEN})(?![a-zA-Z])\s*[\.\):]+"          
    r")"
    r"\s*\)?\s*(.*)$",
    re.IGNORECASE,
)

def _match_numbering(line: str):
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
    choice_group: Optional[str] = None   
    choice_required: Optional[int] = None

def parse_question_paper(text: str, default_marks_if_unspecified: float = 1.0) -> List[ParsedQuestion]:
    lines = [l for l in text.split("\n")]
    questions: List[ParsedQuestion] = []

    current_section = None
    current_section_default_marks = None
    current_section_choice_required = None
    current_section_group_id = None

    buffer_qnum, buffer_lines = None, []
    last_flushed_qnum = None
    expecting_or_alt_for = None

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

        # attempt any N of M"
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
                expecting_or_alt_for = None
        elif buffer_qnum is not None:
            buffer_lines.append(line)

    flush()
    return questions

def parse_numbered_text(text: str) -> Dict[str, str]:
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
    numbered = parse_numbered_text(student_text)
    if numbered:
        return numbered

    # Fallback: split on blank lines into paragraphs, assign positionally.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", student_text) if p.strip()]
    return dict(zip(ordered_question_ids, paragraphs))


def build_question_bank(paper_text: str, answer_key_text: str):
    from rubric_scoring import Question

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