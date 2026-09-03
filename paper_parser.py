import csv
import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SECTION_RE = re.compile(r"^\s*SECTION[\s\-]*([A-Z0-9]+)\b(.*)$", re.IGNORECASE)
# "Part I: Multiple Choice Questions (1 Mark Each)" style headers — a second,
# equally common convention alongside "SECTION A (N marks each)". Captures
# the part label, the descriptive title (used to detect MCQ vs Short Answer),
# and the marks-each figure.
PART_RE = re.compile(
    r"^\s*Part\s+([IVXLC]+|\d+)\s*[:\.\-]\s*(.*?)\s*(?:\((\d+(?:\.\d+)?)\s*Marks?\s*Each\))?\s*$",
    re.IGNORECASE,
)
SECTION_MARKS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*marks?\s*each", re.IGNORECASE)
ATTEMPT_ANY_RE = re.compile(r"attempt\s+any\s+(\d+)|answer\s+any\s+(\d+)|do\s+any\s+(\d+)", re.IGNORECASE)
OR_LINE_RE = re.compile(r"^\s*-{0,3}\s*OR\s*-{0,3}\s*$", re.IGNORECASE)

NUMBERING_RE = re.compile(
    r"^\s*\(?\s*(?:Q\.?\s*|Ans(?:wer)?\.?\s*)?(\d+(?:_OR)?)\s*[\.\):_\-]+\s*(.*)$",
    re.IGNORECASE,
)

_ROMAN = r"(?:viii|iii|vii|ii|iv|vi|ix|i|v|x)"
# The plain-letter branch is deliberately restricted to LOWERCASE only
# (?-i: turns off the pattern's global IGNORECASE for just this group),
# even though the rest of this regex is case-insensitive. This matters:
# real sub-part labels conventionally use lowercase ("1.a", "1(b)"),
# while MCQ answer-key entries conventionally use uppercase option
# letters ("1.B" meaning "the answer to Q1 is option B"). Without this
# distinction, an answer key line like "3.A" would be misread as
# "question 3, sub-part a" with an EMPTY answer, silently losing the
# actual answer content.
_SUBPART_TOKEN = rf"(?:{_ROMAN}|(?-i:[a-h]))"
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

INLINE_MARKS_RE = re.compile(r"[\(\[]\s*(\d+(?:\.\d+)?)\s*(?:marks?)?\s*[\)\]]\s*$", re.IGNORECASE)

# Extracts "A. High multicollinearity" style options out of a question's
# joined text (option lines from the paper get absorbed into the question
# text as continuation lines, but the "A." "B." "C." "D." markers survive
# intact — this pulls them back out into a clean {letter: text} mapping).
MCQ_OPTION_RE = re.compile(r"\b([A-D])[\.\)]\s*(.+?)(?=\s+[A-D][\.\)]\s|$)")


def parse_mcq_options(question_text: str) -> Optional[dict]:
    """
    Extracts MCQ option text from a question's raw text, e.g.
    "...? A. High multicollinearity B. Little or no autocorrelation..."
    -> {"A": "High multicollinearity", "B": "Little or no autocorrelation", ...}
    Returns None if fewer than 2 options are found (not a real MCQ, or
    the paper didn't format options this way).
    """
    matches = MCQ_OPTION_RE.findall(question_text)
    if len(matches) < 2:
        return None
    return {letter.upper(): text.strip() for letter, text in matches}


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


@dataclass
class ParsedQuestion:
    question_number: str
    section: Optional[str]
    question_text: str
    max_marks: float
    choice_group: Optional[str] = None
    choice_required: Optional[int] = None
    question_type: str = "Short_Answer"  # "MCQ" or "Short_Answer", detected from the Part/Section header


def parse_question_paper(text: str, default_marks_if_unspecified: float = 1.0) -> List[ParsedQuestion]:
    lines = [l for l in text.split("\n")]
    questions: List[ParsedQuestion] = []
    current_section = None
    current_section_default_marks = None
    current_section_choice_required = None
    current_section_group_id = None
    current_question_type = "Short_Answer"
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

            questions.append(ParsedQuestion(
                question_number=qid, section=current_section, question_text=full_text,
                max_marks=marks, choice_group=choice_group, choice_required=choice_required,
                question_type=current_question_type,
            ))
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

        part_match = PART_RE.match(line)
        if part_match:
            flush()
            part_label, part_title, part_marks = part_match.group(1), part_match.group(2), part_match.group(3)
            current_section = f"Part {part_label.upper()}"
            current_section_default_marks = float(part_marks) if part_marks else None
            current_question_type = "MCQ" if "multiple choice" in (part_title or "").lower() or "mcq" in (part_title or "").lower() else "Short_Answer"
            current_section_choice_required = None
            current_section_group_id = None
            continue

        if OR_LINE_RE.match(line):
            flush()
            expecting_or_alt_for = last_flushed_qnum
            continue

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
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", student_text) if p.strip()]
    return dict(zip(ordered_question_ids, paragraphs))


# ---------------------------------------------------------------------
# CSV-formatted answer key support (Question_Number,Type,Correct_Answer)
# ---------------------------------------------------------------------
# THE FIX: answer keys shaped like Mendeley's MCQ+Short-Answer format
# ("1,MCQ,B" / "21,Short_Answer,\"A type of ML where...\"") could not be
# understood by the plain-text answer-key parser above at all — its
# question-numbering regex requires punctuation like "." or ")" right
# after the number, and a bare CSV comma doesn't match that, so EVERY
# line silently failed to parse and the system had no model answers to
# grade against. build_question_bank() below now auto-detects this CSV
# shape and routes it through a dedicated CSV parser instead, and also
# carries the MCQ/Short_Answer type through so MCQs are graded by exact
# letter match rather than NLP similarity.

def _try_parse_csv_answer_key(text: str) -> Optional[Dict[str, dict]]:
    """
    Detects and parses a CSV-formatted answer key with columns
    Question_Number, Type, Correct_Answer. Returns
    {question_number: {"type": ..., "answer": ...}}, or None if the text
    doesn't look like this CSV shape (so callers can fall back to the
    plain-text parser instead).
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        reader = csv.DictReader(io.StringIO(stripped))
        fieldnames = reader.fieldnames or []
        normalized = {f.strip().lower() for f in fieldnames}
        required = {"question_number", "type", "correct_answer"}
        if not required.issubset(normalized):
            return None
        field_map = {f.strip().lower(): f for f in fieldnames}

        result = {}
        for row in reader:
            qnum = (row.get(field_map["question_number"]) or "").strip()
            qtype = (row.get(field_map["type"]) or "").strip()
            answer = (row.get(field_map["correct_answer"]) or "").strip()
            if qnum:
                result[qnum] = {"type": qtype, "answer": answer}
        return result if result else None
    except Exception:
        return None


def build_question_bank(paper_text: str, answer_key_text: str):
    """
    Combines the parsed question paper + answer key into ready-to-grade
    Question objects. The answer key is auto-detected as either the
    CSV format (Question_Number,Type,Correct_Answer) or the plain
    numbered-text format ("1. Mitochondria is...") — no manual
    configuration needed either way.
    """
    from rubric_scoring import Question

    parsed_questions = parse_question_paper(paper_text)

    csv_key = _try_parse_csv_answer_key(answer_key_text)
    if csv_key is not None:
        answer_key = {qnum: entry["answer"] for qnum, entry in csv_key.items()}
        type_map = {qnum: entry["type"] for qnum, entry in csv_key.items()}
    else:
        answer_key = parse_numbered_text(answer_key_text)
        type_map = {}

    bank = {}
    missing_answers = []
    for pq in parsed_questions:
        model_answer = answer_key.get(pq.question_number, "")
        if not model_answer:
            missing_answers.append(pq.question_number)

        # Question TYPE priority: what the question paper itself declares
        # (from "Part I: Multiple Choice Questions" style headers) wins,
        # since it's detected independently of whatever answer-key format
        # is in use. Only fall back to the answer key's Type column (CSV
        # format) if the paper didn't specify — this is what makes MCQ
        # grading work correctly regardless of which answer-key format
        # you upload (CSV with a Type column, or a plain "1.B" list).
        if pq.question_type == "MCQ":
            question_type = "MCQ"
        else:
            raw_type = type_map.get(pq.question_number, "Short_Answer")
            question_type = "MCQ" if raw_type.strip().upper() == "MCQ" else "Short_Answer"

        bank[pq.question_number] = Question(
            question_id=pq.question_number,
            question_text=pq.question_text,
            max_marks=pq.max_marks,
            model_answer=model_answer,
            choice_group=pq.choice_group,
            choice_required=pq.choice_required,
            question_type=question_type,
            options=parse_mcq_options(pq.question_text) if question_type == "MCQ" else None,
        )

    return bank, missing_answers