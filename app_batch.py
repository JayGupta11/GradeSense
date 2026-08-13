"""
app_batch.py
------------
GradeSense — web app for grading a WHOLE CLASS at once, no terminal needed.

Workflow:
    1. Upload the question paper — .txt OR .pdf (typed or scanned, handled
        independently of the answer key).
    2. Upload the answer key — .txt OR .pdf, independently of the paper.
        Sections, marks per question, and choice groups (OR-choices,
        "attempt N of M") are all auto-detected from the paper.
    3. Upload every student's answer sheet (.txt/.jpg/.png/.pdf), named by
        roll number. Answers are matched to questions by number regardless
        of the order they were written in, including lettered sub-parts
        (e.g. "1(a)"), with a positional fallback if a student writes no
        numbering at all.
    4. Grade the whole class in one pass, with class-level statistics and
        a downloadable CSV.

Run with:
    streamlit run app_batch.py
"""

import io
import os
import tempfile
import pandas as pd
import streamlit as st

from blur_detection import check_image_quality
from ocr_module import extract_text
from pdf_utils import ocr_pdf, extract_pdf_text, roll_number_from_filename
from batch_grade import grade_paper_texts, try_load_trained_model, DEFAULT_MODEL_PATH
from rubric_scoring import using_trained_model

st.set_page_config(page_title="GradeSense", page_icon="✅", layout="wide")

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

_model_active = try_load_trained_model(DEFAULT_MODEL_PATH)

# ---------------------------------------------------------------------
# Sidebar: branding + settings (Issue 4 — more options, better UI)
# ---------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=72)
    st.markdown("## GradeSense")
    st.caption("AI-assisted handwritten answer evaluation")
    st.divider()

    st.markdown("**Scoring engine**")
    if _model_active:
        st.success("Trained ML model active", icon="🤖")
    else:
        st.info("Rule-based scoring (no trained model found)", icon="⚙️")

    st.divider()
    st.markdown("**Settings**")
    blur_threshold = st.slider(
        "Blur sensitivity for scanned pages",
        min_value=20.0, max_value=200.0, value=80.0, step=5.0,
        help="Lower = stricter. A page below this sharpness score is still OCR'd but flagged as low quality.",
    )
    show_choice_notes = st.checkbox("Show choice-group resolution details", value=True)
    show_point_breakdown = st.checkbox("Show per-point rubric breakdown", value=True)
    sort_by = st.selectbox("Sort class results by", ["Roll Number", "Marks Obtained (high to low)", "Marks Obtained (low to high)", "Percentage"])

st.title("✅ GradeSense — Batch Grade a Whole Class")
st.caption(
    "Upload the question paper and answer key once (text or PDF), then upload "
    "every student's answer sheet — named by roll number — to grade the entire "
    "class in one pass."
)


def _read_text_or_pdf(uploaded_file):
    """Reads an uploaded .txt or .pdf file into plain text. For PDFs, uses
    the real text layer when available (typed question papers/answer keys)
    and falls back to OCR per page automatically for scanned pages."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    raw = uploaded_file.read()
    if ext == ".pdf":
        return extract_pdf_text(raw)
    return raw.decode("utf-8")


# ---------------------------------------------------------------------
# Step 1 & 2: Question paper + answer key — INDEPENDENT uploads, PDF or txt
# ---------------------------------------------------------------------

tab_setup, tab_students, tab_results = st.tabs(["1. Question Paper & Answer Key", "2. Student Answer Sheets", "3. Results"])

with tab_setup:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Question paper")
        paper_file = st.file_uploader("Upload question paper (.txt or .pdf)", type=["txt", "pdf"], key="paper_upload")
        paper_text_area = st.text_area(
            "...or paste the question paper text here",
            height=220,
            placeholder=(
                "SECTION A (7 marks each)\n"
                "1. Explain the concept of deadlock.\n\n"
                "SECTION B (10 marks each) [Attempt any 2 of the following 3]\n"
                "2. Explain process scheduling.\n"
                "3. Describe memory management.\n"
                "4. Explain virtual memory."
            ),
        )
        paper_text = _read_text_or_pdf(paper_file) if paper_file else paper_text_area
        if paper_file:
            st.success(f"Loaded question paper from {paper_file.name}")

    with col2:
        st.subheader("🔑 Answer key")
        key_file = st.file_uploader("Upload answer key (.txt or .pdf)", type=["txt", "pdf"], key="key_upload")
        key_text_area = st.text_area(
            "...or paste the answer key text here",
            height=220,
            placeholder=(
                "1. A deadlock occurs when processes wait indefinitely for resources...\n"
                "2. Process scheduling algorithms include FCFS, SJF, Round Robin..."
            ),
        )
        answer_key_text = _read_text_or_pdf(key_file) if key_file else key_text_area
        if key_file:
            st.success(f"Loaded answer key from {key_file.name}")

    st.info(
        "Question paper and answer key are read **independently** — upload each as "
        "PDF (typed or scanned) or plain text, in any combination. Sections, marks "
        "per question (any value — 1, 2, 3, 5, 7, 10...), and optional/choice "
        "questions are auto-detected from the paper.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------------
# Step 3: Student answer sheets (many at once)
# ---------------------------------------------------------------------

with tab_students:
    st.subheader("🧑‍🎓 Student answer sheets")
    st.caption(
        "Name each file by roll number, e.g. **12345.pdf**, **22011.jpg**, **A101.txt**. "
        "Answers are matched to questions by number regardless of the order they were "
        "written in — including lettered sub-parts like **1(a)**, **1(b)** — and a "
        "student may skip numbering entirely if they answer strictly in question order."
    )

    student_files = st.file_uploader(
        "Upload student answer sheets",
        type=["txt", "jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        key="student_uploads",
    )

    if student_files:
        st.info(f"{len(student_files)} student file(s) ready: " + ", ".join(f.name for f in student_files))

    grade_clicked = st.button(
        "🚀 Grade All Students",
        type="primary",
        disabled=not (paper_text and answer_key_text and student_files),
        use_container_width=True,
    )
    if not (paper_text and answer_key_text and student_files):
        st.caption("Upload the question paper, answer key, and at least one student file to enable grading.")

# ---------------------------------------------------------------------
# Step 4: Grade everyone + results tab
# ---------------------------------------------------------------------

with tab_results:
    if "class_rows" not in st.session_state:
        st.session_state.class_rows = None
        st.session_state.per_student_details = None
        st.session_state.quality_warnings = None

    if grade_clicked:
        class_rows = []
        per_student_details = {}
        quality_warnings = []

        progress = st.progress(0.0, text="Starting...")

        for i, sfile in enumerate(student_files):
            roll = roll_number_from_filename(sfile.name)
            progress.progress(i / len(student_files), text=f"Processing {roll}...")

            ext = os.path.splitext(sfile.name)[1].lower()
            raw_bytes = sfile.read()

            try:
                if ext == ".txt":
                    student_sheet_text = raw_bytes.decode("utf-8")
                elif ext == ".pdf":
                    student_sheet_text, page_results = ocr_pdf(raw_bytes, blur_threshold=blur_threshold)
                    for pr in page_results:
                        if pr.quality_warning:
                            quality_warnings.append(
                                f"{roll}: page {pr.page_number} is low quality (sharpness={pr.blur_score}) — verify manually."
                            )
                elif ext in (".jpg", ".jpeg", ".png"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(raw_bytes)
                        tmp_path = tmp.name
                    quality = check_image_quality(tmp_path, threshold=blur_threshold)
                    if not quality["accepted"]:
                        quality_warnings.append(
                            f"{roll}: image too blurry (sharpness={quality['blur_score']}) — SKIPPED, please re-upload a sharper photo."
                        )
                        os.unlink(tmp_path)
                        continue
                    student_sheet_text = extract_text(tmp_path)
                    os.unlink(tmp_path)
                else:
                    quality_warnings.append(f"{roll}: unsupported file type '{ext}', skipped.")
                    continue

                results, total_awarded, total_max, notes, missing_answers, unanswered = grade_paper_texts(
                    paper_text, answer_key_text, student_sheet_text
                )

                class_rows.append({
                    "Roll Number": roll,
                    "Marks Obtained": round(total_awarded, 2),
                    "Max Marks": total_max,
                    "Percentage": round(100 * total_awarded / total_max, 1) if total_max else 0,
                    "Unanswered Questions": ", ".join(unanswered) if unanswered else "-",
                })
                per_student_details[roll] = (results, notes)

                if missing_answers:
                    quality_warnings.append(
                        f"Answer key is missing entries for question(s) {missing_answers} — these could not be graded for any student."
                    )

            except Exception as e:
                quality_warnings.append(f"{roll}: FAILED to process ({e})")

        progress.progress(1.0, text="Done.")
        st.session_state.class_rows = class_rows
        st.session_state.per_student_details = per_student_details
        st.session_state.quality_warnings = quality_warnings

    class_rows = st.session_state.class_rows
    per_student_details = st.session_state.per_student_details
    quality_warnings = st.session_state.quality_warnings

    if quality_warnings:
        with st.expander(f"⚠️ {len(quality_warnings)} warning(s) — click to review", expanded=False):
            for w in quality_warnings:
                st.write(f"- {w}")

    if class_rows:
        df = pd.DataFrame(class_rows)
        sort_map = {
            "Roll Number": ("Roll Number", True),
            "Marks Obtained (high to low)": ("Marks Obtained", False),
            "Marks Obtained (low to high)": ("Marks Obtained", True),
            "Percentage": ("Percentage", False),
        }
        sort_col, ascending = sort_map[sort_by]
        df = df.sort_values(sort_col, ascending=ascending)

        st.subheader("📊 Class summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students graded", len(df))
        c2.metric("Class average", f"{df['Percentage'].mean():.1f}%")
        c3.metric("Highest", f"{df['Percentage'].max():.1f}%")
        c4.metric("Lowest", f"{df['Percentage'].min():.1f}%")

        st.bar_chart(df.set_index("Roll Number")["Percentage"])

        st.subheader("📋 Class results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download class results (CSV)",
            data=csv_bytes,
            file_name="class_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("🔍 Per-student breakdown")
        selected_roll = st.selectbox("View detailed breakdown for:", sorted(per_student_details.keys()))
        if selected_roll:
            results, notes = per_student_details[selected_roll]
            for r in results:
                with st.expander(f"{r.question_id} — {r.awarded_marks}/{r.max_marks} marks"):
                    st.write(f"**Expected length:** ~{r.expected_length_words} words | **Actual:** {r.actual_length_words} words")
                    if show_point_breakdown:
                        for pr in r.point_results:
                            st.write(f"- [{pr.awarded_marks}/{pr.point_max_marks}] {pr.point_text}")
                    st.write(f"**Feedback:** {r.feedback}")
            if notes and show_choice_notes:
                st.write("**Choice-group resolution:**")
                for n in notes:
                    st.write(f"- {n}")
    elif grade_clicked:
        st.warning("No students were successfully graded — check the warnings above.")
    else:
        st.info("Upload files and click **Grade All Students** to see results here.")