"""
app_batch.py
------------
GradeSense — lightweight Streamlit frontend for whole-class grading.

The frontend contains the UI, authentication, MongoDB integration, history,
and result display. Heavy OCR/model inference is delegated to a separate
GPU backend through GRADESENSE_BACKEND_URL.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import requests
import database as db

import os
import tempfile
import pandas as pd
import streamlit as st

import requests
import database as db
import auth

st.set_page_config(
    page_title="GradeSense | AI Answer Evaluation",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --gs-primary: #6366f1;
    --gs-primary-2: #8b5cf6;
    --gs-success: #10b981;
    --gs-warning: #f59e0b;
    --gs-danger: #ef4444;
    --gs-border: rgba(148, 163, 184, .20);
    --gs-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(99,102,241,.12), transparent 30%),
        radial-gradient(circle at 92% 8%, rgba(139,92,246,.10), transparent 28%),
        linear-gradient(180deg, rgba(15,23,42,.03), transparent 25%);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--gs-border);
    background: rgba(15, 23, 42, .20);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

.gs-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 20px;
}

.gs-brand-mark {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    font-size: 22px;
    background: linear-gradient(135deg, var(--gs-primary), var(--gs-primary-2));
    box-shadow: 0 10px 30px rgba(99,102,241,.28);
}

.gs-brand-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.4px;
}

.gs-brand-sub {
    color: var(--gs-muted);
    font-size: 11px;
    margin-top: 1px;
}

.gs-hero {
    position: relative;
    overflow: hidden;
    padding: 28px 30px;
    border: 1px solid var(--gs-border);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(99,102,241,.13), rgba(139,92,246,.07) 45%, rgba(15,23,42,.10));
    box-shadow: 0 18px 55px rgba(2,6,23,.10);
    animation: gsFadeUp .55s ease both;
}

.gs-hero:after {
    content: '';
    position: absolute;
    width: 230px;
    height: 230px;
    right: -90px;
    top: -100px;
    border-radius: 50%;
    background: rgba(99,102,241,.14);
    filter: blur(8px);
    animation: gsFloat 5s ease-in-out infinite;
}

.gs-eyebrow {
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 7px;
}

.gs-title {
    font-size: clamp(28px, 4vw, 42px);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -1.3px;
    margin: 0;
}

.gs-subtitle {
    color: var(--gs-muted);
    max-width: 760px;
    margin-top: 10px;
    font-size: 14px;
    line-height: 1.65;
}

.gs-status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}

.gs-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border-radius: 999px;
    border: 1px solid var(--gs-border);
    background: rgba(15,23,42,.20);
    font-size: 11px;
    color: #cbd5e1;
}

.gs-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--gs-success);
    box-shadow: 0 0 0 4px rgba(16,185,129,.12);
}

.gs-dot-purple {
    background: var(--gs-primary);
    box-shadow: 0 0 0 4px rgba(99,102,241,.12);
}

.gs-progress {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 22px 0 28px;
    animation: gsFadeUp .65s ease both;
}

.gs-step {
    position: relative;
    padding: 13px 15px;
    border-radius: 15px;
    border: 1px solid var(--gs-border);
    background: rgba(15,23,42,.12);
    transition: transform .22s ease, border-color .22s ease, background .22s ease;
}

.gs-step:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,.38);
}

.gs-step.active {
    border-color: rgba(99,102,241,.55);
    background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(139,92,246,.08));
    box-shadow: 0 10px 30px rgba(99,102,241,.10);
}

.gs-step.done {
    border-color: rgba(16,185,129,.30);
    background: rgba(16,185,129,.07);
}

.gs-step-num {
    display: inline-grid;
    place-items: center;
    width: 26px;
    height: 26px;
    border-radius: 9px;
    font-size: 11px;
    font-weight: 800;
    margin-right: 8px;
    background: rgba(148,163,184,.12);
}

.gs-step.active .gs-step-num {
    background: var(--gs-primary);
    color: white;
}

.gs-step.done .gs-step-num {
    background: var(--gs-success);
    color: white;
}

.gs-step-label {
    font-size: 12px;
    font-weight: 700;
}

.gs-step-state {
    color: var(--gs-muted);
    font-size: 10px;
    margin-top: 5px;
    padding-left: 34px;
}

.gs-section {
    margin: 10px 0 18px;
    animation: gsFadeUp .45s ease both;
}

.gs-section-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -.5px;
    margin-bottom: 4px;
}

.gs-section-sub {
    color: var(--gs-muted);
    font-size: 12px;
}

.gs-card {
    border: 1px solid var(--gs-border);
    border-radius: 20px;
    padding: 20px;
    background: rgba(15,23,42,.12);
    box-shadow: 0 12px 38px rgba(2,6,23,.07);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: gsFadeUp .5s ease both;
}

.gs-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 45px rgba(2,6,23,.11);
    border-color: rgba(99,102,241,.28);
}

.gs-card-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    font-size: 19px;
    background: rgba(99,102,241,.12);
    margin-bottom: 13px;
}

.gs-card-title {
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 5px;
}

.gs-card-text {
    color: var(--gs-muted);
    font-size: 11px;
    line-height: 1.6;
}

.gs-metric {
    border: 1px solid var(--gs-border);
    border-radius: 18px;
    padding: 17px 18px;
    background: rgba(15,23,42,.11);
    transition: transform .2s ease;
    animation: gsFadeUp .5s ease both;
}

.gs-metric:hover { transform: translateY(-2px); }
.gs-metric-label { color: var(--gs-muted); font-size: 11px; font-weight: 600; }
.gs-metric-value { font-size: 25px; font-weight: 800; margin-top: 5px; letter-spacing: -.7px; }

.gs-alert {
    border-radius: 15px;
    border: 1px solid rgba(245,158,11,.28);
    background: rgba(245,158,11,.07);
    padding: 12px 15px;
    font-size: 12px;
    color: #fbbf24;
    margin: 10px 0;
}

.gs-success {
    border-color: rgba(16,185,129,.25);
    background: rgba(16,185,129,.07);
    color: #6ee7b7;
}

.gs-empty {
    text-align: center;
    border: 1px dashed rgba(148,163,184,.28);
    border-radius: 20px;
    padding: 34px 20px;
    color: var(--gs-muted);
}

.gs-empty-icon { font-size: 30px; margin-bottom: 7px; }
.gs-empty-title { color: inherit; font-weight: 800; font-size: 14px; }
.gs-empty-text { font-size: 11px; margin-top: 4px; }

div[data-testid="stFileUploader"] {
    border-radius: 16px;
    transition: transform .2s ease, border-color .2s ease;
}

div[data-testid="stFileUploader"]:hover { transform: translateY(-1px); }

.stButton > button, .stDownloadButton > button {
    border-radius: 12px !important;
    min-height: 42px;
    font-weight: 700 !important;
    border: 1px solid var(--gs-border) !important;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 9px 25px rgba(2,6,23,.12);
    border-color: rgba(99,102,241,.35) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gs-primary), var(--gs-primary-2)) !important;
    border: none !important;
    box-shadow: 0 9px 26px rgba(99,102,241,.22);
}

.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
    border-radius: 11px !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--gs-primary), var(--gs-primary-2));
}

[data-testid="stMetricValue"] { font-weight: 800; }

@keyframes gsFadeUp {
    from { opacity: 0; transform: translateY(9px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes gsFloat {
    0%, 100% { transform: translate(0,0); }
    50% { transform: translate(-15px, 12px); }
}

@media (max-width: 760px) {
    .gs-progress { grid-template-columns: 1fr; }
    .gs-hero { padding: 22px; border-radius: 19px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Auth gate — must sign in before anything else renders
# ---------------------------------------------------------------------
auth.require_login()

if "step" not in st.session_state:
    st.session_state.step = 1
if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""
if "answer_key_text" not in st.session_state:
    st.session_state.answer_key_text = ""
if "class_rows" not in st.session_state:
    st.session_state.class_rows = None
    st.session_state.per_student_details = None
    st.session_state.quality_warnings = None

BACKEND_API_URL = os.environ.get("GRADESENSE_BACKEND_URL", "").rstrip("/")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=58)

    st.markdown(
        """
        <div class="gs-brand">
            <div class="gs-brand-mark">✓</div>
            <div>
                <div class="gs-brand-title">GradeSense</div>
                <div class="gs-brand-sub">AI answer evaluation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Signed in as **{st.session_state.username}**")
    auth.logout_button()
    st.divider()

    st.markdown("**System status**")
    if BACKEND_API_URL:
        st.success("AI backend connected", icon="🤖")
    else:
        st.warning("AI backend URL not configured", icon="⚠️")

    st.divider()
    st.markdown("**Grading settings**")
    blur_threshold = st.slider(
        "Blur sensitivity for scanned pages",
        20.0,
        200.0,
        80.0,
        5.0,
    )
    show_choice_notes = st.checkbox(
        "Show choice-group resolution details",
        value=True,
    )
    show_point_breakdown = st.checkbox(
        "Show per-point rubric breakdown",
        value=True,
    )
    sort_by = st.selectbox(
        "Sort class results by",
        [
            "Roll Number",
            "Marks Obtained (high to low)",
            "Marks Obtained (low to high)",
            "Percentage",
        ],
    )

    st.divider()
    with st.expander("📜 My grading history"):
        try:
            history = db.get_results_for_user(
                st.session_state.username,
                limit=20,
            )
            if history:
                hist_df = pd.DataFrame([
                    {
                        "Paper": h.get("paper_name", "-"),
                        "Roll": h["roll_number"],
                        "Marks": h["marks_obtained"],
                        "Max": h["max_marks"],
                        "%": h["percentage"],
                    }
                    for h in history
                ])
                st.dataframe(
                    hist_df,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No graded results yet.")
        except Exception as e:
            st.caption(f"History unavailable: {e}")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
step = st.session_state.step

step_data = [
    (1, "Setup", "Question paper & key"),
    (2, "Students", "Answer sheets"),
    (3, "Results", "Class performance"),
]

progress_html = '<div class="gs-progress">'
for number, short, detail in step_data:
    if step > number:
        cls = "done"
        state = "Completed"
        num = "✓"
    elif step == number:
        cls = "active"
        state = "In progress"
        num = str(number)
    else:
        cls = ""
        state = "Upcoming"
        num = str(number)

    progress_html += (
        f'<div class="gs-step {cls}">'
        f'<span class="gs-step-num">{num}</span>'
        f'<span class="gs-step-label">{short}</span>'
        f'<div class="gs-step-state">{detail} · {state}</div>'
        f'</div>'
    )
progress_html += '</div>'

st.markdown(
    f"""
    <div class="gs-hero">
        <div class="gs-eyebrow">AI-assisted assessment workspace</div>
        <div class="gs-title">GradeSense</div>
        <div class="gs-subtitle">
            Evaluate an entire class from one workspace. Upload the paper and key once,
            add student answer sheets, and get structured marks, feedback, unanswered
            questions, and class-level performance.
        </div>
        <div class="gs-status-row">
            <span class="gs-status"><span class="gs-dot"></span> Secure session</span>
            <span class="gs-status"><span class="gs-dot-purple"></span> Qwen OCR ready</span>
            <span class="gs-status">⚡ Batch grading</span>
        </div>
    </div>
    {progress_html}
    """,
    unsafe_allow_html=True,
)


def _uploaded_bytes(uploaded_file):
    if not uploaded_file:
        return None
    return uploaded_file.getvalue()


def _backend_evaluate(paper_bytes, paper_name, key_bytes, key_name, student_files, settings):
    if not BACKEND_API_URL:
        raise RuntimeError(
            "GRADESENSE_BACKEND_URL is not configured. Set it to the URL of the GPU/API backend."
        )

    files = [
        ("paper", (paper_name, paper_bytes, "application/octet-stream")),
        ("answer_key", (key_name, key_bytes, "application/octet-stream")),
    ]

    for student in student_files:
        files.append(
            (
                "students",
                (student.name, student.getvalue(), student.type or "application/octet-stream"),
            )
        )

    data = {
        "username": st.session_state.username,
        "blur_threshold": str(settings["blur_threshold"]),
        "show_choice_notes": str(settings["show_choice_notes"]).lower(),
        "show_point_breakdown": str(settings["show_point_breakdown"]).lower(),
    }

    response = requests.post(
        f"{BACKEND_API_URL}/evaluate",
        files=files,
        data=data,
        timeout=1800,
    )
    response.raise_for_status()
    payload = response.json()

    if not payload.get("success", True):
        raise RuntimeError(payload.get("message", "Backend evaluation failed."))

    return payload


def _section(title, subtitle):
    st.markdown(
        f"""
        <div class="gs-section">
            <div class="gs-section-title">{title}</div>
            <div class="gs-section-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _card(icon, title, text):
    st.markdown(
        f"""
        <div class="gs-card">
            <div class="gs-card-icon">{icon}</div>
            <div class="gs-card-title">{title}</div>
            <div class="gs-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =======================================================================
# STEP 1: Question paper & answer key
# =======================================================================
if st.session_state.step == 1:
    _section(
        "Assessment setup",
        "Prepare the reference material GradeSense will use for the whole class.",
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            '<div class="gs-card-icon">📄</div><div class="gs-card-title">Question paper</div>'
            '<div class="gs-card-text">Upload a PDF/TXT or paste the paper text below.</div>',
            unsafe_allow_html=True,
        )
        paper_file = st.file_uploader(
            "Upload question paper",
            type=["txt", "pdf"],
            key="paper_upload",
            label_visibility="collapsed",
        )
        paper_text_area = st.text_area(
            "Paste question paper",
            height=210,
            placeholder=(
                "SECTION A (7 marks each)\n"
                "1. Explain the concept of deadlock.\n\n"
                "SECTION B (10 marks each) [Attempt any 2 of the following 3]\n"
                "2. Explain process scheduling.\n"
                "3. Describe memory management.\n"
                "4. Explain virtual memory."
            ),
            key="paper_text_area",
            label_visibility="collapsed",
        )
        resolved_paper_text = paper_text_area
        if paper_file:
            st.session_state.paper_upload_bytes = paper_file.getvalue()
            st.session_state.paper_upload_name = paper_file.name
        if paper_file:
            st.success(
                f"Loaded question paper from {paper_file.name}",
                icon="✓",
            )
            db.save_uploaded_file_metadata(
                st.session_state.username,
                paper_file.name,
                "question_paper",
                paper_file.size,
            )

    with col2:
        st.markdown(
            '<div class="gs-card-icon">🔑</div><div class="gs-card-title">Answer key</div>'
            '<div class="gs-card-text">Provide the expected answers and marking reference.</div>',
            unsafe_allow_html=True,
        )
        key_file = st.file_uploader(
            "Upload answer key",
            type=["txt", "pdf"],
            key="key_upload",
            label_visibility="collapsed",
        )
        key_text_area = st.text_area(
            "Paste answer key",
            height=210,
            placeholder=(
                "1. A deadlock occurs when processes wait indefinitely for resources...\n"
                "2. Process scheduling algorithms include FCFS, SJF, Round Robin..."
            ),
            key="key_text_area",
            label_visibility="collapsed",
        )
        resolved_key_text = key_text_area
        if key_file:
            st.session_state.key_upload_bytes = key_file.getvalue()
            st.session_state.key_upload_name = key_file.name
        if key_file:
            st.success(
                f"Loaded answer key from {key_file.name}",
                icon="✓",
            )
            db.save_uploaded_file_metadata(
                st.session_state.username,
                key_file.name,
                "answer_key",
                key_file.size,
            )

    st.markdown(
        """
        <div class="gs-alert gs-success">
            <strong>Flexible assessment detection:</strong> question sections, marks,
            MCQs, short answers, and optional/choice questions are detected from the
            supplied reference material rather than hard-coded question counts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    paper_ready = bool(paper_file or resolved_paper_text.strip())
    key_ready = bool(key_file or resolved_key_text.strip())
    ready = paper_ready and key_ready

    col_back, _, col_next = st.columns([1, 2.5, 1], gap="medium")
    with col_next:
        if st.button(
            "Continue to students  →",
            type="primary",
            width="stretch",
            disabled=not ready,
            key="step1_next",
        ):
            st.session_state.paper_text = resolved_paper_text
            st.session_state.answer_key_text = resolved_key_text
            if not paper_file:
                st.session_state.paper_upload_bytes = resolved_paper_text.encode("utf-8")
                st.session_state.paper_upload_name = "question_paper.txt"
            if not key_file:
                st.session_state.key_upload_bytes = resolved_key_text.encode("utf-8")
                st.session_state.key_upload_name = "answer_key.txt"
            st.session_state.paper_name = (paper_file.name if paper_file else "question_paper.txt")
            st.session_state.step = 2
            st.rerun()

    if not ready:
        st.caption("Add both the question paper and answer key to continue.")


# =======================================================================
# STEP 2: Student answer sheets
# =======================================================================
elif st.session_state.step == 2:
    _section(
        "Student answer sheets",
        "Upload one or more student files. GradeSense matches detected answers by question number.",
    )

    st.markdown(
        """
        <div class="gs-card">
            <div class="gs-card-icon">🧑‍🎓</div>
            <div class="gs-card-title">Batch upload</div>
            <div class="gs-card-text">
                Use filenames such as <strong>12345.pdf</strong>, <strong>22011.jpg</strong>,
                or <strong>A101.txt</strong>. Students may skip questions, answer out of order,
                or use lettered sub-parts such as 1(a) and 1(b).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    student_files = st.file_uploader(
        "Upload student answer sheets",
        type=["txt", "jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        key="student_uploads",
        label_visibility="collapsed",
    )

    if student_files:
        st.markdown(
            f"<div class='gs-alert gs-success'><strong>{len(student_files)}</strong> student file(s) ready for grading.</div>",
            unsafe_allow_html=True,
        )

        preview_cols = st.columns(min(4, len(student_files)))
        for i, sfile in enumerate(student_files[:4]):
            with preview_cols[i % len(preview_cols)]:
                st.markdown(
                    f"<div class='gs-card'><div class='gs-card-title'>📎 {sfile.name}</div>"
                    f"<div class='gs-card-text'>{sfile.size / 1024:.1f} KB</div></div>",
                    unsafe_allow_html=True,
                )
        if len(student_files) > 4:
            st.caption(f"+ {len(student_files) - 4} more file(s)")

    st.divider()

    col_back, _, col_grade = st.columns([1, 2, 1], gap="medium")
    with col_back:
        if st.button("← Back to setup", width="stretch", key="step2_back"):
            st.session_state.step = 1
            st.rerun()

    with col_grade:
        grade_clicked = st.button(
            "🚀  Grade all students",
            type="primary",
            width="stretch",
            disabled=not student_files,
            key="grade_all",
        )

    if not student_files:
        st.markdown(
            """
            <div class="gs-empty">
                <div class="gs-empty-icon">📥</div>
                <div class="gs-empty-title">Waiting for answer sheets</div>
                <div class="gs-empty-text">Upload at least one student file to begin batch grading.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if grade_clicked:
        if not BACKEND_API_URL:
            st.error(
                "AI backend is not configured. Add GRADESENSE_BACKEND_URL in Render environment variables."
            )
            st.stop()

        try:
            with st.spinner("Sending files to the AI evaluation backend…"):
                payload = _backend_evaluate(
                    st.session_state.get("paper_upload_bytes", b""),
                    st.session_state.get("paper_upload_name", "question_paper.txt"),
                    st.session_state.get("key_upload_bytes", b""),
                    st.session_state.get("key_upload_name", "answer_key.txt"),
                    student_files,
                    {
                        "blur_threshold": blur_threshold,
                        "show_choice_notes": show_choice_notes,
                        "show_point_breakdown": show_point_breakdown,
                    },
                )

            class_rows = payload.get("class_rows", [])
            per_student_details = payload.get("per_student_details", {})
            quality_warnings = payload.get("quality_warnings", [])

            # Store returned results in MongoDB from the lightweight frontend.
            for row in class_rows:
                roll = str(row.get("Roll Number", row.get("roll_number", "Unknown")))
                marks = float(row.get("Marks Obtained", row.get("marks_obtained", 0)))
                max_marks = float(row.get("Max Marks", row.get("max_marks", 0)))
                details = per_student_details.get(roll, {})
                db.save_grading_result(
                    username=st.session_state.username,
                    paper_name=st.session_state.get("paper_name", "Assessment"),
                    roll_number=roll,
                    marks_obtained=marks,
                    max_marks=max_marks,
                    details=details if isinstance(details, dict) else {"details": details},
                )

            st.session_state.class_rows = class_rows
            st.session_state.per_student_details = per_student_details
            st.session_state.quality_warnings = quality_warnings
            st.session_state.step = 3
            st.rerun()

        except requests.RequestException as exc:
            st.error(f"Could not reach the AI backend: {exc}")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")


# =======================================================================
# STEP 3: Results
# =======================================================================
elif st.session_state.step == 3:
    class_rows = st.session_state.class_rows
    per_student_details = st.session_state.per_student_details
    quality_warnings = st.session_state.quality_warnings

    _section(
        "Class performance",
        "Review the batch outcome, inspect individual answers, and export the class report.",
    )

    col_back, col_new = st.columns([1, 1], gap="medium")
    with col_back:
        if st.button("← Back to student sheets", width="stretch", key="results_back"):
            st.session_state.step = 2
            st.rerun()
    with col_new:
        if st.button("＋ New grading session", width="stretch", key="results_new_top"):
            st.session_state.step = 1
            st.session_state.paper_text = ""
            st.session_state.answer_key_text = ""
            st.session_state.class_rows = None
            st.session_state.per_student_details = None
            st.session_state.quality_warnings = None
            st.rerun()

    if quality_warnings:
        with st.expander(
            f"⚠️ {len(quality_warnings)} quality warning(s) — review before finalizing",
            expanded=False,
        ):
            for w in quality_warnings:
                st.write(f"• {w}")

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

        avg = df["Percentage"].mean()
        highest = df["Percentage"].max()
        lowest = df["Percentage"].min()

        st.markdown("### 📊 Overview")
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(
                f"<div class='gs-metric'><div class='gs-metric-label'>STUDENTS GRADED</div>"
                f"<div class='gs-metric-value'>{len(df)}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='gs-metric'><div class='gs-metric-label'>CLASS AVERAGE</div>"
                f"<div class='gs-metric-value'>{avg:.1f}%</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='gs-metric'><div class='gs-metric-label'>HIGHEST</div>"
                f"<div class='gs-metric-value'>{highest:.1f}%</div></div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div class='gs-metric'><div class='gs-metric-label'>LOWEST</div>"
                f"<div class='gs-metric-value'>{lowest:.1f}%</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 📈 Performance")
        st.bar_chart(
            df.set_index("Roll Number")["Percentage"],
            height=330,
        )

        st.markdown("### 📋 Class results")
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download class results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="class_results.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("### 🔍 Per-student breakdown")
        selected_roll = st.selectbox(
            "View detailed breakdown for:",
            sorted(per_student_details.keys()),
        )

        if selected_roll:
            detail = per_student_details[selected_roll]
            if isinstance(detail, dict):
                results = detail.get("results", [])
                notes = detail.get("notes", [])
            elif isinstance(detail, (list, tuple)) and len(detail) == 2:
                results, notes = detail
            else:
                results, notes = [], []

            for r in results:
                if isinstance(r, dict):
                    question_id = r.get("question_id", r.get("questionId", "Question"))
                    awarded = r.get("awarded_marks", 0)
                    max_marks = r.get("max_marks", 0)
                    expected = r.get("expected_length_words")
                    actual = r.get("actual_length_words")
                    feedback = r.get("feedback", "")
                    point_results = r.get("point_results", [])
                else:
                    question_id = getattr(r, "question_id", "Question")
                    awarded = getattr(r, "awarded_marks", 0)
                    max_marks = getattr(r, "max_marks", 0)
                    expected = getattr(r, "expected_length_words", None)
                    actual = getattr(r, "actual_length_words", None)
                    feedback = getattr(r, "feedback", "")
                    point_results = getattr(r, "point_results", [])

                with st.expander(f"{question_id}  ·  {awarded}/{max_marks} marks"):
                    if expected is not None or actual is not None:
                        st.write(
                            f"**Expected length:** ~{expected or 0} words | "
                            f"**Actual:** {actual or 0} words"
                        )

                    if show_point_breakdown:
                        for pr in point_results:
                            if isinstance(pr, dict):
                                pa = pr.get("awarded_marks", 0)
                                pm = pr.get("point_max_marks", 0)
                                pt = pr.get("point_text", "")
                            else:
                                pa = getattr(pr, "awarded_marks", 0)
                                pm = getattr(pr, "point_max_marks", 0)
                                pt = getattr(pr, "point_text", "")
                            st.write(f"- [{pa}/{pm}] {pt}")

                    st.write(f"**Feedback:** {feedback}")

            if notes and show_choice_notes:
                st.write("**Choice-group resolution:**")
                for n in notes:
                    st.write(f"- {n}")

    else:
        st.markdown(
            """
            <div class="gs-empty">
                <div class="gs-empty-icon">📊</div>
                <div class="gs-empty-title">No grading results available</div>
                <div class="gs-empty-text">Check the quality warnings and try another grading session.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
