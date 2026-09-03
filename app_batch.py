import os
import tempfile
import uuid
import datetime as dt
from dataclasses import is_dataclass, asdict

import pandas as pd
import streamlit as st
from streamlit_cropper import st_cropper

import time

import io
from PIL import Image, ImageDraw

from blur_detection import check_image_quality
from ocr_module import extract_text
from pdf_utils import (
    ocr_pdf,
    extract_pdf_text,
    roll_number_from_filename,
)
from batch_grade import (
    grade_paper_texts,
    try_load_trained_model,
    DEFAULT_MODEL_PATH,
)

import database as db


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "logo.png",
)


# ============================================================
# PAGE CONFIG
# IMPORTANT: keep this before importing auth
# ============================================================

st.set_page_config(
    page_title="GradeSense | AI Answer Evaluation",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


import auth


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
        GLOBAL BACKGROUND
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 8%,
                rgba(37, 99, 235, 0.28),
                transparent 28%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(124, 58, 237, 0.28),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 88%,
                rgba(236, 72, 153, 0.18),
                transparent 28%
            ),
            linear-gradient(
                125deg,
                #070a12,
                #10172c,
                #21163b,
                #111a36,
                #070a12
            );

        background-size: 180% 180%;

        animation:
            gradesenseGradient 18s ease infinite;

        min-height: 100vh;
    }


    @keyframes gradesenseGradient {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }


    [data-testid="stHeader"] {
        background: transparent !important;
    }


    /* ======================================================
        MAIN WIDTH
       ====================================================== */

    [data-testid="stMainBlockContainer"] {
        max-width: 1240px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
    }


    /* ======================================================
        SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(17, 24, 39, 0.98),
                rgba(15, 23, 42, 0.98)
            ) !important;

        border-right:
            1px solid
            rgba(148, 163, 184, 0.12);
    }


    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }


    /* ======================================================
        TYPOGRAPHY
       ====================================================== */

    h1,
    h2,
    h3,
    h4 {
        letter-spacing: -0.5px;
    }


    h1 {
        font-weight: 850 !important;
    }


    h2,
    h3 {
        font-weight: 780 !important;
    }


    p,
    label,
    span {
        letter-spacing: 0.05px;
    }


    /* ======================================================
        TOP RIBBON
       ====================================================== */

    .gs-ribbon {
        padding: 12px 18px;
        border-radius: 15px;

        background:
            linear-gradient(
                100deg,
                rgba(37, 99, 235, 0.22),
                rgba(124, 58, 237, 0.22),
                rgba(236, 72, 153, 0.16)
            );

        border:
            1px solid
            rgba(129, 140, 248, 0.20);

        box-shadow:
            0 12px 35px
            rgba(0, 0, 0, 0.15);

        margin-bottom: 18px;
    }


    .gs-ribbon-title {
        font-size: 17px;
        font-weight: 800;
        color: #ffffff;
    }


    .gs-ribbon-subtitle {
        font-size: 11px;
        color: #aab4ca;
        margin-top: 2px;
    }


    /* ======================================================
        HERO
       ====================================================== */

    .gs-hero {
        padding: 28px;
        border-radius: 24px;

        background:
            linear-gradient(
                120deg,
                rgba(30, 64, 175, 0.58),
                rgba(79, 70, 229, 0.55),
                rgba(126, 34, 206, 0.45),
                rgba(190, 24, 93, 0.32)
            );

        background-size: 220% 220%;

        animation:
            gradesenseHero 12s ease infinite;

        border:
            1px solid
            rgba(165, 180, 252, 0.20);

        box-shadow:
            0 22px 55px
            rgba(0, 0, 0, 0.20);
    }


    @keyframes gradesenseHero {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    .gs-hero-kicker {
        color: #c7d2fe;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }


    .gs-hero-title {
        color: #ffffff;
        font-size: 38px;
        font-weight: 850;
        letter-spacing: -1.5px;
    }


    .gs-hero-text {
        color: #e0e7ff;
        font-size: 14px;
        line-height: 1.65;
        max-width: 820px;
        margin-top: 8px;
    }


    /* ======================================================
        STATUS CARDS
       ====================================================== */

    .gs-status {
        padding: 12px 15px;
        border-radius: 13px;

        border:
            1px solid
            rgba(148, 163, 184, 0.14);

        background:
            rgba(15, 23, 42, 0.55);

        color: #e5e7eb;

        font-size: 12px;
        font-weight: 650;
    }


    /* ======================================================
        CARDS
       ====================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 17px !important;
        border-color:
            rgba(148, 163, 184, 0.15) !important;

        background:
            rgba(15, 23, 42, 0.42) !important;
    }


    /* ======================================================
        BUTTONS
       ====================================================== */

    .stButton > button {
        min-height: 44px !important;
        border-radius: 11px !important;

        font-weight: 700 !important;

        border:
            1px solid
            rgba(148, 163, 184, 0.16) !important;

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            border-color 0.18s ease !important;
    }


    .stButton > button:hover {
        transform: translateY(-2px) !important;

        border-color:
            rgba(129, 140, 248, 0.50) !important;

        box-shadow:
            0 10px 28px
            rgba(99, 102, 241, 0.18) !important;
    }


    .stButton > button[kind="primary"] {
        color: #ffffff !important;

        background:
            linear-gradient(
                110deg,
                #2563eb,
                #6366f1,
                #8b5cf6,
                #ec4899,
                #2563eb
            ) !important;

        background-size: 300% 300% !important;

        animation:
            gradesenseButton 8s ease infinite !important;

        border: none !important;
    }


    @keyframes gradesenseButton {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }

    }


    /* ======================================================
        INPUTS
       ====================================================== */

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 11px !important;

        background:
            rgba(30, 32, 43, 0.92) !important;

        color: #ffffff !important;

        border:
            1px solid
            rgba(148, 163, 184, 0.18) !important;
    }


    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color:
            rgba(99, 102, 241, 0.80) !important;

        box-shadow:
            0 0 0 3px
            rgba(99, 102, 241, 0.12) !important;
    }


    /* ======================================================
        FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        border-radius: 14px !important;
    }


    [data-testid="stFileUploaderDropzone"] {
        background:
            rgba(30, 32, 43, 0.75) !important;

        border:
            1px dashed
            rgba(129, 140, 248, 0.32) !important;

        border-radius: 13px !important;
    }


    /* ======================================================
        METRICS
       ====================================================== */

    [data-testid="stMetric"] {
        padding: 16px;
        border-radius: 15px;

        background:
            rgba(15, 23, 42, 0.44);

        border:
            1px solid
            rgba(148, 163, 184, 0.12);
    }


    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
    }


    /* ======================================================
        DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 13px !important;
        overflow: hidden;
    }


    /* ======================================================
        PROGRESS
       ====================================================== */

    [data-testid="stProgressBar"] {
        height: 12px !important;
    }


    /* ======================================================
        FOOTER
       ====================================================== */

    .gs-footer {
        text-align: center;

        color: #7f8aa1;

        font-size: 12px;

        padding-top: 22px;

        white-space: nowrap;
    }


    /* ======================================================
        MOBILE
       ====================================================== */

    @media (max-width: 800px) {

        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .gs-hero-title {
            font-size: 29px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AUTHENTICATION
# ============================================================

auth.require_login()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "gs_page": "Dashboard",
    "gs_step": 1,
    "paper_text": "",
    "answer_key_text": "",
    "class_rows": None,
    "per_student_details": None,
    "quality_warnings": None,
    "session_id": None,
    "paper_name": "Untitled assessment",
    "blur_threshold": 80.0,
    "show_choice_notes": True,
    "show_point_breakdown": True,
    "default_sort": "Roll Number",
    "evaluation_version": str(uuid.uuid4()),
    "display_name": None,
}

for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODEL
# ============================================================

try:

    model_active = try_load_trained_model(
        DEFAULT_MODEL_PATH
    )

except Exception:

    model_active = False


# ============================================================
# HELPERS
# ============================================================

def _safe(value):

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):

        return {
            str(k): _safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            _safe(v)
            for v in value
        ]

    if hasattr(value, "__dict__"):

        return {
            str(k): _safe(v)
            for k, v in vars(value).items()
        }

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


def _history():

    try:

        return db.get_results_for_user(
            st.session_state.username,
            limit=500,
        )

    except Exception:

        return []


def _fmt_date(value):

    if isinstance(
        value,
        dt.datetime,
    ):

        return value.strftime(
            "%d %b %Y · %I:%M %p"
        )

    return "Date unavailable"


def _session_groups(history):

    groups = {}

    for item in history:

        details = (
            item.get("details")
            or {}
        )

        session_id = details.get(
            "session_id"
        )

        created = item.get(
            "created_at"
        )

        if session_id:

            key = str(
                session_id
            )

        elif isinstance(
            created,
            dt.datetime,
        ):

            key = (
                f"legacy-"
                f"{item.get('paper_name', 'Assessment')}-"
                f"{created.strftime('%Y%m%d%H%M')}"
            )

        else:

            key = (
                f"legacy-"
                f"{item.get('paper_name', 'Assessment')}"
            )

        groups.setdefault(
            key,
            [],
        ).append(item)

    return sorted(
        groups.values(),
        key=lambda rows: max(
            (
                r.get("created_at")
                for r in rows
                if isinstance(
                    r.get("created_at"),
                    dt.datetime,
                )
            ),
            default=dt.datetime.min,
        ),
        reverse=True,
    )


def _reset_evaluation():

    st.session_state.gs_step = 1

    st.session_state.paper_text = ""

    st.session_state.answer_key_text = ""

    st.session_state.class_rows = None

    st.session_state.per_student_details = None

    st.session_state.quality_warnings = None

    st.session_state.session_id = None

    st.session_state.paper_name = (
        "Untitled assessment"
    )

    st.session_state.evaluation_version = (
        str(uuid.uuid4())
    )

    st.session_state.gs_page = (
        "New Evaluation"
    )


def _read_uploaded_file(
    uploaded_file,
):

    if not uploaded_file:
        return ""

    ext = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    raw = uploaded_file.getvalue()

    if ext == ".pdf":

        return extract_pdf_text(
            raw
        )

    return raw.decode(
        "utf-8",
        errors="replace",
    )

def _get_profile_picture():
    username = st.session_state.get("username")

    if not username:
        return None

    try:
        user = (
            db.get_db()
            .users
            .find_one(
                {"username": username},
                {"profile_picture": 1}
            )
            or {}
        )

        picture = user.get("profile_picture")

        if picture:
            return picture

    except Exception:
        pass

    return None


def _make_circular_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGBA")

    image.thumbnail((256, 256))

    canvas = Image.new(
        "RGBA",
        (256, 256),
        (0, 0, 0, 0)
    )

    x = (256 - image.width) // 2
    y = (256 - image.height) // 2

    canvas.paste(
        image,
        (x, y),
        image
    )

    mask = Image.new(
        "L",
        (256, 256),
        0
    )

    draw = ImageDraw.Draw(mask)

    draw.ellipse(
        (0, 0, 255, 255),
        fill=255
    )

    canvas.putalpha(mask)

    output = io.BytesIO()

    canvas.save(
        output,
        format="PNG"
    )

    return output.getvalue()

def _display_name():

    return (
        st.session_state.get(
            "display_name"
        )
        or st.session_state.get(
            "username"
        )
        or "User"
    )


# ============================================================
# TOP RIBBON
# ============================================================

def _top_ribbon():

    logo_col, brand_col, workspace_col, profile_col = st.columns(
        [1, 2, 6, 2],
        vertical_alignment="center",
    )

    with logo_col:
        if os.path.exists(LOGO_PATH):
            st.image(
                LOGO_PATH,
                width=58,
            )

    with brand_col:
        st.subheader(
            "GradeSense"
        )

    with workspace_col:
        st.write("")
        st.markdown(
            "##### AI-powered handwritten answer evaluation workspace"
        )

    with profile_col:

        profile_picture = _get_profile_picture()

        if profile_picture:
            st.image(
                profile_picture,
                width=58,
            )
        else:
            st.markdown(
                f"### 👤 {_display_name()}"
            )


# ============================================================
# HERO
# ============================================================

def _hero(
    title,
    subtitle,
):

    st.header(
        title
    )

    st.write(
        subtitle
    )

    st.write("")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.success(
            "Secure session",
            icon="🔐",
        )

    with p2:
        if model_active:
            st.success(
                "Qwen OCR ready",
                icon="⚙️",
            )
        else:
            st.info(
                "Rule-based scoring",
                icon="⚙️",
            )

    with p3:
        st.info(
            "Batch evaluation",
            icon="⚡",
        )


# ============================================================
# PAGE HEADER
# ============================================================

def _page_header(
    title,
    subtitle,
):

    st.title(
        title
    )

    st.caption(
        subtitle
    )

    st.divider()


# ============================================================
# WORKFLOW STEPPER
# ============================================================

def _stepper(step):

    c1, c2, c3 = st.columns(3)

    items = [
        (
            c1,
            1,
            "Reference",
            "Question paper + answer key",
        ),
        (
            c2,
            2,
            "Students",
            "Student answer sheets",
        ),
        (
            c3,
            3,
            "Results",
            "Class performance",
        ),
    ]

    for container, number, title, subtitle in items:

        with container:

            if step > number:

                st.success(
                    f"{number}. {title}"
                )

                st.caption(
                    f"{subtitle} · Completed"
                )

            elif step == number:

                st.info(
                    f"{number}. {title}",
                    icon="⚡",
                )

                st.caption(
                    f"{subtitle} · In progress"
                )

            else:

                st.info(
                    f"{number}. {title}",
                    icon="⏳",
                )

                st.caption(
                    f"{subtitle} · Upcoming"
                )


# ============================================================
# PROGRESS
# ============================================================

def _progress(
    progress_box,
    value,
    title,
    stage,
):

    value = max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )

    progress_box.progress(
        value,
        text=(
            f"{int(value * 100)}% · {title}"
        ),
    )

    st.caption(
        stage
    )


# ============================================================
# QUESTION DETAIL
# ============================================================

def _student_question_detail(
    results,
):

    if not results:

        st.info(
            "No question-level details were stored.",
            icon="ℹ️",
        )

        return

    for result in results:

        question_id = getattr(
            result,
            "question_id",
            "?",
        )

        awarded = getattr(
            result,
            "awarded_marks",
            0,
        )

        maximum = getattr(
            result,
            "max_marks",
            0,
        )

        with st.expander(
            f"Question {question_id} · {awarded}/{maximum}"
        ):

            expected_length = getattr(
                result,
                "expected_length_words",
                None,
            )

            actual_length = getattr(
                result,
                "actual_length_words",
                None,
            )

            if expected_length is not None:

                st.write(
                    f"Expected length: ~{expected_length} words"
                )

            if actual_length is not None:

                st.write(
                    f"Detected length: {actual_length} words"
                )

            point_results = getattr(
                result,
                "point_results",
                None,
            )

            if (
                point_results
                and st.session_state.show_point_breakdown
            ):

                st.write(
                    "**Rubric breakdown**"
                )

                for point in point_results:

                    point_awarded = getattr(
                        point,
                        "awarded_marks",
                        0,
                    )

                    point_max = getattr(
                        point,
                        "point_max_marks",
                        0,
                    )

                    point_text = getattr(
                        point,
                        "point_text",
                        "",
                    )

                    st.write(
                        f"{point_awarded}/{point_max} — {point_text}"
                    )

            feedback = getattr(
                result,
                "feedback",
                "",
            )

            if feedback:

                st.info(
                    feedback,
                    icon="💬",
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if os.path.exists(
        LOGO_PATH
    ):

        st.image(
            LOGO_PATH,
            width=125,
        )

    st.markdown(
        "## GradeSense"
    )

    st.caption(
        "AI-powered handwritten answer evaluation"
    )

    st.divider()

    profile_picture = _get_profile_picture()

    if profile_picture:
        st.image(
            profile_picture,
            width=80,
        )
    else:
        st.markdown(
            "### 👤"
        )

    display_name = _display_name()

    st.write(
        f"**{display_name}**"
    )

    st.caption(
        f"Username: {st.session_state.username}"
    )

    st.divider()

    st.markdown(
        "### Workspace"
    )

    navigation = [
        (
            "Dashboard",
            "Dashboard",
        ),
        (
            "New Evaluation",
            "New Evaluation",
        ),
        (
            "Evaluation History",
            "Evaluation History",
        ),
        (
            "Profile",
            "Profile",
        ),
        (
            "About Us",
            "About Us",
        ),
        (
            "Settings",
            "Settings",
        ),
    ]

    for label, page_name in navigation:

        if (
            st.session_state.gs_page
            == page_name
        ):

            button_type = "primary"

        else:

            button_type = "secondary"

        if st.button(
            label,
            type=button_type,
            width="stretch",
            key=f"nav_{page_name}",
        ):

            st.session_state.gs_page = (
                page_name
            )

            st.rerun()

    st.divider()

    st.markdown(
        "### System status"
    )

    if model_active:

        st.success(
            "Trained ML model active",
            icon="⚙️",
        )

    else:

        st.warning(
            "Rule-based scoring active",
            icon="⚙️",
        )

    st.divider()

    st.markdown(
        "### Grading settings"
    )

    st.session_state.blur_threshold = (
        st.slider(
            "Blur sensitivity for scanned pages",
            min_value=20.0,
            max_value=200.0,
            value=float(
                st.session_state.blur_threshold
            ),
            step=5.0,
        )
    )

    st.session_state.show_choice_notes = (
        st.checkbox(
            "Show choice-group resolution details",
            value=st.session_state.show_choice_notes,
        )
    )

    st.session_state.show_point_breakdown = (
        st.checkbox(
            "Show per-point rubric breakdown",
            value=st.session_state.show_point_breakdown,
        )
    )

    st.session_state.default_sort = (
        st.selectbox(
            "Default class-results sorting",
            [
                "Roll Number",
                "Marks Obtained (high to low)",
                "Marks Obtained (low to high)",
                "Percentage",
            ],
            index=[
                "Roll Number",
                "Marks Obtained (high to low)",
                "Marks Obtained (low to high)",
                "Percentage",
            ].index(
                st.session_state.default_sort
            ),
        )
    )

    st.divider()

    auth.logout_button()


# ============================================================
# DASHBOARD
# ============================================================

def _render_dashboard():

    history = _history()

    groups = _session_groups(
        history
    )

    _hero(
        f"Welcome back, {_display_name()}",
        "A complete workspace for class-wide handwritten answer evaluation, automated grading, reports, and performance analysis.",
    )

    st.write("")
    st.divider()

    student_count = len(
        history
    )

    evaluation_count = len(
        groups
    )

    average = (
        sum(
            float(
                item.get(
                    "percentage",
                    0,
                )
            )
            for item in history
        )
        / student_count
        if student_count
        else 0
    )

    best = max(
        (
            float(
                item.get(
                    "percentage",
                    0,
                )
            )
            for item in history
        ),
        default=0,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Student results",
            student_count,
            "Saved results",
        )

    with c2:

        st.metric(
            "Evaluations",
            evaluation_count,
            "Grading sessions",
        )

    with c3:

        st.metric(
            "Average score",
            f"{average:.1f}%",
            "Across saved results",
        )

    with c4:

        st.metric(
            "Best score",
            f"{best:.1f}%",
            "Highest student result",
        )

    st.divider()

    left, right = st.columns(
        [1.7, 1],
        gap="large",
    )

    with left:

        st.subheader(
            "Recent evaluations"
        )

        st.caption(
            "Your latest grading sessions."
        )

        if not groups:

            st.info(
                "No evaluations have been completed yet. Start your first evaluation from New Evaluation.",
                icon="📊",
            )

        else:

            for rows in groups[:6]:

                first = rows[0]

                paper = first.get(
                    "paper_name",
                    "Assessment",
                )

                when = max(
                    (
                        row.get("created_at")
                        for row in rows
                        if isinstance(
                            row.get("created_at"),
                            dt.datetime,
                        )
                    ),
                    default=None,
                )

                session_average = (
                    sum(
                        float(
                            row.get(
                                "percentage",
                                0,
                            )
                        )
                        for row in rows
                    )
                    / len(rows)
                )

                with st.container(
                    border=True
                ):

                    c1, c2 = st.columns(
                        [4, 1]
                    )

                    with c1:

                        st.write(
                            f"**{os.path.splitext(paper)[0]}**"
                        )

                        st.caption(
                            f"{_fmt_date(when)} · {len(rows)} student result(s)"
                        )

                    with c2:

                        st.caption(
                            "Average"
                        )

                        st.subheader(
                            f"{session_average:.1f}%"
                        )

    with right:

        st.subheader(
            "Quick actions"
        )

        st.caption(
            "Frequently used GradeSense actions."
        )

        if st.button(
            "Start new evaluation",
            type="primary",
            width="stretch",
            key="dashboard_new",
        ):

            _reset_evaluation()

            st.rerun()

        if st.button(
            "Open evaluation history",
            width="stretch",
            key="dashboard_history",
        ):

            st.session_state.gs_page = (
                "Evaluation History"
            )

            st.rerun()

        if st.button(
            "Open profile",
            width="stretch",
            key="dashboard_profile",
        ):

            st.session_state.gs_page = (
                "Profile"
            )

            st.rerun()

        st.divider()

        st.info(
            "GradeSense stores marks, unanswered questions, answer-key warnings, and question-level grading details when available.",
            icon="💾",
        )


# ============================================================
# NEW EVALUATION
# ============================================================

def _render_new_evaluation():

    _hero(
        "New class evaluation",
        "Upload the reference material once, add the complete class, and grade every student from one workflow.",
    )

    st.write("")

    _stepper(
        st.session_state.gs_step
    )

    st.divider()

    step = (
        st.session_state.gs_step
    )

    # ========================================================
    # STEP 1
    # ========================================================

    if step == 1:

        st.subheader(
            "Reference material"
        )

        st.caption(
            "Provide the question paper and answer key that will be used for the complete class."
        )

        c1, c2 = st.columns(
            2,
            gap="large",
        )

        paper_key = (
            f"paper_upload_"
            f"{st.session_state.evaluation_version}"
        )

        key_key = (
            f"answer_key_upload_"
            f"{st.session_state.evaluation_version}"
        )

        paper_text_key = (
            f"paper_text_"
            f"{st.session_state.evaluation_version}"
        )

        key_text_key = (
            f"answer_key_text_"
            f"{st.session_state.evaluation_version}"
        )

        with c1:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Question paper"
                )

                st.caption(
                    "Upload a PDF/TXT file or paste the paper text."
                )

                paper_file = st.file_uploader(
                    "Question paper",
                    type=[
                        "pdf",
                        "txt",
                    ],
                    key=paper_key,
                )

                paper_text = st.text_area(
                    "Question paper text",
                    height=240,
                    key=paper_text_key,
                    placeholder=(
                        "Paste the question paper text here..."
                    ),
                )

                if paper_file:

                    resolved_paper = (
                        _read_uploaded_file(
                            paper_file
                        )
                    )

                    st.success(
                        f"Loaded: {paper_file.name}",
                        icon="📄",
                    )

                    try:

                        db.save_uploaded_file_metadata(
                            st.session_state.username,
                            paper_file.name,
                            "question_paper",
                            paper_file.size,
                        )

                    except Exception:

                        pass

                else:

                    resolved_paper = (
                        paper_text
                    )

        with c2:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Answer key"
                )

                st.caption(
                    "Upload a PDF/TXT file or paste the expected answers."
                )

                key_file = st.file_uploader(
                    "Answer key",
                    type=[
                        "pdf",
                        "txt",
                    ],
                    key=key_key,
                )

                answer_key_text = st.text_area(
                    "Answer key text",
                    height=240,
                    key=key_text_key,
                    placeholder=(
                        "Paste the answer key and marking reference here..."
                    ),
                )

                if key_file:

                    resolved_key = (
                        _read_uploaded_file(
                            key_file
                        )
                    )

                    st.success(
                        f"Loaded: {key_file.name}",
                        icon="🔑",
                    )

                    try:

                        db.save_uploaded_file_metadata(
                            st.session_state.username,
                            key_file.name,
                            "answer_key",
                            key_file.size,
                        )

                    except Exception:

                        pass

                else:

                    resolved_key = (
                        answer_key_text
                    )

        st.divider()

        st.info(
            "Question detection is flexible. Students may skip questions, answer out of order, use optional questions, and use different answer structures.",
            icon="🧠",
        )

        ready = bool(
            resolved_paper.strip()
            and resolved_key.strip()
        )

        if ready:

            st.success(
                "Reference material is ready.",
                icon="🟢",
            )

        else:

            st.warning(
                "Both the question paper and answer key are required.",
                icon="⚠️",
            )

        c1, c2, c3 = st.columns(
            [1, 2, 1]
        )

        with c3:

            if st.button(
                "Continue to students",
                type="primary",
                width="stretch",
                disabled=not ready,
                key="reference_continue",
            ):

                st.session_state.paper_text = (
                    resolved_paper
                )

                st.session_state.answer_key_text = (
                    resolved_key
                )

                if paper_file:

                    st.session_state.paper_name = (
                        paper_file.name
                    )

                else:

                    st.session_state.paper_name = (
                        "Untitled assessment"
                    )

                st.session_state.gs_step = 2

                st.rerun()

    # ========================================================
    # STEP 2
    # ========================================================

    elif step == 2:

        st.subheader(
            "Student answer sheets"
        )

        st.caption(
            "Upload one or more student answer sheets. The filename is used to identify the roll number."
        )

        with st.container(
            border=True
        ):

            st.info(
                "Students do not need to answer every question. Answers can appear out of order and the system does not assume a fixed answer count.",
                icon="🧠",
            )

            student_key = (
                f"student_uploads_"
                f"{st.session_state.evaluation_version}"
            )

            student_files = st.file_uploader(
                "Student answer sheets",
                type=[
                    "pdf",
                    "jpg",
                    "jpeg",
                    "png",
                    "txt",
                ],
                accept_multiple_files=True,
                key=student_key,
            )

        if student_files:

            st.success(
                f"{len(student_files)} student file(s) ready.",
                icon="🧑‍🎓",
            )

            preview_columns = st.columns(
                min(
                    4,
                    len(student_files),
                )
            )

            for index, student_file in enumerate(
                student_files[:8]
            ):

                with preview_columns[
                    index % len(preview_columns)
                ]:

                    with st.container(
                        border=True
                    ):

                        st.write(
                            f"**{student_file.name}**"
                        )

                        st.caption(
                            f"{student_file.size / 1024:.1f} KB"
                        )

            if len(student_files) > 8:

                st.caption(
                    f"+ {len(student_files) - 8} more file(s)"
                )

        else:

            st.warning(
                "Upload at least one student answer sheet.",
                icon="📥",
            )

        st.divider()

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "Back to reference",
                width="stretch",
                key="students_back",
            ):

                st.session_state.gs_step = 1

                st.rerun()

        with c2:

            grade_clicked = st.button(
                "Grade all students",
                type="primary",
                width="stretch",
                disabled=not student_files,
                key="grade_all_students",
            )

        if grade_clicked:

            session_id = str(
                uuid.uuid4()
            )

            st.session_state.session_id = (
                session_id
            )

            class_rows = []

            details = {}

            warnings = []

            progress_box = st.empty()

            status_box = st.empty()

            total_students = len(
                student_files
            )

            progress_box.progress(
                0.0,
                text="0% · Preparing evaluation",
            )

            status_box.info(
                "Preparing the grading pipeline...",
                icon="⚙️",
            )

            for index, student_file in enumerate(
                student_files
            ):

                roll = roll_number_from_filename(
                    student_file.name
                )

                base = (
                    index
                    / total_students
                )

                span = (
                    1.0
                    / total_students
                )

                _progress(
                    progress_box,
                    base + span * 0.05,
                    f"Student {index + 1}/{total_students} · {roll}",
                    "Reading answer-sheet file",
                )

                status_box.info(
                    f"Processing **{roll}** — loading input",
                    icon="📄",
                )

                ext = os.path.splitext(
                    student_file.name
                )[1].lower()

                raw_bytes = (
                    student_file.getvalue()
                )

                try:

                    try:

                        db.save_uploaded_file_metadata(
                            st.session_state.username,
                            student_file.name,
                            "student_sheet",
                            len(raw_bytes),
                        )

                    except Exception:

                        pass

                    _progress(
                        progress_box,
                        base + span * 0.12,
                        f"Student {index + 1}/{total_students} · {roll}",
                        "Input loaded",
                    )

                    # ------------------------------------------------
                    # TXT
                    # ------------------------------------------------

                    if ext == ".txt":

                        student_text = (
                            raw_bytes.decode(
                                "utf-8",
                                errors="replace",
                            )
                        )

                        _progress(
                            progress_box,
                            base + span * 0.50,
                            f"Student {index + 1}/{total_students} · {roll}",
                            "Text loaded · preparing grading",
                        )

                    # ------------------------------------------------
                    # PDF
                    # ------------------------------------------------

                    elif ext == ".pdf":

                        status_box.info(
                            f"Processing **{roll}** — Qwen OCR is analyzing PDF pages",
                            icon="🔎",
                        )

                        _progress(
                            progress_box,
                            base + span * 0.25,
                            f"Student {index + 1}/{total_students} · {roll}",
                            "Qwen OCR analyzing handwritten pages",
                        )

                        student_text, page_results = (
                            ocr_pdf(
                                raw_bytes,
                                blur_threshold=(
                                    st.session_state.blur_threshold
                                ),
                            )
                        )

                        for page_result in page_results:

                            if getattr(
                                page_result,
                                "quality_warning",
                                False,
                            ):

                                warnings.append(
                                    f"{roll}: page "
                                    f"{page_result.page_number} "
                                    f"is low quality "
                                    f"(sharpness="
                                    f"{page_result.blur_score}) — "
                                    f"verify manually."
                                )

                        _progress(
                            progress_box,
                            base + span * 0.58,
                            f"Student {index + 1}/{total_students} · {roll}",
                            "OCR complete · grading detected answers",
                        )

                    # ------------------------------------------------
                    # IMAGE
                    # ------------------------------------------------

                    elif ext in (
                        ".jpg",
                        ".jpeg",
                        ".png",
                    ):

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=ext,
                        ) as temp_file:

                            temp_file.write(
                                raw_bytes
                            )

                            temp_path = (
                                temp_file.name
                            )

                        try:

                            quality = (
                                check_image_quality(
                                    temp_path,
                                    threshold=(
                                        st.session_state.blur_threshold
                                    ),
                                )
                            )

                            if not quality.get(
                                "accepted",
                                True,
                            ):

                                warnings.append(
                                    f"{roll}: image quality below threshold "
                                    f"(sharpness="
                                    f"{quality.get('blur_score')}) — skipped."
                                )

                                _progress(
                                    progress_box,
                                    base + span,
                                    f"Student {index + 1}/{total_students} · {roll}",
                                    "Skipped because image quality was too low",
                                )

                                continue

                            _progress(
                                progress_box,
                                base + span * 0.25,
                                f"Student {index + 1}/{total_students} · {roll}",
                                "Image quality passed · running Qwen OCR",
                            )

                            student_text = extract_text(
                                temp_path
                            )

                        finally:

                            try:

                                os.unlink(
                                    temp_path
                                )

                            except OSError:

                                pass

                        _progress(
                            progress_box,
                            base + span * 0.58,
                            f"Student {index + 1}/{total_students} · {roll}",
                            "OCR complete · grading detected answers",
                        )

                    else:

                        warnings.append(
                            f"{roll}: unsupported file type "
                            f"'{ext}', skipped."
                        )

                        continue

                    # ------------------------------------------------
                    # GRADING
                    # ------------------------------------------------

                    status_box.info(
                        f"Processing **{roll}** — calculating marks",
                        icon="🧠",
                    )

                    _progress(
                        progress_box,
                        base + span * 0.70,
                        f"Student {index + 1}/{total_students} · {roll}",
                        "Calculating marks and rubric feedback",
                    )

                    (
                        results,
                        total_awarded,
                        total_max,
                        notes,
                        missing_answers,
                        unanswered,
                    ) = grade_paper_texts(
                        st.session_state.paper_text,
                        st.session_state.answer_key_text,
                        student_text,
                    )

                    _progress(
                        progress_box,
                        base + span * 0.86,
                        f"Student {index + 1}/{total_students} · {roll}",
                        "Saving detailed grading result",
                    )

                    percentage = (
                        round(
                            100
                            * total_awarded
                            / total_max,
                            1,
                        )
                        if total_max
                        else 0
                    )

                    class_rows.append(
                        {
                            "Roll Number": roll,
                            "Marks Obtained": round(
                                total_awarded,
                                2,
                            ),
                            "Max Marks": total_max,
                            "Percentage": percentage,
                            "Unanswered Questions": (
                                ", ".join(
                                    map(
                                        str,
                                        unanswered,
                                    )
                                )
                                if unanswered
                                else "-"
                            ),
                        }
                    )

                    details[roll] = (
                        results,
                        notes,
                    )

                    serialized_results = []

                    for result in results:

                        safe_result = _safe(
                            result
                        )

                        if isinstance(
                            safe_result,
                            dict,
                        ):

                            serialized_results.append(
                                safe_result
                            )

                    try:

                        db.save_grading_result(
                            username=(
                                st.session_state.username
                            ),
                            paper_name=(
                                st.session_state.paper_name
                            ),
                            roll_number=roll,
                            marks_obtained=(
                                total_awarded
                            ),
                            max_marks=(
                                total_max
                            ),
                            details={
                                "session_id": session_id,
                                "unanswered": unanswered,
                                "missing_answer_key_entries": (
                                    missing_answers
                                ),
                                "choice_notes": notes,
                                "question_results": (
                                    serialized_results
                                ),
                            },
                        )

                    except Exception as db_error:

                        warnings.append(
                            f"{roll}: result was graded, "
                            f"but could not be saved to history: "
                            f"{db_error}"
                        )

                    if missing_answers:

                        warnings.append(
                            f"{roll}: answer key is missing "
                            f"entries for {missing_answers}."
                        )

                    _progress(
                        progress_box,
                        base + span,
                        f"Student {index + 1}/{total_students} · {roll}",
                        "Student completed",
                    )

                    status_box.success(
                        f"{roll} completed successfully.",
                        icon="🟢",
                    )

                except Exception as exc:

                    warnings.append(
                        f"{roll}: FAILED to process ({exc})"
                    )

                    _progress(
                        progress_box,
                        base + span,
                        f"Student {index + 1}/{total_students} · {roll}",
                        "Student failed · continuing with next student",
                    )

                    status_box.error(
                        f"{roll} could not be processed. "
                        f"The remaining students will continue.",
                        icon="❌",
                    )

            st.session_state.class_rows = (
                class_rows
            )

            st.session_state.per_student_details = (
                details
            )

            st.session_state.quality_warnings = (
                warnings
            )

            st.session_state.gs_step = 3

            progress_box.progress(
                1.0,
                text="100% · Batch evaluation complete",
            )

            st.rerun()

    # ========================================================
    # STEP 3
    # ========================================================

    elif step == 3:

        rows = (
            st.session_state.class_rows
            or []
        )

        details = (
            st.session_state.per_student_details
            or {}
        )

        warnings = (
            st.session_state.quality_warnings
            or []
        )

        st.subheader(
            "Class performance"
        )

        st.caption(
            "Review the batch outcome and inspect individual student grading."
        )

        if warnings:

            with st.expander(
                f"{len(warnings)} warning(s)",
                expanded=False,
            ):

                for warning in warnings:

                    st.warning(
                        warning
                    )

        if rows:

            df = pd.DataFrame(
                rows
            )

            sort_map = {
                "Roll Number": (
                    "Roll Number",
                    True,
                ),
                "Marks Obtained (high to low)": (
                    "Marks Obtained",
                    False,
                ),
                "Marks Obtained (low to high)": (
                    "Marks Obtained",
                    True,
                ),
                "Percentage": (
                    "Percentage",
                    False,
                ),
            }

            sort_column, ascending = (
                sort_map[
                    st.session_state.default_sort
                ]
            )

            df = df.sort_values(
                sort_column,
                ascending=ascending,
            )

            average = df[
                "Percentage"
            ].mean()

            highest = df[
                "Percentage"
            ].max()

            lowest = df[
                "Percentage"
            ].min()

            c1, c2, c3, c4 = st.columns(
                4
            )

            with c1:

                st.metric(
                    "Students graded",
                    len(df),
                )

            with c2:

                st.metric(
                    "Class average",
                    f"{average:.1f}%",
                )

            with c3:

                st.metric(
                    "Highest",
                    f"{highest:.1f}%",
                )

            with c4:

                st.metric(
                    "Lowest",
                    f"{lowest:.1f}%",
                )

            st.divider()

            st.subheader(
                "Performance distribution"
            )

            st.bar_chart(
                df.set_index(
                    "Roll Number"
                )["Percentage"],
                height=320,
            )

            st.subheader(
                "Class results"
            )

            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
            )

            csv_data = (
                df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )
            )

            st.download_button(
                "Download class results CSV",
                data=csv_data,
                file_name=(
                    "gradesense_class_results.csv"
                ),
                mime="text/csv",
                width="stretch",
            )

            st.divider()

            st.subheader(
                "Student details"
            )

            student_options = sorted(
                details.keys(),
                key=str,
            )

            if student_options:

                selected_roll = st.selectbox(
                    "Select a student",
                    student_options,
                )

                if selected_roll:

                    result_data = (
                        details.get(
                            selected_roll
                        )
                    )

                    if result_data:

                        results, notes = (
                            result_data
                        )

                        st.write(
                            f"### Student {selected_roll}"
                        )

                        _student_question_detail(
                            results
                        )

                        if (
                            notes
                            and st.session_state.show_choice_notes
                        ):

                            st.subheader(
                                "Choice-group resolution"
                            )

                            for note in notes:

                                st.info(
                                    note
                                )

        else:

            st.error(
                "No student was successfully graded.",
                icon="❌",
            )

            if warnings:

                st.info(
                    "Review the warnings above and try another evaluation.",
                    icon="ℹ️",
                )

        st.divider()

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "Back to students",
                width="stretch",
                key="results_back",
            ):

                st.session_state.gs_step = 2

                st.rerun()

        with c2:

            if st.button(
                "New evaluation",
                type="primary",
                width="stretch",
                key="results_new",
            ):

                _reset_evaluation()

                st.rerun()


# ============================================================
# EVALUATION HISTORY
# ============================================================

def _render_history():

    history = _history()

    groups = _session_groups(
        history
    )

    _hero(
        "Evaluation history",
        "Return to previous grading sessions and inspect saved student and question-level results.",
    )

    st.write("")

    if not groups:

        st.info(
            "No saved evaluations yet.",
            icon="📊",
        )

        if st.button(
            "Start your first evaluation",
            type="primary",
            width="stretch",
            key="history_start",
        ):

            _reset_evaluation()

            st.rerun()

        return

    labels = []

    for rows in groups:

        first = rows[0]

        paper = first.get(
            "paper_name",
            "Assessment",
        )

        when = max(
            (
                row.get("created_at")
                for row in rows
                if isinstance(
                    row.get("created_at"),
                    dt.datetime,
                )
            ),
            default=None,
        )

        average = (
            sum(
                float(
                    row.get(
                        "percentage",
                        0,
                    )
                )
                for row in rows
            )
            / len(rows)
        )

        labels.append(
            (
                f"{os.path.splitext(paper)[0]} · "
                f"{_fmt_date(when)} · "
                f"{len(rows)} students · "
                f"{average:.1f}%",
                rows,
            )
        )

    selected_label = st.selectbox(
        "Choose an evaluation",
        [
            label
            for label, _ in labels
        ],
    )

    selected_rows = next(
        rows
        for label, rows in labels
        if label == selected_label
    )

    average = (
        sum(
            float(
                row.get(
                    "percentage",
                    0,
                )
            )
            for row in selected_rows
        )
        / len(selected_rows)
    )

    highest = max(
        float(
            row.get(
                "percentage",
                0,
            )
        )
        for row in selected_rows
    )

    lowest = min(
        float(
            row.get(
                "percentage",
                0,
            )
        )
        for row in selected_rows
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "Students",
            len(selected_rows),
        )

    with c2:

        st.metric(
            "Average",
            f"{average:.1f}%",
        )

    with c3:

        st.metric(
            "Highest",
            f"{highest:.1f}%",
        )

    with c4:

        st.metric(
            "Lowest",
            f"{lowest:.1f}%",
        )

    st.divider()

    st.subheader(
        "Saved student results"
    )

    for row in sorted(
        selected_rows,
        key=lambda item: str(
            item.get(
                "roll_number",
                "",
            )
        ),
    ):

        roll = row.get(
            "roll_number",
            "-",
        )

        marks = row.get(
            "marks_obtained",
            0,
        )

        maximum = row.get(
            "max_marks",
            0,
        )

        percentage = float(
            row.get(
                "percentage",
                0,
            )
        )

        details = (
            row.get(
                "details"
            )
            or {}
        )

        with st.expander(
            f"{roll} · {marks}/{maximum} · {percentage:.1f}%"
        ):

            unanswered = (
                details.get(
                    "unanswered"
                )
                or []
            )

            missing = (
                details.get(
                    "missing_answer_key_entries"
                )
                or []
            )

            if unanswered:

                st.warning(
                    "Unanswered / not detected: "
                    + ", ".join(
                        map(
                            str,
                            unanswered,
                        )
                    ),
                    icon="⚠️",
                )

            if missing:

                st.warning(
                    f"Missing answer-key entries: {missing}",
                    icon="🔑",
                )

            question_results = (
                details.get(
                    "question_results"
                )
                or []
            )

            if not question_results:

                st.info(
                    "Question-level details were not stored for this older evaluation.",
                    icon="ℹ️",
                )

            else:

                for question in question_results:

                    question_id = question.get(
                        "question_id",
                        "?",
                    )

                    awarded = question.get(
                        "awarded_marks",
                        0,
                    )

                    maximum_marks = question.get(
                        "max_marks",
                        0,
                    )

                    actual_length = question.get(
                        "actual_length_words",
                        "",
                    )

                    feedback = question.get(
                        "feedback",
                        "",
                    )

                    with st.container(
                        border=True
                    ):

                        c1, c2 = st.columns(
                            [4, 1]
                        )

                        with c1:

                            st.write(
                                f"**Question {question_id}**"
                            )

                            if actual_length != "":

                                st.caption(
                                    f"Detected answer length: "
                                    f"{actual_length} words"
                                )

                        with c2:

                            st.metric(
                                "Marks",
                                f"{awarded}/{maximum_marks}",
                            )

                        if feedback:

                            st.info(
                                feedback,
                                icon="💬",
                            )

                        point_results = (
                            question.get(
                                "point_results"
                            )
                            or []
                        )

                        if (
                            point_results
                            and st.session_state.show_point_breakdown
                        ):

                            st.write(
                                "**Rubric breakdown**"
                            )

                            for point in point_results:

                                st.write(
                                    f"{point.get('awarded_marks', 0)}/"
                                    f"{point.get('point_max_marks', 0)} — "
                                    f"{point.get('point_text', '')}"
                                )

            notes = (
                details.get(
                    "choice_notes"
                )
                or []
            )

            if (
                notes
                and st.session_state.show_choice_notes
            ):

                st.write(
                    "**Choice-group resolution**"
                )

                for note in notes:

                    st.info(
                        note,
                        icon="🔀",
                    )


# ============================================================
# PROFILE
# ============================================================

def _render_profile():

    _page_header(
        "Your profile",
        "Account information and GradeSense usage summary.",
    )

    try:

        user = (
            db.get_db()
            .users
            .find_one(
                {
                    "username": (
                        st.session_state.username
                    )
                }
            )
            or {}
        )

    except Exception as exc:

        user = {}

        st.warning(
            f"Could not load account details: {exc}"
        )

    history = _history()

    groups = _session_groups(
        history
    )

    average = (
        sum(
            float(
                item.get(
                    "percentage",
                    0,
                )
            )
            for item in history
        )
        / len(history)
        if history
        else 0
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Student results",
            len(history),
        )

    with c2:

        st.metric(
            "Evaluations",
            len(groups),
        )

    with c3:

        st.metric(
            "Average score",
            f"{average:.1f}%",
        )

    st.divider()

    st.subheader(
        "Account information"
    )

    with st.container(
        border=True
    ):

        display_name = (
            st.session_state.get(
                "display_name"
            )
            or user.get(
                "display_name"
            )
            or user.get(
                "name"
            )
            or st.session_state.get(
                "username"
            )
            or "User"
        )

        st.write(
            f"**Name:** {display_name}"
        )

        st.write(
            f"**Username:** "
            f"{st.session_state.get('username', 'Not available')}"
        )

        st.write(
            f"**Email:** "
            f"{user.get('email', 'Not available')}"
        )

        st.write(
            f"**Account created:** "
            f"{_fmt_date(user.get('created_at'))}"
        )

        verified = user.get(
            "email_verified",
            False,
        )

        if verified:

            st.success(
                "Email verified",
                icon="🟢",
            )

        else:

            st.warning(
                "Email not verified",
                icon="⚠️",
            )

    st.divider()

    st.subheader(
        "Profile picture"
    )

    st.caption(
        "Upload a photo and crop the area you want to use as your profile picture."
    )

    profile_picture = st.file_uploader(
        "Upload profile picture",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        key="profile_picture",
    )

    if profile_picture:

        profile_image = Image.open(
            profile_picture
        ).convert("RGB")

        cropped_image = st_cropper(
            profile_image,
            realtime_update=True,
            box_color="#7c3aed",
            aspect_ratio=(1, 1),
            return_type="image",
        )

        st.write("### Preview")

        st.image(
            cropped_image,
            width=180,
        )

        if st.button(
            "Save Profile Picture",
            type="primary",
            width="stretch",
            key="save_profile_picture",
        ):
            try:

                output = io.BytesIO()

                cropped_image.save(
                    output,
                    format="PNG",
                )

                picture_bytes = output.getvalue()

                db.get_db().users.update_one(
                    {
                        "username": st.session_state.username
                    },
                    {
                        "$set": {
                            "profile_picture": picture_bytes
                        }
                    },
                )

                st.success(
                    "Profile picture saved.",
                    icon="🟢",
                )
                time.sleep(1)

                st.session_state.gs_page = "Dashboard"

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not save profile picture: {exc}"
                )

# ============================================================
# ABOUT US
# ============================================================

def _render_about():

    _hero(
        "About GradeSense",
        "An AI-assisted workspace designed to reduce repetitive handwritten answer evaluation work.",
    )

    st.write("")

    st.subheader(
        "What is GradeSense?"
    )

    st.write(
        "GradeSense is a handwritten answer evaluation system designed for teachers, educators, and assessment teams. It combines document processing, OCR, structured grading, and class-level reporting in one workspace."
    )

    st.divider()

    st.subheader(
        "Core capabilities"
    )

    c1, c2 = st.columns(2)

    with c1:

        with st.container(
            border=True
        ):

            st.write(
                "### Reference-based evaluation"
            )

            st.write(
                "Use a question paper and answer key as the reference for the complete class."
            )

            st.write(
                "### Handwritten OCR"
            )

            st.write(
                "Qwen-based OCR extracts student answers from supported handwritten documents."
            )

            st.write(
                "### Batch processing"
            )

            st.write(
                "Upload multiple student answer sheets and process them in one evaluation."
            )

    with c2:

        with st.container(
            border=True
        ):

            st.write(
                "### Question-level grading"
            )

            st.write(
                "Review marks, rubric points, feedback, and detected answer information."
            )

            st.write(
                "### Flexible answers"
            )

            st.write(
                "Students can skip questions, answer out of order, and use different answer structures."
            )

            st.write(
                "### Class reporting"
            )

            st.write(
                "Review class averages, highest scores, lowest scores, and individual performance."
            )

    st.divider()

    st.subheader(
        "Flexible answer evaluation"
    )

    st.info(
        "GradeSense does not assume that every student answers every question in the same order. The grading pipeline uses detected question identifiers and the supplied reference material.",
        icon="💡",
    )

    st.divider()

    st.subheader(
        "Technology"
    )

    st.write(
        "GradeSense uses Python and Streamlit for the application interface, MongoDB for account and grading-history storage, Qwen OCR for handwritten document processing, and the existing grading pipeline for structured evaluation."
    )

    st.success(
        "Built for educators and assessment teams.",
        icon="🎓",
    )


# ============================================================
# SETTINGS
# ============================================================

def _render_settings():

    _page_header(
        "Workspace settings",
        "Configure display and grading preferences for the current session.",
    )

    st.subheader(
        "Grading preferences"
    )

    st.session_state.blur_threshold = (
        st.slider(
            "Blur sensitivity for scanned pages",
            min_value=20.0,
            max_value=200.0,
            value=float(
                st.session_state.blur_threshold
            ),
            step=5.0,
        )
    )

    st.session_state.show_choice_notes = (
        st.checkbox(
            "Show choice-group resolution details",
            value=(
                st.session_state.show_choice_notes
            ),
        )
    )

    st.session_state.show_point_breakdown = (
        st.checkbox(
            "Show per-point rubric breakdown",
            value=(
                st.session_state.show_point_breakdown
            ),
        )
    )

    st.session_state.default_sort = (
        st.selectbox(
            "Default class-results sorting",
            [
                "Roll Number",
                "Marks Obtained (high to low)",
                "Marks Obtained (low to high)",
                "Percentage",
            ],
            index=[
                "Roll Number",
                "Marks Obtained (high to low)",
                "Marks Obtained (low to high)",
                "Percentage",
            ].index(
                st.session_state.default_sort
            ),
        )
    )

    st.divider()

    st.subheader(
        "System information"
    )

    c1, c2 = st.columns(2)

    with c1:

        if model_active:

            st.success(
                "Trained ML model active",
                icon="⚙️",
            )

        else:

            st.warning(
                "Rule-based scoring active",
                icon="⚙️",
            )

    with c2:

        st.info(
            "Qwen OCR is handled by the existing OCR module.",
            icon="🔎",
        )

    st.divider()

    st.success(
        "Settings are active for the current Streamlit session.",
        icon="🟢",
    )


# ============================================================
# TOP RIBBON
# ============================================================

_top_ribbon()


# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.gs_page == "Dashboard":

    _render_dashboard()

elif st.session_state.gs_page == "New Evaluation":

    _render_new_evaluation()

elif st.session_state.gs_page == "Evaluation History":

    _render_history()

elif st.session_state.gs_page == "Profile":

    _render_profile()

elif st.session_state.gs_page == "About Us":

    _render_about()

elif st.session_state.gs_page == "Settings":

    _render_settings()

else:

    st.session_state.gs_page = "Dashboard"

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_col = st.columns([1, 2, 1])[1]

with footer_col:
    st.caption(
        "GradeSense · AI-assisted handwritten answer evaluation · © 2026 GradeSense"
    )