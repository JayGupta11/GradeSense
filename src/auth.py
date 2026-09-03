import os
import re
import streamlit as st
import database as db
import email_utils


BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 8% 12%,
                rgba(37, 99, 235, 0.34),
                transparent 30%
            ),
            radial-gradient(
                circle at 92% 14%,
                rgba(124, 58, 237, 0.34),
                transparent 32%
            ),
            radial-gradient(
                circle at 50% 95%,
                rgba(236, 72, 153, 0.24),
                transparent 35%
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
        animation: gsAuthBackground 14s ease infinite;
        min-height: 100vh;
    }

    @keyframes gsAuthBackground {

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

    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer {
        width: 920px !important;
        max-width: 920px !important;
        min-width: 920px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMainBlockContainer"] > div,
    .stMainBlockContainer > div {
        width: 100% !important;
        max-width: 100% !important;
    }

    @media (max-width: 980px) {

        [data-testid="stMainBlockContainer"],
        .stMainBlockContainer {
            width: calc(100% - 32px) !important;
            max-width: calc(100% - 32px) !important;
            min-width: 0 !important;
            margin-left: 16px !important;
            margin-right: 16px !important;
        }

    }

    .gs-brand-title {
        font-size: 42px;
        font-weight: 850;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    .gs-brand-subtitle {
        color: #aeb8ce;
        font-size: 15px;
        margin-bottom: 12px;
    }

    .gs-section-title {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.7px;
        margin-top: 8px;
        margin-bottom: 6px;
    }

    .gs-section-subtitle {
        color: #9ba6bd;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .gs-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        color: #dbeafe;
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.28);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.4px;
    }

    .gs-intro {
        padding: 22px;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.16);

        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 75, 0.70),
                rgba(30, 20, 58, 0.55)
            );

        box-shadow:
            0 18px 55px rgba(0, 0, 0, 0.20);
    }

    .gs-intro-title {
        color: #ffffff;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .gs-intro-text {
        color: #b8c2d7;
        font-size: 13px;
        line-height: 1.65;
    }

    .gs-feature {
        min-height: 120px;
        padding: 17px;
        border-radius: 17px;
        border: 1px solid rgba(148, 163, 184, 0.15);

        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 75, 0.72),
                rgba(24, 20, 48, 0.62)
            );

        transition:
            transform 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }

    .gs-feature:hover {
        transform: translateY(-3px);
        border-color: rgba(129, 140, 248, 0.45);
        box-shadow:
            0 15px 35px rgba(99, 102, 241, 0.16);
    }

    .gs-feature-icon {
        font-size: 22px;
        margin-bottom: 7px;
    }

    .gs-feature-title {
        color: #ffffff;
        font-size: 14px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .gs-feature-text {
        color: #9ba6bd;
        font-size: 11px;
        line-height: 1.5;
    }

    .stTextInput label {
        color: #e5e7eb !important;
        font-weight: 650 !important;
    }

    .stTextInput input {
        min-height: 46px !important;
        border-radius: 12px !important;

        background: rgba(35, 37, 49, 0.96) !important;

        border:
            1px solid
            rgba(148, 163, 184, 0.20) !important;

        color: #ffffff !important;
    }

    .stTextInput input::placeholder {
        color: #8d96aa !important;
    }

    .stTextInput input:focus {
        border-color:
            rgba(99, 102, 241, 0.85) !important;

        box-shadow:
            0 0 0 3px
            rgba(99, 102, 241, 0.14) !important;
    }

    .stButton > button {
        min-height: 46px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;

        box-shadow:
            0 12px 28px
            rgba(99, 102, 241, 0.20) !important;
    }

    .stButton > button[kind="primary"] {
        color: #ffffff !important;

        background:
            linear-gradient(
                120deg,
                #2563eb,
                #6366f1,
                #8b5cf6,
                #ec4899,
                #2563eb
            ) !important;

        background-size: 300% 300% !important;

        animation:
            gsButtonGradient 7s ease infinite !important;

        border: none !important;
    }

    @keyframes gsButtonGradient {

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

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    button[data-baseweb="tab"] {
        font-weight: 700 !important;
    }

    hr {
        border-color:
            rgba(148, 163, 184, 0.16) !important;
    }

    .gs-footer {
        text-align: center;
        color: #7f8aa1;
        font-size: 12px;
        margin-top: 28px;
        white-space: nowrap;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

def _username_rules(username):
    username = username.strip()

    if not username:
        return False, "Username is required."

    if len(username) < 5:
        return False, "Username must be at least 5 characters."

    if len(username) > 30:
        return False, "Username must not exceed 30 characters."

    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+",
        username,
    ):
        return (
            False,
            "Only letters, numbers, _, . and - are allowed.",
        )

    return True, ""

def _name_rules(name):
    name = name.strip()

    if not name:
        return False, "Full name is required."

    if len(name) < 2:
        return False, "Full name must contain at least 2 characters."

    if len(name) > 60:
        return False, "Full name must not exceed 60 characters."

    if not re.fullmatch(
        r"[A-Za-zÀ-ÖØ-öø-ÿ .'-]+",
        name,
    ):
        return (
            False,
            "Full name can only contain letters, spaces, . ' and -.",
        )

    return True, ""

def _password_rules(password):
    if not password:
        return False, "Password is required."

    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain an uppercase letter."

    if not re.search(r"[a-z]", password):
        return False, "Password must contain a lowercase letter."

    if not re.search(r"[0-9]", password):
        return False, "Password must contain a number."

    return True, ""

def _email_rules(email):
    email = email.strip()

    if not email:
        return False, "Email is required."

    if not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+",
        email,
    ):
        return False, "Enter a valid email address."

    return True, ""

def _register_username_check(username):

    username = username.strip()

    if not username:
        return False

    valid, message = _username_rules(username)

    if not valid:
        st.error(
            message,
            icon="❌",
        )
        return False

    try:
        available, message = db.is_username_available(
            username
        )

    except Exception as exc:
        st.error(
            f"Username availability check failed: {exc}",
            icon="⚠️",
        )
        return False

    if available:
        st.success(
            "Username is available.",
            icon="🟢",
        )
        return True

    st.error(
        message or "Username is already taken.",
        icon="❌",
    )

    return False

def _login_username_check(username):

    username = username.strip()

    if not username:
        return False

    valid, message = _username_rules(username)

    if not valid:
        st.error(
            message,
            icon="❌",
        )
        return False

    try:
        user = (
            db.get_db()
            .users
            .find_one(
                {
                    "username": username
                },
                {
                    "_id": 1,
                    "email": 1,
                    "email_verified": 1,
                },
            )
        )

    except Exception as exc:
        st.error(
            f"Could not check username: {exc}",
            icon="⚠️",
        )
        return False

    if user:
        st.success(
            "Username found.",
            icon="🟢",
        )
        return True

    st.error(
        "No account exists with this username.",
        icon="❌",
    )

    return False

def _password_checklist(password):

    if not password:
        return

    st.caption("Password requirements")

    c1, c2 = st.columns(2)

    with c1:

        if len(password) >= 8:
            st.success(
                "8+ characters",
                icon="🟢",
            )
        else:
            st.error(
                "8+ characters",
                icon="🔴",
            )

        if any(
            char.isupper()
            for char in password
        ):
            st.success(
                "Uppercase letter",
                icon="🟢",
            )
        else:
            st.error(
                "Uppercase letter",
                icon="🔴",
            )

    with c2:

        if any(
            char.islower()
            for char in password
        ):
            st.success(
                "Lowercase letter",
                icon="🟢",
            )
        else:
            st.error(
                "Lowercase letter",
                icon="🔴",
            )

        if any(
            char.isdigit()
            for char in password
        ):
            st.success(
                "Number",
                icon="🟢",
            )
        else:
            st.error(
                "Number",
                icon="🔴",
            )

def _otp_form():

    username = st.session_state.get(
        "pending_username",
        "",
    )

    email = st.session_state.get(
        "pending_email",
        "",
    )

    st.subheader(
        "Verify your email"
    )

    st.caption(
        "Enter the verification code sent to your email."
    )

    st.info(
        f"A 6-digit verification code was sent to {email}.",
        icon="📨",
    )

    otp = st.text_input(
        "Verification code",
        max_chars=6,
        placeholder="Enter 6-digit code",
        key="otp_input",
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Verify email",
            type="primary",
            width="stretch",
            key="verify_otp",
        ):

            clean_otp = otp.strip()

            if (
                not clean_otp
                or not clean_otp.isdigit()
                or len(clean_otp) != 6
            ):
                st.error(
                    "OTP must contain exactly 6 digits.",
                    icon="❌",
                )

            else:

                result = db.verify_otp(
                    username,
                    clean_otp,
                )

                if result.get("success"):

                    st.session_state.just_verified_username = (
                        username
                    )

                    st.session_state.pop(
                        "pending_username",
                        None,
                    )

                    st.session_state.pop(
                        "pending_email",
                        None,
                    )

                    st.session_state.pop(
                        "otp_input",
                        None,
                    )

                    st.rerun()

                else:

                    st.error(
                        result.get(
                            "message",
                            "OTP verification failed.",
                        ),
                        icon="❌",
                    )

    with c2:

        if st.button(
            "Resend OTP",
            width="stretch",
            key="resend_otp",
        ):

            result = db.resend_otp(
                username
            )

            if result.get("success"):

                st.success(
                    result.get(
                        "message",
                        "OTP resent.",
                    ),
                    icon="📨",
                )

            else:

                st.error(
                    result.get(
                        "message",
                        "Could not resend OTP.",
                    ),
                    icon="❌",
                )

    st.divider()

    if st.button(
        "Use a different account",
        width="stretch",
        key="different_account",
    ):

        st.session_state.pop(
            "pending_username",
            None,
        )

        st.session_state.pop(
            "pending_email",
            None,
        )

        st.session_state.pop(
            "otp_input",
            None,
        )

        st.rerun()

def _register_form():

    st.subheader(
        "Create your account"
    )

    full_name = st.text_input(
        "Full name",
        key="reg_full_name",
        placeholder="Enter your full name",
        max_chars=60,
    )

    name_ok = False

    if full_name:

        name_ok, name_message = _name_rules(
            full_name
        )

        if name_ok:

            st.success(
                "Name looks good.",
                icon="🟢",
            )

        else:

            st.error(
                name_message,
                icon="❌",
            )

    username = st.text_input(
        "Username",
        key="reg_username",
        placeholder="Choose a unique username",
        max_chars=30,
    )

    username_ok = _register_username_check(
        username
    )

    email = st.text_input(
        "Email address",
        key="reg_email",
        placeholder="you@example.com",
    )

    email_ok = False

    if email:

        email_ok, email_message = _email_rules(
            email
        )

        if email_ok:

            st.success(
                "Email format is valid.",
                icon="🟢",
            )

        else:

            st.error(
                email_message,
                icon="❌",
            )

    password = st.text_input(
        "Password",
        type="password",
        key="reg_password",
        placeholder="Create a strong password",
    )

    _password_checklist(
        password
    )

    password_ok, password_message = (
        _password_rules(password)
    )

    confirm_password = st.text_input(
        "Confirm password",
        type="password",
        key="reg_confirm",
        placeholder="Enter password again",
    )

    passwords_match = (
        bool(confirm_password)
        and confirm_password == password
    )

    if confirm_password:

        if passwords_match:

            st.success(
                "Passwords match.",
                icon="🟢",
            )

        else:

            st.error(
                "Passwords do not match.",
                icon="❌",
            )

    ready = (
        name_ok
        and username_ok
        and email_ok
        and password_ok
        and passwords_match
    )

    st.divider()

    if ready:

        st.success(
            "All registration checks passed.",
            icon="🎉",
        )

    else:

        st.info(
            "Complete all required checks before creating your account.",
            icon="ℹ️",
        )

    if st.button(
        "Send OTP & Create Account",
        type="primary",
        width="stretch",
        key="register_submit",
    ):

        clean_name = full_name.strip()
        clean_username = username.strip()
        clean_email = email.strip()

        valid_name, name_error = _name_rules(
            clean_name
        )

        if not valid_name:

            st.error(
                name_error,
                icon="❌",
            )

            return

        valid_username, username_error = (
            _username_rules(
                clean_username
            )
        )

        if not valid_username:

            st.error(
                username_error,
                icon="❌",
            )

            return

        try:

            available, availability_message = (
                db.is_username_available(
                    clean_username
                )
            )

        except Exception as exc:

            st.error(
                f"Could not check username availability: {exc}",
                icon="⚠️",
            )

            return

        if not available:

            st.error(
                availability_message
                or "Username is already taken.",
                icon="❌",
            )

            return

        valid_email, email_error = _email_rules(
            clean_email
        )

        if not valid_email:

            st.error(
                email_error,
                icon="❌",
            )

            return

        valid_password, password_error = (
            _password_rules(
                password
            )
        )

        if not valid_password:

            st.error(
                password_error,
                icon="❌",
            )

            return

        if password != confirm_password:

            st.error(
                "Passwords do not match.",
                icon="❌",
            )

            return

        result = db.register_user(
            clean_username,
            password,
            clean_email,
        )

        if not result.get("success"):

            st.error(
                result.get(
                    "message",
                    "Could not create account.",
                ),
                icon="❌",
            )

            return

        try:

            db.get_db().users.update_one(
                {
                    "username": clean_username
                },
                {
                    "$set": {
                        "name": clean_name,
                        "display_name": clean_name,
                    }
                },
            )

        except Exception as exc:

            st.warning(
                f"Account created, but the name could not be saved: {exc}",
                icon="⚠️",
            )

        st.session_state.pending_username = (
            clean_username
        )

        st.session_state.pending_email = (
            clean_email
        )

        if result.get("email_sent"):

            st.success(
                "Account created and verification code sent to your email.",
                icon="📨",
            )

        else:

            st.warning(
                "Account created, but the verification email could not be sent. "
                "Check your email configuration.",
                icon="⚠️",
            )

        st.rerun()

def _login_form():

    st.subheader(
        "Welcome back"
    )

    st.caption(
        "Sign in to continue to your GradeSense assessment workspace."
    )

    username = st.text_input(
        "Username",
        key="login_username",
        placeholder="Enter your username",
        max_chars=30,
    )

    username_ok = _login_username_check(
        username
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_password",
        placeholder="Enter your password",
    )
    _password_checklist(password)

    st.divider()

    if st.button(
        "Sign in to GradeSense",
        type="primary",
        width="stretch",
        key="login_submit",
    ):

        clean_username = username.strip()

        valid_username, username_error = (
            _username_rules(
                clean_username
            )
        )

        if not valid_username:

            st.error(
                username_error,
                icon="❌",
            )

            return

        if not password:

            st.error(
                "Password is required.",
                icon="❌",
            )

            return

        try:

            user = (
                db.get_db()
                .users
                .find_one(
                    {
                        "username": clean_username
                    }
                )
            )

        except Exception as exc:

            st.error(
                f"Database error: {exc}",
                icon="⚠️",
            )

            return

        if not user:

            st.error(
                "No account exists with this username.",
                icon="❌",
            )

            return

        result = db.authenticate_user(
            clean_username,
            password,
        )

        if result.get("success"):

            display_name = (
                user.get("display_name")
                or user.get("name")
                or clean_username
            )

            st.session_state.authenticated = True

            st.session_state.username = (
                clean_username
            )

            st.session_state.display_name = (
                display_name
            )

            st.session_state.gs_page = (
                "Dashboard"
            )

            st.session_state.gs_step = 1

            st.rerun()

        elif result.get("needs_otp"):

            st.error(
                result.get(
                    "message",
                    "Email verification is required.",
                ),
                icon="📧",
            )

            if st.button(
                "Enter verification code",
                width="stretch",
                key="login_enter_otp",
            ):

                st.session_state.pending_username = (
                    clean_username
                )

                st.session_state.pending_email = (
                    user.get(
                        "email",
                        "",
                    )
                )

                st.rerun()

        else:

            st.error(
                result.get(
                    "message",
                    "Invalid username or password.",
                ),
                icon="❌",
            )

def require_login():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "username" not in st.session_state:
        st.session_state.username = None

    if "display_name" not in st.session_state:
        st.session_state.display_name = None

    if st.session_state.authenticated:
        return

    if not db.check_connection():

        st.error(
            "Could not connect to MongoDB. "
            "Make sure MongoDB is running and MONGODB_URI is configured.",
            icon="🔌",
        )

        st.stop()

    if "pending_username" in st.session_state:

        if os.path.exists(LOGO_PATH):

            st.image(
                LOGO_PATH,
                width=105,
            )

        st.subheader(
            "GradeSense"
        )

        st.caption(
            "AI-Powered Assessment Intelligence"
        )

        st.divider()

        _otp_form()

        st.stop()

    brand_left, brand_right = st.columns(
        [1, 3]
    )

    with brand_left:

        if os.path.exists(LOGO_PATH):

            st.image(
                LOGO_PATH,
                width=125,
            )

    with brand_right:

        st.subheader(
            "GradeSense"
        )

        st.caption(
            "AI-Powered Assessment Intelligence"
        )

        st.info(
            "SMART · SECURE · SCALABLE",
            icon="💡",
        )

    st.divider()

    st.subheader(
        "Intelligent evaluation. Better insights."
    )

    st.write(
        "Grade handwritten answer sheets with an AI-assisted workflow designed for teachers, educators and assessment teams."
    )

    st.write("")

    f1, f2, f3 = st.columns(3)

    with f1:

        with st.container(
            border=True
        ):

            st.write(
                "### 🧠 Smart Evaluation"
            )

            st.caption(
                "AI-assisted evaluation of handwritten student responses."
            )

    with f2:

        with st.container(
            border=True
        ):

            st.write(
                "### ⚡ Batch Grading"
            )

            st.caption(
                "Evaluate complete classes from one assessment workspace."
            )

    with f3:

        with st.container(
            border=True
        ):

            st.write(
                "### 📊 Clear Insights"
            )

            st.caption(
                "Review marks, performance and detailed evaluation history."
            )

    st.write("")

    if "just_verified_username" in st.session_state:

        verified_username = (
            st.session_state.just_verified_username
        )

        st.success(
            f"Email verified successfully. "
            f"You can now log in as {verified_username}.",
            icon="🟢",
        )

        st.session_state.pop(
            "just_verified_username",
            None,
        )

    try:

        email_configured = (
            email_utils.is_configured()
        )

    except Exception:

        email_configured = True

    if not email_configured:

        st.warning(
            "Email sending is not configured. "
            "Registration can create an account, but OTP delivery "
            "requires email configuration.",
            icon="📧",
        )

    login_tab, register_tab = st.tabs(
        [
            "Login",
            "Create account",
        ]
    )

    with login_tab:

        _login_form()

    with register_tab:

        _register_form()

    footer_col = st.columns([1, 2, 1])[1]

    with footer_col:
        st.caption(
            "GradeSense · AI-assisted handwritten answer evaluation · © 2026 GradeSense"
        )

    st.stop()

def logout_button():

    if st.sidebar.button(
        "Log out",
        width="stretch",
        key="logout_button",
    ):

        st.session_state.authenticated = False

        st.session_state.username = None

        st.session_state.display_name = None

        st.session_state.gs_page = "Dashboard"

        st.session_state.gs_step = 1

        for key in [
            "paper_text",
            "answer_key_text",
            "class_rows",
            "per_student_details",
            "quality_warnings",
            "session_id",
            "paper_name",
            "pending_username",
            "pending_email",
            "otp_input",
            "reg_full_name",
            "reg_username",
            "reg_email",
            "reg_password",
            "reg_confirm",
            "login_username",
            "login_password",
        ]:

            st.session_state.pop(
                key,
                None,
            )

        st.rerun()
