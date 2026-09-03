import os
import datetime
import hashlib
import re
import random
import secrets
from typing import Optional, List

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

import email_utils

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("GRADESENSE_DB_NAME", "gradesense")

_client = None
_db = None


def get_db():
    global _client, _db

    if _db is None:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _db = _client[DB_NAME]

        _db.users.create_index("username", unique=True)
        _db.users.create_index("email", unique=True)
        _db.results.create_index("created_at")
        _db.results.create_index("username")
        _db.logs.create_index("timestamp")

    return _db


def check_connection() -> bool:
    try:
        get_db().client.admin.command("ping")
        return True
    except Exception:
        return False


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)

    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt,
        200_000
    )

    return salt.hex() + ":" + pwd_hash.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)

        check_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt,
            200_000
        ).hex()

        return secrets.compare_digest(check_hash, hash_hex)
    except (ValueError, TypeError, AttributeError):
        return False


def validate_username(username: str) -> tuple:
    """Returns (is_valid, error_message)."""

    username = username.strip()

    if not username:
        return False, "Username is required."

    if len(username) < 5:
        return False, "Username must be at least 5 characters long."

    if len(username) > 30:
        return False, "Username must not exceed 30 characters."

    if not re.match(r"^[A-Za-z0-9_.\-]+$", username):
        return False, "Username can only contain letters, numbers, and _ . - characters."

    return True, ""


def is_username_available(username: str) -> tuple:
    """
    Returns (is_available, message).

    The username is validated first and then checked against
    the users collection in MongoDB.
    """

    username = username.strip()

    valid, error = validate_username(username)

    if not valid:
        return False, error

    db = get_db()

    if db.users.find_one({"username": username}, {"_id": 1}):
        return False, "Username is already taken."

    return True, "Username is available."


def validate_password(password: str) -> tuple:
    if not password:
        return False, "Password is required."

    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."

    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number."

    return True, ""


def validate_email(email: str) -> tuple:
    email = email.strip()

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Enter a valid email address."

    return True, ""


OTP_VALIDITY_MINUTES = 10


def _generate_otp() -> str:
    return f"{random.randint(0, 999999):06d}"


def register_user(username: str, password: str, email: str) -> dict:
    username = username.strip()
    email = email.strip()

    db = get_db()

    if db.users.find_one({"username": username}):
        return {
            "success": False,
            "message": "Username already exists."
        }

    if db.users.find_one({"email": email}):
        return {
            "success": False,
            "message": "An account with this email already exists."
        }

    otp = _generate_otp()

    db.users.insert_one({
        "username": username,
        "password_hash": _hash_password(password),
        "email": email,
        "email_verified": False,
        "otp_hash": _hash_password(otp),
        "otp_expiry": datetime.datetime.utcnow()
        + datetime.timedelta(minutes=OTP_VALIDITY_MINUTES),
        "created_at": datetime.datetime.utcnow(),
    })

    email_sent = email_utils.send_otp_email(email, otp)

    log_event(
        "register",
        username,
        {
            "email": email,
            "email_sent": email_sent
        }
    )

    return {
        "success": True,
        "message": "Account created.",
        "email_sent": email_sent
    }


def verify_otp(username: str, otp: str) -> dict:
    username = username.strip()
    otp = otp.strip()

    db = get_db()

    user = db.users.find_one({"username": username})

    if not user:
        return {
            "success": False,
            "message": "User not found."
        }

    if user.get("email_verified"):
        return {
            "success": True,
            "message": "Email already verified."
        }

    if datetime.datetime.utcnow() > user.get(
        "otp_expiry",
        datetime.datetime.min
    ):
        return {
            "success": False,
            "message": "OTP expired. Request a new one."
        }

    if not _verify_password(otp, user.get("otp_hash", "")):
        return {
            "success": False,
            "message": "Incorrect OTP."
        }

    db.users.update_one(
        {"username": username},
        {
            "$set": {
                "email_verified": True
            },
            "$unset": {
                "otp_hash": "",
                "otp_expiry": ""
            }
        },
    )

    log_event("email_verified", username, {})

    return {
        "success": True,
        "message": "Email verified! You can now log in."
    }


def resend_otp(username: str) -> dict:
    username = username.strip()

    db = get_db()

    user = db.users.find_one({"username": username})

    if not user:
        return {
            "success": False,
            "message": "User not found."
        }

    if user.get("email_verified"):
        return {
            "success": True,
            "message": "Email already verified."
        }

    otp = _generate_otp()

    db.users.update_one(
        {"username": username},
        {
            "$set": {
                "otp_hash": _hash_password(otp),
                "otp_expiry": datetime.datetime.utcnow()
                + datetime.timedelta(minutes=OTP_VALIDITY_MINUTES),
            }
        },
    )

    email_sent = email_utils.send_otp_email(user["email"], otp)

    log_event(
        "otp_resent",
        username,
        {
            "email_sent": email_sent
        }
    )

    return {
        "success": email_sent,
        "message": (
            "OTP resent — check your email."
            if email_sent
            else
            "Failed to send OTP email. Check the email server configuration."
        )
    }


def authenticate_user(username: str, password: str) -> dict:
    username = username.strip()

    db = get_db()

    user = db.users.find_one({"username": username})

    if not user or not _verify_password(
        password,
        user.get("password_hash", "")
    ):
        log_event("login_failed", username, {})

        return {
            "success": False,
            "message": "Invalid username or password."
        }

    if not user.get("email_verified", False):
        return {
            "success": False,
            "message": "Please verify your email first (check your inbox for the OTP).",
            "needs_otp": True
        }

    log_event("login", username, {})

    return {
        "success": True,
        "message": "Logged in."
    }


def save_grading_result(
    username: str,
    paper_name: str,
    roll_number: str,
    marks_obtained: float,
    max_marks: float,
    details: dict
) -> str:
    db = get_db()

    doc = {
        "username": username,
        "paper_name": paper_name,
        "roll_number": roll_number,
        "marks_obtained": marks_obtained,
        "max_marks": max_marks,
        "percentage": (
            round(100 * marks_obtained / max_marks, 2)
            if max_marks
            else 0
        ),
        "details": details,
        "created_at": datetime.datetime.utcnow(),
    }

    result = db.results.insert_one(doc)

    return str(result.inserted_id)


def get_results_for_user(
    username: str,
    limit: int = 200
) -> List[dict]:
    db = get_db()

    return list(
        db.results.find(
            {"username": username}
        ).sort(
            "created_at",
            -1
        ).limit(limit)
    )


def save_uploaded_file_metadata(
    username: str,
    filename: str,
    file_type: str,
    size_bytes: int
) -> str:
    db = get_db()

    doc = {
        "username": username,
        "filename": filename,
        "file_type": file_type,
        "size_bytes": size_bytes,
        "uploaded_at": datetime.datetime.utcnow(),
    }

    return str(
        db.uploaded_files.insert_one(doc).inserted_id
    )


def log_event(
    event_type: str,
    username: str,
    data: dict
):
    db = get_db()

    db.logs.insert_one({
        "event_type": event_type,
        "username": username,
        "data": data,
        "timestamp": datetime.datetime.utcnow(),
    })


def get_logs(limit: int = 200) -> List[dict]:
    db = get_db()

    return list(
        db.logs.find().sort(
            "timestamp",
            -1
        ).limit(limit)
    )