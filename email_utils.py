import os
import smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "jayawsb74@gmail.com")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD", "oxmp otzq uspr jxvn")

def is_configured() -> bool:
    return bool(EMAIL_SENDER and EMAIL_APP_PASSWORD)

def send_otp_email(to_email: str, otp: str) -> bool:
    if not is_configured():
        print("[email_utils] EMAIL_SENDER/EMAIL_APP_PASSWORD not set — cannot send OTP email.")
        return False

    subject = "GradeSense — Your verification code"
    body = (
        f"Your GradeSense verification code is: {otp}\n\n"
        f"This code expires in 10 minutes. If you didn't request this, you can ignore this email."
    )
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_SENDER, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[email_utils] Failed to send OTP email: {e}")
        return False