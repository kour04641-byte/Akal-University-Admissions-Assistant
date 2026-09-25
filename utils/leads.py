import csv
import re
from datetime import datetime
from pathlib import Path

LEADS_FILE = Path(__file__).resolve().parent.parent / "leads.csv"
FIELDS = ["timestamp", "name", "phone", "email", "program"]


def normalize_phone(phone):
    """Return a 10-digit Indian mobile number, or None if invalid."""
    digits = re.sub(r"\D", "", phone or "")
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


def valid_email(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""))


def _safe(value):
    """Stop spreadsheet formula injection when the CSV is opened in Excel."""
    value = (value or "").strip()
    return "'" + value if value[:1] in ("=", "+", "-", "@") else value


def save_lead(name, phone, email, program):
    is_new = not LEADS_FILE.exists()
    with open(LEADS_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FIELDS)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            _safe(name), _safe(phone), _safe(email), _safe(program),
        ])