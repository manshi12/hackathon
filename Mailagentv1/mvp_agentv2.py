# mvp_agent.py
# MVP: Automated email triage and reply using Python + Azure OpenAI

import os
import re
import ssl
import time
import json
import imaplib
import smtplib
import logging
import email
from typing import List, Tuple, Optional
from email.header import decode_header, make_header
from email.utils import parseaddr
from email.message import EmailMessage

# Optional .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -----------------------
# Configuration
# -----------------------
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_USER = os.getenv("EMAIL_USER", EMAIL_ADDRESS)
EMAIL_PASS = os.getenv("EMAIL_PASS", "")

IMAP_HOST = os.getenv("IMAP_HOST", "")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") == "1"

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
MAX_EMAILS_PER_CYCLE = int(os.getenv("MAX_EMAILS_PER_CYCLE", "10"))
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"

# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

STATE_FILE = ".processed_ids.json"

# Basic assertions to fail fast on missing config
assert EMAIL_ADDRESS and EMAIL_USER and EMAIL_PASS, "Email credentials missing"
assert IMAP_HOST and SMTP_HOST, "IMAP/SMTP host configuration missing"
# assert AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT, "Azure OpenAI config missing"

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

# -----------------------
# State (processed Message-IDs)
# -----------------------
try:
    PROCESSED = set(json.load(open(STATE_FILE)))
except Exception:
    PROCESSED = set()

def save_state():
    try:
        json.dump(sorted(PROCESSED), open(STATE_FILE, "w"))
    except Exception:
        pass

# -----------------------
# Utilities
# -----------------------
NO_REPLY_PATTERNS = [
    r"no-?reply",
    r"donotreply",
    r"do-?not-?reply",
]

def is_no_reply(addr: str) -> bool:
    local = (addr or "").split("@")[0].lower()
    return any(re.search(p, local) for p in NO_REPLY_PATTERNS)

def decode_mime_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value

def extract_text_from_message(msg: email.message.Message) -> str:
    """Prefer text/plain, fallback to text/html with crude tag stripping."""
    def strip_html(html: str) -> str:
        html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
        html = re.sub(r"(?is)<br\s*/?>", "\n", html)
        html = re.sub(r"(?is)</p>", "\n\n", html)
        text = re.sub(r"(?s)<.*?>", "", html)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/plain":
                raw = part.get_payload(decode=True) or b""
                return raw.decode(part.get_content_charset() or "utf-8", "replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                raw = part.get_payload(decode=True) or b""
                html = raw.decode(part.get_content_charset() or "utf-8", "replace")
                return strip_html(html)
        return ""
    else:
        ctype = msg.get_content_type()
        raw = msg.get_payload(decode=True) or b""
        text = raw.decode(msg.get_content_charset() or "utf-8", "replace")
        return strip_html(text) if ctype == "text/html" else text

def parseaddr_safe(value: str) -> Tuple[str, str]:
    name, addr = parseaddr(value or "")
    return name, addr

def threading_headers(original: email.message.Message):
    in_reply_to = original.get("Message-ID")
    references = original.get("References")
    refs = f"{references} {in_reply_to}" if (references and in_reply_to) else (in_reply_to or references)
    return {"In-Reply-To": in_reply_to, "References": refs}

def truncate_for_prompt(text: str, max_chars: int = 6000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return f"{head}\n\n...[snip]...\n\n{tail}"

# -----------------------
# IMAP fetch
# -----------------------
def imap_fetch_unseen(limit=10) -> List[email.message.Message]:
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    imap.login(EMAIL_USER, EMAIL_PASS)
    imap.select("INBOX")
    _, data = imap.search(None, "(UNSEEN)")
    ids = (data[0].split() if data and data[0] else [])[:limit]
    messages = []
    for num in ids:
        _, msg_data = imap.fetch(num, "(RFC822)")
        if not msg_data:
            continue
        raw = msg_data[0][1]
        messages.append(email.message_from_bytes(raw))
    imap.close()
    imap.logout()
    return messages

# -----------------------
# SMTP send
# -----------------------
def smtp_send(to_addr: str, subject: str, body: str, headers: dict | None = None):
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_addr
    msg["Subject"] = subject
    for k, v in (headers or {}).items():
        if v:
            msg[k] = v
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        if SMTP_STARTTLS:
            s.starttls(context=context)
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)

# -----------------------
# Azure OpenAI (triage + draft)
# -----------------------
import requests

# def azure_chat_complete(user_prompt: str) -> dict:
#     """
#     Calls Azure OpenAI Chat Completions and returns strict JSON fields:
#     reply_needed (bool), summary (str), proposed_subject (str), proposed_body (str)
#     """
#     url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
#     params = {"api-version": AZURE_OPENAI_API_VERSION}
#     headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_API_KEY}
#     system_prompt = (
#         "You are an email triage and drafting assistant. "
#         "Given an email's subject, sender, and body, decide if a reply is needed. "
#         "If a reply is needed, draft a concise, professional reply. "
#         "Return ONLY a strict JSON object with keys: "
#         "{"
#         "\"reply_needed\": true|false, "
#         "\"summary\": \"string\", "
#         "\"proposed_subject\": \"string\", "
#         "\"proposed_body\": \"string\""
#         "} "
#         "No markdown, no commentary."
#     )
#     payload = {
#         "temperature": 0.4,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#     }
#     r = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
#     r.raise_for_status()
#     content = r.json()["choices"][0]["message"]["content"]

#     # Strict JSON parse with fallback
#     try:
#         return json.loads(content)
#     except Exception:
#         m = re.search(r"\{[\s\S]*\}", content)
#         if not m:
#             raise ValueError("Model did not return JSON.")
#         return json.loads(m.group(0))




groq_client = Groq(api_key=GROQ_API_KEY)

def groq_chat_complete(user_prompt: str) -> dict:
    """
    Calls Groq LLaMA‑3 and returns strict JSON fields:
    reply_needed (bool), summary (str), proposed_subject (str), proposed_body (str)
    """
    system_prompt = (
        "You are an email triage and drafting assistant. "
        "Given an email's subject, sender, and body, decide if a reply is needed. "
        "If a reply is needed, draft a concise, professional reply. "
        "Return ONLY a strict JSON object with keys: "
        "{"
        "\"reply_needed\": true|false, "
        "\"summary\": \"string\", "
        "\"proposed_subject\": \"string\", "
        "\"proposed_body\": \"string\""
        "} "
        "No markdown, no commentary."
    )

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=800,
    )

    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            raise ValueError("Model did not return JSON.")
        return json.loads(m.group(0))


def build_prompt(msg: email.message.Message) -> str:
    subject = decode_mime_header(msg.get("Subject", "")) or "(no subject)"
    sender = decode_mime_header(msg.get("From", "")) or "(unknown sender)"
    body = extract_text_from_message(msg)
    body_trim = truncate_for_prompt(body, 6500)
    return f"""Email metadata:
- Subject: {subject}
- From: {sender}

Email body:
\"\"\"{body_trim}\"\"\"

Decide if a reply is needed (ignore spam/automated notifications). If yes, draft a concise, professional reply.
Include clarifying questions only if essential.
"""

# -----------------------
# Orchestration
# -----------------------
def process_one_message(msg: email.message.Message):
    subject = decode_mime_header(msg.get("Subject", ""))
    from_name, from_addr = parseaddr_safe(msg.get("From", ""))
    reply_to_name, reply_to_addr = parseaddr_safe(msg.get("Reply-To") or msg.get("From", ""))

    # Guards
    if not from_addr:
        logging.info("Skip: no From")
        return
    if is_no_reply(from_addr) or (reply_to_addr and is_no_reply(reply_to_addr)):
        logging.info("Skip no-reply sender: %s", from_addr)
        return

    body = extract_text_from_message(msg).strip()
    if not body:
        logging.info("Skip: empty body")
        return

    logging.info("Incoming | from=%s | subject=%s", from_addr, subject)

    # AI triage + draft
    try:
        # decision = azure_chat_complete(build_prompt(msg))
        decision = groq_chat_complete(build_prompt(msg))
    except Exception as e:
        logging.error("AI error: %s", e)
        return

    reply_needed = bool(decision.get("reply_needed", False))
    summary = (decision.get("summary") or "").strip()
    proposed_subject = (decision.get("proposed_subject") or "").strip() or f"Re: {subject}"
    proposed_body = (decision.get("proposed_body") or "").strip()

    logging.info("AI summary: %s", summary or "(none)")
    if not reply_needed:
        logging.info("Decision: no reply")
        return

    # Prepare send
    to_addr = reply_to_addr or from_addr
    if to_addr.lower() == EMAIL_ADDRESS.lower():
        logging.info("Skip: would reply to self")
        return

    headers = threading_headers(msg)
    final_subject = proposed_subject if proposed_subject.lower().startswith("re:") else f"Re: {subject}"

    logging.info("Draft → %s | %s\n---\n%s\n---", to_addr, final_subject, proposed_body)

    # Dry run or send
    if DRY_RUN:
        logging.info("DRY_RUN=1 (not sending). Set DRY_RUN=0 to send.")
        return

    try:
        smtp_send(to_addr, final_subject, proposed_body, headers)
        logging.info("Sent reply to %s", to_addr)
    except Exception as e:
        logging.error("Send failed: %s", e)

def run_cycle():
    try:
        msgs = imap_fetch_unseen(limit=MAX_EMAILS_PER_CYCLE)
        if not msgs:
            logging.info("No unread.")
            return
        for m in msgs:
            msg_id = (m.get("Message-ID") or "").strip()
            if msg_id and msg_id in PROCESSED:
                logging.info("Skip (already processed): %s", msg_id)
                continue
            process_one_message(m)
            if msg_id:
                PROCESSED.add(msg_id)
        save_state()
    except Exception as e:
        logging.error("Cycle error: %s", e)

if __name__ == "__main__":
    logging.info("Agent starting (DRY_RUN=%s, POLL_SECONDS=%s)", int(DRY_RUN), POLL_SECONDS)
    while True:
        run_cycle()
        time.sleep(POLL_SECONDS)