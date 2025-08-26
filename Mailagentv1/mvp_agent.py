# mvp_agent.py
# Enhanced Email Agent with Dashboard Integration
# Works with both Gmail and Outlook/Office 365

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
from datetime import datetime, timedelta
import platform
import subprocess

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

# Provider-agnostic Outlook/Corporate features
REQUIRE_REVIEW_HIGH_IMPORTANCE = os.getenv("REQUIRE_REVIEW_HIGH_IMPORTANCE", "1") == "1"
REQUIRE_REVIEW_EXTERNAL = os.getenv("REQUIRE_REVIEW_EXTERNAL", "1") == "1"
COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "").lower()  # e.g., "yourcompany.com"
VIP_SENDERS_FILE = os.getenv("VIP_SENDERS_FILE", "vip_senders.json")
AUTO_REPLY_CATEGORIES = os.getenv("AUTO_REPLY_CATEGORIES", "newsletter,notification,automated").split(",")

# AI Configuration
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

STATE_FILE = ".processed_ids.json"
PENDING_REVIEW_FILE = ".pending_review.json"

# Basic assertions to fail fast on missing config
assert EMAIL_ADDRESS and EMAIL_USER and EMAIL_PASS, "Email credentials missing"
assert IMAP_HOST and SMTP_HOST, "IMAP/SMTP host configuration missing"
assert GROQ_API_KEY, "Groq API key missing"

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

# -----------------------
# State Management
# -----------------------
try:
    PROCESSED = set(json.load(open(STATE_FILE)))
except Exception:
    PROCESSED = set()

try:
    PENDING_REVIEW = json.load(open(PENDING_REVIEW_FILE))
except Exception:
    PENDING_REVIEW = []

try:
    VIP_SENDERS = set(json.load(open(VIP_SENDERS_FILE)))
except Exception:
    VIP_SENDERS = set()
    # Create default VIP file
    try:
        json.dump([], open(VIP_SENDERS_FILE, 'w'))
    except Exception:
        pass

def save_state():
    """Save all state to files"""
    try:
        json.dump(sorted(PROCESSED), open(STATE_FILE, "w"))
        json.dump(PENDING_REVIEW, open(PENDING_REVIEW_FILE, "w"), indent=2)
    except Exception as e:
        logging.error(f"Failed to save state: {e}")

# -----------------------
# Desktop Notification Function
# -----------------------
def desktop_notify(title: str, message: str):
    """Send cross-platform desktop notification using stdlib where possible"""
    plat = platform.system()
    try:
        if plat == 'Darwin':  # macOS
            cmd = ['osascript', '-e', f'display notification "{message}" with title "{title}"']
            subprocess.call(cmd)
            logging.info("Sent macOS notification")
        elif plat == 'Windows':
            from ctypes import windll
            windll.user32.MessageBoxW(0, message, title, 0x00001000)  # MB_SYSTEMMODAL
            logging.info("Displayed Windows message box")
        elif plat == 'Linux':
            # Requires notify-send installed (common on most distros)
            subprocess.call(['notify-send', title, message])
            logging.info("Sent Linux notification")
        else:
            logging.warning("Desktop notifications not supported on this platform")
    except Exception as e:
        logging.error(f"Failed to send desktop notification: {e}")

# -----------------------
# Email Provider Utilities
# -----------------------
NO_REPLY_PATTERNS = [
    r"no-?reply",
    r"donotreply",
    r"do-?not-?reply",
    r"noreply",
]

def is_no_reply(addr: str) -> bool:
    """Check if address is a no-reply address"""
    local = (addr or "").split("@")[0].lower()
    return any(re.search(p, local) for p in NO_REPLY_PATTERNS)

def get_email_importance(msg: email.message.Message) -> str:
    """Extract importance level (works for both Gmail and Outlook)"""
    # Outlook headers
    importance = msg.get("Importance", "").lower()
    priority = msg.get("X-Priority", "")
    
    # Gmail priority indicators
    x_gmail_labels = msg.get("X-Gmail-Labels", "").lower()
    
    if importance == "high" or priority == "1" or "important" in x_gmail_labels:
        return "high"
    elif importance == "low" or priority == "5":
        return "low"
    return "normal"

def get_email_sensitivity(msg: email.message.Message) -> str:
    """Extract sensitivity level (mainly Outlook)"""
    sensitivity = msg.get("Sensitivity", "").lower()
    return sensitivity if sensitivity in ["personal", "private", "confidential"] else "normal"

def is_external_sender(sender_addr: str) -> bool:
    """Check if sender is external to company"""
    if not COMPANY_DOMAIN or not sender_addr:
        return False
    return not sender_addr.lower().endswith(f"@{COMPANY_DOMAIN}")

def is_vip_sender(sender_addr: str) -> bool:
    """Check if sender is in VIP list"""
    return sender_addr.lower() in VIP_SENDERS

def detect_email_category(msg: email.message.Message) -> str:
    """Detect email category based on content and headers"""
    subject = decode_mime_header(msg.get("Subject", "")).lower()
    sender = msg.get("From", "").lower()
    body = extract_text_from_message(msg).lower()
    
    # Newsletter/Marketing indicators
    if any(keyword in subject or keyword in sender for keyword in 
           ["newsletter", "unsubscribe", "marketing", "promotion", "offer", "deal"]):
        return "newsletter"
    
    # Automated system indicators
    if any(keyword in sender for keyword in 
           ["noreply", "no-reply", "donotreply", "automated", "system", "notification"]):
        return "automated"
    
    # Meeting/Calendar indicators
    if any(keyword in subject for keyword in 
           ["meeting", "calendar", "appointment", "schedule", "invite", "call"]):
        return "meeting"
    
    # Urgent/Support indicators
    if any(keyword in subject or keyword in body[:500] for keyword in 
           ["urgent", "asap", "emergency", "critical", "help", "support", "issue"]):
        return "urgent"
    
    # Customer service indicators
    if any(keyword in subject or keyword in body[:500] for keyword in 
           ["customer", "client", "complaint", "feedback", "inquiry"]):
        return "customer"
    
    return "business"

def extract_meeting_info(msg: email.message.Message) -> dict:
    """Extract meeting information from email"""
    body = extract_text_from_message(msg)
    meeting_info = {}
    
    # Look for date/time patterns
    date_patterns = [
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}"
    ]
    
    time_patterns = [
        r"(\d{1,2}:\d{2}\s*(?:am|pm|AM|PM))",
        r"(\d{1,2}\s*(?:am|pm|AM|PM))"
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            meeting_info["date"] = match.group(1)
            break
    
    for pattern in time_patterns:
        match = re.search(pattern, body)
        if match:
            meeting_info["time"] = match.group(1)
            break
    
    return meeting_info

def requires_manual_review(msg: email.message.Message, ai_decision: dict) -> Tuple[bool, str]:
    """Determine if email requires manual review before sending"""
    reasons = []
    
    # High importance emails
    if REQUIRE_REVIEW_HIGH_IMPORTANCE and get_email_importance(msg) == "high":
        reasons.append("High importance email")
    
    # External senders
    from_addr = parseaddr_safe(msg.get("From", ""))[1]
    if REQUIRE_REVIEW_EXTERNAL and is_external_sender(from_addr):
        reasons.append("External sender")
    
    # VIP senders
    if is_vip_sender(from_addr):
        reasons.append("VIP sender")
    
    # Confidential content
    if get_email_sensitivity(msg) in ["private", "confidential"]:
        reasons.append("Confidential/Private email")
    
    # AI detected high urgency
    if ai_decision.get("urgency_score", 0) > 0.8:
        reasons.append("High urgency detected")
    
    # Meeting requests
    category = detect_email_category(msg)
    if category == "meeting":
        reasons.append("Meeting request")
    
    # Low confidence AI responses
    if ai_decision.get("confidence", 1.0) < 0.7:
        reasons.append("Low AI confidence")
    
    return len(reasons) > 0, "; ".join(reasons)

# -----------------------
# Text Processing Utilities
# -----------------------
def decode_mime_header(value: Optional[str]) -> str:
    """Decode MIME-encoded headers"""
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
    """Safely parse email address"""
    name, addr = parseaddr(value or "")
    return name, addr

def threading_headers(original: email.message.Message):
    """Generate proper threading headers for replies"""
    in_reply_to = original.get("Message-ID")
    references = original.get("References")
    refs = f"{references} {in_reply_to}" if (references and in_reply_to) else (in_reply_to or references)
    return {"In-Reply-To": in_reply_to, "References": refs}

def truncate_for_prompt(text: str, max_chars: int = 6000) -> str:
    """Truncate text for AI prompt while preserving context"""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return f"{head}\n\n...[snip]...\n\n{tail}"

# -----------------------
# IMAP Email Fetching
# -----------------------
def imap_fetch_unseen(limit=10) -> List[email.message.Message]:
    """Fetch unread emails, prioritizing high importance"""
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        imap.login(EMAIL_USER, EMAIL_PASS)
        imap.select("INBOX")
        
        # Try to prioritize high importance emails (works better with Outlook)
        try:
            _, high_data = imap.search(None, "(UNSEEN HEADER Importance high)")
            _, normal_data = imap.search(None, "(UNSEEN NOT HEADER Importance high)")
            
            high_ids = high_data[0].split() if high_data and high_data[0] else []
            normal_ids = normal_data[0].split() if normal_data and normal_data[0] else []
            
            # Process high importance first
            ids = (high_ids + normal_ids)[:limit]
        except:
            # Fallback to simple unseen search (works for Gmail)
            _, data = imap.search(None, "(UNSEEN)")
            ids = (data[0].split() if data and data[0] else [])[:limit]
        
        messages = []
        for num in ids:
            try:
                _, msg_data = imap.fetch(num, "(RFC822)")
                if not msg_data:
                    continue
                raw = msg_data[0][1]
                messages.append(email.message_from_bytes(raw))
            except Exception as e:
                logging.error(f"Failed to fetch message {num}: {e}")
                continue
        
        imap.close()
        imap.logout()
        return messages
        
    except Exception as e:
        logging.error(f"IMAP fetch error: {e}")
        return []

# -----------------------
# SMTP Email Sending
# -----------------------
def smtp_send(to_addr: str, subject: str, body: str, headers: dict = None):
    """Send email via SMTP"""
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
# AI Processing
# -----------------------
groq_client = Groq(api_key=GROQ_API_KEY)

def groq_chat_complete(user_prompt: str) -> dict:
    """Enhanced Groq completion with corporate email analysis"""
    system_prompt = (
        "You are an advanced email triage assistant for corporate environments. "
        "Analyze emails considering corporate context, urgency, sender relationships, and professional communication standards. "
        "Return ONLY a strict JSON object with keys: "
        "{"
        "\"reply_needed\": true|false, "
        "\"urgency_score\": 0.0-1.0, "
        "\"category\": \"urgent|meeting|business|customer|internal|automated|newsletter\", "
        "\"sentiment\": \"positive|negative|neutral\", "
        "\"requires_action\": true|false, "
        "\"summary\": \"string\", "
        "\"key_points\": [\"point1\", \"point2\"], "
        "\"proposed_subject\": \"string\", "
        "\"proposed_body\": \"string\", "
        "\"confidence\": 0.0-1.0"
        "} "
        "For corporate emails, be professional, concise, and action-oriented. "
        "Consider the importance level, sender relationship, and business context. "
        "No markdown, no commentary outside the JSON."
    )

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Lower for more consistent corporate responses
            max_tokens=1000,
        )

        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise ValueError("Model did not return JSON.")
            return json.loads(m.group(0))
            
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        # Return safe fallback
        return {
            "reply_needed": False,
            "urgency_score": 0.5,
            "category": "business",
            "sentiment": "neutral",
            "requires_action": False,
            "summary": "AI analysis failed",
            "key_points": [],
            "proposed_subject": "",
            "proposed_body": "",
            "confidence": 0.0
        }

def build_enhanced_prompt(msg: email.message.Message) -> str:
    """Build enhanced prompt with provider-agnostic metadata"""
    subject = decode_mime_header(msg.get("Subject", "")) or "(no subject)"
    sender = decode_mime_header(msg.get("From", "")) or "(unknown sender)"
    sender_addr = parseaddr_safe(msg.get("From", ""))[1]
    
    importance = get_email_importance(msg)
    sensitivity = get_email_sensitivity(msg)
    is_external = is_external_sender(sender_addr)
    is_vip = is_vip_sender(sender_addr)
    category = detect_email_category(msg)
    
    body = extract_text_from_message(msg)
    body_trim = truncate_for_prompt(body, 5500)  # Leave room for metadata
    
    meeting_info = extract_meeting_info(msg) if category == "meeting" else {}
    
    metadata = f"""Email metadata:
- Subject: {subject}
- From: {sender}
- Importance: {importance}
- Sensitivity: {sensitivity}
- External sender: {is_external}
- VIP sender: {is_vip}
- Detected category: {category}
- Company domain: {COMPANY_DOMAIN or 'not set'}"""

    if meeting_info:
        metadata += f"\n- Meeting info: {meeting_info}"

    return f"""{metadata}

Email body:
\"\"\"{body_trim}\"\"\"

Analyze this email and provide triage decision with appropriate professional response if needed.
Consider the importance level, sender relationship, and corporate communication standards.
"""

# -----------------------
# Email Processing Logic
# -----------------------
def add_to_pending_review(msg: email.message.Message, decision: dict, reason: str):
    """Add email to pending review queue and trigger desktop notification"""
    pending_item = {
        "timestamp": datetime.now().isoformat(),
        "message_id": msg.get("Message-ID"),
        "from": msg.get("From"),
        "subject": decode_mime_header(msg.get("Subject", "")),
        "importance": get_email_importance(msg),
        "category": decision.get("category", "unknown"),
        "reason": reason,
        "ai_decision": decision,
        "body_preview": extract_text_from_message(msg)[:200] + "..."
    }
    PENDING_REVIEW.append(pending_item)
    logging.info("Added to review queue: %s - %s", pending_item["from"], reason)
    
    # Trigger desktop notification
    notify_title = "Email Pending Review"
    notify_message = f"From: {pending_item['from']}\nSubject: {pending_item['subject']}\nReason: {reason}"
    desktop_notify(notify_title, notify_message)

def process_one_message(msg: email.message.Message):
    """Process a single email message"""
    subject = decode_mime_header(msg.get("Subject", ""))
    from_name, from_addr = parseaddr_safe(msg.get("From", ""))
    reply_to_name, reply_to_addr = parseaddr_safe(msg.get("Reply-To") or msg.get("From", ""))
    
    importance = get_email_importance(msg)
    category = detect_email_category(msg)

    # Guards
    if not from_addr:
        logging.info("Skip: no From address")
        return
    if is_no_reply(from_addr) or (reply_to_addr and is_no_reply(reply_to_addr)):
        logging.info("Skip no-reply sender: %s", from_addr)
        return

    body = extract_text_from_message(msg).strip()
    if not body:
        logging.info("Skip: empty body")
        return

    logging.info("Processing | from=%s | subject=%s | importance=%s | category=%s", 
                from_addr, subject, importance, category)

    # Skip auto-reply categories unless high importance
    if category in AUTO_REPLY_CATEGORIES and importance != "high":
        logging.info("Skip: auto-reply category (%s) with normal importance", category)
        return

    # AI triage + draft
    try:
        decision = groq_chat_complete(build_enhanced_prompt(msg))
    except Exception as e:
        logging.error("AI processing error: %s", e)
        return

    reply_needed = bool(decision.get("reply_needed", False))
    summary = (decision.get("summary") or "").strip()
    confidence = decision.get("confidence", 0.5)
    
    logging.info("AI analysis | reply_needed=%s | confidence=%.2f | category=%s | urgency=%.2f", 
                reply_needed, confidence, decision.get("category"), decision.get("urgency_score", 0))
    logging.info("Summary: %s", summary)

    if not reply_needed:
        logging.info("Decision: no reply needed")
        return

    # Check if requires manual review
    needs_review, review_reason = requires_manual_review(msg, decision)
    
    if needs_review:
        add_to_pending_review(msg, decision, review_reason)
        return

    # Auto-send logic
    proposed_subject = (decision.get("proposed_subject") or "").strip() or f"Re: {subject}"
    proposed_body = (decision.get("proposed_body") or "").strip()

    if not proposed_body:
        logging.info("Skip: AI did not generate reply body")
        return

    to_addr = reply_to_addr or from_addr
    if to_addr.lower() == EMAIL_ADDRESS.lower():
        logging.info("Skip: would reply to self")
        return

    headers = threading_headers(msg)
    final_subject = proposed_subject if proposed_subject.lower().startswith("re:") else f"Re: {subject}"

    logging.info("Auto-reply → %s | %s | confidence=%.2f\n---\n%s\n---", 
                to_addr, final_subject, confidence, proposed_body)

    if DRY_RUN:
        logging.info("DRY_RUN=1 (not sending). Set DRY_RUN=0 to send.")
        return

    try:
        smtp_send(to_addr, final_subject, proposed_body, headers)
        logging.info("Sent auto-reply to %s", to_addr)
    except Exception as e:
        logging.error("Send failed: %s", e)

def run_cycle():
    """Run one processing cycle"""
    try:
        msgs = imap_fetch_unseen(limit=MAX_EMAILS_PER_CYCLE)
        if not msgs:
            logging.info("No unread emails.")
            return
        
        logging.info("Processing %d unread emails", len(msgs))
        for m in msgs:
            msg_id = (m.get("Message-ID") or "").strip()
            if msg_id and msg_id in PROCESSED:
                logging.info("Skip (already processed): %s", msg_id)
                continue
            process_one_message(m)
            if msg_id:
                PROCESSED.add(msg_id)
        save_state()
        
        if PENDING_REVIEW:
            logging.info("=== %d emails pending manual review ===", len(PENDING_REVIEW))
            
    except Exception as e:
        logging.error("Cycle error: %s", e)

def show_pending_review():
    """Display emails pending manual review"""
    if not PENDING_REVIEW:
        print("No emails pending review.")
        return
    
    print(f"\n=== {len(PENDING_REVIEW)} Emails Pending Review ===")
    for i, item in enumerate(PENDING_REVIEW, 1):
        print(f"\n{i}. From: {item['from']}")
        print(f"   Subject: {item['subject']}")
        print(f"   Importance: {item['importance']}")
        print(f"   Reason: {item['reason']}")
        print(f"   Preview: {item['body_preview']}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    logging.info("Email Agent starting (DRY_RUN=%s, POLL_SECONDS=%s)", int(DRY_RUN), POLL_SECONDS)
    logging.info("Provider: %s | Company: %s", SMTP_HOST, COMPANY_DOMAIN)
    logging.info("Review settings: HIGH_IMPORTANCE=%s, EXTERNAL=%s", 
                REQUIRE_REVIEW_HIGH_IMPORTANCE, REQUIRE_REVIEW_EXTERNAL)
    
    # Show any existing pending reviews on startup
    if PENDING_REVIEW:
        show_pending_review()
    
    try:
        while True:
            run_cycle()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        logging.info("Agent stopped by user")
        save_state()
    except Exception as e:
        logging.error("Agent crashed: %s", e)
        save_state()