# mvp_agent.py - Enhanced with proper thread handling and summary fixes
# Enhanced Email Agent with Dashboard Integration and proper thread reply handling

import os
import re
import ssl
import time
import json
import imaplib
import smtplib
import logging
import email
from typing import List, Tuple, Optional, Dict
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
# Configuration (same as before)
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

REQUIRE_REVIEW_HIGH_IMPORTANCE = os.getenv("REQUIRE_REVIEW_HIGH_IMPORTANCE", "1") == "1"
REQUIRE_REVIEW_EXTERNAL = os.getenv("REQUIRE_REVIEW_EXTERNAL", "1") == "1"
COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "").lower()
VIP_SENDERS_FILE = os.getenv("VIP_SENDERS_FILE", "vip_senders.json")
AUTO_REPLY_CATEGORIES = os.getenv("AUTO_REPLY_CATEGORIES", "newsletter,notification,automated").split(",")

from groq import Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

STATE_FILE = ".processed_ids.json"
PENDING_REVIEW_FILE = ".pending_review.json"

assert EMAIL_ADDRESS and EMAIL_USER and EMAIL_PASS, "Email credentials missing"
assert IMAP_HOST and SMTP_HOST, "IMAP/SMTP host configuration missing"
assert GROQ_API_KEY, "Groq API key missing"

# -----------------------
# Logging and State (same as before)
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

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
    try:
        json.dump([], open(VIP_SENDERS_FILE, 'w'))
    except Exception:
        pass

# -----------------------
# Email Provider Utilities
# -----------------------
NO_REPLY_PATTERNS = [
    r"no-?reply",
    r"donotreply",
    r"do-?not-?reply",
    r"noreply",
]

def save_state():
    """Save all state to files"""
    try:
        json.dump(sorted(PROCESSED), open(STATE_FILE, "w"))
        json.dump(PENDING_REVIEW, open(PENDING_REVIEW_FILE, "w"), indent=2)
    except Exception as e:
        logging.error(f"Failed to save state: {e}")

# -----------------------
# Helper Functions (Enhanced)
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
    """Extract text content from email message"""
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

def is_thread_reply(msg: email.message.Message) -> bool:
    """Check if email is part of an existing thread"""
    in_reply_to = msg.get("In-Reply-To")
    references = msg.get("References") 
    subject = decode_mime_header(msg.get("Subject", "")).strip()
    has_re_pattern = bool(re.match(r"^(re|fw|fwd):\s*", subject, re.IGNORECASE))
    
    return bool(in_reply_to or references or has_re_pattern)

def extract_original_subject(subject: str) -> str:
    """Extract original subject from Re: Fw: patterns"""
    if not subject:
        return ""
    
    clean_subject = re.sub(r"^(re|fw|fwd):\s*", "", subject, flags=re.IGNORECASE)
    clean_subject = re.sub(r"^\[.*?\]\s*", "", clean_subject)
    
    return clean_subject.strip()

def get_thread_id(msg: email.message.Message) -> str:
    """Generate consistent thread ID for grouping related emails"""
    in_reply_to = msg.get("In-Reply-To", "").strip()
    if in_reply_to:
        return in_reply_to
    
    references = msg.get("References", "").strip()
    if references:
        first_ref = references.split()[0] if references else ""
        if first_ref:
            return first_ref
    
    subject = extract_original_subject(decode_mime_header(msg.get("Subject", "")))
    sender = parseaddr_safe(msg.get("From", ""))[1]
    
    thread_key = f"{subject.lower()}:{sender.lower()}"
    return f"thread-{hash(thread_key)}"

def threading_headers(original: email.message.Message) -> Dict[str, str]:
    """Generate proper threading headers for replies - FIXED VERSION"""
    headers = {}
    
    # Get the Message-ID from original email
    original_msg_id = original.get("Message-ID", "").strip()
    if original_msg_id:
        headers["In-Reply-To"] = original_msg_id
    
    # Build References chain properly
    existing_refs = original.get("References", "").strip()
    if existing_refs and original_msg_id:
        # Add original Message-ID to existing references
        headers["References"] = f"{existing_refs} {original_msg_id}"
    elif original_msg_id:
        # Start new references chain
        headers["References"] = original_msg_id
    elif existing_refs:
        # Keep existing references even without new Message-ID
        headers["References"] = existing_refs
    
    return headers

def clean_header(value: str) -> str:
    """Clean header value by removing invalid characters like \r and \n"""
    return re.sub(r'[\r\n]', ' ', value.strip())

def is_no_reply(addr: str) -> bool:
    """Check if address is a no-reply address"""
    local = (addr or "").split("@")[0].lower()
    return any(re.search(p, local) for p in NO_REPLY_PATTERNS)

def get_email_importance(msg: email.message.Message) -> str:
    """Extract importance level"""
    importance = msg.get("Importance", "").lower()
    priority = msg.get("X-Priority", "")
    x_gmail_labels = msg.get("X-Gmail-Labels", "").lower()
    
    if importance == "high" or priority == "1" or "important" in x_gmail_labels:
        return "high"
    elif importance == "low" or priority == "5":
        return "low"
    return "normal"

def is_external_sender(sender_addr: str) -> bool:
    """Check if sender is external to company"""
    if not COMPANY_DOMAIN or not sender_addr:
        return False
    return not sender_addr.lower().endswith(f"@{COMPANY_DOMAIN}")

def is_vip_sender(sender_addr: str) -> bool:
    """Check if sender is in VIP list"""
    return sender_addr.lower() in VIP_SENDERS

def detect_email_category(msg: email.message.Message) -> str:
    """Detect email category"""
    subject = decode_mime_header(msg.get("Subject", "")).lower()
    sender = msg.get("From", "").lower()
    body = extract_text_from_message(msg).lower()
    
    if any(keyword in subject or keyword in sender for keyword in 
           ["newsletter", "unsubscribe", "marketing", "promotion"]):
        return "newsletter"
    
    if any(keyword in sender for keyword in 
           ["noreply", "no-reply", "donotreply", "automated", "system"]):
        return "automated"
    
    if any(keyword in subject for keyword in 
           ["meeting", "calendar", "appointment", "schedule"]):
        return "meeting"
    
    if any(keyword in subject or keyword in body[:500] for keyword in 
           ["urgent", "asap", "emergency", "critical"]):
        return "urgent"
    
    return "business"

# -----------------------
# Desktop Notification
# -----------------------
def desktop_notify(title: str, message: str):
    """Send cross-platform desktop notification using stdlib where possible"""
    plat = platform.system()
    try:
        if plat == 'Darwin':
            cmd = ['osascript', '-e', f'display notification "{message}" with title "{title}"']
            subprocess.call(cmd)
            logging.info("Sent macOS notification")
        elif plat == 'Windows':
            from ctypes import windll
            windll.user32.MessageBoxW(0, message, title, 0x00001000)
            logging.info("Displayed Windows message box")
        elif plat == 'Linux':
            subprocess.call(['notify-send', title, message])
            logging.info("Sent Linux notification")
        else:
            logging.warning("Desktop notifications not supported on this platform")
    except Exception as e:
        logging.error(f"Failed to send desktop notification: {e}")

def fetch_thread_context(msg: email.message.Message, limit: int = 5) -> List[Dict]:
    """Fetch thread context with better error handling and structure"""
    thread_context = []
    
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        imap.login(EMAIL_USER, EMAIL_PASS)
        imap.select("INBOX")
        
        references = []
        in_reply_to = msg.get("In-Reply-To", "").strip()
        existing_refs = msg.get("References", "").strip()
        
        if in_reply_to:
            references.append(in_reply_to)
        
        if existing_refs:
            refs = [ref.strip() for ref in existing_refs.split() if ref.strip()]
            references.extend(refs[-limit:])
        
        original_subject = extract_original_subject(decode_mime_header(msg.get("Subject", "")))
        sender_addr = parseaddr_safe(msg.get("From", ""))[1]
        
        seen_message_ids = set()
        
        # Search by Message-ID references first
        for ref_id in references:
            if ref_id in seen_message_ids:
                continue
                
            try:
                search_query = f'HEADER Message-ID "{ref_id}"'
                _, data = imap.search(None, search_query)
                
                if data and data[0]:
                    msg_nums = data[0].split()
                    for msg_num in msg_nums[:1]:
                        _, fetch_data = imap.fetch(msg_num, "(RFC822)")
                        if fetch_data and fetch_data[0] and len(fetch_data[0]) > 1:
                            raw_msg = fetch_data[0][1]
                            parsed_msg = email.message_from_bytes(raw_msg)
                            
                            thread_context.append({
                                'date': parsed_msg.get("Date", ""),
                                'from': decode_mime_header(parsed_msg.get("From", "")),
                                'subject': decode_mime_header(parsed_msg.get("Subject", "")),
                                'body': extract_text_from_message(parsed_msg)[:1000],
                                'message_id': parsed_msg.get("Message-ID", "")
                            })
                            seen_message_ids.add(ref_id)
                            
            except Exception as e:
                logging.debug(f"Failed to fetch reference {ref_id}: {e}")
                continue
        
        # Subject-based search fallback
        if len(thread_context) < 2 and original_subject:
            try:
                search_query = f'FROM "{sender_addr}" SUBJECT "{original_subject}"'
                _, data = imap.search(None, search_query)
                
                if data and data[0]:
                    msg_nums = data[0].split()[-5:]
                    for msg_num in msg_nums:
                        try:
                            _, fetch_data = imap.fetch(msg_num, "(RFC822)")
                            if fetch_data and fetch_data[0] and len(fetch_data[0]) > 1:
                                raw_msg = fetch_data[0][1]
                                parsed_msg = email.message_from_bytes(raw_msg)
                                
                                msg_id = parsed_msg.get("Message-ID", "")
                                if msg_id not in seen_message_ids:
                                    thread_context.append({
                                        'date': parsed_msg.get("Date", ""),
                                        'from': decode_mime_header(parsed_msg.get("From", "")),
                                        'subject': decode_mime_header(parsed_msg.get("Subject", "")),
                                        'body': extract_text_from_message(parsed_msg)[:1000],
                                        'message_id': msg_id
                                    })
                                    seen_message_ids.add(msg_id)
                                    
                        except Exception as e:
                            logging.debug(f"Failed to fetch thread message {msg_num}: {e}")
                            
            except Exception as e:
                logging.debug(f"Subject-based thread search failed: {e}")
        
        imap.close()
        imap.logout()
        
        thread_context.sort(key=lambda x: x.get('date', ''))
        return thread_context[-limit:]
        
    except Exception as e:
        logging.error(f"Thread context fetch failed: {e}")
        return []

def summarize_thread(thread_context: List[Dict]) -> str:
    """Generate AI summary of thread context - ENHANCED VERSION"""
    if not thread_context:
        return ""
    
    thread_text = ""
    for ctx in thread_context:
        thread_text += f"[{ctx['date']}] From: {ctx['from']}\nSubject: {ctx['subject']}\n{ctx['body']}\n\n"
    
    if len(thread_text) > 4000:
        thread_text = thread_text[-4000:]
    
    summary_prompt = (
        "You are a thread summary assistant. Summarize this email conversation history in 2-4 sentences. "
        "Focus on: 1) What the conversation is about, 2) Key decisions or requests made, 3) Current status/next steps. "
        "Be concise but informative. Output only the summary text."
    )

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": thread_text},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        summary = resp.choices[0].message.content.strip()
        logging.info(f"Generated thread summary: {summary[:100]}...")
        return summary
    except Exception as e:
        logging.error(f"Thread summary failed: {e}")
        return f"Thread conversation with {len(thread_context)} previous messages. Unable to generate detailed summary."

def build_thread_aware_prompt(msg: email.message.Message, thread_context: List[Dict]) -> str:
    """Build AI prompt with full thread context and summary"""
    subject = decode_mime_header(msg.get("Subject", "")) or "(no subject)"
    sender = decode_mime_header(msg.get("From", "")) or "(unknown sender)"
    sender_addr = parseaddr_safe(msg.get("From", ""))[1]
    
    importance = get_email_importance(msg)
    is_external = is_external_sender(sender_addr)
    is_vip = is_vip_sender(sender_addr)
    category = detect_email_category(msg)
    is_reply = is_thread_reply(msg)
    
    current_body = extract_text_from_message(msg)
    
    # Generate thread summary
    thread_summary = summarize_thread(thread_context)
    
    # Build context section
    context_section = ""
    if thread_context:
        context_section = f"\n=== THREAD HISTORY SUMMARY ===\n{thread_summary}\n"
        context_section += "\n=== RECENT THREAD MESSAGES ===\n"
        for i, ctx in enumerate(thread_context[-3:], 1):
            context_section += f"\n{i}. [{ctx['date']}] From: {ctx['from']}\n"
            context_section += f"   Subject: {ctx['subject']}\n"
            context_section += f"   Body: {ctx['body'][:300]}...\n"
        context_section += "\n=== END THREAD HISTORY ===\n"
    
    metadata = f"""Email Analysis Request:

CURRENT EMAIL:
- Subject: {subject}
- From: {sender}
- Importance: {importance}
- External sender: {is_external}
- VIP sender: {is_vip}
- Category: {category}
- Is thread reply: {is_reply}
- Company domain: {COMPANY_DOMAIN or 'not set'}

{context_section}

CURRENT EMAIL BODY:
\"\"\"{current_body[:4000]}\"\"\"

INSTRUCTIONS:
This email {'is part of an ongoing thread' if is_reply else 'appears to be a new conversation'}. 
{'Consider the full thread context and summary when generating your response.' if thread_context else 'Generate an appropriate response based on this email.'}

Provide a JSON response with thread-aware analysis and reply generation.
"""
    
    return metadata

# -----------------------
# Enhanced AI Processing
# -----------------------
groq_client = Groq(api_key=GROQ_API_KEY)

def groq_chat_complete_with_thread(user_prompt: str, is_thread_reply: bool = False) -> dict:
    """Enhanced Groq completion with thread awareness"""
    thread_context = "thread continuation" if is_thread_reply else "new conversation"
    
    system_prompt = f"""You are an advanced email assistant specializing in {thread_context}. 
    
When processing thread replies:
- Consider the full conversation history
- Maintain context and reference previous points
- Provide continuity in communication tone
- Address specific questions or requests from the thread
- Avoid repeating information already covered

When processing new conversations:
- Focus on the current email content
- Provide comprehensive initial responses
- Set appropriate tone for the relationship

Return ONLY a strict JSON object with keys:
{{
  "reply_needed": true|false,
  "urgency_score": 0.0-1.0,
  "category": "urgent|meeting|business|customer|internal|automated|newsletter",
  "sentiment": "positive|negative|neutral", 
  "requires_action": true|false,
  "summary": "string",
  "key_points": ["point1", "point2"],
  "proposed_subject": "string",
  "proposed_body": "string",
  "confidence": 0.0-1.0,
  "thread_context_used": true|false,
  "is_thread_continuation": {str(is_thread_reply).lower()}
}}

Be professional, contextually aware, and ensure replies maintain thread continuity.
"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
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
            "confidence": 0.0,
            "thread_context_used": False,
            "is_thread_continuation": is_thread_reply
        }

# -----------------------
# Enhanced SMTP with proper threading
# -----------------------
def smtp_send(to_addr: str, subject: str, body: str, headers: dict = None):
    """Send email via SMTP with proper headers"""
    msg = EmailMessage()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_addr
    msg["Subject"] = subject
    
    # Add threading headers with cleaning
    for k, v in (headers or {}).items():
        if v:
            clean_value = clean_header(str(v))
            msg[k] = clean_value
            logging.info(f"Added header {k}: {clean_value[:50]}...")
    
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        if SMTP_STARTTLS:
            s.starttls(context=context)
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)

# -----------------------
# Enhanced Message Processing
# -----------------------
def requires_manual_review(msg: email.message.Message, ai_decision: dict) -> Tuple[bool, str]:
    """Determine if email requires manual review"""
    reasons = []
    
    if REQUIRE_REVIEW_HIGH_IMPORTANCE and get_email_importance(msg) == "high":
        reasons.append("High importance email")
    
    from_addr = parseaddr_safe(msg.get("From", ""))[1]
    if REQUIRE_REVIEW_EXTERNAL and is_external_sender(from_addr):
        reasons.append("External sender")
    
    if is_vip_sender(from_addr):
        reasons.append("VIP sender")
    
    if ai_decision.get("urgency_score", 0) > 0.8:
        reasons.append("High urgency detected")
    
    if ai_decision.get("confidence", 1.0) < 0.7:
        reasons.append("Low AI confidence")
    
    if ai_decision.get("is_thread_continuation") and ai_decision.get("confidence", 1.0) < 0.8:
        reasons.append("Complex thread continuation")
    
    return len(reasons) > 0, "; ".join(reasons)

def add_to_pending_review(msg: email.message.Message, decision: dict, reason: str, thread_summary: str = "", thread_context: List[Dict] = None):
    """Add email to pending review queue with complete thread information"""
    
    # Store complete original message for proper threading later
    original_headers = {}
    for header_name in ['Message-ID', 'In-Reply-To', 'References', 'Date']:
        header_value = msg.get(header_name)
        if header_value:
            original_headers[header_name] = header_value
    
    pending_item = {
        "timestamp": datetime.now().isoformat(),
        "message_id": msg.get("Message-ID"),
        "from": msg.get("From"),
        "subject": decode_mime_header(msg.get("Subject", "")),
        "importance": get_email_importance(msg),
        "category": decision.get("category", "unknown"),
        "reason": reason,
        "ai_decision": decision,
        "body_preview": extract_text_from_message(msg)[:200] + "...",
        "is_thread_reply": is_thread_reply(msg),
        "thread_context_used": decision.get("thread_context_used", False),
        "thread_summary": thread_summary,
        "original_headers": original_headers,  # Store for proper threading
        "thread_context": thread_context or []  # Store full context if needed
    }
    PENDING_REVIEW.append(pending_item)
    logging.info("Added to review queue: %s - %s", pending_item["from"], reason)
    
    notify_title = "Email Pending Review"
    notify_message = f"From: {pending_item['from']}\nSubject: {pending_item['subject']}\nReason: {reason}"
    desktop_notify(notify_title, notify_message)

def process_one_message(msg: email.message.Message):
    """Enhanced message processing with thread awareness"""
    subject = decode_mime_header(msg.get("Subject", ""))
    from_name, from_addr = parseaddr_safe(msg.get("From", ""))
    reply_to_name, reply_to_addr = parseaddr_safe(msg.get("Reply-To") or msg.get("From", ""))
    
    importance = get_email_importance(msg)
    category = detect_email_category(msg)
    is_reply = is_thread_reply(msg)
    
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

    logging.info("Processing | from=%s | subject=%s | importance=%s | category=%s | thread_reply=%s", 
                from_addr, subject, importance, category, is_reply)

    # Fetch thread context if this is a reply
    thread_context = []
    thread_summary = ""
    if is_reply:
        thread_context = fetch_thread_context(msg, limit=5)
        thread_summary = summarize_thread(thread_context)
        logging.info("Thread context: %d previous messages found", len(thread_context))
        if thread_summary:
            logging.info("Thread summary: %s", thread_summary[:100])

    # AI analysis with thread context
    try:
        prompt = build_thread_aware_prompt(msg, thread_context)
        decision = groq_chat_complete_with_thread(prompt, is_reply)
        
        if decision.get("thread_context_used"):
            logging.info("AI used thread context for analysis")
            
    except Exception as e:
        logging.error("AI processing error: %s", e)
        return

    reply_needed = bool(decision.get("reply_needed", False))
    confidence = decision.get("confidence", 0.5)
    
    logging.info("AI analysis | reply_needed=%s | confidence=%.2f | thread_continuation=%s", 
                reply_needed, confidence, decision.get("is_thread_continuation"))

    if not reply_needed:
        logging.info("Decision: no reply needed")
        return

    # Check manual review requirements
    needs_review, review_reason = requires_manual_review(msg, decision)
    
    if needs_review:
        add_to_pending_review(msg, decision, review_reason, thread_summary, thread_context)
        return

    # Prepare reply
    proposed_subject = decision.get("proposed_subject", "").strip()
    proposed_body = decision.get("proposed_body", "").strip()

    if not proposed_body:
        logging.info("Skip: AI did not generate reply body")
        return

    # Generate proper reply subject
    if is_reply:
        original_subject = extract_original_subject(subject)
        final_subject = f"Re: {original_subject}" if original_subject else f"Re: {subject}"
    else:
        final_subject = proposed_subject if proposed_subject else f"Re: {subject}"

    to_addr = reply_to_addr or from_addr
    if to_addr.lower() == EMAIL_ADDRESS.lower():
        logging.info("Skip: would reply to self")
        return

    # Generate proper threading headers
    headers = threading_headers(msg)
    
    # Clean headers before sending
    cleaned_headers = {k: clean_header(v) for k, v in headers.items()}
    
    logging.info("Thread-aware reply → %s | %s | confidence=%.2f | headers=%s", 
                to_addr, final_subject, confidence, list(headers.keys()))
    logging.info("Reply body preview: %s", proposed_body[:200] + "...")

    if DRY_RUN:
        logging.info("DRY_RUN=1 (not sending). Set DRY_RUN=0 to send.")
        return

    try:
        smtp_send(to_addr, final_subject, proposed_body, cleaned_headers)
        logging.info("Sent thread-aware reply to %s", to_addr)
    except Exception as e:
        logging.error("Send failed: %s", e)

# -----------------------
# IMAP and Main Functions (same as before)
# -----------------------
def imap_fetch_unseen(limit=10) -> List[email.message.Message]:
    """Fetch unread emails, prioritizing high importance"""
    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        imap.login(EMAIL_USER, EMAIL_PASS)
        imap.select("INBOX")
        
        try:
            _, high_data = imap.search(None, "(UNSEEN HEADER Importance high)")
            _, normal_data = imap.search(None, "(UNSEEN NOT HEADER Importance high)")
            
            high_ids = high_data[0].split() if high_data and high_data[0] else []
            normal_ids = normal_data[0].split() if normal_data and normal_data[0] else []
            
            ids = (high_ids + normal_ids)[:limit]
        except:
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
        print(f"   Thread Reply: {item.get('is_thread_reply', False)}")
        print(f"   Reason: {item['reason']}")
        print(f"   Preview: {item['body_preview']}")
        if item.get("thread_summary"):
            print(f"   Thread Summary: {item['thread_summary'][:200]}...")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    logging.info("Enhanced Email Agent starting (DRY_RUN=%s, POLL_SECONDS=%s)", int(DRY_RUN), POLL_SECONDS)
    logging.info("Provider: %s | Company: %s", SMTP_HOST, COMPANY_DOMAIN)
    logging.info("Review settings: HIGH_IMPORTANCE=%s, EXTERNAL=%s", 
                REQUIRE_REVIEW_HIGH_IMPORTANCE, REQUIRE_REVIEW_EXTERNAL)
    logging.info("Thread handling: ENABLED with context  fetch")
    
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
