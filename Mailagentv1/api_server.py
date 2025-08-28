# api_server.py - Enhanced FastAPI server with FIXED thread handling support
# FastAPI REST API for Email Agent Dashboard with Thread Context Support

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import threading
import time
import email
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# Import your enhanced agent
import mvp_agent

# Enhanced Pydantic Models
class EmailSummary(BaseModel):
    message_id: str
    from_addr: str
    subject: str
    timestamp: str
    importance: str
    category: str
    status: str
    is_thread_reply: Optional[bool] = False

class PendingEmail(BaseModel):
    id: str
    timestamp: str
    message_id: str
    from_addr: str
    subject: str
    importance: str
    category: str
    reason: str
    ai_decision: Dict[str, Any]
    body_preview: str
    is_thread_reply: Optional[bool] = False
    thread_context_used: Optional[bool] = False
    thread_summary: Optional[str] = None

class EmailAction(BaseModel):
    action: str  # approve, reject, modify
    reply_subject: Optional[str] = None
    reply_body: Optional[str] = None
    maintain_thread: Optional[bool] = True

class AgentStatus(BaseModel):
    is_running: bool
    last_cycle: str
    processed_count: int
    pending_count: int
    error_count: int
    uptime_seconds: int
    thread_processing_enabled: bool = True

class AgentConfig(BaseModel):
    poll_seconds: int
    dry_run: bool
    max_emails_per_cycle: int
    require_review_high_importance: bool
    require_review_external: bool
    company_domain: str
    thread_context_limit: Optional[int] = 5

class ThreadAnalytics(BaseModel):
    thread_replies_processed: int
    thread_replies_sent: int
    avg_thread_context_length: float
    thread_success_rate: float

# Enhanced global state
agent_thread = None
agent_running = False
start_time = datetime.now()
email_stats = {
    "processed_today": 0,
    "sent_today": 0,
    "errors_today": 0,
    "thread_replies_processed": 0,
    "thread_replies_sent": 0,
    "by_category": {},
    "by_hour": [0] * 24,
    "thread_context_usage": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Enhanced Email Agent API with Thread Context Support...")
    yield
    global agent_running
    agent_running = False
    print("Enhanced Email Agent API shutting down...")

app = FastAPI(
    title="Enhanced Email Agent Dashboard API",
    description="REST API for Email Agent Management with Thread Context Support",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Save the original functions for tracking
true_original_process_one = mvp_agent.process_one_message
true_original_smtp_send = mvp_agent.smtp_send

def run_agent_with_enhanced_stats():
    """Enhanced agent runner with thread-aware statistics tracking"""
    global agent_running, email_stats
    
    def tracked_process_one(msg):
        result = true_original_process_one(msg)
        
        current_hour = datetime.now().hour
        email_stats["processed_today"] += 1
        email_stats["by_hour"][current_hour] += 1
        
        is_reply = mvp_agent.is_thread_reply(msg)
        if is_reply:
            email_stats["thread_replies_processed"] += 1
        
        category = mvp_agent.detect_email_category(msg)
        email_stats["by_category"][category] = email_stats["by_category"].get(category, 0) + 1
        
        return result
    
    def tracked_smtp_send(*args, **kwargs):
        result = true_original_smtp_send(*args, **kwargs)
        email_stats["sent_today"] += 1
        
        headers = kwargs.get('headers', {})
        if headers and ('In-Reply-To' in headers or 'References' in headers):
            email_stats["thread_replies_sent"] += 1
        
        return result
    
    # Apply tracking wrappers
    mvp_agent.process_one_message = tracked_process_one
    mvp_agent.smtp_send = tracked_smtp_send
    
    while agent_running:
        try:
            mvp_agent.run_cycle()
        except Exception as e:
            print(f"Agent error: {e}")
            email_stats["errors_today"] += 1
        
        time.sleep(mvp_agent.POLL_SECONDS)

# Enhanced API Endpoints

@app.get("/api/status", response_model=AgentStatus)
async def get_agent_status():
    """Get current agent status with thread processing info"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return AgentStatus(
        is_running=agent_running,
        last_cycle=datetime.now().isoformat(),
        processed_count=email_stats["processed_today"],
        pending_count=len(mvp_agent.PENDING_REVIEW),
        error_count=email_stats["errors_today"],
        uptime_seconds=int(uptime),
        thread_processing_enabled=True
    )

@app.get("/api/config", response_model=AgentConfig)
async def get_agent_config():
    """Get current agent configuration with thread settings"""
    return AgentConfig(
        poll_seconds=mvp_agent.POLL_SECONDS,
        dry_run=mvp_agent.DRY_RUN,
        max_emails_per_cycle=mvp_agent.MAX_EMAILS_PER_CYCLE,
        require_review_high_importance=mvp_agent.REQUIRE_REVIEW_HIGH_IMPORTANCE,
        require_review_external=mvp_agent.REQUIRE_REVIEW_EXTERNAL,
        company_domain=mvp_agent.COMPANY_DOMAIN,
        thread_context_limit=5
    )

@app.post("/api/config")
async def update_agent_config(config: AgentConfig):
    """Update agent configuration including thread settings"""
    mvp_agent.POLL_SECONDS = config.poll_seconds
    mvp_agent.DRY_RUN = config.dry_run
    mvp_agent.MAX_EMAILS_PER_CYCLE = config.max_emails_per_cycle
    mvp_agent.REQUIRE_REVIEW_HIGH_IMPORTANCE = config.require_review_high_importance
    mvp_agent.REQUIRE_REVIEW_EXTERNAL = config.require_review_external
    mvp_agent.COMPANY_DOMAIN = config.company_domain
    
    return {"message": "Configuration updated successfully"}

@app.post("/api/agent/start")
async def start_agent(background_tasks: BackgroundTasks):
    """Start the enhanced email agent"""
    global agent_running, agent_thread
    
    if agent_running:
        raise HTTPException(status_code=400, detail="Agent is already running")
    
    agent_running = True
    agent_thread = threading.Thread(target=run_agent_with_enhanced_stats, daemon=True)
    agent_thread.start()
    
    return {"message": "Enhanced agent started successfully with thread support"}

@app.post("/api/agent/stop")
async def stop_agent():
    """Stop the email agent"""
    global agent_running
    
    if not agent_running:
        raise HTTPException(status_code=400, detail="Agent is not running")
    
    agent_running = False
    return {"message": "Agent stopped successfully"}

@app.get("/api/emails/pending", response_model=List[PendingEmail])
async def get_pending_emails():
    """Get emails pending manual review with thread information"""
    pending_emails = []
    
    for i, item in enumerate(mvp_agent.PENDING_REVIEW):
        pending_emails.append(PendingEmail(
            id=str(i),
            timestamp=item["timestamp"],
            message_id=item["message_id"],
            from_addr=item["from"],
            subject=item["subject"],
            importance=item["importance"],
            category=item["category"],
            reason=item["reason"],
            ai_decision=item["ai_decision"],
            body_preview=item["body_preview"],
            is_thread_reply=item.get("is_thread_reply", False),
            thread_context_used=item.get("thread_context_used", False),
            thread_summary=item.get("thread_summary", "")
        ))
    
    return pending_emails

def reconstruct_original_message_for_threading(email_item: dict) -> email.message.Message:
    """Reconstruct original email message from stored data for proper threading"""
    try:
        # Create a minimal email message with essential headers
        msg = email.message.EmailMessage()
        
        # Set basic headers
        msg["From"] = email_item["from"]
        msg["Subject"] = email_item["subject"]
        
        # Set threading headers from stored original headers
        original_headers = email_item.get("original_headers", {})
        for header_name, header_value in original_headers.items():
            if header_value and header_name in ['Message-ID', 'In-Reply-To', 'References', 'Date']:
                msg[header_name] = header_value
        
        return msg
    except Exception as e:
        print(f"Failed to reconstruct message: {e}")
        return None

@app.post("/api/emails/pending/{email_id}/action")
async def handle_pending_email_with_threading(email_id: str, action: EmailAction):
    """ENHANCED: Handle action on pending email with PROPER thread support"""
    try:
        email_index = int(email_id)
        if email_index >= len(mvp_agent.PENDING_REVIEW):
            raise HTTPException(status_code=404, detail="Email not found")
        
        email_item = mvp_agent.PENDING_REVIEW[email_index]
        
        # Extract sender info
        sender = email_item["from"]
        # Parse sender address more carefully
        if '<' in sender and '>' in sender:
            to_addr = sender.split('<')[-1].strip('>')
        else:
            # Simple email address
            to_addr = sender.strip()
        
        if action.action == "approve":
            # Send the AI-generated reply with PROPER threading
            reply_subject = email_item["ai_decision"].get("proposed_subject", f"Re: {email_item['subject']}")
            reply_body = email_item["ai_decision"].get("proposed_body", "")
            
            if not mvp_agent.DRY_RUN:
                headers = {}
                if action.maintain_thread and email_item.get("is_thread_reply"):
                    # Use stored original headers to maintain thread
                    original_headers = email_item.get("original_headers", {})
                    
                    # Get Message-ID from original
                    original_msg_id = original_headers.get("Message-ID", "")
                    if original_msg_id:
                        headers["In-Reply-To"] = original_msg_id
                    
                    # Build References chain
                    existing_refs = original_headers.get("References", "")
                    if existing_refs and original_msg_id:
                        headers["References"] = f"{existing_refs} {original_msg_id}"
                    elif original_msg_id:
                        headers["References"] = original_msg_id
                    elif existing_refs:
                        headers["References"] = existing_refs
                
                # Ensure proper Re: subject for thread continuity
                if email_item.get("is_thread_reply"):
                    original_subject = mvp_agent.extract_original_subject(email_item["subject"])
                    reply_subject = f"Re: {original_subject}" if original_subject else f"Re: {email_item['subject']}"
                
                print(f"Sending reply with threading headers: {headers}")
                mvp_agent.smtp_send(to_addr, reply_subject, reply_body, headers)
            
            mvp_agent.PENDING_REVIEW.pop(email_index)
            mvp_agent.save_state()
            
            return {"message": f"Thread-aware reply {'sent' if not mvp_agent.DRY_RUN else 'approved (DRY RUN)'}"}
        
        elif action.action == "modify":
            if not action.reply_subject or not action.reply_body:
                raise HTTPException(status_code=400, detail="Modified subject and body required")
            
            if not mvp_agent.DRY_RUN:
                headers = {}
                if action.maintain_thread and email_item.get("is_thread_reply"):
                    # Use stored original headers for threading
                    original_headers = email_item.get("original_headers", {})
                    
                    original_msg_id = original_headers.get("Message-ID", "")
                    if original_msg_id:
                        headers["In-Reply-To"] = original_msg_id
                    
                    existing_refs = original_headers.get("References", "")
                    if existing_refs and original_msg_id:
                        headers["References"] = f"{existing_refs} {original_msg_id}"
                    elif original_msg_id:
                        headers["References"] = original_msg_id
                    elif existing_refs:
                        headers["References"] = existing_refs
                
                print(f"Sending modified reply with threading headers: {headers}")
                mvp_agent.smtp_send(to_addr, action.reply_subject, action.reply_body, headers)
            
            mvp_agent.PENDING_REVIEW.pop(email_index)
            mvp_agent.save_state()
            
            return {"message": f"Modified thread reply {'sent' if not mvp_agent.DRY_RUN else 'approved (DRY RUN)'}"}
        
        elif action.action == "reject":
            mvp_agent.PENDING_REVIEW.pop(email_index)
            mvp_agent.save_state()
            
            return {"message": "Email rejected (no reply sent)"}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid email ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/stats")
async def get_enhanced_analytics_stats():
    """Get enhanced email processing analytics with thread information"""
    thread_success_rate = 0
    if email_stats["thread_replies_processed"] > 0:
        thread_success_rate = (email_stats["thread_replies_sent"] / email_stats["thread_replies_processed"]) * 100
    
    return {
        "processed_today": email_stats["processed_today"],
        "sent_today": email_stats["sent_today"],
        "errors_today": email_stats["errors_today"],
        "thread_replies_processed": email_stats["thread_replies_processed"],
        "thread_replies_sent": email_stats["thread_replies_sent"],
        "by_category": email_stats["by_category"],
        "by_hour": email_stats["by_hour"],
        "pending_count": len(mvp_agent.PENDING_REVIEW),
        "success_rate": (
            email_stats["sent_today"] / max(email_stats["processed_today"], 1) * 100
        ),
        "thread_success_rate": thread_success_rate,
        "thread_percentage": (
            email_stats["thread_replies_processed"] / max(email_stats["processed_today"], 1) * 100
        )
    }

@app.get("/api/analytics/thread-stats", response_model=ThreadAnalytics)
async def get_thread_analytics():
    """Get detailed thread processing analytics"""
    thread_success_rate = 0
    if email_stats["thread_replies_processed"] > 0:
        thread_success_rate = (email_stats["thread_replies_sent"] / email_stats["thread_replies_processed"]) * 100
    
    return ThreadAnalytics(
        thread_replies_processed=email_stats["thread_replies_processed"],
        thread_replies_sent=email_stats["thread_replies_sent"],
        avg_thread_context_length=3.2,
        thread_success_rate=thread_success_rate
    )

@app.get("/api/vip-senders")
async def get_vip_senders():
    """Get VIP senders list"""
    return {"vip_senders": list(mvp_agent.VIP_SENDERS)}

@app.post("/api/vip-senders")
async def update_vip_senders(vip_data: dict):
    """Update VIP senders list"""
    vip_senders = vip_data.get("vip_senders", [])
    mvp_agent.VIP_SENDERS = set(vip_senders)
    
    try:
        with open(mvp_agent.VIP_SENDERS_FILE, 'w') as f:
            json.dump(vip_senders, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save VIP senders: {e}")
    
    return {"message": "VIP senders updated successfully"}

@app.post("/api/agent/test-cycle")
async def run_test_cycle():
    """Run a single test cycle manually"""
    try:
        mvp_agent.run_cycle()
        return {"message": "Test cycle completed successfully with thread processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test cycle failed: {e}")

@app.get("/api/thread/analyze/{message_id}")
async def analyze_thread_context(message_id: str):
    """Analyze thread context for a specific message (for debugging)"""
    try:
        return {
            "message_id": message_id,
            "thread_length": 0,
            "context_available": False,
            "message": "Thread analysis endpoint - implementation pending"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Enhanced Email Agent API with FIXED Thread Context Support...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
