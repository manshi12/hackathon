# Email Agent Dashboard Setup Guide

## 📁 Project Structure
```
email-agent-dashboard/
├── mvp_agent.py          # Your existing email agent
├── api_server.py         # FastAPI backend (new)
├── dashboard.html        # React dashboard (new)
├── requirements.txt      # Python dependencies (new)
├── .env                  # Environment variables
├── vip_senders.json      # VIP senders list
└── README.md            # This setup guide
```

## 🔧 Step 1: Install Python Dependencies

Create a `requirements.txt` file:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
groq==0.4.1
pydantic==2.5.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Step 2: Update Your Email Agent Code

1. **Copy your existing `mvp_agent.py`** - The API server imports this directly
2. **Make sure your agent has the enhanced Outlook features** from the previous code I provided

## 🔧 Step 3: Set Up Environment Variables

Update your `.env` file with the dashboard-specific settings:

```env
# === Email Configuration (Gmail/Outlook) ===
EMAIL_ADDRESS=your.email@company.com
EMAIL_USER=your.email@company.com
EMAIL_PASS=your_app_password
IMAP_HOST=imap.gmail.com  # or outlook.office365.com
IMAP_PORT=993
SMTP_HOST=smtp.gmail.com  # or smtp.office365.com
SMTP_PORT=587
SMTP_STARTTLS=1

# === AI Configuration ===
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192

# === Dashboard Configuration ===
COMPANY_DOMAIN=yourcompany.com
REQUIRE_REVIEW_HIGH_IMPORTANCE=1
REQUIRE_REVIEW_EXTERNAL=1
AUTO_REPLY_CATEGORIES=newsletter,notification,automated
VIP_SENDERS_FILE=vip_senders.json

# === Agent Behavior ===
POLL_SECONDS=60
DRY_RUN=1
MAX_EMAILS_PER_CYCLE=10
```

## 🔧 Step 4: Create VIP Senders File

Create `vip_senders.json`:
```json
[
  "ceo@yourcompany.com",
  "manager@yourcompany.com", 
  "important.client@clientcompany.com"
]
```

## 🚀 Step 5: Start the Application

### Terminal 1: Start the API Server
```bash
python api_server.py
```
The API will be available at `http://localhost:8000`

You can test the API by visiting `http://localhost:8000/docs` for the interactive documentation.

### Terminal 2: Serve the Dashboard
You have several options to serve the HTML dashboard:

**Option A: Simple HTTP Server (Python)**
```bash
python -m http.server 3000
```

**Option B: Node.js serve**
```bash
npx serve -s . -l 3000
```

**Option C: Open directly in browser**
Just open `dashboard.html` in your browser (some features may not work due to CORS)

The dashboard will be available at `http://localhost:3000`

## 🔍 Step 6: First-Time Setup

1. **Open the dashboard** at `http://localhost:3000`
2. **Configure settings** in the Agent Controls section
3. **Set DRY_RUN=1** initially to test without sending emails
4. **Click "Start Agent"** to begin monitoring
5. **Use "Test Cycle"** to manually process emails
6. **Review pending emails** in the dashboard
7. **Set DRY_RUN=0** when ready for production

## 📊 Dashboard Features

### Real-Time Monitoring
- ✅ Agent status (running/stopped)
- ✅ Live statistics (processed, sent, pending)
- ✅ 24-hour activity chart
- ✅ Auto-refresh every 5 seconds

### Email Management
- ✅ View emails pending review
- ✅ Approve/reject/modify AI-generated replies
- ✅ Real-time preview of email content
- ✅ Inline editing for reply modifications

### Agent Controls
- ✅ Start/stop agent remotely
- ✅ Configure polling intervals
- ✅ Toggle dry-run mode
- ✅ Set company domain and VIP rules
- ✅ Run manual test cycles

### Analytics
- ✅ Success rate tracking
- ✅ Category breakdown
- ✅ Hourly processing patterns
- ✅ Error monitoring

## 🔧 Configuration Options

### High-Level Settings
- **POLL_SECONDS**: How often to check for new emails
- **DRY_RUN**: Test mode (emails processed but not sent)
- **MAX_EMAILS_PER_CYCLE**: Batch size for processing

### Review Requirements
- **REQUIRE_REVIEW_HIGH_IMPORTANCE**: Force review for high-importance emails
- **REQUIRE_REVIEW_EXTERNAL**: Force review for external senders
- **VIP_SENDERS**: Always require review for these addresses

## 🛠️ Troubleshooting

### Common Issues

**1. API Connection Failed**
- Ensure `api_server.py` is running on port 8000
- Check that CORS is configured correctly
- Verify no firewall blocking the connection

**2. Email Authentication Errors**
- Use App Passwords for Gmail/Outlook
- Enable "Allow less secure apps" if required
- Check IMAP/SMTP settings for your provider

**3. Groq API Errors**
- Verify your GROQ_API_KEY is correct
- Check your API usage limits
- Ensure you have sufficient credits

**4. Dashboard Not Loading**
- Check browser console for JavaScript errors
- Ensure all CDN resources are loading
- Try refreshing the page

### Debug Mode
Add this to your `.env` for detailed logging:
```env
LOG_LEVEL=DEBUG
```

## 🔄 Development Workflow

### Testing New Features
1. Set `DRY_RUN=1`
2. Use "Test Cycle" to process emails
3. Check results in dashboard
4. Modify configuration as needed
5. Set `DRY_RUN=0` for production

### Monitoring Production
1. Keep dashboard open for monitoring
2. Check pending emails regularly  
3. Review analytics for performance
4. Adjust settings based on usage patterns

## 🔐 Security Considerations

- **Never commit `.env` files** to version control
- **Use App Passwords** instead of main email passwords
- **Review VIP settings** regularly
- **Monitor for suspicious email patterns**
- **Keep API access restricted** to localhost/internal networks

## 📈 Scaling Up

### For High-Volume Environments
- Increase `MAX_EMAILS_PER_CYCLE`
- Decrease `POLL_SECONDS` for faster processing
- Consider multiple agent instances
- Implement database storage for persistence
- Add email archiving and retention policies

### Enterprise Features (Future)
- User authentication and roles
- Multi-tenant support
- Advanced analytics and reporting
- Integration with calendar systems
- Slack/Teams notifications
- Custom AI model training

## 🆘 Support

If you encounter issues:
1. Check the console logs in both terminals
2. Verify your environment variables
3. Test email credentials manually
4. Check the API documentation at `/docs`
5. Review the dashboard browser console for errors

The dashboard is designed to work with both Gmail and Outlook without code changes - just update the `.env` file with the appropriate settings for your email provider.
