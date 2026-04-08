"""
Static task data for all three difficulty levels.
Each task defines tickets + ground-truth answers used by graders.
"""
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# EASY TASK  —  Classify 5 tickets by department + urgency
# ──────────────────────────────────────────────────────────────────────────────

EASY_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "T001",
        "subject": "My invoice is wrong – I was charged twice!",
        "body": (
            "Hi, I checked my bank statement and I see TWO charges of $49.99 "
            "from your company on October 3rd. I only made one purchase. "
            "Please refund the duplicate charge immediately. This is very urgent."
        ),
        "customer_name": "Alice Chen",
        "customer_email": "alice.chen@email.com",
        "created_at": "2024-10-03T09:15:00Z",
        "metadata": {},
    },
    {
        "id": "T002",
        "subject": "How do I reset my password?",
        "body": (
            "Hello, I forgot my password and the reset email never arrived. "
            "I've checked my spam folder too. Can you help me get back into my account?"
        ),
        "customer_name": "Bob Smith",
        "customer_email": "bob.smith@email.com",
        "created_at": "2024-10-03T10:00:00Z",
        "metadata": {},
    },
    {
        "id": "T003",
        "subject": "I want to return a defective product",
        "body": (
            "I received order #ORD-8821 last week and the item stopped working after "
            "two days. The screen flickers and then goes black. I'd like to return it "
            "and get a replacement or full refund."
        ),
        "customer_name": "Carol Davis",
        "customer_email": "carol.davis@email.com",
        "created_at": "2024-10-03T11:30:00Z",
        "metadata": {},
    },
    {
        "id": "T004",
        "subject": "App keeps crashing on iOS 17",
        "body": (
            "Your app crashes every single time I try to open the 'Reports' section "
            "on my iPhone 14 running iOS 17.0.3. I've reinstalled twice. "
            "This is blocking my workflow – please fix ASAP."
        ),
        "customer_name": "David Lee",
        "customer_email": "david.lee@email.com",
        "created_at": "2024-10-03T12:00:00Z",
        "metadata": {},
    },
    {
        "id": "T005",
        "subject": "What are your office hours?",
        "body": (
            "Hi, just a quick question — what are your customer support hours "
            "and do you have weekend coverage? Thanks!"
        ),
        "customer_name": "Emma Wilson",
        "customer_email": "emma.wilson@email.com",
        "created_at": "2024-10-03T13:00:00Z",
        "metadata": {},
    },
]

EASY_GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    "T001": {"department": "billing",   "urgency": "high"},
    "T002": {"department": "technical", "urgency": "medium"},
    "T003": {"department": "returns",   "urgency": "medium"},
    "T004": {"department": "technical", "urgency": "high"},
    "T005": {"department": "general",   "urgency": "low"},
}

# ──────────────────────────────────────────────────────────────────────────────
# MEDIUM TASK  —  Respond appropriately to 3 tickets
# ──────────────────────────────────────────────────────────────────────────────

MEDIUM_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "M001",
        "subject": "Subscription cancelled but still being charged",
        "body": (
            "I cancelled my Premium subscription on September 15th and received "
            "a confirmation email. Despite this, I was charged $29.99 on October 1st. "
            "I need this charge reversed immediately and confirmation that no further "
            "charges will occur. My account email is frank.jones@email.com."
        ),
        "customer_name": "Frank Jones",
        "customer_email": "frank.jones@email.com",
        "created_at": "2024-10-02T08:00:00Z",
        "metadata": {"subscription_id": "SUB-4421", "plan": "Premium"},
    },
    {
        "id": "M002",
        "subject": "Data export not working – urgent project deadline",
        "body": (
            "I need to export my project data as a CSV for a client presentation "
            "tomorrow morning. Every time I click 'Export', I get error code 500. "
            "I've tried Chrome, Firefox, and Edge. Please help me resolve this today."
        ),
        "customer_name": "Grace Kim",
        "customer_email": "grace.kim@email.com",
        "created_at": "2024-10-02T14:30:00Z",
        "metadata": {"error_code": "500", "browsers_tried": ["Chrome", "Firefox", "Edge"]},
    },
    {
        "id": "M003",
        "subject": "Package arrived damaged",
        "body": (
            "My order (ORD-9934) arrived yesterday with the box crushed and the "
            "product inside cracked. I have photos. I paid $89.99 and expect either "
            "a replacement shipped expedited or a full refund. Please let me know "
            "the next steps."
        ),
        "customer_name": "Henry Park",
        "customer_email": "henry.park@email.com",
        "created_at": "2024-10-02T16:00:00Z",
        "metadata": {"order_id": "ORD-9934", "amount": 89.99},
    },
]

# Keywords/criteria each response MUST include to earn credit
MEDIUM_RESPONSE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "M001": {
        "required_keywords": ["refund", "apologize", "cancel", "charge"],
        "must_acknowledge_issue": True,
        "must_provide_next_steps": True,
        "min_length": 80,
        "tone": "apologetic",
        "description": "Billing dispute — acknowledge, apologize, confirm refund process",
    },
    "M002": {
        "required_keywords": ["export", "error", "workaround", "team"],
        "must_acknowledge_issue": True,
        "must_provide_next_steps": True,
        "min_length": 80,
        "tone": "helpful_urgent",
        "description": "Technical issue with deadline — acknowledge urgency, provide workaround or escalation",
    },
    "M003": {
        "required_keywords": ["replacement", "refund", "apologize", "photo", "damage"],
        "must_acknowledge_issue": True,
        "must_provide_next_steps": True,
        "min_length": 80,
        "tone": "apologetic",
        "description": "Damaged product — acknowledge, apologize, offer replacement or refund",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# HARD TASK  —  Full inbox: classify + respond + route 8 mixed tickets
# ──────────────────────────────────────────────────────────────────────────────

HARD_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "H001",
        "subject": "URGENT: Production API returning 403 for all requests",
        "body": (
            "Our entire production system is down. All API calls return 403 Forbidden "
            "since 2:00 AM UTC. Thousands of users are affected. Our API key is valid — "
            "we regenerated it twice. This is a P0 incident for us."
        ),
        "customer_name": "Iris Thompson",
        "customer_email": "iris@acmecorp.com",
        "created_at": "2024-10-04T02:15:00Z",
        "metadata": {"plan": "Enterprise", "mrr": 5000},
    },
    {
        "id": "H002",
        "subject": "Win a $1000 gift card – click here!",
        "body": (
            "Congratulations! You've been selected to win a $1000 Amazon gift card. "
            "Click the link below to claim your prize: http://suspicious-site.xyz/claim"
        ),
        "customer_name": "Unknown Sender",
        "customer_email": "noreply@suspicious-site.xyz",
        "created_at": "2024-10-04T03:00:00Z",
        "metadata": {},
    },
    {
        "id": "H003",
        "subject": "Request to upgrade to annual plan",
        "body": (
            "Hi, I'm currently on the monthly Pro plan and want to switch to annual "
            "billing to save money. Can you set this up and let me know if there's "
            "a pro-rated credit for the remaining days this month?"
        ),
        "customer_name": "James Nguyen",
        "customer_email": "james.nguyen@email.com",
        "created_at": "2024-10-04T09:00:00Z",
        "metadata": {"current_plan": "Pro Monthly", "customer_since": "2023-01"},
    },
    {
        "id": "H004",
        "subject": "2FA codes not arriving via SMS",
        "body": (
            "I set up two-factor authentication last week and now I can't log in "
            "because the SMS codes never arrive. I've verified my phone number is correct. "
            "I'm locked out of my account completely."
        ),
        "customer_name": "Kate Morris",
        "customer_email": "kate.morris@email.com",
        "created_at": "2024-10-04T09:30:00Z",
        "metadata": {},
    },
    {
        "id": "H005",
        "subject": "Legal complaint – GDPR data deletion request",
        "body": (
            "Under Article 17 of the GDPR, I formally request immediate deletion of all "
            "personal data you hold for me. My account email is liam.brown@email.com. "
            "Please confirm deletion in writing within 30 days as required by law. "
            "Failure to comply will result in a complaint to the ICO."
        ),
        "customer_name": "Liam Brown",
        "customer_email": "liam.brown@email.com",
        "created_at": "2024-10-04T10:00:00Z",
        "metadata": {"legal": True},
    },
    {
        "id": "H006",
        "subject": "Wrong item shipped — order ORD-7751",
        "body": (
            "I ordered the Blue Widget (SKU: BW-100) but received a Red Widget (SKU: RW-200). "
            "The packing slip shows the right item but the box contains the wrong one. "
            "Please arrange collection and send the correct item."
        ),
        "customer_name": "Mia Clark",
        "customer_email": "mia.clark@email.com",
        "created_at": "2024-10-04T10:30:00Z",
        "metadata": {"order_id": "ORD-7751"},
    },
    {
        "id": "H007",
        "subject": "Feedback: great product, minor UI suggestion",
        "body": (
            "Just wanted to say your product has saved me hours every week — thank you! "
            "One small suggestion: it would be great if the dashboard could remember "
            "my last used filter. Keep up the great work!"
        ),
        "customer_name": "Noah Adams",
        "customer_email": "noah.adams@email.com",
        "created_at": "2024-10-04T11:00:00Z",
        "metadata": {"sentiment": "positive"},
    },
    {
        "id": "H008",
        "subject": "Can't download my receipts",
        "body": (
            "The billing portal shows my invoices but when I click 'Download PDF', "
            "nothing happens. I need these for my company expense report due Friday. "
            "Using Chrome on Windows 11."
        ),
        "customer_name": "Olivia Scott",
        "customer_email": "olivia.scott@email.com",
        "created_at": "2024-10-04T11:30:00Z",
        "metadata": {},
    },
]

HARD_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "H001": {
        "department": "technical",
        "urgency": "high",
        "correct_action": "escalate",   # P0 production outage → must escalate
        "reason_keywords": ["production", "outage", "p0", "urgent", "enterprise"],
    },
    "H002": {
        "department": "general",
        "urgency": "low",
        "correct_action": "archive",    # Spam → archive
        "reason_keywords": ["spam", "phishing", "suspicious", "scam"],
    },
    "H003": {
        "department": "billing",
        "urgency": "low",
        "correct_action": "respond",
        "response_keywords": ["annual", "upgrade", "pro-rat", "credit", "billing"],
    },
    "H004": {
        "department": "technical",
        "urgency": "high",
        "correct_action": "respond",    # Locked out → respond with fix or escalate
        "response_keywords": ["2fa", "sms", "locked", "alternative", "backup"],
    },
    "H005": {
        "department": "general",
        "urgency": "high",
        "correct_action": "escalate",   # Legal/GDPR → must escalate to legal team
        "reason_keywords": ["gdpr", "legal", "compliance", "data", "deletion"],
    },
    "H006": {
        "department": "returns",
        "urgency": "medium",
        "correct_action": "respond",
        "response_keywords": ["wrong", "collect", "replacement", "correct", "apologize"],
    },
    "H007": {
        "department": "general",
        "urgency": "low",
        "correct_action": "close",      # Positive feedback → acknowledge and close
        "response_keywords": ["thank", "feedback", "forward", "team"],
    },
    "H008": {
        "department": "technical",
        "urgency": "medium",
        "correct_action": "respond",
        "response_keywords": ["download", "pdf", "browser", "workaround", "investigate"],
    },
}
