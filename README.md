---
title: Customer Support Triage
emoji: ??
colorFrom: purple
colorTo: teal
sdk: docker
app_port: 7860
tags:
  - openenv
license: mit
---

# 🎫 Customer Support Triage — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

## Overview & Motivation

Customer support triage is a task performed **millions of times daily** by human agents worldwide:  
reading incoming tickets, classifying urgency, routing to the right team, drafting responses,  
escalating critical issues, and closing resolved cases.

This OpenEnv environment simulates that full workflow, giving RL and language-model agents a  
**grounded, real-world benchmark** with deterministic, reproducible scoring. Unlike toy grid  
worlds or synthetic benchmarks, an agent trained or evaluated here is learning skills directly  
transferable to production support systems.

---

## Observation Space

Each step the agent receives a structured `Observation` object:

| Field | Type | Description |
|---|---|---|
| `task_name` | `str` | Active task identifier |
| `task_description` | `str` | Natural-language task instructions |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Maximum allowed steps |
| `tickets` | `List[Ticket]` | All tickets in the inbox (id, subject, body, customer info) |
| `pending_ticket_ids` | `List[str]` | Tickets not yet actioned |
| `completed_ticket_ids` | `List[str]` | Tickets already actioned |
| `actions_history` | `List[dict]` | Last 10 actions taken (for context) |
| `score_so_far` | `float` | Running episode score [0.0, 1.0] |
| `done` | `bool` | Whether the episode has ended |

---

## Action Space

All actions target a specific ticket and are typed via Pydantic:

```json
{
  "action_type": "classify | respond | escalate | archive | close | skip",
  "ticket_id": "T001",
  "classification": { "department": "billing | technical | general | returns",
                      "urgency": "low | medium | high" },
  "response": { "response_text": "Dear customer, ..." },
  "escalation": { "reason": "P0 production outage..." }
}
```

| Action | When to use |
|---|---|
| `classify` | Assign department + urgency to a ticket |
| `respond` | Write a professional reply to the customer |
| `escalate` | Send to human/specialist team (P0 incidents, legal/GDPR) |
| `archive` | Mark as spam or irrelevant |
| `close` | Mark as resolved (positive feedback, already-resolved) |
| `skip` | Defer a ticket (small penalty applied) |

---

## Tasks

### 🟢 Easy — Ticket Classification (`ticket_classification`)

**Objective:** Classify 5 customer support tickets by department and urgency.  
**Max steps:** 10  
**Scoring:** Each ticket worth 0.2 of episode score. Department and urgency each contribute 0.1.  
**Expected score for a capable agent:** ~0.90+

Tickets cover a double billing charge, a password reset, a defective product return,  
an app crash, and an office hours inquiry — clear-cut cases designed to test  
basic domain understanding.

---

### 🟡 Medium — Ticket Response (`ticket_response`)

**Objective:** Draft professional, empathetic responses to 3 tickets.  
**Max steps:** 9  
**Scoring per ticket (averaged over 3):**
- Keyword coverage (40%) — domain-specific terms present
- Response length ≥ 80 words (20%)
- Acknowledgment phrases present (20%)
- Next-steps / action language present (20%)

Tickets cover a billing dispute after cancellation, a technical error before a deadline,  
and a damaged delivery — each requiring empathy + concrete resolution path.

**Expected score:** ~0.65 (frontier models), ~0.45 (smaller models)

---

### 🔴 Hard — Full Inbox Triage (`inbox_triage`)

**Objective:** Manage 8 mixed tickets across all action types.  
**Max steps:** 24  
**Scoring breakdown:**
- Classification accuracy: 40%
- Correct routing decision: 40%
- Response quality (for `respond` tickets): 20%

The inbox includes a P0 production outage (must escalate), a phishing spam email (must archive),  
a GDPR data-deletion request (must escalate to legal), a 2FA lockout, a billing upgrade request,  
a wrong-item shipment, positive feedback (must close), and a PDF download bug.  
Agents that fail to recognise legal/P0 urgency or misroute spam will be penalised significantly.

**Expected score:** ~0.55 (frontier models), ~0.35 (smaller models)

---

## Reward Function

Rewards are issued **at every step**, not just at episode end:

| Scenario | Reward |
|---|---|
| Correct classify (dept + urgency) | `+0.40 × accuracy × 1/N_tickets` per step |
| Correct routing action | `+0.40 × 1/N_tickets` per step |
| Good response (respond action) | `+0.20 × quality × 1/N_tickets` bonus |
| Skip action | `-0.01` (small penalty) |
| Re-actioning a completed ticket | `0.0` (no reward) |
| Exceeding max_steps | Final score multiplied by `0.90` |

Episode score (0.0–1.0) is reported in `info["episode_score"]` when `done=True`.

---

## Baseline Performance

Scores produced by running `inference.py` with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Difficulty | Baseline Score |
|---|---|---|
| `ticket_classification` | Easy | **0.72** |
| `ticket_response` | Medium | **0.61** |
| `inbox_triage` | Hard | **0.54** |
| **Average** | — | **0.62** |

---

## Setup & Usage

### Local Python

```bash
git clone <repo-url>
cd customer-support-env
pip install -r requirements.txt

# Run the HTTP server
python server.py
# → Listening on http://localhost:7860

# Run baseline inference (set your key first)
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  customer-support-env
```

### Python API (programmatic)

```python
from customer_support_env import CustomerSupportEnv, Action, ActionType, ClassificationPayload, Department, Urgency

env = CustomerSupportEnv(task_name="ticket_classification")
obs = env.reset()

action = Action(
    action_type=ActionType.CLASSIFY,
    ticket_id="T001",
    classification=ClassificationPayload(
        department=Department.BILLING,
        urgency=Urgency.HIGH,
    )
)
result = env.step(action)
print(result.reward)   # Reward(value=1.0, reason="department='billing' ✓ | urgency='high' ✓", ...)
```

### HTTP API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "ticket_classification"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "classify",
      "ticket_id": "T001",
      "classification": {"department": "billing", "urgency": "high"}
    }
  }'

# State
curl http://localhost:7860/state
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | Hugging Face / API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `TASK_NAME` | No | *(all tasks)* | Run a single task |
| `PORT` | No | `7860` | Server port |

---

## Project Structure

```
customer-support-env/
├── customer_support_env/
│   ├── __init__.py          # Package exports
│   ├── env.py               # CustomerSupportEnv — OpenEnv interface
│   ├── models.py            # Pydantic models (Observation, Action, Reward, ...)
│   ├── data.py              # Task data + ground truth
│   └── tasks/
│       ├── __init__.py      # Task registry
│       ├── easy.py          # Ticket classification grader
│       ├── medium.py        # Response quality grader
│       └── hard.py          # Full inbox triage grader
├── server.py                # FastAPI HTTP server (HF Spaces)
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| Typed `Observation`, `Action`, `Reward` Pydantic models | ✅ |
| `reset()` → initial `Observation` | ✅ |
| `step(action)` → `(observation, reward, done, info)` | ✅ |
| `state()` → full state snapshot | ✅ |
| `openenv.yaml` metadata | ✅ |
| ≥ 3 tasks with programmatic graders | ✅ (3 tasks) |
| Reward in [0.0, 1.0] range | ✅ |
| Incremental reward signal (not just terminal) | ✅ |
| Penalty for undesirable behaviour (skip/overtime) | ✅ |
| Baseline `inference.py` using OpenAI client | ✅ |
| Reads credentials from `HF_TOKEN` env variable | ✅ |
| Dockerfile + `docker build/run` | ✅ |
| HF Spaces deployable (port 7860) | ✅ |

