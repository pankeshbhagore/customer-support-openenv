---
title: Customer Support Triage
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
openenv
license: mit
---
🎫 Customer Support Triage — OpenEnv v2.0
![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-7c6fe0)
![Tests](https://img.shields.io/badge/tests-52%20passed-22c55e)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![openenv-core](https://img.shields.io/badge/openenv--core-0.2.3-orange)
Overview & Motivation
Customer support triage is performed millions of times daily by human agents worldwide. This OpenEnv environment simulates that complete workflow across four tasks of increasing difficulty — from simple ticket classification to expert-level churn prevention requiring genuine business reasoning.
The environment properly extends `openenv-core`'s `Environment` base class with typed Pydantic models for Observation, Action, and State. It supports concurrent sessions and embeds reward directly in observations per the openenv-core convention.
Why this environment fills a real gap:
Unlike toy benchmarks, agents learn skills directly transferable to production support systems
The churn prevention task requires reading between the lines — "How do I export my data?" is a hidden churn signal, not a how-to question
Multi-signal grading (classification + routing + content quality + churn detection) rewards nuanced reasoning
Deterministic, reproducible graders make evaluation fair and consistent
---
Environment Architecture
```
CustomerSupportTriageEnv
  ├── extends: openenv-core Environment[TriageAction, TriageObservation, TriageState]
  ├── SUPPORTS_CONCURRENT_SESSIONS = True
  ├── reset(seed, episode_id) → TriageObservation
  ├── step(TriageAction) → TriageObservation  (reward embedded in obs)
  └── state → TriageState  (full internal snapshot)
```
All models extend openenv-core base classes:
`TriageObservation` extends `openenv.core.env_server.Observation`
`TriageAction` extends `openenv.core.env_server.Action`
`TriageState` extends `openenv.core.env_server.State` (has `episode_id`, `step_count`)
---
Tasks
🟢 Easy — Ticket Classification (`ticket_classification`)
5 tickets | 10 max steps | Baseline ~0.72
Classify each ticket by department and urgency.
Ticket	Issue	Department	Urgency
T001	Double invoice charge	billing	high
T002	Password reset missing	technical	medium
T003	Defective product return	returns	medium
T004	App crash on iPhone	technical	high
T005	Support hours inquiry	general	low
Scoring: Per-ticket score = 0.5×dept_correct + 0.5×urg_correct. Episode = average over all 5.
---
🟡 Medium — Ticket Response (`ticket_response`)
3 tickets | 9 max steps | Baseline ~0.61
Draft professional responses. Graded on 4 signals:
Signal	Weight
Keyword coverage	40%
Length ≥ 80 words	20%
Acknowledgment phrases	20%
Next-steps language	20%
---
🔴 Hard — Full Inbox Triage (`inbox_triage`)
8 tickets | 24 max steps | Baseline ~0.54
Classify every ticket then apply correct routing:
Ticket	Scenario	Correct Action
H001	P0 prod outage (Enterprise)	escalate
H002	Phishing spam	archive
H003	Annual billing upgrade	respond
H004	2FA lockout	respond
H005	GDPR Article 17 deletion	escalate
H006	Wrong item shipped	respond
H007	Positive feedback	close
H008	PDF download broken	respond
Scoring: Classification 40% + Routing+content 60%
---
🟣 Expert — Churn Prevention (`churn_prevention`)
10 tickets | 30 max steps | Baseline ~0.42
The hardest task. Requires reading implicit churn signals:
Ticket	Scenario	Churn Risk	Action
E001	Competitor offering 40% less	critical	escalate
E002	Key API being deprecated	high	respond
E003	5th outage, $5k/incident	critical	escalate
E004	"How do I export all my data?"	high	respond
E005	Charged for deactivated seats ×3mo	high	respond
E006	CEO SaaS consolidation audit	critical	escalate
E007	Team downsized 12→2 people	medium	respond
E008	Salesforce integration breaking	high	escalate
E009	Happy renewal, wants discount	low	respond
E010	SOC2 compliance gap in audit	critical	escalate
E004 is the key test: "Planning to evaluate alternatives... want to export my data" looks like a how-to question but is a high churn signal. Agents that treat it as low-urgency lose significant points.
Scoring: Classification 20% + Routing 20% + Churn detection 40% + Retention quality 20%
---
Observation Space
```python
class TriageObservation(Observation):  # extends openenv-core Observation
    done: bool                         # episode terminated
    reward: float | None               # step reward (openenv-core convention)
    metadata: dict                     # episode_score when done=True
    task_name: str
    task_description: str
    step_count: int
    max_steps: int
    tickets: list[Ticket]
    pending_ticket_ids: list[str]
    completed_ticket_ids: list[str]
    actions_history: list[dict]        # last 10 actions
    score_so_far: float                # running episode score
```
---
Action Space
```python
class TriageAction(Action):            # extends openenv-core Action
    action_type: ActionType            # classify|respond|escalate|archive|close|skip
    ticket_id: str
    classification: ClassificationPayload | None
    response: ResponsePayload | None
    escalation: EscalationPayload | None
```
JSON examples:
```json
// Classify
{"action_type":"classify","ticket_id":"T001",
 "classification":{"department":"billing","urgency":"high"}}

// Respond
{"action_type":"respond","ticket_id":"M001",
 "response":{"response_text":"Dear Frank, I sincerely apologize..."}}

// Escalate (include churn_risk for expert task)
{"action_type":"escalate","ticket_id":"E001",
 "escalation":{"reason":"Critical churn risk: enterprise, competitor pricing",
                "churn_risk":"critical"}}

// Archive / Close
{"action_type":"archive","ticket_id":"H002"}
{"action_type":"close","ticket_id":"H007"}
```
---
Reward Function
Rewards are incremental — issued at every step, never only at episode end.
Scenario	Reward
Correct classify (both dept+urg)	+0.09–0.18
Correct routing + good content	+0.05–0.12
Correct routing + weak content	+0.01–0.04
Wrong routing	+0.01–0.02
Skip action	-0.01
All scores strictly in (0.01, 0.99) — never exactly 0.0 or 1.0. Episode score in `obs.metadata["episode_score"]` when `done=True`.
---
Baseline Performance
Task	Difficulty	Baseline (Qwen-72B)	Oracle Bound
`ticket_classification`	Easy	0.72	0.98
`ticket_response`	Medium	0.61	0.93
`inbox_triage`	Hard	0.54	0.91
`churn_prevention`	Expert	0.42	0.79
Average		0.57	0.90
---
Setup & Usage
Python
```bash
git clone https://github.com/pankeshbhagore/customer-support-openenv
cd customer-support-openenv
pip install -r requirements.txt
python -m uvicorn server.app:app --port 7860
python -m pytest tests/ -v        # 52 tests
```
Docker
```bash
docker build -t customer-support-triage .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN customer-support-triage
```
Inference
```bash
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```
Python SDK
```python
from support_triage_env import (
    CustomerSupportTriageEnv, TriageAction, ActionType,
    ClassificationPayload, Department, Urgency
)

env = CustomerSupportTriageEnv("churn_prevention")
obs = env.reset(episode_id="ep-001")

action = TriageAction(
    action_type=ActionType.CLASSIFY, ticket_id="E001",
    classification=ClassificationPayload(department=Department.BILLING, urgency=Urgency.HIGH)
)
obs = env.step(action)
print(obs.reward)    # 0.036 (per-step reward, embedded in observation)
print(obs.done)      # False
print(env.state.episode_id)  # "ep-001"
```
HTTP API
```bash
# Health
curl https://pankeshbhagore-customer-support-triage.hf.space/health

# Reset
curl -X POST .../reset -H "Content-Type: application/json" \
  -d '{"task_name":"churn_prevention","session_id":"s1"}'

# Step
curl -X POST .../step -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"classify","ticket_id":"E001",
       "classification":{"department":"billing","urgency":"high"}},"session_id":"s1"}'

# State
curl ".../state?session_id=s1"

# Metrics
curl .../metrics
```
---
Project Structure
```
customer-support-openenv/
├── support_triage_env/
│   ├── __init__.py          # Package exports
│   ├── environment.py       # CustomerSupportTriageEnv (extends openenv-core Environment)
│   ├── models.py            # TriageAction, TriageObservation, TriageState
│   ├── graders.py           # Deterministic graders for all 4 tasks
│   ├── task_data.py         # All ticket data + ground truth
│   └── inference.py         # Baseline inference script
├── server/
│   ├── __init__.py
│   └── app.py               # FastAPI server + dashboard + metrics
├── static/
│   └── index.html           # Interactive dashboard
├── tests/
│   ├── __init__.py
│   └── test_environment.py  # 52 unit tests
├── openenv.yaml             # OpenEnv metadata
├── Dockerfile               # Container definition
├── requirements.txt
├── setup.py
└── README.md
```
---
OpenEnv Compliance
Requirement	Status
Extends `openenv-core Environment` base class	✅
Typed Pydantic Observation, Action, State	✅
Observation extends `openenv.core.env_server.Observation`	✅
State extends `openenv.core.env_server.State` (episode_id, step_count)	✅
`reset(seed, episode_id)` signature	✅
`step(action)` → Observation with reward embedded	✅
`state` property → full state	✅
Scores strictly in (0.01, 0.99)	✅
`SUPPORTS_CONCURRENT_SESSIONS = True`	✅
≥ 3 tasks with deterministic graders	✅ (4 tasks)
Incremental reward signal	✅
`openenv.yaml` with spec_version, type, runtime, app	✅
`inference.py` with [START]/[STEP]/[END] format	✅
`HF_TOKEN` no default, `API_BASE_URL`/`MODEL_NAME` with defaults	✅
`LOCAL_IMAGE_NAME` variable	✅
52 unit tests	✅
Dockerfile + `docker build/run`	✅
Interactive dashboard at `/`	✅
`/metrics` endpoint	✅
