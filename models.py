"""
Typed Pydantic models for the Customer Support Triage OpenEnv.
Extends openenv-core base classes (Observation, Action, State).
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State


# ─────────────────────── Enumerations ────────────────────────────────────────

class Department(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    GENERAL   = "general"
    RETURNS   = "returns"


class Urgency(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    RESPOND  = "respond"
    ESCALATE = "escalate"
    ARCHIVE  = "archive"
    CLOSE    = "close"
    SKIP     = "skip"


class ChurnRisk(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ─────────────────────── Ticket data model ───────────────────────────────────

class Ticket(Action):  # Tickets are pure data — reuse pydantic base
    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}

    id: str
    subject: str
    body: str
    customer_name: str
    customer_email: str
    created_at: str
    ticket_metadata: Dict[str, Any] = {}


# ─────────────────────── Action payloads ─────────────────────────────────────

class ClassificationPayload(Action):
    department: Department
    urgency: Urgency


class ResponsePayload(Action):
    response_text: str


class EscalationPayload(Action):
    reason: str
    churn_risk: Optional[ChurnRisk] = None


class TriageAction(Action):
    """Single unified action type for all triage operations."""
    action_type: ActionType
    ticket_id: str
    classification: Optional[ClassificationPayload] = None
    response: Optional[ResponsePayload] = None
    escalation: Optional[EscalationPayload] = None


# ─────────────────────── Observation ─────────────────────────────────────────

class TriageObservation(Observation):
    """What the agent sees at each step — extends openenv Observation."""
    task_name: str
    task_description: str
    step_count: int
    max_steps: int
    tickets: List[Ticket]
    pending_ticket_ids: List[str]
    completed_ticket_ids: List[str]
    actions_history: List[Dict[str, Any]] = []
    score_so_far: float = 0.01


# ─────────────────────── State ───────────────────────────────────────────────

class TriageState(State):
    """Full internal state — extends openenv State (has episode_id, step_count)."""
    task_name: str = ""
    max_steps: int = 0
    tickets: List[Ticket] = []
    ground_truth: Dict[str, Any] = {}
    agent_actions: List[Dict[str, Any]] = []
    cumulative_reward: float = 0.0
    episode_score: float = 0.01
