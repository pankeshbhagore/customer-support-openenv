"""
Pydantic models for the Customer Support Triage Environment.
Defines typed Observation, Action, and Reward models per OpenEnv spec.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────── Enumerations ────────────────────────────────────

class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    RETURNS = "returns"
    UNKNOWN = "unknown"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(str, Enum):
    CLASSIFY = "classify"       # Assign department + urgency to a ticket
    RESPOND = "respond"         # Write a reply to a ticket
    ESCALATE = "escalate"       # Mark ticket for human escalation
    CLOSE = "close"             # Mark ticket as resolved/closed
    ARCHIVE = "archive"         # Archive (spam / irrelevant)
    SKIP = "skip"               # Skip current ticket (penalised)


# ─────────────────────────── Sub-models ──────────────────────────────────────

class Ticket(BaseModel):
    """A single customer support ticket."""
    id: str
    subject: str
    body: str
    customer_name: str
    customer_email: str
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClassificationPayload(BaseModel):
    department: Department
    urgency: Urgency


class ResponsePayload(BaseModel):
    response_text: str


class EscalationPayload(BaseModel):
    reason: str


# ─────────────────────────── Core Models ─────────────────────────────────────

class Action(BaseModel):
    """Agent action — always targets a specific ticket."""
    action_type: ActionType
    ticket_id: str
    classification: Optional[ClassificationPayload] = None
    response: Optional[ResponsePayload] = None
    escalation: Optional[EscalationPayload] = None

    model_config = {"use_enum_values": True}


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_name: str
    task_description: str
    step_count: int
    max_steps: int
    tickets: List[Ticket]
    pending_ticket_ids: List[str]          # tickets not yet actioned
    completed_ticket_ids: List[str]        # tickets already actioned
    actions_history: List[Dict[str, Any]]  # log of past actions
    score_so_far: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(ge=-0.1, le=1.0)   # allows small skip penalty
    reason: str
    breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Return type of env.step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    """Full internal state snapshot (for env.state())."""
    task_name: str
    step_count: int
    max_steps: int
    tickets: List[Ticket]
    ground_truth: Dict[str, Any]
    agent_actions: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool
    episode_score: float
