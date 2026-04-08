"""
MEDIUM Task: Draft appropriate responses to 3 support tickets.
Grader checks keyword presence, acknowledgment, next-steps, and length.
"""
from typing import Any, Dict, List, Tuple

from ..data import MEDIUM_RESPONSE_CRITERIA, MEDIUM_TICKETS
from ..models import Action, ActionType, Observation, Ticket

TASK_NAME = "ticket_response"
MAX_STEPS = 9


def get_initial_observation(step_count: int = 0) -> Observation:
    tickets = [Ticket(**t) for t in MEDIUM_TICKETS]
    return Observation(
        task_name=TASK_NAME,
        task_description=(
            "You are a customer support agent. Draft professional responses."
        ),
        step_count=step_count,
        max_steps=MAX_STEPS,
        tickets=tickets,
        pending_ticket_ids=[t["id"] for t in MEDIUM_TICKETS],
        completed_ticket_ids=[],
        actions_history=[],
        score_so_far=0.0,
        done=False,
    )


def _clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


def _score_response(ticket_id: str, response_text: str) -> Tuple[float, str, Dict[str, float]]:
    criteria = MEDIUM_RESPONSE_CRITERIA.get(ticket_id)
    if criteria is None:
        return 0.01, f"No criteria for ticket {ticket_id}", {}

    text_lower = response_text.lower()
    breakdown: Dict[str, float] = {}

    # 1. Keyword coverage
    keywords = criteria["required_keywords"]
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    kw_score = hits / len(keywords) if keywords else 0.9
    breakdown["keyword_coverage"] = _clamp(kw_score * 0.40)

    # 2. Length
    word_count = len(response_text.split())
    min_len = criteria.get("min_length", 80)
    length_score = min(word_count / min_len, 0.99)
    breakdown["length"] = _clamp(length_score * 0.20)

    # 3. Acknowledgment
    ack_phrases = ["i understand", "sorry", "apologize", "thank you"]
    ack_hit = any(p in text_lower for p in ack_phrases)
    breakdown["acknowledgment"] = 0.20 if ack_hit else 0.01

    # 4. Next steps
    action_phrases = ["we will", "please", "contact", "process", "refund"]
    action_hit = any(p in text_lower for p in action_phrases)
    breakdown["next_steps"] = 0.20 if action_hit else 0.01

    total = sum(breakdown.values())
    total = _clamp(total)

    reason = f"keywords={hits}/{len(keywords)} | words={word_count}/{min_len}"
    return total, reason, breakdown


def grade_action(
    action: Action,
    completed_ids: List[str],
    agent_actions: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:

    tid = action.ticket_id

    if tid not in [t["id"] for t in MEDIUM_TICKETS]:
        return 0.01, f"Unknown ticket id '{tid}'", {}

    if tid in completed_ids:
        return 0.01, f"Ticket {tid} already responded", {}

    if action.action_type != ActionType.RESPOND:
        return 0.01, "Action must be 'respond'", {}

    if action.response is None or not action.response.response_text.strip():
        return 0.01, "No response text provided", {}

    return _score_response(tid, action.response.response_text)


def compute_episode_score(agent_actions: List[Dict[str, Any]]) -> float:
    seen: Dict[str, float] = {}

    for a in agent_actions:
        tid = a.get("ticket_id")
        if a.get("action_type") == "respond" and tid and tid not in seen:
            text = a.get("response_text", "")
            score, _, _ = _score_response(tid, text)
            seen[tid] = _clamp(score)

    if not seen:
        return 0.01

    score = sum(seen.get(t["id"], 0.01) for t in MEDIUM_TICKETS) / len(MEDIUM_TICKETS)
    return _clamp(score)