"""
MEDIUM Task: Draft appropriate responses to 3 support tickets.
Grader checks keyword presence, acknowledgment, next-steps, and length.
"""
from typing import Any, Dict, List, Tuple

from ..data import MEDIUM_RESPONSE_CRITERIA, MEDIUM_TICKETS
from ..models import Action, ActionType, Observation, Ticket

TASK_NAME = "ticket_response"
MAX_STEPS = 9   # 3 tickets × 3 headroom


def get_initial_observation(step_count: int = 0) -> Observation:
    tickets = [Ticket(**t) for t in MEDIUM_TICKETS]
    return Observation(
        task_name=TASK_NAME,
        task_description=(
            "You are a customer support agent. For each of the 3 tickets, "
            "draft a professional, empathetic response that: "
            "(1) acknowledges the customer's issue, "
            "(2) apologizes where appropriate, "
            "(3) provides clear next steps or resolution. "
            "Use action_type='respond' with a 'response' payload. "
            "Responses should be at least 80 words and address the specific situation."
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


def _score_response(ticket_id: str, response_text: str) -> Tuple[float, str, Dict[str, float]]:
    """Score a single response against criteria. Returns (score 0-1, reason, breakdown)."""
    criteria = MEDIUM_RESPONSE_CRITERIA.get(ticket_id)
    if criteria is None:
        return 0.0, f"No criteria for ticket {ticket_id}", {}

    text_lower = response_text.lower()
    breakdown: Dict[str, float] = {}

    # 1. Keyword coverage (40% of score)
    keywords = criteria["required_keywords"]
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    kw_score = hits / len(keywords) if keywords else 1.0
    breakdown["keyword_coverage"] = round(kw_score * 0.40, 4)

    # 2. Minimum length (20% of score)
    word_count = len(response_text.split())
    min_len = criteria.get("min_length", 80)
    length_score = min(word_count / min_len, 1.0)
    breakdown["length"] = round(length_score * 0.20, 4)

    # 3. Acknowledgment heuristic (20%): contains phrases like "I understand", "sorry", "apologize"
    ack_phrases = ["i understand", "i'm sorry", "im sorry", "apologize", "apologies",
                   "sorry to hear", "thank you for", "we understand"]
    ack_hit = any(p in text_lower for p in ack_phrases)
    breakdown["acknowledgment"] = round(0.20 if ack_hit else 0.0, 4)

    # 4. Next-steps heuristic (20%): contains action words
    action_phrases = ["we will", "we'll", "you can", "please", "contact", "team",
                      "investigate", "process", "arrange", "send", "refund", "replace"]
    action_hit = any(p in text_lower for p in action_phrases)
    breakdown["next_steps"] = round(0.20 if action_hit else 0.0, 4)

    total = sum(breakdown.values())
    total = max(0.0, min(1.0, total))

    hits_str = f"{hits}/{len(keywords)} keywords matched"
    reason = (
        f"{hits_str} | words={word_count}/{min_len} | "
        f"ack={'✓' if ack_hit else '✗'} | next_steps={'✓' if action_hit else '✗'}"
    )
    return round(total, 4), reason, breakdown


def grade_action(
    action: Action,
    completed_ids: List[str],
    agent_actions: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:
    tid = action.ticket_id

    if tid not in [t["id"] for t in MEDIUM_TICKETS]:
        return 0.0, f"Unknown ticket id '{tid}'", {}

    if tid in completed_ids:
        return 0.0, f"Ticket {tid} already responded to — no reward for re-responding", {}

    if action.action_type != ActionType.RESPOND:
        return 0.0, "Action must be 'respond' for this task", {}

    if action.response is None or not action.response.response_text.strip():
        return 0.0, "No response text provided", {}

    return _score_response(tid, action.response.response_text)


def compute_episode_score(agent_actions: List[Dict[str, Any]]) -> float:
    """Average response quality across all 3 tickets (missing = 0)."""
    seen: Dict[str, float] = {}
    for a in agent_actions:
        tid = a.get("ticket_id")
        if a.get("action_type") == "respond" and tid and tid not in seen:
            text = a.get("response_text", "")
            score, _, _ = _score_response(tid, text)
            seen[tid] = score

    return sum(seen.get(t["id"], 0.0) for t in MEDIUM_TICKETS) / len(MEDIUM_TICKETS)
