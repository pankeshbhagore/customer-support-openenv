"""
EASY Task: Classify 5 support tickets by department + urgency.
Grader scores: correct classifications / total possible.
"""
from typing import Any, Dict, List, Tuple

from ..data import EASY_GROUND_TRUTH, EASY_TICKETS
from ..models import Action, ActionType, Observation, Reward, Ticket

TASK_NAME = "ticket_classification"
MAX_STEPS = 10   # 5 tickets × 2 headroom


def get_initial_observation(step_count: int = 0) -> Observation:
    tickets = [Ticket(**t) for t in EASY_TICKETS]
    return Observation(
        task_name=TASK_NAME,
        task_description=(
            "You are a customer support triage agent. "
            "Classify each of the 5 tickets by assigning the correct 'department' "
            "(billing | technical | general | returns) and 'urgency' (low | medium | high). "
            "Use action_type='classify' for each ticket. "
            "You will be scored on accuracy."
        ),
        step_count=step_count,
        max_steps=MAX_STEPS,
        tickets=tickets,
        pending_ticket_ids=[t["id"] for t in EASY_TICKETS],
        completed_ticket_ids=[],
        actions_history=[],
        score_so_far=0.0,
        done=False,
    )


def grade_action(
    action: Action,
    completed_ids: List[str],
    agent_actions: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Score a single classify action.
    Returns (reward_value, reason, breakdown).
    """
    tid = action.ticket_id
    gt = EASY_GROUND_TRUTH.get(tid)

    if gt is None:
        return 0.0, f"Unknown ticket id '{tid}'", {}

    if tid in completed_ids:
        return 0.0, f"Ticket {tid} already classified — no reward for re-classifying", {}

    if action.action_type != ActionType.CLASSIFY:
        return 0.0, "Action must be 'classify' for this task", {}

    if action.classification is None:
        return 0.0, "No classification payload provided", {}

    cls = action.classification
    dept_correct = cls.department == gt["department"]
    urg_correct = cls.urgency == gt["urgency"]

    dept_score = 0.5 if dept_correct else 0.0
    urg_score = 0.5 if urg_correct else 0.0
    total = dept_score + urg_score  # 0.0, 0.5, or 1.0

    # Scale to per-ticket weight (5 tickets → each worth 0.2 of episode)
    # Reward returned here is per-step; final score is averaged over tickets.
    reason_parts = []
    if dept_correct:
        reason_parts.append(f"department='{gt['department']}' ✓")
    else:
        reason_parts.append(f"department: got '{cls.department}', expected '{gt['department']}' ✗")
    if urg_correct:
        reason_parts.append(f"urgency='{gt['urgency']}' ✓")
    else:
        reason_parts.append(f"urgency: got '{cls.urgency}', expected '{gt['urgency']}' ✗")

    total = max(0.01, min(0.99, total))
    return total, " | ".join(reason_parts), {"department": dept_score, "urgency": urg_score}


def compute_episode_score(agent_actions: List[Dict[str, Any]]) -> float:
    """
    Final episode score: average per-ticket accuracy across all 5 tickets.
    Only the FIRST classify action per ticket is counted.
    """
    seen: Dict[str, float] = {}
    for a in agent_actions:
        tid = a.get("ticket_id")
        if a.get("action_type") == "classify" and tid and tid not in seen:
            gt = EASY_GROUND_TRUTH.get(tid)
            if gt is None:
                continue
            dept_ok = a.get("department") == gt["department"]
            urg_ok = a.get("urgency") == gt["urgency"]
            seen[tid] = (0.5 * int(dept_ok)) + (0.5 * int(urg_ok))

    if not seen:
        return 0.0
    # Average over ALL 5 tickets (missing ones score 0)
    return sum(seen.get(t["id"], 0.0) for t in EASY_TICKETS) / len(EASY_TICKETS)
