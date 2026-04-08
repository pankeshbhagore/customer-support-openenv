"""
EASY Task: Classify 5 support tickets by department + urgency.
"""

from typing import Any, Dict, List, Tuple

from ..data import EASY_GROUND_TRUTH, EASY_TICKETS
from ..models import Action, ActionType, Observation, Ticket

TASK_NAME = "ticket_classification"
MAX_STEPS = 10


def _clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


def get_initial_observation(step_count: int = 0) -> Observation:
    tickets = [Ticket(**t) for t in EASY_TICKETS]
    return Observation(
        task_name=TASK_NAME,
        task_description="Classify tickets by department and urgency.",
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

    tid = action.ticket_id
    gt = EASY_GROUND_TRUTH.get(tid)

    if gt is None:
        return 0.01, f"Unknown ticket id '{tid}'", {}

    if tid in completed_ids:
        return 0.01, f"Ticket {tid} already classified", {}

    if action.action_type != ActionType.CLASSIFY:
        return 0.01, "Action must be 'classify'", {}

    if action.classification is None:
        return 0.01, "No classification payload", {}

    cls = action.classification

    dept_correct = cls.department == gt["department"]
    urg_correct = cls.urgency == gt["urgency"]

    dept_score = 0.5 if dept_correct else 0.01
    urg_score = 0.5 if urg_correct else 0.01

    total = dept_score + urg_score
    total = _clamp(total)

    return total, "scored", {
        "department": dept_score,
        "urgency": urg_score
    }


def compute_episode_score(agent_actions: List[Dict[str, Any]]) -> float:
    seen: Dict[str, float] = {}

    for a in agent_actions:
        tid = a.get("ticket_id")

        if a.get("action_type") == "classify" and tid and tid not in seen:
            gt = EASY_GROUND_TRUTH.get(tid)
            if gt is None:
                continue

            dept_ok = a.get("department") == gt["department"]
            urg_ok = a.get("urgency") == gt["urgency"]

            score_val = (0.5 if dept_ok else 0.01) + (0.5 if urg_ok else 0.01)
            score_val = _clamp(score_val)

            seen[tid] = score_val

    if not seen:
        return 0.01

    score = sum(seen.get(t["id"], 0.01) for t in EASY_TICKETS) / len(EASY_TICKETS)
    return _clamp(score)