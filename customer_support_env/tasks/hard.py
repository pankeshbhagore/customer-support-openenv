"""
HARD Task: Full inbox triage — OpenEnv SAFE VERSION
"""

from typing import Any, Dict, List, Tuple

from ..data import HARD_GROUND_TRUTH, HARD_TICKETS
from ..models import Action, ActionType, Observation, Ticket

TASK_NAME = "inbox_triage"
MAX_STEPS = 24


def _clamp(x: float) -> float:
    return max(0.01, min(0.99, x))


def get_initial_observation(step_count: int = 0) -> Observation:
    tickets = [Ticket(**t) for t in HARD_TICKETS]
    return Observation(
        task_name=TASK_NAME,
        task_description="Inbox triage system",
        step_count=step_count,
        max_steps=MAX_STEPS,
        tickets=tickets,
        pending_ticket_ids=[t["id"] for t in HARD_TICKETS],
        completed_ticket_ids=[],
        actions_history=[],
        score_so_far=0.0,
        done=False,
    )


def _score_response_keywords(ticket_id: str, response_text: str) -> float:
    gt = HARD_GROUND_TRUTH.get(ticket_id, {})
    keywords = gt.get("response_keywords", [])

    if not keywords:
        return 0.9

    text_lower = response_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)

    score = hits / len(keywords)
    return _clamp(score)


def grade_action(
    action: Action,
    completed_ids: List[str],
    agent_actions: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:

    tid = action.ticket_id
    gt = HARD_GROUND_TRUTH.get(tid)

    if gt is None:
        return 0.01, f"Unknown ticket id '{tid}'", {}

    all_ids = [t["id"] for t in HARD_TICKETS]
    if tid not in all_ids:
        return 0.01, f"Ticket {tid} not in task", {}

    breakdown: Dict[str, float] = {}

    prior_actions = [a for a in agent_actions if a.get("ticket_id") == tid]
    classify_done = any(a.get("action_type") == "classify" for a in prior_actions)
    route_done = any(a.get("action_type") in {"escalate", "archive", "respond", "close"} for a in prior_actions)

    atype = action.action_type
    atype_str = atype.value if hasattr(atype, "value") else str(atype)

    # ── CLASSIFY ──
    if atype_str == "classify":
        if classify_done:
            return 0.01, "Already classified", {}

        if action.classification is None:
            return 0.01, "Missing classification", {}

        cls = action.classification

        dept_ok = cls.department == gt["department"]
        urg_ok = cls.urgency == gt["urgency"]

        dept_score = 0.5 if dept_ok else 0.01
        urg_score = 0.5 if urg_ok else 0.01

        sub_score = _clamp(dept_score + urg_score)

        step_reward = 0.4 * sub_score * (1 / len(HARD_TICKETS))
        return _clamp(step_reward), "classified", {
            "department": dept_score,
            "urgency": urg_score
        }

    # ── ROUTING ──
    if atype_str in {"escalate", "archive", "respond", "close"}:

        if route_done:
            return 0.01, "Already routed", {}

        correct_action = gt.get("correct_action")
        action_correct = (atype_str == correct_action)

        route_score = 0.9 if action_correct else 0.1

        response_quality = 0.01
        if action_correct and atype_str == "respond":
            resp_text = (action.response.response_text if action.response else "")
            response_quality = _score_response_keywords(tid, resp_text)

        route_component = 0.4 * route_score * (1 / len(HARD_TICKETS))
        resp_component = 0.2 * response_quality * (1 / len(HARD_TICKETS))

        step_reward = route_component + resp_component

        if not classify_done:
            step_reward *= 0.7

        return _clamp(step_reward), "routed", {
            "routing": route_score,
            "response_quality": response_quality
        }

    return 0.01, "Invalid action", {}


def compute_episode_score(agent_actions: List[Dict[str, Any]]) -> float:

    cls_scores: Dict[str, float] = {}
    route_scores: Dict[str, float] = {}
    resp_scores: Dict[str, float] = {}

    for a in agent_actions:
        tid = a.get("ticket_id")
        gt = HARD_GROUND_TRUTH.get(tid)

        if gt is None:
            continue

        atype = a.get("action_type")

        if atype == "classify" and tid not in cls_scores:
            dept_ok = a.get("department") == gt["department"]
            urg_ok = a.get("urgency") == gt["urgency"]

            val = (0.5 if dept_ok else 0.01) + (0.5 if urg_ok else 0.01)
            cls_scores[tid] = _clamp(val)

        if atype in {"escalate", "archive", "respond", "close"} and tid not in route_scores:
            correct = gt.get("correct_action")
            route_scores[tid] = 0.9 if atype == correct else 0.1

            if atype == "respond":
                resp_scores[tid] = _score_response_keywords(tid, a.get("response_text", ""))

    n = len(HARD_TICKETS)
    ids = [t["id"] for t in HARD_TICKETS]

    cls_avg = sum(cls_scores.get(tid, 0.01) for tid in ids) / n
    route_avg = sum(route_scores.get(tid, 0.01) for tid in ids) / n
    resp_avg = sum(resp_scores.get(tid, 0.01) for tid in ids) / n

    final = 0.40 * cls_avg + 0.40 * route_avg + 0.20 * resp_avg
    return _clamp(final)
