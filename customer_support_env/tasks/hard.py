"""
HARD Task: Full inbox triage — 8 mixed tickets.
Agent must classify each, then apply the correct action:
  - Escalate P0 outages and legal/GDPR tickets
  - Archive spam
  - Respond to billing / returns / tech issues
  - Close positive feedback
Grader checks: classification accuracy + correct routing + response quality.
"""
from typing import Any, Dict, List, Tuple

from ..data import HARD_GROUND_TRUTH, HARD_TICKETS
from ..models import Action, ActionType, Observation, Ticket

TASK_NAME = "inbox_triage"
MAX_STEPS = 24   # 8 tickets × 3 headroom (classify + act)


# Actions that need a written response to get full credit
_RESPONSE_REQUIRED = {"respond"}
# Legal/GDPR escalation keywords
_ESCALATION_KEYWORDS = ["gdpr", "legal", "compliance", "data deletion", "p0",
                         "production", "outage", "enterprise", "critical"]


def get_initial_observation(step_count: int = 0) -> Observation:
    tickets = [Ticket(**t) for t in HARD_TICKETS]
    return Observation(
        task_name=TASK_NAME,
        task_description=(
            "You are a senior customer support triage agent managing a busy inbox. "
            "For each of the 8 tickets you must:\n"
            "  1. Use action_type='classify' to assign department + urgency.\n"
            "  2. Then use the correct routing action:\n"
            "     - 'escalate' for P0 production incidents or legal/GDPR requests\n"
            "     - 'archive'  for spam or phishing\n"
            "     - 'respond'  for billing, returns, and solvable tech issues\n"
            "     - 'close'    for positive feedback or already-resolved tickets\n"
            "You will be scored on: classification accuracy (40%), "
            "correct routing (40%), and response quality for tickets needing a response (20%)."
        ),
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
    """Score response quality by keyword presence for hard task."""
    gt = HARD_GROUND_TRUTH.get(ticket_id, {})
    keywords = gt.get("response_keywords", [])
    if not keywords:
        return 1.0  # no requirement
    text_lower = response_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 4)


def grade_action(
    action: Action,
    completed_ids: List[str],
    agent_actions: List[Dict[str, Any]],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Grade a single action. Multi-step per ticket expected:
    First classify → partial credit.
    Then correct route action → remaining credit.
    """
    tid = action.ticket_id
    gt = HARD_GROUND_TRUTH.get(tid)

    if gt is None:
        return 0.0, f"Unknown ticket id '{tid}'", {}

    atype = action.action_type
    all_ids = [t["id"] for t in HARD_TICKETS]
    if tid not in all_ids:
        return 0.0, f"Ticket {tid} not in this task", {}

    breakdown: Dict[str, float] = {}

    # Count prior actions on this ticket
    prior_actions = [a for a in agent_actions if a.get("ticket_id") == tid]
    classify_done = any(a.get("action_type") == "classify" for a in prior_actions)
    route_done = any(a.get("action_type") in {"escalate", "archive", "respond", "close"}
                     for a in prior_actions)

    atype_str = atype.value if hasattr(atype, "value") else str(atype)

    # ── CLASSIFY ──
    if atype_str == "classify":
        if classify_done:
            return 0.0, f"Ticket {tid} already classified", {}
        if action.classification is None:
            return 0.0, "No classification payload", {}
        cls = action.classification
        dept_ok = cls.department == gt["department"]
        urg_ok = cls.urgency == gt["urgency"]
        dept_score = 0.5 if dept_ok else 0.0
        urg_score = 0.5 if urg_ok else 0.0
        # Classify contributes 40% of this ticket's share; per-ticket: 1/8
        # Return classification sub-score (0-1); weighting done at episode level
        sub_score = dept_score + urg_score  # 0.0–1.0
        breakdown["department"] = dept_score
        breakdown["urgency"] = urg_score
        reason = (
            f"dept={'✓' if dept_ok else '✗'}({gt['department']}) "
            f"| urgency={'✓' if urg_ok else '✗'}({gt['urgency']})"
        )
        # Step reward: classify is worth 0.4 × sub_score × (1/8) of full episode
        step_reward = round(0.4 * sub_score * (1 / len(HARD_TICKETS)), 4)
        return step_reward, reason, breakdown

    # ── ROUTING ACTIONS ──
    if atype_str in {"escalate", "archive", "respond", "close"}:
        if route_done:
            return 0.0, f"Ticket {tid} already routed — skipping", {}

        correct_action = gt.get("correct_action")
        atype_str = atype.value if hasattr(atype, "value") else str(atype)
        action_correct = (atype_str == correct_action)

        route_score = 1.0 if action_correct else 0.0
        breakdown["routing"] = route_score

        reason_parts = [f"route={'✓' if action_correct else '✗'}(expected={correct_action})"]

        # For respond: also check response quality
        response_quality = 0.0
        if action_correct and atype_str == "respond":
            resp_text = (action.response.response_text if action.response else "")
            response_quality = _score_response_keywords(tid, resp_text)
            breakdown["response_quality"] = response_quality
            reason_parts.append(f"response_quality={response_quality:.2f}")

        # For escalate: check reason keywords
        if action_correct and atype_str == "escalate":
            reason_kws = gt.get("reason_keywords", [])
            esc_reason = (action.escalation.reason if action.escalation else "")
            kw_hits = sum(1 for kw in reason_kws if kw.lower() in esc_reason.lower())
            kw_score = (kw_hits / len(reason_kws)) if reason_kws else 1.0
            breakdown["escalation_reason_quality"] = round(kw_score, 4)
            reason_parts.append(f"escalation_reason={kw_score:.2f}")

        # Step reward: routing worth 0.4; response bonus 0.2 (scaled per ticket)
        route_component = 0.4 * route_score * (1 / len(HARD_TICKETS))
        resp_component = 0.2 * response_quality * (1 / len(HARD_TICKETS))
        step_reward = round(route_component + resp_component, 4)

        # Penalise skipping classify step
        if not classify_done:
            step_reward = round(step_reward * 0.7, 4)
            reason_parts.append("(-30% penalty: no classify step)")

        return step_reward, " | ".join(reason_parts), breakdown

    return 0.0, f"Unexpected action_type '{atype}' — no reward", {}


def compute_episode_score(agent_actions: List[Dict[str, Any]]) -> float:
    """
    Final holistic score 0-1 over 8 tickets.
    Classification (40%) + Routing (40%) + Response quality (20%).
    """
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
            cls_scores[tid] = 0.5 * int(dept_ok) + 0.5 * int(urg_ok)

        if atype in {"escalate", "archive", "respond", "close"} and tid not in route_scores:
            correct = gt.get("correct_action")
            route_scores[tid] = 1.0 if atype == correct else 0.0
            if atype == "respond":
                resp_text = a.get("response_text", "")
                resp_scores[tid] = _score_response_keywords(tid, resp_text)

    n = len(HARD_TICKETS)
    ticket_ids = [t["id"] for t in HARD_TICKETS]

    cls_avg = sum(cls_scores.get(tid, 0.0) for tid in ticket_ids) / n
    route_avg = sum(route_scores.get(tid, 0.0) for tid in ticket_ids) / n
    resp_avg = sum(resp_scores.get(tid, 0.0) for tid in ticket_ids) / n

    return round(0.40 * cls_avg + 0.40 * route_avg + 0.20 * resp_avg, 4)
