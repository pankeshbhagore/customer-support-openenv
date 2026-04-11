"""
Deterministic graders for all 4 tasks.
All scores strictly in (0.01, 0.99) — never exactly 0.0 or 1.0.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .task_data import EASY_GT, MEDIUM_CRITERIA, HARD_GT, EXPERT_GT


def _clamp(x: float) -> float:
    """Strictly clamp to open interval (0.01, 0.99)."""
    return max(0.02, min(0.98, round(x, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# EASY — ticket_classification
# ─────────────────────────────────────────────────────────────────────────────

def grade_classify(ticket_id: str, department: str, urgency: str) -> Tuple[float, str]:
    gt = EASY_GT.get(ticket_id)
    if not gt:
        return 0.01, f"Unknown ticket '{ticket_id}'"
    dept_ok = department == gt["department"]
    urg_ok  = urgency == gt["urgency"]
    score = _clamp((0.5 if dept_ok else 0.01) + (0.5 if urg_ok else 0.01))
    reason = (f"dept={'✓' if dept_ok else '✗'}({gt['department']}) "
              f"urgency={'✓' if urg_ok else '✗'}({gt['urgency']})")
    return score, reason


def episode_score_easy(agent_actions: List[Dict]) -> float:
    from .task_data import EASY_TICKETS
    seen: Dict[str, float] = {}
    for a in agent_actions:
        tid = a.get("ticket_id")
        if a.get("action_type") == "classify" and tid and tid not in seen:
            s, _ = grade_classify(tid, a.get("department", ""), a.get("urgency", ""))
            seen[tid] = s
    n = len(EASY_TICKETS)
    return _clamp(sum(seen.get(t["id"], 0.01) for t in EASY_TICKETS) / n)


# ─────────────────────────────────────────────────────────────────────────────
# MEDIUM — ticket_response
# ─────────────────────────────────────────────────────────────────────────────

def grade_response(ticket_id: str, response_text: str) -> Tuple[float, str]:
    criteria = MEDIUM_CRITERIA.get(ticket_id)
    if not criteria:
        return 0.01, f"No criteria for '{ticket_id}'"
    text = response_text.lower()

    # Keyword coverage 40%
    kws = criteria["keywords"]
    kw_hits = sum(1 for k in kws if k in text)
    kw_score = kw_hits / len(kws) if kws else 0.9

    # Length 20%
    words = len(response_text.split())
    min_w = criteria.get("min_words", 80)
    len_score = min(words / min_w, 0.99)

    # Acknowledgment 20%
    ack_ok = any(p in text for p in criteria.get("ack_phrases", []))

    # Next steps 20%
    act_ok = any(p in text for p in criteria.get("action_phrases", []))

    raw = 0.40 * kw_score + 0.20 * len_score + 0.20 * (0.9 if ack_ok else 0.05) + 0.20 * (0.9 if act_ok else 0.05)
    reason = (f"kw={kw_hits}/{len(kws)} words={words}/{min_w} "
              f"ack={'✓' if ack_ok else '✗'} steps={'✓' if act_ok else '✗'}")
    return _clamp(raw), reason


def episode_score_medium(agent_actions: List[Dict]) -> float:
    from .task_data import MEDIUM_TICKETS
    seen: Dict[str, float] = {}
    for a in agent_actions:
        tid = a.get("ticket_id")
        if a.get("action_type") == "respond" and tid and tid not in seen:
            s, _ = grade_response(tid, a.get("response_text", ""))
            seen[tid] = s
    n = len(MEDIUM_TICKETS)
    return _clamp(sum(seen.get(t["id"], 0.01) for t in MEDIUM_TICKETS) / n)


# ─────────────────────────────────────────────────────────────────────────────
# HARD — inbox_triage
# ─────────────────────────────────────────────────────────────────────────────

def grade_hard_classify(ticket_id: str, department: str, urgency: str) -> Tuple[float, str]:
    gt = HARD_GT.get(ticket_id)
    if not gt:
        return 0.01, f"Unknown ticket '{ticket_id}'"
    dept_ok = department == gt["department"]
    urg_ok  = urgency == gt["urgency"]
    score = _clamp((0.5 if dept_ok else 0.01) + (0.5 if urg_ok else 0.01))
    return score, f"dept={'✓' if dept_ok else '✗'} urgency={'✓' if urg_ok else '✗'}"


def grade_hard_route(ticket_id: str, action_type: str,
                     response_text: str = "", reason: str = "") -> Tuple[float, str]:
    gt = HARD_GT.get(ticket_id)
    if not gt:
        return 0.01, f"Unknown ticket '{ticket_id}'"
    expected = gt["action"]
    route_ok = action_type == expected
    route_s  = 0.9 if route_ok else 0.08

    # Quality bonus for text content
    text = (response_text + " " + reason).lower()
    content_kws = gt.get("response_kws", gt.get("reason_kws", []))
    if content_kws:
        hits = sum(1 for k in content_kws if k in text)
        content_s = hits / len(content_kws)
    else:
        content_s = 0.5

    raw = 0.70 * route_s + 0.30 * content_s
    reason_out = f"route={'✓' if route_ok else '✗'}(exp={expected}) content={content_s:.2f}"
    return _clamp(raw), reason_out


def episode_score_hard(agent_actions: List[Dict]) -> float:
    from .task_data import HARD_TICKETS
    cls_s: Dict[str, float] = {}
    route_s: Dict[str, float] = {}
    for a in agent_actions:
        tid = a.get("ticket_id")
        atype = a.get("action_type", "")
        if atype == "classify" and tid and tid not in cls_s:
            s, _ = grade_hard_classify(tid, a.get("department",""), a.get("urgency",""))
            cls_s[tid] = s
        if atype in {"escalate","archive","respond","close"} and tid and tid not in route_s:
            s, _ = grade_hard_route(tid, atype,
                                    a.get("response_text",""), a.get("escalation_reason",""))
            route_s[tid] = s
    ids = [t["id"] for t in HARD_TICKETS]
    n = len(ids)
    cls_avg   = sum(cls_s.get(i, 0.01) for i in ids) / n
    route_avg = sum(route_s.get(i, 0.01) for i in ids) / n
    return _clamp(0.40 * cls_avg + 0.60 * route_avg)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT — churn_prevention
# ─────────────────────────────────────────────────────────────────────────────

def grade_churn_classify(ticket_id: str, department: str, urgency: str) -> Tuple[float, str]:
    gt = EXPERT_GT.get(ticket_id)
    if not gt:
        return 0.01, f"Unknown ticket '{ticket_id}'"
    dept_ok = department == gt["department"]
    urg_ok  = urgency == gt["urgency"]
    score = _clamp((0.5 if dept_ok else 0.01) + (0.5 if urg_ok else 0.01))
    return score, f"dept={'✓' if dept_ok else '✗'} urgency={'✓' if urg_ok else '✗'}"


def _detect_churn_risk_quality(ticket_id: str, text: str) -> float:
    """Score how well the text captures churn risk signals."""
    gt = EXPERT_GT.get(ticket_id, {})
    expected_risk = gt.get("churn_risk", "medium")
    text_l = text.lower()
    risk_named = expected_risk in text_l
    kws = gt.get("reason_kws", [])
    kw_hits = sum(1 for k in kws if k in text_l) / (len(kws) or 1)
    if risk_named and kw_hits >= 0.5:
        return 0.93
    elif risk_named or kw_hits >= 0.6:
        return 0.68
    elif kw_hits >= 0.3:
        return 0.42
    return 0.12


def _score_retention_response(ticket_id: str, response_text: str) -> float:
    gt = EXPERT_GT.get(ticket_id, {})
    kws = gt.get("response_kws", [])
    text = response_text.lower()
    kw_score = sum(1 for k in kws if k in text) / (len(kws) or 1)
    words = len(response_text.split())
    len_score = min(words / 100, 0.99)
    personal = any(p in text for p in ["your team", "your account", "your business",
                                        "you've been", "your contract", "your plan"])
    return _clamp(0.50 * kw_score + 0.25 * len_score + 0.25 * (0.9 if personal else 0.08))


def grade_churn_route(ticket_id: str, action_type: str,
                      response_text: str = "", reason: str = "") -> Tuple[float, str]:
    gt = EXPERT_GT.get(ticket_id)
    if not gt:
        return 0.01, f"Unknown ticket '{ticket_id}'"
    expected = gt["action"]
    route_ok = action_type == expected
    route_s  = 0.9 if route_ok else 0.08

    combined_text = response_text + " " + reason
    churn_s = _detect_churn_risk_quality(ticket_id, combined_text)
    resp_s  = _score_retention_response(ticket_id, combined_text)

    raw = 0.30 * route_s + 0.40 * churn_s + 0.30 * resp_s
    parts = [f"route={'✓' if route_ok else '✗'}(exp={expected})",
             f"churn_detect={churn_s:.2f}", f"retention={resp_s:.2f}"]
    return _clamp(raw), " | ".join(parts)


def episode_score_expert(agent_actions: List[Dict]) -> float:
    from .task_data import EXPERT_TICKETS
    cls_s: Dict[str, float] = {}
    route_s: Dict[str, float] = {}
    for a in agent_actions:
        tid = a.get("ticket_id")
        atype = a.get("action_type", "")
        if atype == "classify" and tid and tid not in cls_s:
            s, _ = grade_churn_classify(tid, a.get("department",""), a.get("urgency",""))
            cls_s[tid] = s
        if atype in {"escalate","respond","close","archive"} and tid and tid not in route_s:
            s, _ = grade_churn_route(tid, atype,
                                     a.get("response_text",""), a.get("escalation_reason",""))
            route_s[tid] = s
    ids = [t["id"] for t in EXPERT_TICKETS]
    n = len(ids)
    cls_avg   = sum(cls_s.get(i, 0.01) for i in ids) / n
    route_avg = sum(route_s.get(i, 0.01) for i in ids) / n
    return _clamp(0.20 * cls_avg + 0.80 * route_avg)
