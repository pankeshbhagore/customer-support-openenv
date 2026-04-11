"""
CustomerSupportTriageEnv — OpenEnv-compliant environment.
Properly extends openenv-core Environment base class.

Four tasks (easy → expert):
  ticket_classification  — classify 5 tickets
  ticket_response        — draft responses for 3 tickets
  inbox_triage           — full 8-ticket inbox
  churn_prevention       — 10 high-value at-risk customers
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from .graders import (
    episode_score_easy, episode_score_medium,
    episode_score_hard, episode_score_expert,
    grade_classify, grade_response,
    grade_hard_classify, grade_hard_route,
    grade_churn_classify, grade_churn_route,
    _clamp,
)
from .models import (
    ActionType, ChurnRisk, ClassificationPayload,
    Ticket, TriageAction, TriageObservation, TriageState,
)
from .task_data import TASKS_META

_EPISODE_SCORERS = {
    "ticket_classification": episode_score_easy,
    "ticket_response": episode_score_medium,
    "inbox_triage": episode_score_hard,
    "churn_prevention": episode_score_expert,
}

_SKIP_PENALTY = -0.01
_OVERTIME_MULT = 0.90


class CustomerSupportTriageEnv(Environment[TriageAction, TriageObservation, TriageState]):
    """
    OpenEnv-compliant Customer Support Triage Environment.
    Extends openenv-core Environment base class with typed generics.
    Supports concurrent sessions via session isolation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "ticket_classification"):
        super().__init__()
        if task_name not in TASKS_META:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Choose from: {list(TASKS_META.keys())}"
            )
        self._task_name = task_name
        self._meta = TASKS_META[task_name]
        self._scorer = _EPISODE_SCORERS[task_name]
        self._reset_state()

    def _reset_state(self):
        self._episode_id: str = str(uuid.uuid4())
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._agent_actions: List[Dict[str, Any]] = []
        self._completed_ids: List[str] = []
        self._tickets: List[Ticket] = [Ticket(**t) for t in self._meta["tickets"]]

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> TriageObservation:
        """Reset environment. Returns initial observation."""
        self._reset_state()
        if episode_id:
            self._episode_id = episode_id
        return self._build_obs()

    def step(self, action: TriageAction) -> TriageObservation:
        """Apply action. Returns observation with reward embedded."""
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")

        self._step_count += 1
        reward_val, reason = self._grade(action)

        # Record action
        self._agent_actions.append(self._action_to_dict(action))

        # Mark completed
        atype = str(action.action_type.value if hasattr(action.action_type, "value") else action.action_type)
        tid = action.ticket_id
        if self._task_name == "ticket_classification":
            if atype == "classify" and tid not in self._completed_ids:
                self._completed_ids.append(tid)
        else:
            if atype in {"respond", "escalate", "archive", "close"} and tid not in self._completed_ids:
                self._completed_ids.append(tid)

        self._cumulative_reward += reward_val if reward_val > 0 else reward_val

        # Check done
        all_ids = [t.id for t in self._tickets]
        all_done = set(all_ids).issubset(set(self._completed_ids))
        max_reached = self._step_count >= self._meta["max_steps"]
        self._done = all_done or max_reached

        obs = self._build_obs()

        # Embed reward and done into observation (openenv-core convention)
        obs.reward = float(round(reward_val, 4))
        obs.done = self._done

        if self._done:
            score = self._scorer(self._agent_actions)
            if max_reached and not all_done:
                score = _clamp(score * _OVERTIME_MULT)
            obs.metadata["episode_score"] = score
            obs.metadata["episode_id"] = self._episode_id
            obs.metadata["steps_taken"] = self._step_count
            obs.metadata["tickets_completed"] = len(self._completed_ids)
            obs.metadata["tickets_total"] = len(all_ids)
            obs.score_so_far = score

        return obs

    @property
    def state(self) -> TriageState:
        """Return full internal state snapshot."""
        score = self._scorer(self._agent_actions) if self._done else _clamp(
            self._scorer(self._agent_actions)
        )
        return TriageState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            max_steps=self._meta["max_steps"],
            tickets=list(self._tickets),
            ground_truth=self._meta["gt"],
            agent_actions=copy.deepcopy(self._agent_actions),
            cumulative_reward=round(self._cumulative_reward, 4),
            episode_score=score,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _grade(self, action: TriageAction):
        atype = str(action.action_type.value if hasattr(action.action_type, "value") else action.action_type)
        tid = action.ticket_id
        all_ids = [t.id for t in self._tickets]
        prior = [a for a in self._agent_actions if a.get("ticket_id") == tid]
        cls_done   = any(a.get("action_type") == "classify" for a in prior)
        route_done = any(a.get("action_type") in {"escalate","respond","archive","close"} for a in prior)

        if tid not in all_ids:
            return 0.01, f"Unknown ticket '{tid}'"

        if atype == "skip":
            return _SKIP_PENALTY, "SKIP — small penalty"

        task = self._task_name

        # ── easy ──
        if task == "ticket_classification":
            if tid in self._completed_ids:
                return 0.01, "Already classified"
            if atype != "classify" or not action.classification:
                return 0.01, "Must use classify action"
            cls = action.classification
            d = str(cls.department.value if hasattr(cls.department, "value") else cls.department)
            u = str(cls.urgency.value if hasattr(cls.urgency, "value") else cls.urgency)
            raw, reason = grade_classify(tid, d, u)
            return _clamp(raw * 0.99), reason  # scale to per-step

        # ── medium ──
        if task == "ticket_response":
            if tid in self._completed_ids:
                return 0.01, "Already responded"
            if atype != "respond" or not action.response:
                return 0.01, "Must use respond action"
            raw, reason = grade_response(tid, action.response.response_text)
            return _clamp(raw * 0.99), reason

        # ── hard ──
        if task == "inbox_triage":
            if atype == "classify":
                if cls_done:
                    return 0.01, "Already classified"
                if not action.classification:
                    return 0.01, "No classification payload"
                cls = action.classification
                d = str(cls.department.value if hasattr(cls.department, "value") else cls.department)
                u = str(cls.urgency.value if hasattr(cls.urgency, "value") else cls.urgency)
                raw, reason = grade_hard_classify(tid, d, u)
                n = len(self._tickets)
                return _clamp(0.40 * raw * (1/n)), reason
            if atype in {"escalate","archive","respond","close"}:
                if route_done:
                    return 0.01, "Already routed"
                resp_text = action.response.response_text if action.response else ""
                esc_reason = action.escalation.reason if action.escalation else ""
                raw, reason = grade_hard_route(tid, atype, resp_text, esc_reason)
                n = len(self._tickets)
                penalty = 0.7 if not cls_done else 1.0
                return _clamp(0.60 * raw * penalty * (1/n)), reason
            return 0.01, f"Unhandled action '{atype}'"

        # ── expert ──
        if task == "churn_prevention":
            if atype == "classify":
                if cls_done:
                    return 0.01, "Already classified"
                if not action.classification:
                    return 0.01, "No classification payload"
                cls = action.classification
                d = str(cls.department.value if hasattr(cls.department, "value") else cls.department)
                u = str(cls.urgency.value if hasattr(cls.urgency, "value") else cls.urgency)
                raw, reason = grade_churn_classify(tid, d, u)
                n = len(self._tickets)
                return _clamp(0.20 * raw * (1/n)), reason
            if atype in {"escalate","archive","respond","close"}:
                if route_done:
                    return 0.01, "Already routed"
                resp_text = action.response.response_text if action.response else ""
                esc_reason = action.escalation.reason if action.escalation else ""
                raw, reason = grade_churn_route(tid, atype, resp_text, esc_reason)
                n = len(self._tickets)
                penalty = 0.7 if not cls_done else 1.0
                return _clamp(0.80 * raw * penalty * (1/n)), reason
            return 0.01, f"Unhandled action '{atype}'"

        return 0.01, "Unknown task"

    def _build_obs(self) -> TriageObservation:
        all_ids = [t.id for t in self._tickets]
        pending = [i for i in all_ids if i not in self._completed_ids]
        score = _clamp(self._scorer(self._agent_actions))
        desc = self._meta["description"]
        return TriageObservation(
            task_name=self._task_name,
            task_description=desc,
            step_count=self._step_count,
            max_steps=self._meta["max_steps"],
            tickets=list(self._tickets),
            pending_ticket_ids=pending,
            completed_ticket_ids=list(self._completed_ids),
            actions_history=copy.deepcopy(self._agent_actions[-10:]),
            score_so_far=score,
            done=self._done,
            reward=None,
            metadata={"episode_id": self._episode_id},
        )

    @staticmethod
    def _action_to_dict(action: TriageAction) -> Dict[str, Any]:
        atype = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        rec: Dict[str, Any] = {"action_type": atype, "ticket_id": action.ticket_id}
        if action.classification:
            cls = action.classification
            rec["department"] = cls.department.value if hasattr(cls.department, "value") else str(cls.department)
            rec["urgency"] = cls.urgency.value if hasattr(cls.urgency, "value") else str(cls.urgency)
        if action.response:
            rec["response_text"] = action.response.response_text
        if action.escalation:
            rec["escalation_reason"] = action.escalation.reason
            if action.escalation.churn_risk:
                rec["churn_risk"] = str(action.escalation.churn_risk.value
                                        if hasattr(action.escalation.churn_risk, "value")
                                        else action.escalation.churn_risk)
        return rec
