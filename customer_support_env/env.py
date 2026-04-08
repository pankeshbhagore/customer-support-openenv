"""
CustomerSupportEnv — OpenEnv-compliant environment for customer support triage.

Implements:
  reset()  → Observation
  step()   → StepResult (observation, reward, done, info)
  state()  → EnvState
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .models import (
    Action,
    ActionType,
    EnvState,
    Observation,
    Reward,
    StepResult,
    Ticket,
)
from .tasks import TASKS

# Penalty for burning steps without progress
_SKIP_PENALTY = -0.01
# Penalty multiplier for exceeding max_steps (applied to final score)
_OVERTIME_PENALTY = 0.90


class CustomerSupportEnv:
    """
    OpenEnv-compliant customer support triage environment.

    Supports three tasks:
      - ticket_classification  (easy)
      - ticket_response        (medium)
      - inbox_triage           (hard)
    """

    SUPPORTED_TASKS = list(TASKS.keys())

    def __init__(self, task_name: str = "ticket_classification"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {self.SUPPORTED_TASKS}"
            )
        self._task_name = task_name
        self._task = TASKS[task_name]
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._agent_actions: List[Dict[str, Any]] = []
        self._completed_ids: List[str] = []
        self._obs: Optional[Observation] = None

    # ─────────────────────────── OpenEnv Interface ───────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns initial observation."""
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._agent_actions = []
        self._completed_ids = []
        self._obs = self._task["get_obs"](step_count=0)
        return copy.deepcopy(self._obs)

    def step(self, action: Action) -> StepResult:
        """
        Apply action to environment.
        Returns (observation, reward, done, info).
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before step()")

        if self._done:
            raise RuntimeError("Episode is done — call reset() to start a new episode")

        self._step_count += 1
        info: Dict[str, Any] = {"step": self._step_count}

        # ── Validate action ──
        if not isinstance(action, Action):
            try:
                action = Action(**action) if isinstance(action, dict) else action
            except Exception as exc:
                reward = Reward(value=0.0, reason=f"Invalid action: {exc}")
                obs = self._build_obs()
                return StepResult(observation=obs, reward=reward, done=self._done, info=info)

        # ── Grade the action ──
        reward_value, reason, breakdown = self._task["grade"](
            action, self._completed_ids, self._agent_actions
        )

        # ── Handle skip action ──
        if action.action_type == ActionType.SKIP:
            reward_value = _SKIP_PENALTY
            reason = "SKIP action — small penalty applied"

        # ── Record action ──
        action_record = self._action_to_dict(action)
        self._agent_actions.append(action_record)

        # ── Update completed tickets ──
        # A ticket is "completed" when a routing action (not classify) is applied
        # For easy task, classify IS the completion action
        tid = action.ticket_id
        atype_val = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        if self._task_name == "ticket_classification":
            if atype_val == "classify" and tid not in self._completed_ids:
                self._completed_ids.append(tid)
        else:
            if atype_val in {"respond", "escalate", "archive", "close"} and tid not in self._completed_ids:
                self._completed_ids.append(tid)

        # ── Accumulate reward ──
        reward_value = max(0.0, min(1.0, reward_value)) if reward_value >= 0 else reward_value
        self._cumulative_reward += reward_value

        reward = Reward(
            value=float(round(max(-1.0, min(1.0, reward_value)), 4)),
            reason=reason,
            breakdown=breakdown,
        )

        # ── Check done conditions ──
        all_tickets = self._get_all_ticket_ids()
        all_done = set(all_tickets).issubset(set(self._completed_ids))
        max_steps_reached = self._step_count >= self._task["max_steps"]

        self._done = all_done or max_steps_reached
        info["all_tickets_done"] = all_done
        info["max_steps_reached"] = max_steps_reached

        if self._done:
            episode_score = self._task["episode_score"](self._agent_actions)
            # Small penalty for running out of steps without finishing
            if max_steps_reached and not all_done:
                episode_score = round(episode_score * _OVERTIME_PENALTY, 4)
            info["episode_score"] = episode_score
            info["completed_tickets"] = len(self._completed_ids)
            info["total_tickets"] = len(all_tickets)

        obs = self._build_obs()
        return StepResult(observation=obs, reward=reward, done=self._done, info=info)

    def state(self) -> EnvState:
        """Return full internal state (for debugging / evaluation)."""
        from .data import EASY_GROUND_TRUTH, MEDIUM_RESPONSE_CRITERIA, HARD_GROUND_TRUTH
        gt_map = {
            "ticket_classification": EASY_GROUND_TRUTH,
            "ticket_response": MEDIUM_RESPONSE_CRITERIA,
            "inbox_triage": HARD_GROUND_TRUTH,
        }
        episode_score = self._task["episode_score"](self._agent_actions) if self._done else 0.0
        return EnvState(
            task_name=self._task_name,
            step_count=self._step_count,
            max_steps=self._task["max_steps"],
            tickets=self._obs.tickets if self._obs else [],
            ground_truth=gt_map[self._task_name],
            agent_actions=copy.deepcopy(self._agent_actions),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            episode_score=episode_score,
        )

    # ─────────────────────────── Helpers ─────────────────────────────────────

    def _build_obs(self) -> Observation:
        all_ids = self._get_all_ticket_ids()
        pending = [tid for tid in all_ids if tid not in self._completed_ids]
        score = self._task["episode_score"](self._agent_actions)
        self._obs = Observation(
            task_name=self._task_name,
            task_description=self._obs.task_description if self._obs else "",
            step_count=self._step_count,
            max_steps=self._task["max_steps"],
            tickets=self._obs.tickets if self._obs else [],
            pending_ticket_ids=pending,
            completed_ticket_ids=list(self._completed_ids),
            actions_history=copy.deepcopy(self._agent_actions[-10:]),  # last 10
            score_so_far=round(score, 4),
            done=self._done,
        )
        return copy.deepcopy(self._obs)

    def _get_all_ticket_ids(self) -> List[str]:
        if self._obs is None:
            return []
        return [t.id for t in self._obs.tickets]

    @staticmethod
    def _action_to_dict(action: Action) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "action_type": action.action_type.value
            if hasattr(action.action_type, "value") else action.action_type,
            "ticket_id": action.ticket_id,
        }
        if action.classification:
            record["department"] = (
                action.classification.department.value
                if hasattr(action.classification.department, "value")
                else action.classification.department
            )
            record["urgency"] = (
                action.classification.urgency.value
                if hasattr(action.classification.urgency, "value")
                else action.classification.urgency
            )
        if action.response:
            record["response_text"] = action.response.response_text
        if action.escalation:
            record["escalation_reason"] = action.escalation.reason
        return record
