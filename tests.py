"""
Test suite for Customer Support Triage OpenEnv.
Run: python -m pytest tests.py -v
"""
import pytest

from customer_support_env import (
    Action,
    ActionType,
    ClassificationPayload,
    CustomerSupportEnv,
    Department,
    EscalationPayload,
    ResponsePayload,
    Urgency,
)
from customer_support_env.tasks import TASKS


# ──────────────────────────────────────────────────────────────────────────────
# Environment init
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_task_raises():
    with pytest.raises(ValueError):
        CustomerSupportEnv(task_name="nonexistent")


def test_step_before_reset_raises():
    env = CustomerSupportEnv()
    with pytest.raises(RuntimeError):
        env.step(Action(action_type=ActionType.SKIP, ticket_id="T001"))


# ──────────────────────────────────────────────────────────────────────────────
# EASY task
# ──────────────────────────────────────────────────────────────────────────────

class TestEasyTask:
    def setup_method(self):
        self.env = CustomerSupportEnv(task_name="ticket_classification")
        self.obs = self.env.reset()

    def test_reset_returns_5_tickets(self):
        assert len(self.obs.tickets) == 5

    def test_reset_all_pending(self):
        assert len(self.obs.pending_ticket_ids) == 5
        assert self.obs.completed_ticket_ids == []

    def test_correct_classify_gives_full_reward(self):
        action = Action(
            action_type=ActionType.CLASSIFY,
            ticket_id="T001",
            classification=ClassificationPayload(
                department=Department.BILLING, urgency=Urgency.HIGH
            ),
        )
        result = self.env.step(action)
        assert result.reward.value == 1.0

    def test_partial_credit_wrong_urgency(self):
        action = Action(
            action_type=ActionType.CLASSIFY,
            ticket_id="T001",
            classification=ClassificationPayload(
                department=Department.BILLING, urgency=Urgency.LOW  # wrong urgency
            ),
        )
        result = self.env.step(action)
        assert result.reward.value == 0.5  # only dept correct

    def test_no_reward_for_duplicate_classify(self):
        action = Action(
            action_type=ActionType.CLASSIFY,
            ticket_id="T001",
            classification=ClassificationPayload(
                department=Department.BILLING, urgency=Urgency.HIGH
            ),
        )
        self.env.step(action)
        result2 = self.env.step(action)
        assert result2.reward.value == 0.0

    def test_episode_done_after_all_5(self):
        data = [
            ("T001", "billing", "high"),
            ("T002", "technical", "medium"),
            ("T003", "returns", "medium"),
            ("T004", "technical", "high"),
            ("T005", "general", "low"),
        ]
        for i, (tid, dept, urg) in enumerate(data):
            action = Action(
                action_type=ActionType.CLASSIFY,
                ticket_id=tid,
                classification=ClassificationPayload(
                    department=Department(dept), urgency=Urgency(urg)
                ),
            )
            result = self.env.step(action)
            if i < 4:
                assert not result.done
            else:
                assert result.done

    def test_perfect_episode_score_is_1(self):
        data = [
            ("T001", "billing", "high"),
            ("T002", "technical", "medium"),
            ("T003", "returns", "medium"),
            ("T004", "technical", "high"),
            ("T005", "general", "low"),
        ]
        for tid, dept, urg in data:
            self.env.step(
                Action(
                    action_type=ActionType.CLASSIFY,
                    ticket_id=tid,
                    classification=ClassificationPayload(
                        department=Department(dept), urgency=Urgency(urg)
                    ),
                )
            )
        state = self.env.state()
        assert state.episode_score == 1.0

    def test_all_wrong_score_is_0(self):
        data = [
            ("T001", "general", "low"),
            ("T002", "billing", "high"),
            ("T003", "technical", "low"),
            ("T004", "returns", "medium"),
            ("T005", "technical", "high"),
        ]
        for tid, dept, urg in data:
            self.env.step(
                Action(
                    action_type=ActionType.CLASSIFY,
                    ticket_id=tid,
                    classification=ClassificationPayload(
                        department=Department(dept), urgency=Urgency(urg)
                    ),
                )
            )
        state = self.env.state()
        assert state.episode_score == 0.0

    def test_step_after_done_raises(self):
        data = [
            ("T001", "billing", "high"),
            ("T002", "technical", "medium"),
            ("T003", "returns", "medium"),
            ("T004", "technical", "high"),
            ("T005", "general", "low"),
        ]
        for tid, dept, urg in data:
            self.env.step(
                Action(
                    action_type=ActionType.CLASSIFY, ticket_id=tid,
                    classification=ClassificationPayload(
                        department=Department(dept), urgency=Urgency(urg)
                    ),
                )
            )
        with pytest.raises(RuntimeError):
            self.env.step(Action(action_type=ActionType.SKIP, ticket_id="T001"))


# ──────────────────────────────────────────────────────────────────────────────
# MEDIUM task
# ──────────────────────────────────────────────────────────────────────────────

class TestMediumTask:
    def setup_method(self):
        self.env = CustomerSupportEnv(task_name="ticket_response")
        self.obs = self.env.reset()

    def test_reset_returns_3_tickets(self):
        assert len(self.obs.tickets) == 3

    def test_good_response_scores_high(self):
        action = Action(
            action_type=ActionType.RESPOND,
            ticket_id="M001",
            response=ResponsePayload(
                response_text=(
                    "Dear Frank, I sincerely apologize for this billing issue. "
                    "We can confirm your subscription was cancelled and you should not have been "
                    "charged. We will process a full refund immediately and ensure no further "
                    "charges occur on your account. Our team has been notified and will follow up."
                )
            ),
        )
        result = self.env.step(action)
        assert result.reward.value >= 0.7

    def test_empty_response_scores_zero(self):
        action = Action(
            action_type=ActionType.RESPOND,
            ticket_id="M001",
            response=ResponsePayload(response_text=""),
        )
        result = self.env.step(action)
        assert result.reward.value == 0.0

    def test_wrong_action_type_scores_zero(self):
        action = Action(action_type=ActionType.CLASSIFY, ticket_id="M001",
                        classification=ClassificationPayload(
                            department=Department.BILLING, urgency=Urgency.HIGH))
        result = self.env.step(action)
        assert result.reward.value == 0.0

    def test_reward_in_valid_range(self):
        action = Action(
            action_type=ActionType.RESPOND,
            ticket_id="M002",
            response=ResponsePayload(
                response_text=(
                    "Dear Grace, I understand the urgency of your export issue before your deadline. "
                    "There is a known issue with the export function — our engineering team is investigating. "
                    "As a workaround, please try selecting fewer columns. We will prioritize this fix."
                )
            ),
        )
        result = self.env.step(action)
        assert 0.0 <= result.reward.value <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# HARD task
# ──────────────────────────────────────────────────────────────────────────────

class TestHardTask:
    def setup_method(self):
        self.env = CustomerSupportEnv(task_name="inbox_triage")
        self.obs = self.env.reset()

    def test_reset_returns_8_tickets(self):
        assert len(self.obs.tickets) == 8

    def test_p0_outage_must_be_escalated(self):
        # classify first
        self.env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id="H001",
            classification=ClassificationPayload(department=Department.TECHNICAL, urgency=Urgency.HIGH)
        ))
        # escalate
        result = self.env.step(Action(
            action_type=ActionType.ESCALATE, ticket_id="H001",
            escalation=EscalationPayload(reason="P0 production outage for enterprise customer")
        ))
        assert result.reward.value > 0

    def test_spam_must_be_archived(self):
        self.env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id="H002",
            classification=ClassificationPayload(department=Department.GENERAL, urgency=Urgency.LOW)
        ))
        result = self.env.step(Action(action_type=ActionType.ARCHIVE, ticket_id="H002"))
        assert result.reward.value > 0

    def test_wrong_route_scores_zero(self):
        self.env.step(Action(
            action_type=ActionType.CLASSIFY, ticket_id="H002",
            classification=ClassificationPayload(department=Department.GENERAL, urgency=Urgency.LOW)
        ))
        # Should archive, but we respond — wrong routing
        result = self.env.step(Action(
            action_type=ActionType.RESPOND, ticket_id="H002",
            response=ResponsePayload(response_text="Thank you for your message")
        ))
        assert result.reward.value == 0.0

    def test_skip_gives_small_negative(self):
        result = self.env.step(Action(action_type=ActionType.SKIP, ticket_id="H001"))
        assert result.reward.value < 0

    def test_state_returns_all_fields(self):
        state = self.env.state()
        assert state.task_name == "inbox_triage"
        assert state.step_count == 0
        assert isinstance(state.ground_truth, dict)
        assert len(state.tickets) == 8


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_action_serialization(self):
        action = Action(
            action_type=ActionType.CLASSIFY,
            ticket_id="T001",
            classification=ClassificationPayload(
                department=Department.BILLING, urgency=Urgency.HIGH
            ),
        )
        d = action.model_dump()
        assert d["ticket_id"] == "T001"

    def test_action_from_dict(self):
        d = {
            "action_type": "classify",
            "ticket_id": "T001",
            "classification": {"department": "billing", "urgency": "high"},
        }
        action = Action(**d)
        assert action.ticket_id == "T001"

    def test_reward_range_enforced(self):
        from customer_support_env.models import Reward
        with pytest.raises(Exception):
            Reward(value=1.5, reason="too high")
        # negative skip penalty IS allowed
        r = Reward(value=-0.01, reason="skip")
        assert r.value == -0.01

    def test_all_tasks_registered(self):
        assert "ticket_classification" in TASKS
        assert "ticket_response" in TASKS
        assert "inbox_triage" in TASKS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
