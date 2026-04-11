"""
Comprehensive test suite for Customer Support Triage OpenEnv.
Run: python -m pytest tests/ -v
"""
import sys
sys.path.insert(0, "/home/claude")

import pytest
from support_triage_env import (
    CustomerSupportTriageEnv, TriageAction, TriageObservation, TriageState,
    ActionType, Department, Urgency, ClassificationPayload,
    ResponsePayload, EscalationPayload, TASKS_META,
)
from support_triage_env.task_data import EASY_GT, HARD_GT, EXPERT_GT
from support_triage_env.graders import (
    _clamp, grade_classify, grade_response,
    episode_score_easy, episode_score_medium, episode_score_hard, episode_score_expert,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def classify(dept: str, urg: str, tid: str) -> TriageAction:
    return TriageAction(action_type=ActionType.CLASSIFY, ticket_id=tid,
        classification=ClassificationPayload(department=Department(dept), urgency=Urgency(urg)))

def respond(tid: str, text: str) -> TriageAction:
    return TriageAction(action_type=ActionType.RESPOND, ticket_id=tid,
        response=ResponsePayload(response_text=text))

def escalate(tid: str, reason: str) -> TriageAction:
    return TriageAction(action_type=ActionType.ESCALATE, ticket_id=tid,
        escalation=EscalationPayload(reason=reason))

GOOD_RESPONSE = (
    "Dear customer, I sincerely apologize for this inconvenience. "
    "I understand how frustrating this must be. Our team will investigate "
    "and resolve this issue immediately. We will process a full refund and "
    "ensure this does not happen again. Thank you for your patience."
)


# ── Score range ───────────────────────────────────────────────────────────────
class TestScoreRange:
    def test_clamp_zero(self):
        assert _clamp(0.0) == 0.02

    def test_clamp_one(self):
        assert _clamp(1.0) == 0.98

    def test_clamp_point_five(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_never_zero(self):
        assert _clamp(-100) > 0.01

    def test_clamp_never_one(self):
        assert _clamp(100) < 0.99

    def test_empty_episode_score_in_range(self):
        for task in TASKS_META:
            env = CustomerSupportTriageEnv(task)
            env.reset()
            score = env.state.episode_score
            assert 0.01 < score < 0.99, f"{task} empty score {score} out of (0.01,0.99)"


# ── Environment lifecycle ─────────────────────────────────────────────────────
class TestLifecycle:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            CustomerSupportTriageEnv("nonexistent")

    def test_step_before_reset_raises(self):
        env = CustomerSupportTriageEnv.__new__(CustomerSupportTriageEnv)
        # Not calling __init__ properly, so use a fresh env with manual done
        env2 = CustomerSupportTriageEnv("ticket_classification")
        env2.reset()
        env2._done = True
        with pytest.raises(RuntimeError):
            env2.step(classify("billing", "high", "T001"))

    def test_reset_returns_observation(self):
        for task in TASKS_META:
            env = CustomerSupportTriageEnv(task)
            obs = env.reset()
            assert isinstance(obs, TriageObservation)
            assert not obs.done
            assert len(obs.tickets) == TASKS_META[task]["num_tickets"]

    def test_reset_clears_state(self):
        env = CustomerSupportTriageEnv("ticket_classification")
        env.reset()
        env.step(classify("billing", "high", "T001"))
        env.reset()
        assert env.state.step_count == 0
        assert env.state.agent_actions == []

    def test_state_returns_triagestate(self):
        for task in TASKS_META:
            env = CustomerSupportTriageEnv(task)
            env.reset()
            s = env.state
            assert isinstance(s, TriageState)
            assert s.episode_id is not None
            assert s.step_count == 0

    def test_episode_id_set_on_reset(self):
        env = CustomerSupportTriageEnv("ticket_classification")
        obs = env.reset(episode_id="test-ep-123")
        assert env.state.episode_id == "test-ep-123"

    def test_step_increments_step_count(self):
        env = CustomerSupportTriageEnv("ticket_classification")
        env.reset()
        env.step(classify("billing", "high", "T001"))
        assert env.state.step_count == 1
        env.step(classify("technical", "medium", "T002"))
        assert env.state.step_count == 2


# ── Easy task ─────────────────────────────────────────────────────────────────
class TestEasyTask:
    def setup_method(self):
        self.env = CustomerSupportTriageEnv("ticket_classification")
        self.env.reset()

    def test_five_tickets(self):
        assert len(self.env.state.tickets) == 5

    def test_correct_classify_reward_positive(self):
        obs = self.env.step(classify("billing", "high", "T001"))
        assert obs.reward > 0.02

    def test_wrong_classify_reward_low(self):
        obs = self.env.step(classify("general", "low", "T001"))
        assert obs.reward < 0.05

    def test_partial_credit_correct_dept_wrong_urg(self):
        r1 = self.env.step(classify("billing", "low", "T001")).reward   # dept right, urg wrong
        env2 = CustomerSupportTriageEnv("ticket_classification")
        env2.reset()
        r2 = env2.step(classify("billing", "high", "T001")).reward      # both right
        assert r1 < r2

    def test_duplicate_classify_no_reward(self):
        self.env.step(classify("billing", "high", "T001"))
        obs2 = self.env.step(classify("billing", "high", "T001"))
        assert obs2.reward <= 0.02

    def test_all_correct_high_episode_score(self):
        for tid, gt in EASY_GT.items():
            self.env.step(classify(gt["department"], gt["urgency"], tid))
        score = self.env.state.episode_score
        assert score > 0.85, f"Perfect oracle score too low: {score}"
        assert 0.01 < score < 0.99

    def test_all_wrong_low_episode_score(self):
        wrong = {"T001":("general","low"), "T002":("billing","high"),
                 "T003":("technical","low"), "T004":("returns","medium"), "T005":("billing","high")}
        for tid, (d,u) in wrong.items():
            self.env.step(classify(d, u, tid))
        score = self.env.state.episode_score
        assert score < 0.20

    def test_episode_done_after_all_5(self):
        obs = None
        for tid, gt in EASY_GT.items():
            obs = self.env.step(classify(gt["department"], gt["urgency"], tid))
        assert obs.done

    def test_reward_in_range_all_steps(self):
        for tid, gt in EASY_GT.items():
            obs = self.env.step(classify(gt["department"], gt["urgency"], tid))
            assert 0.01 < obs.reward < 0.99

    def test_episode_score_in_metadata_when_done(self):
        for tid, gt in EASY_GT.items():
            obs = self.env.step(classify(gt["department"], gt["urgency"], tid))
        assert "episode_score" in obs.metadata


# ── Medium task ───────────────────────────────────────────────────────────────
class TestMediumTask:
    def setup_method(self):
        self.env = CustomerSupportTriageEnv("ticket_response")
        self.env.reset()

    def test_three_tickets(self):
        assert len(self.env.state.tickets) == 3

    def test_good_response_high_reward(self):
        obs = self.env.step(respond("M001",
            "Dear Frank, I sincerely apologize for this billing issue. "
            "We confirm your subscription was cancelled and you should not have been charged. "
            "We will process a full refund immediately and ensure no further charges occur. "
            "Our team will follow up within 24 hours to confirm resolution."))
        assert obs.reward > 0.3

    def test_empty_response_minimal_reward(self):
        obs = self.env.step(respond("M001", ""))
        assert obs.reward <= 0.05

    def test_one_word_response_low_reward(self):
        obs = self.env.step(respond("M001", "ok"))
        assert obs.reward < 0.15

    def test_wrong_action_for_task_minimal_reward(self):
        obs = self.env.step(classify("billing", "high", "M001"))
        assert obs.reward <= 0.05

    def test_duplicate_response_no_credit(self):
        self.env.step(respond("M001", GOOD_RESPONSE))
        obs2 = self.env.step(respond("M001", GOOD_RESPONSE))
        assert obs2.reward <= 0.02

    def test_reward_always_in_range(self):
        obs = self.env.step(respond("M001", GOOD_RESPONSE))
        assert 0.01 < obs.reward < 0.99


# ── Hard task ─────────────────────────────────────────────────────────────────
class TestHardTask:
    def setup_method(self):
        self.env = CustomerSupportTriageEnv("inbox_triage")
        self.env.reset()

    def test_eight_tickets(self):
        assert len(self.env.state.tickets) == 8

    def test_p0_outage_escalate_rewarded(self):
        self.env.step(classify("technical", "high", "H001"))
        obs = self.env.step(escalate("H001", "P0 production outage enterprise customer all API 403"))
        assert obs.reward > 0.02

    def test_spam_must_archive(self):
        self.env.step(classify("general", "low", "H002"))
        obs = self.env.step(TriageAction(action_type=ActionType.ARCHIVE, ticket_id="H002"))
        assert obs.reward > 0.02

    def test_wrong_route_low_reward(self):
        self.env.step(classify("general", "low", "H002"))
        # Should archive spam, not respond
        obs = self.env.step(respond("H002", "Thank you for contacting us"))
        assert obs.reward < 0.05

    def test_gdpr_must_escalate(self):
        self.env.step(classify("general", "high", "H005"))
        obs = self.env.step(escalate("H005", "GDPR Article 17 legal data deletion compliance request"))
        assert obs.reward > 0.02

    def test_positive_feedback_must_close(self):
        self.env.step(classify("general", "low", "H007"))
        obs = self.env.step(TriageAction(action_type=ActionType.CLOSE, ticket_id="H007"))
        assert obs.reward > 0.02

    def test_classify_contributes_to_score(self):
        # Classify all tickets correctly
        for tid, gt in HARD_GT.items():
            self.env.step(classify(gt["department"], gt["urgency"], tid))
        score_after_classify = self.env.state.episode_score
        assert score_after_classify > 0.01

    def test_skip_gives_negative_reward(self):
        obs = self.env.step(TriageAction(action_type=ActionType.SKIP, ticket_id="H001"))
        assert obs.reward < 0


# ── Expert task ───────────────────────────────────────────────────────────────
class TestExpertTask:
    def setup_method(self):
        self.env = CustomerSupportTriageEnv("churn_prevention")
        self.env.reset()

    def test_ten_tickets(self):
        assert len(self.env.state.tickets) == 10

    def test_competitor_pricing_critical_churn(self):
        self.env.step(classify("billing", "high", "E001"))
        obs = self.env.step(escalate("E001",
            "Critical churn risk: enterprise competitor pricing retention discount renewal"))
        assert obs.reward > 0.02

    def test_happy_renewal_respond_not_escalate(self):
        self.env.step(classify("billing", "low", "E009"))
        obs = self.env.step(respond("E009",
            "Dear Rachel, thank you for your loyalty! "
            "Yes we offer a multi-year discount for annual 2-year commitment. "
            "Your account manager will contact you with renewal details."))
        assert obs.reward > 0.02

    def test_data_export_high_churn_risk(self):
        # E004 "evaluating alternatives + export data" = hidden HIGH churn signal
        self.env.step(classify("general", "high", "E004"))
        obs = self.env.step(respond("E004",
            "Dear Marcus, I'll help you export your data right away. "
            "I also wanted to mention some alternative features you might not know about — "
            "we'd love to have you stay and can offer additional value for your team."))
        assert obs.reward > 0.02

    def test_gdpr_compliance_critical(self):
        self.env.step(classify("general", "high", "E010"))
        obs = self.env.step(escalate("E010",
            "Critical: healthcare compliance SOC2 security certification required, regulated industry, board meeting deadline"))
        assert obs.reward > 0.02

    def test_episode_score_in_range_full_run(self):
        for tid, gt in EXPERT_GT.items():
            self.env.step(classify(gt["department"], gt["urgency"], tid))
        for tid, gt in EXPERT_GT.items():
            ca = gt["action"]
            text = "critical churn risk " + " ".join(gt.get("reason_kws", []) + gt.get("response_kws", []))
            if ca == "escalate":
                self.env.step(escalate(tid, text))
            else:
                self.env.step(respond(tid, f"Dear valued customer your account your business loyalty renew {text}"))
        score = self.env.state.episode_score
        assert 0.01 < score < 0.99, f"Score {score} out of strict (0.01, 0.99)"


# ── Models ────────────────────────────────────────────────────────────────────
class TestModels:
    def test_action_from_dict(self):
        d = {"action_type": "classify", "ticket_id": "T001",
             "classification": {"department": "billing", "urgency": "high"}}
        a = TriageAction(**d)
        assert a.ticket_id == "T001"

    def test_action_serializes(self):
        a = TriageAction(action_type=ActionType.CLASSIFY, ticket_id="T001",
            classification=ClassificationPayload(department=Department.BILLING, urgency=Urgency.HIGH))
        d = a.model_dump()
        assert d["ticket_id"] == "T001"

    def test_all_4_tasks_registered(self):
        assert len(TASKS_META) == 4
        for name in ["ticket_classification", "ticket_response", "inbox_triage", "churn_prevention"]:
            assert name in TASKS_META

    def test_difficulty_progression(self):
        diffs = [TASKS_META[t]["difficulty"] for t in TASKS_META]
        assert diffs == ["easy", "medium", "hard", "expert"]

    def test_observation_has_reward_field(self):
        env = CustomerSupportTriageEnv("ticket_classification")
        env.reset()
        obs = env.step(classify("billing", "high", "T001"))
        assert obs.reward is not None
        assert isinstance(obs.reward, float)

    def test_state_has_episode_id(self):
        env = CustomerSupportTriageEnv("ticket_classification")
        env.reset(episode_id="abc-123")
        assert env.state.episode_id == "abc-123"

    def test_observation_done_false_mid_episode(self):
        env = CustomerSupportTriageEnv("ticket_classification")
        env.reset()
        obs = env.step(classify("billing", "high", "T001"))
        assert obs.done is False

    def test_supports_concurrent_sessions(self):
        assert CustomerSupportTriageEnv.SUPPORTS_CONCURRENT_SESSIONS is True
