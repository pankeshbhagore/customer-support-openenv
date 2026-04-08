from .easy import TASK_NAME as EASY_TASK_NAME, get_initial_observation as easy_obs, grade_action as easy_grade, compute_episode_score as easy_score, MAX_STEPS as EASY_MAX_STEPS
from .medium import TASK_NAME as MEDIUM_TASK_NAME, get_initial_observation as medium_obs, grade_action as medium_grade, compute_episode_score as medium_score, MAX_STEPS as MEDIUM_MAX_STEPS
from .hard import TASK_NAME as HARD_TASK_NAME, get_initial_observation as hard_obs, grade_action as hard_grade, compute_episode_score as hard_score, MAX_STEPS as HARD_MAX_STEPS

TASKS = {
    "ticket_classification": {
        "name": EASY_TASK_NAME,
        "difficulty": "easy",
        "get_obs": easy_obs,
        "grade": easy_grade,
        "episode_score": easy_score,
        "max_steps": EASY_MAX_STEPS,
    },
    "ticket_response": {
        "name": MEDIUM_TASK_NAME,
        "difficulty": "medium",
        "get_obs": medium_obs,
        "grade": medium_grade,
        "episode_score": medium_score,
        "max_steps": MEDIUM_MAX_STEPS,
    },
    "inbox_triage": {
        "name": HARD_TASK_NAME,
        "difficulty": "hard",
        "get_obs": hard_obs,
        "grade": hard_grade,
        "episode_score": hard_score,
        "max_steps": HARD_MAX_STEPS,
    },
}

__all__ = ["TASKS"]
