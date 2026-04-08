"""
Inference Script — Customer Support Triage OpenEnv
===================================================
MANDATORY FORMAT
  [START] task=<task_name> env=customer_support_triage model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
  export HF_TOKEN=hf_...
  export API_BASE_URL=https://router.huggingface.co/v1   # or your endpoint
  export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
  python inference.py
"""
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

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
# Configuration — mandatory variable spec (OpenEnv Hackathon)
# ──────────────────────────────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")           # no default — must be set by user
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional: only needed with from_docker_image()

BENCHMARK   = "customer_support_triage"
MAX_STEPS   = 30
TEMPERATURE = 0.2

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support triage agent.
You must process support tickets by returning a JSON action object.

VALID ACTION TYPES:
  classify  — assign department + urgency to a ticket
  respond   — write a professional customer reply
  escalate  — escalate to human team (requires reason)
  archive   — mark as spam/irrelevant
  close     — mark as resolved

DEPARTMENTS: billing | technical | general | returns
URGENCY:     low | medium | high

JSON FORMAT (return ONLY valid JSON, no markdown fences):
{
  "action_type": "classify",
  "ticket_id": "T001",
  "classification": {"department": "billing", "urgency": "high"}
}

For respond:
{
  "action_type": "respond",
  "ticket_id": "T001",
  "response": {"response_text": "Dear customer, ..."}
}

For escalate:
{
  "action_type": "escalate",
  "ticket_id": "T001",
  "escalation": {"reason": "P0 production outage requires immediate engineering attention"}
}

For archive or close:
{"action_type": "archive", "ticket_id": "T001"}

Think carefully before each action. Return ONLY the JSON object.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def obs_to_prompt(obs_dict: Dict[str, Any]) -> str:
    """Convert observation dict to a concise prompt string for the LLM."""
    lines = [
        f"TASK: {obs_dict.get('task_name')}",
        f"DESCRIPTION: {obs_dict.get('task_description')}",
        f"Step {obs_dict.get('step_count')}/{obs_dict.get('max_steps')}",
        f"Score so far: {obs_dict.get('score_so_far', 0.0):.2f}",
        f"Pending tickets: {obs_dict.get('pending_ticket_ids')}",
        f"Completed tickets: {obs_dict.get('completed_ticket_ids')}",
        "",
        "TICKETS:",
    ]
    for t in obs_dict.get("tickets", []):
        lines.append(
            f"  [{t['id']}] Subject: {t['subject']}\n"
            f"       From: {t['customer_name']} <{t['customer_email']}>\n"
            f"       Body: {t['body'][:300]}"
        )
    if obs_dict.get("actions_history"):
        lines.append("\nRECENT ACTIONS:")
        for a in obs_dict["actions_history"][-5:]:
            lines.append(f"  {a}")
    lines.append("\nReturn the next JSON action:")
    return "\n".join(lines)


def call_llm(messages: List[Dict]) -> str:
    """Call the LLM and return raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> Optional[Action]:
    """Parse LLM JSON output into an Action object."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()
    try:
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return None


def run_episode(task_name: str) -> Dict[str, Any]:
    """Run one full episode for the given task. Returns result dict."""
    env = CustomerSupportEnv(task_name=task_name)
    obs = env.reset()
    obs_dict = obs.model_dump()

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    last_error: Optional[str] = None
    step = 0
    done = False
    episode_score = 0.0

    try:
        while not done and step < MAX_STEPS:
            step += 1
            user_prompt = obs_to_prompt(obs_dict)
            messages.append({"role": "user", "content": user_prompt})

            # Get LLM response
            raw = call_llm(messages)
            messages.append({"role": "assistant", "content": raw})

            # Parse action
            action = parse_action(raw)
            if action is None:
                last_error = f"JSON parse failed: {raw[:120]}"
                action_str = f"PARSE_ERROR"
                # Penalise with a skip action on first pending ticket
                pending = obs_dict.get("pending_ticket_ids", [])
                if pending:
                    action = Action(action_type=ActionType.SKIP, ticket_id=pending[0])
                else:
                    break
            else:
                last_error = None
                action_str = f"{action.action_type}:{action.ticket_id}"

            # Step environment
            result = env.step(action)
            obs_dict = result.observation.model_dump()
            reward_val = result.reward.value
            done = result.done
            rewards.append(reward_val)

            if done:
                episode_score = result.info.get("episode_score", 0.0)

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward_val:.2f} done={str(done).lower()} "
                f"error={last_error if last_error else 'null'}",
                flush=True,
            )

    except Exception as exc:
        last_error = str(exc)
        traceback.print_exc(file=sys.stderr)

    # Final episode score (compute even if loop exited early)
    state = env.state()
    episode_score = state.episode_score

    success = episode_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={episode_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task": task_name,
        "score": episode_score,
        "steps": step,
        "success": success,
        "rewards": rewards,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    task_name = os.getenv("TASK_NAME", "")
    if task_name and task_name in TASKS:
        tasks_to_run = [task_name]
    else:
        tasks_to_run = list(TASKS.keys())  # run all three

    results = []
    for task in tasks_to_run:
        result = run_episode(task)
        results.append(result)
        print("", flush=True)  # blank line between tasks

    # Summary
    print("=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(f"  {r['task']:30s}  score={r['score']:.4f}  steps={r['steps']}", flush=True)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  AVERAGE SCORE: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
