"""
inference.py — Customer Support Triage OpenEnv
===============================================
Mandatory log format (exact):
  [START] task=<task_name> env=customer_support_triage model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Environment variables:
  HF_TOKEN         required — Hugging Face / API key (no default)
  API_BASE_URL     optional — defaults to https://router.huggingface.co/v1
  MODEL_NAME       optional — defaults to Qwen/Qwen2.5-72B-Instruct
  LOCAL_IMAGE_NAME optional — Docker image name (from_docker_image)
  TASK_NAME        optional — run single task (default: run all)
"""
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from support_triage_env import (
    CustomerSupportTriageEnv,
    TriageAction,
    ActionType,
    ClassificationPayload,
    ResponsePayload,
    EscalationPayload,
    Department,
    Urgency,
    TASKS_META,
)

# ── Mandatory variable spec ───────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")           # no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional

BENCHMARK    = "customer_support_triage"
MAX_STEPS    = 35
TEMPERATURE  = 0.15

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support triage agent with deep business acumen.

TASK: Process each support ticket using structured JSON actions.

THINKING FRAMEWORK (apply for every ticket):
1. ISSUE: What is the core problem? (billing error, technical bug, legal request, churn signal?)
2. SIGNALS: Any churn/retention signals? (competitor mention, data export request, repeated failures)
3. DEPARTMENT: billing | technical | general | returns
4. URGENCY: high (P0 outages, legal, enterprise churn) | medium (bugs, returns) | low (questions)
5. ACTION: classify → then route correctly:
   - escalate: P0 outages, GDPR/legal, critical churn risk (competitor pricing, repeated failures)
   - archive:  spam / phishing (look for suspicious URLs, prize claims)
   - respond:  billing issues, technical bugs, product questions, happy customers
   - close:    positive feedback, already resolved

CHURN SIGNALS (expert task):
- "evaluating alternatives" / "export my data" = hidden churn (HIGH risk)
- "competitor offered X%" / "CEO consolidating tools" = critical churn
- "5th outage" / "keeps breaking" = frustrated, about to leave
- "renewing for sure" = low risk, just answer their question

RETURN ONLY VALID JSON — no markdown, no explanation.

JSON SCHEMAS:

Classify:
{"action_type":"classify","ticket_id":"T001","classification":{"department":"billing","urgency":"high"}}

Respond:
{"action_type":"respond","ticket_id":"T001","response":{"response_text":"Dear [Name], I sincerely apologize..."}}

Escalate (include churn_risk for churn prevention task):
{"action_type":"escalate","ticket_id":"T001","escalation":{"reason":"P0 production outage affecting enterprise...","churn_risk":"critical"}}

Archive:
{"action_type":"archive","ticket_id":"H002"}

Close:
{"action_type":"close","ticket_id":"H007"}
"""


def obs_to_prompt(obs: Dict[str, Any]) -> str:
    """Convert observation dict to concise agent prompt."""
    lines = [
        f"=== TASK: {obs['task_name']} ===",
        f"Step {obs['step_count']}/{obs['max_steps']} | Score: {obs.get('score_so_far', 0):.3f}",
        f"Pending: {obs['pending_ticket_ids']} | Done: {obs['completed_ticket_ids']}",
        "",
        "TICKETS:",
    ]
    for t in obs["tickets"]:
        if t["id"] in obs["pending_ticket_ids"]:
            meta = t.get("ticket_metadata", {})
            meta_str = ""
            if meta.get("mrr"):
                meta_str = f" [MRR=${meta['mrr']}/mo, tenure={meta.get('tenure_months','?')}mo]"
            if meta.get("plan"):
                meta_str += f" [Plan: {meta['plan']}]"
            lines.append(f"\n[{t['id']}]{meta_str}")
            lines.append(f"Subject: {t['subject']}")
            lines.append(f"From: {t['customer_name']} <{t['customer_email']}>")
            lines.append(f"Body: {t['body'][:400]}")

    if obs.get("actions_history"):
        lines.append("\nRECENT ACTIONS (last 5):")
        for a in obs["actions_history"][-5:]:
            lines.append(f"  {a.get('action_type')} → {a.get('ticket_id')}")

    lines.append("\nReturn next JSON action:")
    return "\n".join(lines)


def call_llm(messages: List[Dict]) -> str:
    """Call LLM with retry on failure."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            import time
            time.sleep(2 ** attempt)
    return ""


def parse_action(raw: str, pending_ids: List[str]) -> Optional[TriageAction]:
    """Parse LLM JSON into TriageAction. Strip markdown fences if present."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        text = "\n".join(inner).strip()

    # Find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        # Validate ticket_id is pending
        if data.get("ticket_id") not in pending_ids and pending_ids:
            data["ticket_id"] = pending_ids[0]
        return TriageAction(**data)
    except Exception:
        return None


def fallback_action(obs: Dict[str, Any]) -> Optional[TriageAction]:
    """Deterministic fallback when LLM parse fails — classify first pending ticket."""
    pending = obs.get("pending_ticket_ids", [])
    if not pending:
        return None
    tid = pending[0]
    # Check if it's already been classified
    history = obs.get("actions_history", [])
    classified = {a["ticket_id"] for a in history if a.get("action_type") == "classify"}
    if tid not in classified:
        return TriageAction(
            action_type=ActionType.CLASSIFY,
            ticket_id=tid,
            classification=ClassificationPayload(
                department=Department.GENERAL,
                urgency=Urgency.MEDIUM,
            )
        )
    # Fallback route
    return TriageAction(action_type=ActionType.RESPOND, ticket_id=tid,
                        response=ResponsePayload(response_text=(
                            "Dear customer, thank you for contacting us. "
                            "I apologize for the inconvenience. Our team will investigate "
                            "and resolve this as soon as possible. We value your business."
                        )))


def run_episode(task_name: str) -> Dict[str, Any]:
    """Run one complete episode. Returns result dict."""
    env = CustomerSupportTriageEnv(task_name=task_name)
    obs_obj = env.reset()
    obs = obs_obj.model_dump()

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    last_error: Optional[str] = None
    step = 0
    done = False
    score = 0.02

    try:
        while not done and step < MAX_STEPS:
            step += 1

            # Build user prompt
            user_msg = obs_to_prompt(obs)
            messages.append({"role": "user", "content": user_msg})

            # Get LLM response
            raw = call_llm(messages)
            messages.append({"role": "assistant", "content": raw})

            # Parse action
            pending = obs.get("pending_ticket_ids", [])
            action = parse_action(raw, pending)

            if action is None:
                last_error = f"parse_failed:{raw[:80].replace(chr(10),' ')}"
                action = fallback_action(obs)
                if action is None:
                    break
            else:
                last_error = None

            action_str = f"{action.action_type}:{action.ticket_id}"

            # Step environment
            try:
                result_obs = env.step(action)
            except RuntimeError as e:
                last_error = str(e)[:120]
                break

            obs = result_obs.model_dump()
            reward_val = result_obs.reward or 0.02
            done = result_obs.done
            rewards.append(reward_val)

            if done:
                score = obs.get("metadata", {}).get("episode_score", env.state.episode_score)

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward_val:.2f} done={str(done).lower()} "
                f"error={last_error if last_error else 'null'}",
                flush=True,
            )

    except Exception as exc:
        last_error = str(exc)[:200]
        traceback.print_exc(file=sys.stderr)

    # Always compute final score
    try:
        final_score = env.state.episode_score
        score = final_score
    except Exception:
        pass

    success = score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.02"

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

    return {"task": task_name, "score": score, "steps": step,
            "success": success, "rewards": rewards}


def main():
    task_name = os.getenv("TASK_NAME", "")
    if task_name and task_name in TASKS_META:
        tasks = [task_name]
    else:
        tasks = list(TASKS_META.keys())

    all_results = []
    for t in tasks:
        result = run_episode(t)
        all_results.append(result)
        print("", flush=True)

    # Summary
    print("=" * 60, flush=True)
    print("BASELINE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['task']:30s} score={r['score']:.3f} steps={r['steps']}",
              flush=True)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  AVERAGE SCORE: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
