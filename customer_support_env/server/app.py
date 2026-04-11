"""FastAPI server — Customer Support Triage OpenEnv"""
from __future__ import annotations
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from support_triage_env import CustomerSupportTriageEnv, TriageAction, TASKS_META

STATIC_DIR = Path(__file__).parent.parent / "static"

_sessions: Dict[str, CustomerSupportTriageEnv] = {}
_metrics: Dict[str, List[float]] = defaultdict(list)
_total_resets: int = 0
_DEFAULT = "default"

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description="Real-world customer support triage environment. 4 tasks: easy→expert.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _get(sid: str) -> CustomerSupportTriageEnv:
    if sid not in _sessions:
        raise HTTPException(404, f"Session '{sid}' not found. POST /reset first.")
    return _sessions[sid]


class ResetRequest(BaseModel):
    task_name: str = "ticket_classification"
    session_id: str = _DEFAULT
    episode_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = _DEFAULT


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return HTMLResponse(content=html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Customer Support Triage OpenEnv v2.0</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "tasks": len(TASKS_META), "active_sessions": len(_sessions)}


@app.get("/tasks")
def list_tasks():
    return {"tasks": [{"name": k, "difficulty": v["difficulty"], "max_steps": v["max_steps"],
                       "num_tickets": v["num_tickets"], "description": v["description"]}
                      for k, v in TASKS_META.items()]}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    global _total_resets
    if request is None:
        request = ResetRequest()
    if request.task_name not in TASKS_META:
        raise HTTPException(400, f"Unknown task. Valid: {list(TASKS_META)}")
    env = CustomerSupportTriageEnv(task_name=request.task_name)
    obs = env.reset(seed=request.seed, episode_id=request.episode_id)
    _sessions[request.session_id] = env
    _total_resets += 1
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest):
    env = _get(request.session_id)
    try:
        action = TriageAction(**request.action)
        obs = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(422, f"Invalid action: {e}")
    if obs.done:
        score = obs.metadata.get("episode_score", env.state.episode_score)
        _metrics[env._task_name].append(float(score))
    return obs.model_dump()


@app.get("/state")
def state(session_id: str = Query(default=_DEFAULT)):
    return _get(session_id).state.model_dump()


@app.get("/metrics")
def metrics():
    stats = {}
    for task, scores in _metrics.items():
        if scores:
            stats[task] = {"episodes": len(scores),
                           "avg_score": round(sum(scores)/len(scores), 4),
                           "min_score": round(min(scores), 4),
                           "max_score": round(max(scores), 4)}
    return {"total_resets": _total_resets, "active_sessions": len(_sessions), "tasks": stats}


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
