"""
FastAPI server exposing OpenEnv HTTP endpoints for Hugging Face Spaces deployment.

Endpoints:
  POST /reset          → Observation
  POST /step           → StepResult
  GET  /state          → EnvState
  GET  /health         → {"status": "ok"}
  GET  /tasks          → list of available tasks
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from customer_support_env import (
    Action,
    CustomerSupportEnv,
    Observation,
    StepResult,
    EnvState,
)
from customer_support_env.tasks import TASKS

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating "
        "AI agents on real-world customer support triage tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Per-session environment store (keyed by session_id) ──
_envs: Dict[str, CustomerSupportEnv] = {}
_DEFAULT_SESSION = "default"


def _get_env(session_id: str) -> CustomerSupportEnv:
    if session_id not in _envs:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return _envs[session_id]


# ─────────────────────────── Request/Response models ─────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "ticket_classification"
    session_id: str = _DEFAULT_SESSION


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = _DEFAULT_SESSION


# ─────────────────────────── Endpoints ───────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": k,
                "difficulty": v["difficulty"],
                "max_steps": v["max_steps"],
            }
            for k, v in TASKS.items()
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """Reset the environment for the given task. Returns initial observation."""
    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Valid tasks: {list(TASKS.keys())}",
        )
    env = CustomerSupportEnv(task_name=request.task_name)
    obs = env.reset()
    _envs[request.session_id] = env
    return obs


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    """Apply an action to the environment. Returns (observation, reward, done, info)."""
    env = _get_env(request.session_id)
    try:
        action = Action(**request.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action format: {exc}")
    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/state", response_model=EnvState)
def state(session_id: str = Query(default=_DEFAULT_SESSION)):
    """Return the full internal state of the environment."""
    env = _get_env(session_id)
    return env.state()


@app.get("/")
def root():
    return {
        "name": "Customer Support Triage OpenEnv",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
        "docs": "/docs",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
