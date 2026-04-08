"""
FastAPI server exposing OpenEnv HTTP endpoints for Hugging Face Spaces deployment.
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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store ──
_envs: Dict[str, CustomerSupportEnv] = {}
_DEFAULT_SESSION = "default"


def _get_env(session_id: str) -> CustomerSupportEnv:
    if session_id not in _envs:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return _envs[session_id]


# ───────── Models ─────────
class ResetRequest(BaseModel):
    task_name: str = "ticket_classification"
    session_id: str = _DEFAULT_SESSION


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = _DEFAULT_SESSION


# ───────── Endpoints ─────────
@app.get("/health")
def health():
    return {"status": "ok"}


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
def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest()

    if request.task_name not in TASKS:
        raise HTTPException(status_code=400, detail="Invalid task")

    env = CustomerSupportEnv(task_name=request.task_name)
    obs = env.reset()
    _envs[request.session_id] = env
    return obs


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    env = _get_env(request.session_id)

    try:
        action = Action(**request.action)
        result = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.get("/state", response_model=EnvState)
def state(session_id: str = Query(default=_DEFAULT_SESSION)):
    env = _get_env(session_id)
    return env.state()


@app.get("/")
def root():
    return {
        "message": "Customer Support OpenEnv Running",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }


# ✅ REQUIRED FOR OPENENV
def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


# ✅ ENTRY POINT
if __name__ == "__main__":
    main()