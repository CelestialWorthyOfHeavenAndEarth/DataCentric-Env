"""
server/main.py — Production FastAPI application (v0.3).

Endpoints:
  POST /reset              — Start new episode, returns session_id
  POST /step               — Take action (requires session_id)
  GET  /state/{session_id} — Get current observation
  GET  /health             — Health check
  GET  /metrics            — Session + episode metrics
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
from server.environment import DataCentricEnvironment
from server.session_manager import session_manager
from server.config import cfg
from server.logger import get_logger, log_event

logger = get_logger("api")

app = FastAPI(
    title="DataCentric-Env",
    version=cfg.ENV_VERSION,
    description=(
        "RL environment: LLM acts as data engineer. "
        "Query specialist agents for recommendations, apply them to fix a dataset, "
        "hit the accuracy target."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

VALID_ACTIONS = {
    "query_cleaner", "query_augmenter", "query_balancer",
    "query_validator", "query_analyst", "apply",
}


# ── Request models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v):
        if v is not None and v not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")
        return v


class ActionRequest(BaseModel):
    session_id: str
    action: str
    rec_id: Optional[str] = None
    target_class: Optional[int] = None

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        if v not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{v}'. Must be one of: {sorted(VALID_ACTIONS)}"
            )
        return v

    @field_validator("target_class")
    @classmethod
    def validate_target_class(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("target_class must be 0 or 1")
        return v


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/reset", summary="Start a new episode")
def reset(body: ResetRequest = None):
    difficulty = body.difficulty if body else None

    # Create new session + environment
    session_id = "pending"  # placeholder before create_session
    env = DataCentricEnvironment(session_id="pending", episode_count=0)
    session_id = session_manager.create_session(env)
    env.session_id = session_id  # patch in the real ID

    obs = env.reset(difficulty=difficulty)
    log_event(logger, "api_reset", session_id=session_id, difficulty=obs.get("difficulty"))
    return obs


@app.post("/step", summary="Take an action in the environment")
def step(body: ActionRequest):
    env = session_manager.get_env(body.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found or expired. Call /reset first."
        )

    action_dict = {"action": body.action}
    if body.rec_id:
        action_dict["rec_id"] = body.rec_id
    if body.target_class is not None:
        action_dict["target_class"] = body.target_class

    result = env.step(action_dict)
    session_manager.increment_steps(body.session_id)
    return result


@app.get("/state/{session_id}", summary="Get current observation")
def state(session_id: str):
    env = session_manager.get_env(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state()


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "version": cfg.ENV_VERSION,
        "active_sessions": session_manager.metrics()["active_sessions"],
    }


@app.get("/metrics", summary="Episode and session metrics")
def metrics():
    return {
        "version": cfg.ENV_VERSION,
        "config": {
            "max_budget": cfg.MAX_BUDGET,
            "max_concurrent_sessions": cfg.MAX_CONCURRENT_SESSIONS,
            "session_ttl_seconds": cfg.SESSION_TTL_SECONDS,
            "golden_row_count": cfg.GOLDEN_ROW_COUNT,
            "max_same_action_streak": cfg.MAX_SAME_ACTION_STREAK,
        },
        "sessions": session_manager.metrics(),
    }
