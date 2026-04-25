"""
server/main.py — Production FastAPI application (v0.5).

Endpoints:
  POST /reset                   — Start new episode (returns session_id + full observation)
  POST /step                    — Take action (query | apply | rollback)
  GET  /state/{session_id}      — Current observation
  GET  /trajectory/{session_id} — Full episode trace with all rewards and effects
  GET  /health                  — Health check + version
  GET  /metrics                 — Session counts + config
  GET  /docs                    — Swagger UI (auto-generated)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional

from server.environment import DataCentricEnvironment, _registry
from server.session_manager import session_manager
from server.config import cfg
from server.logger import get_logger, log_event

logger = get_logger("api")

app = FastAPI(
    title="DataCentric-Env",
    version=cfg.ENV_VERSION,
    description=(
        "RL environment: an LLM acts as a data engineer. "
        "Given a real, messy tabular dataset (UCI Adult, Pima Diabetes, German Credit, etc.), "
        "the agent queries specialist agents for recommendations and applies them to fix the data "
        "until the frozen classifier hits the accuracy target. "
        "All scores compared against published academic baselines.\n\n"
        "**New in v0.5:** Rollback action, episode reasoning trace, feature importance, "
        "regression explanations, benchmark comparisons."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load all 5 real datasets in a background thread so the first /reset is instant."""
    _registry.warmup()


VALID_ACTIONS = {
    "query_cleaner", "query_augmenter", "query_balancer",
    "query_validator", "query_analyst", "apply", "rollback",
}


# ── Request models ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None
    seed: Optional[int] = None

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
            raise ValueError(f"Invalid action '{v}'. Valid: {sorted(VALID_ACTIONS)}")
        return v

    @field_validator("target_class")
    @classmethod
    def validate_target_class(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError("target_class must be 0 or 1")
        return v


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.post("/reset", summary="Start a new episode")
def reset(body: ResetRequest = None):
    """
    Creates a new episode on a real dataset. Returns `session_id` + full observation.

    The observation includes:
    - Dataset name, domain, and documented known quality issues
    - Current accuracy vs target vs published benchmark vs majority-class baseline
    - Dataset statistics (missing %, class balance ratio)
    - Feature importance (empty until first apply)
    - Episode trace (empty at start)
    - All pending recommendations (empty until first query)
    """
    difficulty = body.difficulty if body else None
    seed = body.seed if body else None

    env = DataCentricEnvironment(session_id="pending", episode_count=0)
    session_id = session_manager.create_session(env)
    env.session_id = session_id

    obs = env.reset(difficulty=difficulty, seed=seed)
    log_event(logger, "api_reset", session_id=session_id, difficulty=obs.get("difficulty"))
    return obs


@app.post("/step", summary="Take an action")
def step(body: ActionRequest):
    """
    Take one action in the environment.

    **Query actions** (cost 1-2 budget, return recommendations):
    - `query_cleaner` (cost 1) — missing value + zero-as-missing analysis, domain-aware
    - `query_augmenter` (cost 1) — minority class synthesis via SMOTE-like interpolation
    - `query_balancer` (cost 1) — class resampling with explicit tradeoff explanation
    - `query_validator` (cost 2) — duplicate + outlier detection (conservative IQR for medical)
    - `query_analyst` (cost 2) — holistic diagnosis + prioritized plan + published baseline

    **Apply action** (modifies dataset, no budget cost):
    - `apply` with `rec_id` — apply a recommendation by its ID from any previous query
    - Response includes: feature importance (LogReg coefs), regression explanation if accuracy drops

    **Rollback action** (cost 1 budget, max 3/episode):
    - `rollback` — undo the last apply and restore the previous dataset state
    """
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

    if "error" in result and "exploit" not in str(result):
        log_event(logger, "step_error", session_id=body.session_id, error=result["error"])

    session_manager.increment_steps(body.session_id)
    return result


@app.get("/state/{session_id}", summary="Get current observation")
def state(session_id: str):
    """Current full observation including episode trace, benchmarks, and feature importance."""
    env = session_manager.get_env(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state()


@app.get("/trajectory/{session_id}", summary="Full episode trajectory")
def trajectory(session_id: str):
    """
    Complete episode trace — every step with reward, accuracy delta, and effect label.

    Useful for:
    - Offline reward model training
    - Debugging agent decisions
    - Comparing strategy effectiveness across episodes
    """
    env = session_manager.get_env(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.episode_summary()


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "version": cfg.ENV_VERSION,
        "active_sessions": session_manager.metrics()["active_sessions"],
        "real_datasets": [
            "UCI Adult Census Income",
            "Pima Indians Diabetes",
            "Wisconsin Breast Cancer Diagnostic",
            "German Credit Risk",
            "Cleveland Heart Disease",
        ],
    }


@app.get("/metrics", summary="Server metrics")
def metrics():
    return {
        "version": cfg.ENV_VERSION,
        "config": {
            "max_budget": cfg.MAX_BUDGET,
            "max_concurrent_sessions": cfg.MAX_CONCURRENT_SESSIONS,
            "session_ttl_seconds": cfg.SESSION_TTL_SECONDS,
            "max_same_action_streak": cfg.MAX_SAME_ACTION_STREAK,
            "max_row_deletion_pct": 0.10,
            "max_rollbacks_per_episode": 3,
        },
        "sessions": session_manager.metrics(),
    }
