from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any
from server.environment import DataCentricEnvironment

app = FastAPI(
    title="DataCentric-Env",
    version="0.2.0",
    description=(
        "RL environment where an LLM acts as a data engineer. "
        "Query specialist agents for recommendations, then apply them to fix a noisy dataset."
    ),
)
env = DataCentricEnvironment()


class ResetRequest(BaseModel):
    difficulty: Optional[str] = None


class ActionRequest(BaseModel):
    action: str                         # query_cleaner | query_augmenter | ... | apply
    rec_id: Optional[str] = None        # required for action="apply"
    target_class: Optional[int] = None  # optional for query_augmenter


@app.post("/reset")
def reset(body: ResetRequest = None):
    difficulty = body.difficulty if body else None
    return env.reset(difficulty=difficulty)


@app.post("/step")
def step(body: ActionRequest):
    action_dict = body.model_dump(exclude_none=True)
    return env.step(action_dict)


@app.get("/state")
def state():
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}
