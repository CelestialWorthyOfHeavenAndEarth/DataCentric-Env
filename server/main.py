from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from server.environment import DataCentricEnvironment

app = FastAPI(title="DataCentric-Env", version="0.1.0")
env = DataCentricEnvironment()


class ResetRequest(BaseModel):
    difficulty: Optional[str] = None


@app.post("/reset")
def reset(body: ResetRequest = None):
    difficulty = body.difficulty if body else None
    return env.reset(difficulty=difficulty)


@app.post("/step")
def step(action: dict):
    return env.step(action)


@app.get("/state")
def state():
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}
