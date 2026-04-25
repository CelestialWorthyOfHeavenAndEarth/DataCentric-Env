"""
client/client.py — OpenEnv client for DataCentric-Env v0.3.
Communicates via HTTP only — never imports from server/.
"""
import requests
from typing import Optional


class DataCentricClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def reset(self, difficulty: str = None) -> dict:
        payload = {}
        if difficulty:
            payload["difficulty"] = difficulty
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        self.session_id = data.get("session_id")
        return data

    def step(self, action: str, rec_id: str = None, target_class: int = None) -> dict:
        if not self.session_id:
            raise RuntimeError("Call reset() first to get a session_id.")
        payload = {"session_id": self.session_id, "action": action}
        if rec_id:
            payload["rec_id"] = rec_id
        if target_class is not None:
            payload["target_class"] = target_class
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        if not self.session_id:
            raise RuntimeError("Call reset() first.")
        r = requests.get(f"{self.base_url}/state/{self.session_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def metrics(self) -> dict:
        r = requests.get(f"{self.base_url}/metrics", timeout=10)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = DataCentricClient("http://localhost:8000")

    # Demo episode
    obs = client.reset(difficulty="easy")
    print(f"Reset: session={obs['session_id']}, acc={obs['current_accuracy']}, target={obs['target_accuracy']}")

    result = client.step("query_analyst")
    plan = result.get("query_result", {}).get("action_plan", [])
    print(f"Analyst plan: {[p['action'] for p in plan]}")

    result = client.step("query_cleaner")
    recs = list(result.get("observation", {}).get("pending_recommendations", {}).keys())
    print(f"Cleaner recs: {recs}")

    if recs:
        result = client.step("apply", rec_id=recs[0])
        print(f"Apply: acc {result['info']['prev_accuracy']} -> {result['info']['new_accuracy']} | reward={result['reward']}")

    print(f"Metrics: {client.metrics()['sessions']}")
