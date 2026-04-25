import requests


class DataCentricClient:
    """
    OpenEnv client for DataCentric-Env.
    Communicates via HTTP only — never imports from server/.
    """
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, difficulty: str = None) -> dict:
        payload = {"difficulty": difficulty} if difficulty else {}
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return r.json()


# Usage
if __name__ == "__main__":
    client = DataCentricClient("http://localhost:8000")
    obs = client.reset(difficulty="easy")
    print("Initial obs:", obs)
    result = client.step({"agent": "cleaner", "target": "all", "strategy": "median_impute"})
    print("Step result:", result)
