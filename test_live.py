import requests
import json

BASE = "https://aswini-kumar-datacentric-env.hf.space"

print("=== Testing live HuggingFace Space ===")

# Health check
h = requests.get(f"{BASE}/health", timeout=30)
print(f"Health: {h.status_code} -> {h.json()}")

# Reset
obs = requests.post(f"{BASE}/reset", json={}, timeout=60)
print(f"Reset: {obs.status_code}")
data = obs.json()
print(json.dumps(data, indent=2))

# Step
result = requests.post(
    f"{BASE}/step",
    json={"agent": "cleaner", "target": "all", "strategy": "median_impute"},
    timeout=60,
)
print(f"Step status: {result.status_code}")
r = result.json()
reward = r["reward"]
done = r["done"]
acc = r["info"]["new_accuracy"]
print(f"reward={reward}  done={done}  accuracy={acc}")

assert 0.0 < reward < 1.0, f"Reward {reward} out of range!"
assert isinstance(done, bool), "done must be bool"
print("\nLIVE endpoint: ALL CHECKS PASSED")
