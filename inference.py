"""
inference.py — Required by Phase 1 automated checks.
Runs a complete episode against the local environment server.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def run_episode(base_url: str = BASE_URL):
    print(f"Connecting to environment at {base_url}")

    # Phase 1: reset
    obs = requests.post(f"{base_url}/reset", json={}, timeout=30)
    assert obs.status_code == 200, f"Reset failed with status {obs.status_code}"
    obs = obs.json()
    assert "current_accuracy" in obs, "Missing current_accuracy in reset response"
    assert "budget_remaining" in obs, "Missing budget_remaining in reset response"
    assert "available_tools" in obs, "Missing available_tools in reset response"
    print(f"Phase 1 reset: PASS | initial accuracy={obs['current_accuracy']}")

    # Phase 2: run a full episode
    tools = ["cleaner", "augmenter", "balancer", "validator"]
    step_count = 0

    while True:
        # Simple greedy heuristic: pick tool based on dataset stats
        stats = obs.get("dataset_stats", {})
        if stats.get("missing_pct", 0) > 0.05:
            action = {"agent": "cleaner", "target": "all", "strategy": "median_impute"}
        elif stats.get("balance_ratio", 1.0) < 0.3:
            action = {"agent": "balancer", "strategy": "undersample"}
        else:
            action = {"agent": "augmenter"}

        result = requests.post(f"{base_url}/step", json=action, timeout=30)
        assert result.status_code == 200, f"Step failed with status {result.status_code}"
        result = result.json()

        reward = result.get("reward")
        done = result.get("done")
        info = result.get("info", {})
        step_count += 1

        assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
        assert 0.0 < reward < 1.0, f"Reward {reward} out of valid range (0.0, 1.0)"
        assert isinstance(done, bool), f"Done must be bool, got {type(done)}"

        print(f"  Step {step_count:02d}: agent={action['agent']} | reward={reward:.4f} | "
              f"accuracy={info.get('new_accuracy', '?')} | done={done}")

        obs = result.get("observation", obs)

        if done or step_count >= 10:
            break

    success = info.get("success", False)
    print(f"\nEpisode complete — {step_count} steps | success={success}")
    print("All automated checks passed.")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    run_episode(url)
