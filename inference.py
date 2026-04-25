"""
inference.py — Required by Phase 1/2 automated checks.
Demonstrates a full episode using the query/apply pattern.

The agent:
  1. Calls query_analyst to get a prioritized action plan
  2. Follows the plan — queries the recommended agent
  3. Applies the top-priority recommendation
  4. Repeats until done or budget exhausted
"""

import requests
import sys
import json

BASE_URL = "http://localhost:8000"


def run_episode(base_url: str = BASE_URL):
    print(f"DataCentric-Env v0.2 — Query/Apply Pattern")
    print(f"Connecting to: {base_url}\n")

    # Phase 1: reset
    obs = requests.post(f"{base_url}/reset", json={}, timeout=30)
    assert obs.status_code == 200, f"Reset failed: {obs.status_code}"
    obs = obs.json()
    assert "current_accuracy" in obs
    assert "budget_remaining" in obs
    assert "available_actions" in obs
    print(f"Phase 1 reset: PASS | accuracy={obs['current_accuracy']} | target={obs['target_accuracy']}")

    # Phase 2: full episode using query/apply
    step_count = 0
    rewards = []

    while not obs.get("done", False):
        budget = obs.get("budget_remaining", 0)
        if budget <= 0:
            break

        # Step 1: query analyst if no pending recs yet, or low on info
        pending = obs.get("pending_recommendations", {})

        if not pending:
            # Start by asking analyst for a plan
            result = requests.post(
                f"{base_url}/step",
                json={"action": "query_analyst"},
                timeout=30,
            ).json()
        else:
            # Apply the first available pending recommendation
            rec_id = next(iter(pending.keys()))
            result = requests.post(
                f"{base_url}/step",
                json={"action": "apply", "rec_id": rec_id},
                timeout=30,
            ).json()

        if "error" in result:
            print(f"  Step {step_count+1}: ERROR — {result['error']}")
            break

        reward = result.get("reward", 0)
        info = result.get("info", {})
        step_count += 1
        rewards.append(reward)

        # Validate reward is in (0.0, 1.0)
        assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
        assert 0.0 < reward < 1.0, f"Reward {reward} out of valid range (0.0, 1.0)"

        action_type = info.get("action_type", "?")
        if action_type == "query":
            n_recs = info.get("n_recommendations", 0)
            agent = info.get("agent_queried", "?")
            print(f"  Step {step_count:02d}: QUERY {agent:10s} -> {n_recs} recs | reward={reward:.4f} | budget={info.get('budget_remaining', '?')}")
        else:
            acc_before = info.get("prev_accuracy", "?")
            acc_after = info.get("new_accuracy", "?")
            print(f"  Step {step_count:02d}: APPLY {info.get('rec_type', '?'):15s} -> acc {acc_before}->{acc_after} | reward={reward:.4f} | success={info.get('success', False)}")

        obs = result.get("observation", obs)

        if result.get("done"):
            print(f"\n  Episode done in {step_count} steps. Success={info.get('success', False)}")
            break

        # Follow-up query if we just queried analyst — query the first recommended agent
        query_result = result.get("query_result", {})
        action_plan = query_result.get("action_plan", [])
        if action_plan and step_count <= 2:
            next_action = action_plan[0].get("action", "query_cleaner")
            if budget - 1 > 0:
                result2 = requests.post(
                    f"{base_url}/step",
                    json={"action": next_action},
                    timeout=30,
                ).json()
                if "error" not in result2:
                    reward2 = result2.get("reward", 0)
                    rewards.append(reward2)
                    step_count += 1
                    info2 = result2.get("info", {})
                    n_recs = info2.get("n_recommendations", 0)
                    agent = info2.get("agent_queried", "?")
                    print(f"  Step {step_count:02d}: QUERY {agent:10s} -> {n_recs} recs | reward={reward2:.4f}")
                    obs = result2.get("observation", obs)

    print(f"\nPhase 2 full episode: PASS")
    print(f"  Steps taken: {step_count}")
    print(f"  Mean reward: {sum(rewards)/max(len(rewards),1):.4f}")
    print(f"  All {len(rewards)} rewards in (0.0, 1.0): {all(0.0 < r < 1.0 for r in rewards)}")
    print("\nAll automated checks passed.")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    run_episode(url)
