"""
inference.py — Phase 1/2 automated check entry point (v0.3).
Demonstrates the full query/apply episode with session management.
"""
import requests
import sys
import json

BASE_URL = "http://localhost:8000"


def run_episode(base_url: str = BASE_URL):
    base_url = base_url.rstrip("/")
    print(f"DataCentric-Env v0.3 | {base_url}\n")

    # Phase 1: reset — get session_id
    resp = requests.post(f"{base_url}/reset", json={}, timeout=30)
    assert resp.status_code == 200, f"Reset failed: {resp.status_code} {resp.text}"
    obs = resp.json()
    assert "session_id" in obs, "Missing session_id in reset response"
    assert "current_accuracy" in obs, "Missing current_accuracy"
    assert "budget_remaining" in obs, "Missing budget_remaining"
    assert "available_actions" in obs, "Missing available_actions"

    session_id = obs["session_id"]
    print(f"Phase 1 reset: PASS")
    print(f"  session_id     = {session_id}")
    print(f"  accuracy       = {obs['current_accuracy']} -> target {obs['target_accuracy']}")
    print(f"  budget         = {obs['budget_remaining']}")
    print(f"  missing_pct    = {obs['dataset_stats']['missing_pct']}")
    print(f"  balance_ratio  = {obs['dataset_stats']['balance_ratio']}\n")

    # Phase 2: run a full episode
    rewards = []
    step_count = 0
    queried_agents = set()

    while True:
        budget = obs.get("budget_remaining", 0)
        if budget <= 0 or obs.get("done"):
            break

        pending = obs.get("pending_recommendations", {})
        stats = obs.get("dataset_stats", {})

        # Strategy: query analyst first, then follow plan, then apply
        if not queried_agents:
            action = {"session_id": session_id, "action": "query_analyst"}
        elif not pending:
            # Pick next agent based on dataset stats
            if stats.get("missing_pct", 0) > 0.05 and "cleaner" not in queried_agents:
                action = {"session_id": session_id, "action": "query_cleaner"}
            elif stats.get("balance_ratio", 1.0) < 0.45 and "balancer" not in queried_agents:
                action = {"session_id": session_id, "action": "query_balancer"}
            elif "augmenter" not in queried_agents:
                action = {"session_id": session_id, "action": "query_augmenter", "target_class": 1}
            else:
                break  # out of ideas
        else:
            # Apply highest priority pending rec
            best_rec_id = min(pending, key=lambda k: pending[k].get("priority", 99))
            action = {"session_id": session_id, "action": "apply", "rec_id": best_rec_id}

        result = requests.post(f"{base_url}/step", json=action, timeout=30)
        assert result.status_code == 200, f"Step failed: {result.status_code} {result.text}"
        result = result.json()

        if "error" in result and "exploit_detected" not in result:
            print(f"  Step {step_count+1}: ERROR - {result['error']}")
            break

        reward = result.get("reward", 0.0)
        info = result.get("info", {})
        step_count += 1
        rewards.append(reward)

        assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
        assert 0.0 < reward < 1.0, f"Reward {reward} out of range (0.0, 1.0)"

        action_type = info.get("action_type", "?")
        if action_type == "query":
            agent = info.get("agent_queried", "?")
            queried_agents.add(agent)
            n_recs = info.get("n_recommendations", 0)
            print(f"  Step {step_count:02d}: QUERY  {agent:12s} -> {n_recs} recs | reward={reward:.4f} | budget={info.get('budget_remaining','?')}")
        else:
            print(f"  Step {step_count:02d}: APPLY  {info.get('rec_type','?'):15s} -> {info.get('prev_accuracy','?'):.4f}->{info.get('new_accuracy','?'):.4f} | reward={reward:.4f} | success={info.get('success',False)}")

        obs = result.get("observation", obs)

        if result.get("done"):
            print(f"\n  Episode done. Success={info.get('success', False)}")
            break

    # Verify metrics endpoint
    m = requests.get(f"{base_url}/metrics", timeout=10).json()
    assert "sessions" in m, "Missing sessions in /metrics"

    print(f"\nPhase 2 full episode: PASS")
    print(f"  Steps: {step_count} | Mean reward: {sum(rewards)/max(len(rewards),1):.4f}")
    print(f"  All {len(rewards)} rewards in (0.0, 1.0): {all(0.0 < r < 1.0 for r in rewards)}")
    print(f"  Active sessions: {m['sessions']['active_sessions']}")
    print("\nAll automated checks passed.")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    run_episode(url)
