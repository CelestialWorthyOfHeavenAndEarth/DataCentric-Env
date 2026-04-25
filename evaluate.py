"""
evaluate.py — Baseline vs trained agent comparison
Run before and after training to measure improvement.

Usage:
    python evaluate.py --url http://localhost:8000
    python evaluate.py --url https://your-hf-username-datacentric-env.hf.space
"""

import requests
import json
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Evaluate DataCentric-Env agent")
parser.add_argument("--url", default="https://aswini-kumar-datacentric-env.hf.space", help="Environment server URL")
parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
args = parser.parse_args()

ENV_URL = args.url.rstrip("/")
N_EPISODES = args.episodes


def random_agent_episode():
    """Baseline: random tool selection."""
    obs = requests.post(f"{ENV_URL}/reset").json()
    tools = ["cleaner", "augmenter", "balancer", "validator"]
    total_reward = 0.0
    success = False
    for _ in range(10):
        action = {"agent": random.choice(tools), "target": "all"}
        result = requests.post(f"{ENV_URL}/step", json=action).json()
        total_reward += result.get("reward", 0)
        if result.get("done"):
            success = result.get("info", {}).get("success", False)
            break
    return total_reward, success


# ─── Run baseline ─────────────────────────────────────────────────────────────
print(f"Running {N_EPISODES} baseline (random) episodes against {ENV_URL}...")
baseline_rewards = []
baseline_successes = []
for i in range(N_EPISODES):
    reward, success = random_agent_episode()
    baseline_rewards.append(reward)
    baseline_successes.append(success)
    print(f"  Episode {i+1:02d}: reward={reward:.3f} success={success}")

mean_baseline = sum(baseline_rewards) / len(baseline_rewards)
success_rate_baseline = sum(baseline_successes) / len(baseline_successes)
print(f"\nBaseline mean reward:  {mean_baseline:.3f}")
print(f"Baseline success rate: {success_rate_baseline:.1%}")

# ─── Plot reward curve ────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, N_EPISODES + 1), baseline_rewards, marker="o", color="#5B8FF9", label="Random baseline")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Baseline Reward per Episode")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
mean_trained = mean_baseline * 1.0  # placeholder — replace with trained agent result
plt.bar(["Random baseline", "Trained agent"],
        [mean_baseline, mean_trained],
        color=["#5B8FF9", "#5AD8A6"])
plt.ylabel("Mean Episode Reward")
plt.title("Baseline vs Trained Agent")
plt.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("\nSaved results.png")
