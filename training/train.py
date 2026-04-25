"""
training/train.py — GRPO training for DataCentric-Env (v0.5).

Trains Qwen2.5-3B-Instruct with GRPO via TRL + Unsloth.
Run end-to-end in Colab (T4 GPU is sufficient).

Before running:
  1. Deploy the environment to HF Spaces (run deploy_to_hf.py locally)
  2. Set ENV_URL below to your HF Space URL
  3. Runtime → Run all

What the agent learns:
  - Given a real messy dataset (UCI Adult Census, Pima Diabetes, etc.)
  - Query specialist agents to diagnose issues (domain-aware analysis)
  - Apply recommended fixes to improve classifier accuracy on a frozen holdout
  - Navigate: when to rollback a bad apply, how to interpret feature importance,
    how to prioritize domain-specific issues (zeros-as-missing in medical data)
"""

import os
import json
import time
import requests
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# ── Configuration ──────────────────────────────────────────────────────────────
ENV_URL = "https://aswini-kumar-datacentric-env.hf.space"  # ← your HF Space URL
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 2048
N_ROLLOUT_EPISODES = 60
MAX_STEPS_PER_EPISODE = 12
LORA_RANK = 16

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert data engineer agent. You are given a real-world dataset \
with known quality issues and must fix it so a frozen classifier achieves the target accuracy.

You work by querying specialist agents for analysis, then deciding which recommendation to apply.

WORKFLOW:
1. Start by calling query_analyst (cost 2) — it gives you a prioritized action plan and \
references the published benchmark accuracy for this dataset.
2. Then call the specific agent it recommends (query_cleaner, query_balancer, etc.)
3. Apply the best recommendation using its rec_id
4. If accuracy dropped after an apply, use rollback to undo it (max 3 per episode)
5. Read feature_importance in the response — it shows what the model actually learned

DOMAIN RULES (critical):
- In medical datasets, zero values for physiological measurements are IMPOSSIBLE — they mean \
missing data. Always apply zero_to_nan_impute before other cleaning.
- In financial datasets, heavily skewed features (like capital-gain) should be log-transformed.
- Removing rows is dangerous — data integrity limit is 10% of training rows max.
- Large augmentation (>200 rows) may overfit training set and HURT holdout accuracy. \
If accuracy drops after augmentation, rollback and try balancer instead.

OUTPUT FORMAT — respond with valid JSON only, no explanation:
For queries: {"action": "query_analyst"} or {"action": "query_cleaner"} etc.
For apply:   {"action": "apply", "rec_id": "<exact_id_from_recommendations>"}
For rollback: {"action": "rollback"}"""


# ── Model setup ────────────────────────────────────────────────────────────────
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)
print(f"Model loaded: {MODEL_NAME} with LoRA r={LORA_RANK}")


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(obs: dict) -> str:
    """
    Build a compact but information-rich prompt from the observation.
    Excludes the full pending_recommendations dict (too verbose) —
    only includes rec_ids and their reason.
    """
    # Compact observation for the prompt
    compact = {
        "dataset": obs.get("dataset", {}).get("name", "unknown"),
        "domain": obs.get("dataset", {}).get("domain", ""),
        "known_issues": obs.get("dataset", {}).get("known_issues", [])[:2],
        "current_accuracy": obs.get("current_accuracy"),
        "target_accuracy": obs.get("target_accuracy"),
        "accuracy_gap": obs.get("accuracy_gap"),
        "benchmarks": obs.get("benchmarks", {}),
        "budget_remaining": obs.get("budget_remaining"),
        "dataset_stats": obs.get("dataset_stats", {}),
        "pending_recommendations": {
            rid: {
                "agent": info.get("agent"),
                "type": info.get("type"),
                "reason": info.get("reason", "")[:120],  # truncate
                "domain_informed": info.get("domain_informed", False),
            }
            for rid, info in obs.get("pending_recommendations", {}).items()
        },
        "episode_trace": obs.get("episode_trace", [])[-3:],  # last 3 steps
        "feature_importance": obs.get("feature_importance", {}).get("top_positive", [])[:2],
        "available_actions": obs.get("available_actions"),
    }
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nCurrent environment state:\n{json.dumps(compact, indent=2)}\n"
        f"<|assistant|>\n"
    )


# ── Episode rollout ────────────────────────────────────────────────────────────
def run_episode(difficulty: str = "easy") -> list[dict]:
    """
    Run one full episode. Returns list of (prompt, response, reward) tuples.
    """
    # Reset — get session_id and initial observation
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty},
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  Reset failed: {e}")
        return []

    obs = resp.json()
    session_id = obs.get("session_id")
    if not session_id:
        print("  No session_id in reset response.")
        return []

    trajectories = []

    for step_num in range(MAX_STEPS_PER_EPISODE):
        prompt = build_prompt(obs)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse and validate action
        try:
            action = json.loads(response)
            if "action" not in action:
                raise ValueError("missing 'action' key")
        except Exception:
            # Invalid JSON — format grader will penalize this
            action = {"action": "query_analyst"}

        # Always inject session_id
        payload = {"session_id": session_id, **action}

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json=payload,
                timeout=30,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as e:
            print(f"  Step failed: {e}")
            break

        reward = float(result.get("reward", 0.001))
        trajectories.append({
            "prompt": prompt,
            "response": response,
            "reward": reward,
        })

        obs = result.get("observation", obs)
        if result.get("done"):
            break

    return trajectories


# ── Collect rollouts across difficulties ───────────────────────────────────────
print(f"\nCollecting {N_ROLLOUT_EPISODES} episodes...")
all_trajectories = []
episode_rewards = []
difficulty_schedule = (
    ["easy"] * 20 + ["medium"] * 20 + ["hard"] * 20
)

for ep_idx, difficulty in enumerate(difficulty_schedule):
    trajs = run_episode(difficulty=difficulty)
    if trajs:
        ep_reward = np.mean([t["reward"] for t in trajs])
        episode_rewards.append(ep_reward)
        all_trajectories.extend(trajs)
        if ep_idx % 10 == 0:
            print(f"  Episode {ep_idx}/{N_ROLLOUT_EPISODES} | "
                  f"difficulty={difficulty} | mean_reward={ep_reward:.4f} | "
                  f"n_steps={len(trajs)}")
    time.sleep(0.5)  # avoid hammering the server

print(f"\nTotal training samples: {len(all_trajectories)}")
print(f"Mean reward across all episodes: {np.mean(episode_rewards):.4f}")

if len(all_trajectories) < 10:
    raise RuntimeError("Too few training samples collected. Check ENV_URL and server status.")


# ── Build GRPO training dataset ────────────────────────────────────────────────
# GRPO needs: prompt, completion (response), reward
train_dataset = Dataset.from_list([
    {
        "prompt": t["prompt"],
        "completion": t["response"],
        "reward": t["reward"],
    }
    for t in all_trajectories
    if t["reward"] > 0.001  # filter degenerate samples
])
print(f"Training dataset: {len(train_dataset)} samples")


# ── GRPO training ──────────────────────────────────────────────────────────────
config = GRPOConfig(
    output_dir="./datacentric-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=50,
    report_to="none",
    max_grad_norm=0.3,
    fp16=True,
    dataloader_num_workers=0,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    reward_funcs=[],  # rewards come from environment, already in dataset
)

print("\nStarting GRPO training...")
train_result = trainer.train()
print(f"Training complete. Final loss: {train_result.training_loss:.4f}")


# ── Sample inspection — check for reward hacking ──────────────────────────────
print("\n--- Sampling 3 agent generations (reward hacking check) ---")
for i in range(3):
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": "easy"}, timeout=60)
        obs = resp.json()
        session_id = obs["session_id"]
        prompt = build_prompt(obs)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  Sample {i+1}: {response[:200]}")

        try:
            action = json.loads(response.strip())
            payload = {"session_id": session_id, **action}
            step_r = requests.post(f"{ENV_URL}/step", json=payload, timeout=30).json()
            print(f"  → reward={step_r.get('reward')} | accuracy={step_r.get('observation', {}).get('current_accuracy')}")
        except Exception as e:
            print(f"  → parse/step failed: {e}")
    except Exception as e:
        print(f"  Sample {i+1} failed: {e}")


# ── Save model — Unsloth merge path (NOT naive save_pretrained) ───────────────
print("\nSaving model (Unsloth merged_16bit path)...")
model.save_pretrained_merged(
    "datacentric-grpo-final",
    tokenizer,
    save_method="merged_16bit",
)
print("Model saved to ./datacentric-grpo-final")


# ── Plot training curves → results.png ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("DataCentric-Env — GRPO Training Results", fontsize=14, fontweight="bold")

# Episode rewards
ax1 = axes[0]
ax1.plot(episode_rewards, color="#4f46e5", linewidth=1.5, alpha=0.6, label="Episode mean reward")
if len(episode_rewards) >= 5:
    smoothed = np.convolve(episode_rewards, np.ones(5)/5, mode="valid")
    ax1.plot(range(4, len(episode_rewards)), smoothed,
             color="#4f46e5", linewidth=2.5, label="5-ep moving avg")
ax1.axvline(x=20, color="gray", linestyle="--", alpha=0.5, label="→ medium")
ax1.axvline(x=40, color="gray", linestyle=":", alpha=0.5, label="→ hard")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Mean Reward")
ax1.set_title("Reward Progression Over Episodes")
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(alpha=0.3)

# Reward distribution
ax2 = axes[1]
rewards_array = [t["reward"] for t in all_trajectories]
ax2.hist(rewards_array, bins=30, color="#7c3aed", alpha=0.7, edgecolor="white")
ax2.axvline(np.mean(rewards_array), color="#ef4444", linewidth=2,
            label=f"Mean={np.mean(rewards_array):.3f}")
ax2.axvline(np.median(rewards_array), color="#f97316", linewidth=2,
            linestyle="--", label=f"Median={np.median(rewards_array):.3f}")
ax2.set_xlabel("Reward")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of Step Rewards")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("results.png saved.")
plt.show()

print("\n✅ All done. Submit results.png + datacentric-grpo-final/ directory.")
