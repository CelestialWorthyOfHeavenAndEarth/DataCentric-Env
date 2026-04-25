"""
GRPO training script — DataCentric-Env
Trains Qwen2.5-3B-Instruct with GRPO via TRL + Unsloth.

Run in Colab (GPU required). Make sure the environment server is deployed
to HuggingFace Spaces and set ENV_URL below before running.
"""

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import requests
import json
import torch

# ─── Configuration ───────────────────────────────────────────────────────────
ENV_URL = "https://aswini-kumar-datacentric-env.hf.space"  # HuggingFace Space URL

SYSTEM_PROMPT = """You are a data quality agent. You receive dataset statistics and must choose which specialist tool to call to improve the dataset so a downstream classifier performs better.

Always respond with valid JSON in this exact format:
{"agent": "<tool_name>", "target": "<column_or_all>", "strategy": "<strategy_name>"}

Available tools: cleaner, augmenter, balancer, relabeler, validator
Cleaner strategies: median_impute, mean_impute, drop_rows
Balancer strategies: undersample
Relabeler: use when labels are noisy, costs 2 budget points."""

# ─── Model setup ─────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)


# ─── Rollout function ─────────────────────────────────────────────────────────
def build_prompt(obs):
    obs_text = json.dumps(obs, indent=2)
    return f"{SYSTEM_PROMPT}\n\nCurrent state:\n{obs_text}\n\nYour action:"


def rollout(prompt="start"):
    """Run one episode and return (prompt, response, reward) tuples."""
    obs = requests.post(f"{ENV_URL}/reset").json()

    trajectories = []

    for step in range(10):
        full_prompt = build_prompt(obs)

        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse action
        try:
            action = json.loads(response.strip())
        except Exception:
            action = {"agent": "validator"}  # fallback

        result = requests.post(f"{ENV_URL}/step", json=action).json()
        reward = result.get("reward", -1.0)

        trajectories.append({
            "prompt": full_prompt,
            "response": response,
            "reward": reward,
        })

        obs = result.get("observation", obs)
        if result.get("done"):
            break

    return trajectories


# ─── Collect rollouts ─────────────────────────────────────────────────────────
print("Collecting rollouts...")
all_trajectories = []
for episode in range(50):
    all_trajectories.extend(rollout("start"))
    if episode % 10 == 0:
        print(f"  Episode {episode}/50 collected")

# ─── Build training dataset ───────────────────────────────────────────────────
dataset = Dataset.from_list([
    {"prompt": t["prompt"], "chosen": t["response"], "reward": t["reward"]}
    for t in all_trajectories
])

# ─── GRPO config ─────────────────────────────────────────────────────────────
config = GRPOConfig(
    output_dir="./datacentric-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    report_to="none",  # swap to "wandb" if you want live curves
)

# ─── Monitor logging ──────────────────────────────────────────────────────────
def log_sample(step):
    """Log a live episode sample every 20 steps — watch for reward hacking."""
    obs = requests.post(f"{ENV_URL}/reset").json()
    print(f"\n--- Generation sample at step {step} ---")
    for t in range(5):
        inputs = tokenizer(build_prompt(obs), return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  Step {t}: agent output = {response[:120]}")
        try:
            action = json.loads(response.strip())
        except Exception:
            print("  WARNING: agent produced invalid JSON — format reward not working")
            action = {"agent": "validator"}
        result = requests.post(f"{ENV_URL}/step", json=action).json()
        print(f"  Reward: {result.get('reward')} | Accuracy: {result['info']['new_accuracy']} | Done: {result.get('done')}")
        obs = result.get("observation", obs)
        if result.get("done"):
            break


# ─── Train ────────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ─── Save — use Unsloth merge path, NOT naive save_pretrained ────────────────
# IMPORTANT: do NOT upcast 4-bit model to 16-bit then merge naively.
# That damages model quality. Use the Unsloth merge path instead.
model.save_pretrained_merged(
    "datacentric-grpo-final",
    tokenizer,
    save_method="merged_16bit",  # correct merge path via Unsloth
)
print("Training complete. Test inference immediately before moving on.")
