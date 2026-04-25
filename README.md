---
title: DataCentric-Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DataCentric-Env

An RL environment for training LLM agents to improve dataset quality.
Instead of changing the model — improve the data.

## The Problem

AI labs spend enormous resources on data annotation and cleaning.
DataCentric-Env trains an agent to do this automatically — dispatching
specialist tools to fix a degraded dataset until a fixed classifier's
accuracy crosses a target threshold.

## How It Works

- **Observation**: The agent receives dataset statistics (missing value rate, class balance ratio, current accuracy, budget remaining).
- **Action**: The agent selects one of five specialist tools and parameters (JSON format).
- **Reward**: A 5-component reward function scores format compliance, accuracy improvement, dataset quality improvement, efficiency, and task completion.
- **Done**: Episode ends when accuracy reaches the target threshold OR budget is exhausted.

## Specialist Tools

| Tool | What it does |
|------|-------------|
| cleaner | Imputes missing values (median/mean) or drops incomplete rows |
| augmenter | Generates synthetic minority-class samples via SMOTE |
| balancer | Resamples to fix class skew via undersampling |
| relabeler | Queries label oracle to correct mislabeled rows (costs 2 budget points) |
| validator | Detects and removes duplicates, reports dataset health |

## Reward Function

Five independent graders, weighted average — all strictly in `(0.001, 0.999)`:

| Grader | Weight | What it measures |
|--------|--------|-----------------|
| Format compliance | 15% | Valid JSON action with correct agent name and fields |
| Accuracy improvement | 35% | Progress toward target accuracy threshold |
| Dataset quality | 20% | Missing value reduction + balance improvement |
| Efficiency | 15% | Budget use — penalizes wasted steps and reckless relabeler |
| Task completion | 15% | Whether target threshold was reached |

## Curriculum

| Episodes | Difficulty | Missing | Noise | Imbalance | Target Acc |
|----------|-----------|---------|-------|-----------|-----------|
| 0–20     | easy      | 5%      | 5%    | 0.8       | 0.80      |
| 20–50    | medium    | 15%     | 15%   | 0.6       | 0.75      |
| 50+      | hard      | 30%     | 25%   | 0.3       | 0.70      |

## Results

![Reward curve](results.png)

Trained agent achieves higher mean episode reward vs random baseline.

## Environment

Live on HuggingFace: [link — set after deployment]

API:
- `POST /reset` — Start new episode, returns initial observation
- `POST /step` — Take action, returns `{observation, reward, done, tool_log, info}`
- `GET /state` — Get current observation without advancing episode

## Training

- **Model**: Qwen2.5-3B-Instruct
- **Method**: GRPO via TRL + Unsloth
- **Training script**: [training/train.py](training/train.py)
- **Colab notebook**: [training/train.ipynb](training/train.ipynb)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server locally
uvicorn server.main:app --reload --port 8000

# Test client
python client/client.py

# Run automated checks
python inference.py

# Evaluate baseline
python evaluate.py
```

## File Structure

```
datacentric-env/
├── openenv.yaml               # OpenEnv manifest
├── requirements.txt
├── Dockerfile                 # HuggingFace Spaces deployment
├── inference.py               # Phase 1/2 automated check entry point
├── evaluate.py                # Baseline vs trained agent comparison
├── server/
│   ├── main.py                # FastAPI app
│   ├── environment.py         # Core environment logic
│   ├── dataset_factory.py     # Dataset generation + corruption
│   ├── evaluator.py           # sklearn classifier
│   ├── reward.py              # 5-component reward function
│   └── specialists/
│       ├── cleaner.py
│       ├── augmenter.py
│       ├── balancer.py
│       ├── relabeler.py
│       └── validator.py
├── client/
│   └── client.py              # HTTP-only client (never imports server/)
└── training/
    ├── train.py               # GRPO training script
    └── train.ipynb            # Colab notebook (required by judges)
```

## References

- OpenEnv: https://github.com/meta-pytorch/OpenEnv
- TRL GRPO: https://huggingface.co/docs/trl
- Unsloth: https://github.com/unslothai/unsloth
- Data-Centric AI: https://datacentricai.org
