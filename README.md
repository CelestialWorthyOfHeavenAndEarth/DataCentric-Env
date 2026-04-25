# DataCentric-Env

**An RL environment that trains an LLM to act as a data engineer.**

The agent receives a real, messy tabular dataset and a frozen classifier it cannot touch. Its only job: fix the data until the classifier hits the accuracy target — measured against published academic benchmarks.

---

## The Problem This Solves

Most RL environments for LLMs test reasoning on synthetic puzzles. Real data engineering requires **domain reasoning** — knowing that `Glucose=0` is medically impossible, that `capital-gain` needs a log transform, that removing 30% of rows will hurt generalization even if it improves cross-validation accuracy.

This environment forces the agent to develop that domain knowledge by grounding rewards in **published accuracy benchmarks** on real UCI datasets.

---

## Live Demo

**Environment server:** https://huggingface.co/spaces/Aswini-Kumar/datacentric-env

- `GET /docs` — Interactive Swagger UI
- `GET /health` — Status + active sessions
- `POST /reset` — Start a new episode

---

## The 5 Real Datasets

| Dataset | Domain | Published Baseline | Key Issues |
|---|---|---|---|
| [UCI Adult Census](https://archive.ics.uci.edu/dataset/2/adult) | Income prediction | **87.1%** | 14% `?` missing, capital-gain 97% zero, education/education-num redundant |
| [Pima Indians Diabetes](https://www.openml.org/d/37) | Medical diagnosis | **77.0%** | Glucose=0, BloodPressure=0, BMI=0 are medically impossible (zeros = missing) |
| [Wisconsin Breast Cancer](https://scikit-learn.org/stable/datasets/toy_dataset.html) | Medical imaging | **97.3%** | Correlated feature groups, outliers represent real rare tumors |
| [German Credit Risk](https://www.openml.org/d/31) | Credit risk | **76.8%** | Mixed categorical + numeric, 70/30 imbalance |
| [Cleveland Heart Disease](https://www.openml.org/d/1497) | Medical diagnosis | **85.5%** | 303 rows, real missing values in `ca` and `thal` |

Datasets download automatically on first run and are cached locally. The server pre-loads all 5 at startup via a background thread.

---

## Architecture

```
POST /reset  →  Load real dataset  →  80/20 train/holdout split
               Agent sees train set (domain + known issues)
               Holdout is FROZEN — agent never sees or modifies it

POST /step   →  Query a specialist agent
               Agent reads recommendations (domain-informed)
               Agent applies the best recommendation

               Score = accuracy on FROZEN holdout
               Compared against published benchmark
```

### 5 Specialist Agents

| Agent | Action | What it does |
|---|---|---|
| **CleanerAgent** | `query_cleaner` | Missing values + zero-as-missing (domain-aware) + log-transform for skewed features |
| **AugmenterAgent** | `query_augmenter` | SMOTE-like interpolation to synthesize minority class rows |
| **BalancerAgent** | `query_balancer` | Oversample/undersample with explicit tradeoff explanation |
| **ValidatorAgent** | `query_validator` (cost 2) | Duplicates + outlier clipping (conservative 5× IQR for medical domains) |
| **AnalystAgent** | `query_analyst` (cost 2) | Holistic diagnosis + prioritized action plan + published baseline reference |

### What's Domain-Aware

The CleanerAgent knows:
- In `medical_diagnosis` datasets: zeros in physiological measurements are impossible — they're missing values → `zero_to_nan_impute`
- In `income_prediction` datasets: `capital-gain` has 97% zeros with heavy right skew → `log1p` transform
- Redundant features (e.g. `education` + `education-num`) → recommend dropping one

The ValidatorAgent knows:
- In medical domains, use 5× IQR instead of 3× — outliers may be real rare conditions
- In credit/income domains, use standard 3× IQR

---

## Reward Structure

All rewards strictly in `(0.001, 0.999)`. Every `/step` returns a full decomposition:

| Grader | Weight | What it measures |
|---|---|---|
| Format | 15% | Valid action with required fields |
| Accuracy | 35% | Progress toward target on **frozen holdout** |
| Quality | 20% | Missing% reduction + class balance improvement |
| Efficiency | 15% | Penalizes wasted steps and low-budget expensive queries |
| Completion | 15% | Bonus for hitting target, scaled by remaining budget |

---

## New in v0.5

### Rollback Action
```json
{"action": "rollback", "session_id": "..."}
```
Undoes the last apply. Max 3 per episode. Costs 1 budget. Real data engineers do this.

### Episode Reasoning Trace
Every observation includes the last 5 steps with effects:
```json
"episode_trace": [
  {"step": 2, "type": "apply", "accuracy_delta": 0.031, "effect": "improved"},
  {"step": 3, "type": "apply", "accuracy_delta": -0.018, "effect": "hurt"}
]
```

### Feature Importance
Returned after every apply — LogisticRegression coefficients after StandardScaler:
```json
"feature_importance": {
  "top_positive": [{"feature": "Glucose", "coef": 0.84}],
  "top_negative": [{"feature": "BMI_raw", "coef": -0.32}]
}
```

### Regression Explanation
When accuracy drops after an apply:
```json
"regression_explanation": {
  "likely_cause": "large_augmentation_overfitting",
  "suggestion": "Synthetic rows don't generalise to holdout. Try undersample_majority or rollback."
}
```

### Benchmark Comparison
```json
"benchmarks": {
  "majority_class_baseline": 0.6510,
  "starting_accuracy": 0.8095,
  "improvement_over_start": 0.0231,
  "published_baseline": 0.8710
}
```

---

## API Reference

```
POST /reset                         Start a new episode
  body: {difficulty: "easy"|"medium"|"hard", seed?: int}

POST /step                          Take an action
  body: {session_id, action, rec_id?, target_class?}
  actions: query_cleaner | query_augmenter | query_balancer |
           query_validator | query_analyst | apply | rollback

GET  /state/{session_id}            Current observation
GET  /trajectory/{session_id}       Full episode trace (for offline analysis)
GET  /health                        Health check
GET  /metrics                       Server metrics + config
GET  /docs                          Swagger UI
```

---

## Training

The training script (`training/train.py`) runs GRPO via TRL + Unsloth on Colab (T4 GPU).

```python
# Set your HF Space URL
ENV_URL = "https://aswini-kumar-datacentric-env.hf.space"

# Then run training/train.py
# - Collects 60 episodes across easy/medium/hard difficulty
# - Trains Qwen2.5-3B-Instruct with LoRA r=16
# - Saves results.png with reward progression + distribution charts
# - Saves merged model to ./datacentric-grpo-final
```

---

## Anti-Exploit Rules

| Rule | What it blocks |
|---|---|
| `action_spam` | Same query 3+ times in a row |
| `low_budget_expensive_query` | Cost-2 queries when budget ≤ 2 |
| `duplicate_apply` | Applying the same rec_id twice |
| `invalid_rec_id` | Applying a rec_id that doesn't exist |
| `data_integrity_violation` | Deleting >10% of training rows in one operation |

---

## Project Structure

```
datacentric-env/
├── server/
│   ├── main.py               # FastAPI app (endpoints + startup warmup)
│   ├── environment.py        # Session-aware RL environment (v0.5)
│   ├── dataset_registry.py   # Real dataset loader + CSV cache + warmup
│   ├── evaluator.py          # Train/holdout split evaluator + feature importance
│   ├── specialist_agents.py  # 5 domain-aware expert systems
│   ├── reward.py             # 5-component reward function
│   ├── session_manager.py    # Thread-safe UUID session management
│   ├── anti_exploit.py       # 5 anti-exploit rules
│   ├── config.py             # Centralized configuration
│   └── logger.py             # Structured JSON logging
├── datasets/                 # Cached real datasets (CSV, git-ignored)
├── training/
│   └── train.py              # GRPO training script (Colab)
├── inference.py              # Automated end-to-end test
├── openenv.yaml              # Full environment spec
├── requirements.txt
└── Dockerfile
```
