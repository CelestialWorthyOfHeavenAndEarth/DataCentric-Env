"""
server/reward.py — Multi-component reward function (v0.3).

CRITICAL: All reward values must be strictly in (0.001, 0.999).

Graders:
  1. Format compliance (15%)  — valid action + required fields
  2. Accuracy improvement (35%) — progress toward target
  3. Dataset quality (20%)    — missing% + balance improvement
  4. Efficiency (15%)         — penalize wasted steps, low-budget queries
  5. Task completion (10%)    — did accuracy reach target?
  6. Golden row integrity (5%) — penalize corrupting canonical rows
"""


def clamp(v: float) -> float:
    return max(0.001, min(0.999, float(v)))


VALID_QUERY_ACTIONS = {
    "query_cleaner", "query_augmenter", "query_balancer",
    "query_validator", "query_analyst",
}


def compute(
    prev_accuracy: float,
    new_accuracy: float,
    prev_stats: dict,
    new_stats: dict,
    action: dict,
    steps_taken: int,
    max_steps: int,
    budget_remaining: int,
    target_accuracy: float,
    step_type: str = "apply",
    n_recs_returned: int = 0,
    golden_penalty: float = 0.0,
) -> float:
    """Returns float strictly in (0.001, 0.999)."""

    # ── Grader 1: Format compliance (INDEPENDENT of all other graders) ────────
    action_type = action.get("action", "")
    if action_type in VALID_QUERY_ACTIONS:
        format_score = 0.999
    elif action_type == "apply" and action.get("rec_id"):
        format_score = 0.999
    elif action_type == "apply":
        format_score = 0.2    # missing rec_id
    else:
        format_score = 0.001  # completely invalid

    # ── Grader 2: Accuracy improvement ────────────────────────────────────────
    delta_acc = new_accuracy - prev_accuracy
    remaining = max(0.001, target_accuracy - prev_accuracy)
    progress = delta_acc / remaining if remaining > 0 else 0.0

    if step_type == "query":
        info_bonus = 0.05 * min(n_recs_returned, 3)
        accuracy_score = clamp(0.45 + info_bonus)
    else:
        accuracy_score = clamp(0.5 + progress * 0.49)

    # ── Grader 3: Dataset quality improvement ────────────────────────────────
    missing_improvement = prev_stats["missing_pct"] - new_stats["missing_pct"]
    balance_improvement = new_stats["balance_ratio"] - prev_stats["balance_ratio"]
    quality_delta = (missing_improvement + balance_improvement) / 2.0

    if step_type == "query":
        quality_score = clamp(0.45)
    else:
        quality_score = clamp(0.5 + quality_delta * 2.0)

    # ── Grader 4: Efficiency ──────────────────────────────────────────────────
    nothing_changed = (
        delta_acc <= 0 and missing_improvement <= 0 and balance_improvement <= 0
    )
    low_budget = budget_remaining <= 2

    if step_type == "query" and low_budget:
        efficiency_score = 0.15
    elif step_type == "apply" and nothing_changed:
        efficiency_score = 0.1
    elif step_type == "query":
        efficiency_score = clamp(0.5 + (budget_remaining / max_steps) * 0.3)
    else:
        efficiency_score = clamp(0.5 + (budget_remaining / max_steps) * 0.49)

    # ── Grader 5: Task completion ─────────────────────────────────────────────
    if new_accuracy >= target_accuracy:
        completion_score = clamp(0.9 + (budget_remaining / max_steps) * 0.09)
    elif new_accuracy > prev_accuracy:
        completion_score = clamp(0.5 + (new_accuracy / target_accuracy) * 0.4)
    elif step_type == "query" and n_recs_returned > 0:
        completion_score = clamp(0.35)
    else:
        completion_score = 0.1

    # ── Grader 6: Golden row integrity ────────────────────────────────────────
    # golden_penalty is in [0.0, 1.0] where 0 = no corruption, 1 = all golden rows lost
    integrity_score = clamp(1.0 - golden_penalty * 0.9)  # maps [0,1] → [0.1, 0.999]

    # ── Weighted average ──────────────────────────────────────────────────────
    reward = (
        format_score     * 0.15 +
        accuracy_score   * 0.35 +
        quality_score    * 0.20 +
        efficiency_score * 0.15 +
        completion_score * 0.10 +
        integrity_score  * 0.05
    )

    return round(clamp(reward), 4)


def compute_stats(df) -> dict:
    if df is None or len(df) == 0:
        return {"missing_pct": 0.0, "balance_ratio": 0.0}
    feature_cols = [c for c in df.columns if c not in ("_archetype",)]
    missing_pct = float(df[feature_cols].isnull().mean().mean())
    label_counts = df["label"].value_counts(normalize=True)
    balance_ratio = float(label_counts.min()) if len(label_counts) > 1 else 1.0
    return {"missing_pct": missing_pct, "balance_ratio": balance_ratio}
