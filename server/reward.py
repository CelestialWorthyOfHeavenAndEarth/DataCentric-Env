"""
server/reward.py

Multi-component reward function supporting both query and apply steps.

CRITICAL REQUIREMENT: All reward values must be strictly between 0.0 and 1.0.
Neither 0.0 nor 1.0 are valid. Valid range: (0.001 ... 0.999).

Graders (all independent):
  1. Format compliance (15%)  — valid action type and required fields
  2. Accuracy improvement (35%) — progress toward target accuracy
  3. Dataset quality (20%)    — missing% reduction + balance improvement
  4. Efficiency (15%)         — penalize wasted steps and low-budget recklessness
  5. Task completion (15%)    — did accuracy reach or approach the target?
"""


def clamp(v: float) -> float:
    """Clamp to strictly open interval (0.001, 0.999)."""
    return max(0.001, min(0.999, float(v)))


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
    step_type: str = "apply",   # "query" or "apply"
    n_recs_returned: int = 0,   # for query steps
) -> float:
    """
    Compute reward. Returns float strictly in (0.001, 0.999).

    step_type="query"  → data did not change; reward reflects info quality
    step_type="apply"  → data changed; reward reflects accuracy + quality improvement
    """

    # ── Grader 1: Format compliance (independent of all other graders) ────────
    valid_query_actions = {
        "query_cleaner", "query_augmenter", "query_balancer",
        "query_validator", "query_analyst",
    }
    action_type = action.get("action", "")

    if action_type in valid_query_actions:
        format_score = 0.999  # valid query action
    elif action_type == "apply":
        if action.get("rec_id"):
            format_score = 0.999  # apply with rec_id
        else:
            format_score = 0.2    # apply missing rec_id
    else:
        format_score = 0.001  # completely invalid action

    # ── Grader 2: Accuracy improvement ────────────────────────────────────────
    delta_acc = new_accuracy - prev_accuracy
    remaining = max(0.001, target_accuracy - prev_accuracy)
    progress = delta_acc / remaining if remaining > 0 else 0.0

    if step_type == "query":
        # Query doesn't change data — neutral score
        # Slight bonus if there was useful info returned (n_recs > 0)
        info_bonus = 0.05 * min(n_recs_returned, 3)
        accuracy_score = clamp(0.45 + info_bonus)
    else:
        accuracy_score = clamp(0.5 + progress * 0.49)

    # ── Grader 3: Dataset quality improvement ────────────────────────────────
    missing_improvement = prev_stats["missing_pct"] - new_stats["missing_pct"]
    balance_improvement = new_stats["balance_ratio"] - prev_stats["balance_ratio"]
    quality_delta = (missing_improvement + balance_improvement) / 2.0

    if step_type == "query":
        quality_score = clamp(0.45)  # neutral for queries
    else:
        quality_score = clamp(0.5 + quality_delta * 2.0)

    # ── Grader 4: Efficiency ──────────────────────────────────────────────────
    nothing_changed = (
        delta_acc <= 0
        and missing_improvement <= 0
        and balance_improvement <= 0
    )
    low_budget = budget_remaining <= 2

    if step_type == "query" and low_budget:
        # Querying when almost out of budget is reckless
        efficiency_score = 0.15
    elif step_type == "apply" and nothing_changed:
        efficiency_score = 0.1   # applied a rec that did nothing
    elif step_type == "query" and not low_budget:
        efficiency_score = clamp(0.5 + (budget_remaining / max_steps) * 0.3)
    else:
        efficiency_score = clamp(0.5 + (budget_remaining / max_steps) * 0.49)

    # ── Grader 5: Task completion ─────────────────────────────────────────────
    if new_accuracy >= target_accuracy:
        completion_score = clamp(0.9 + (budget_remaining / max_steps) * 0.09)
    elif new_accuracy > prev_accuracy:
        completion_score = clamp(0.5 + (new_accuracy / target_accuracy) * 0.4)
    elif step_type == "query" and n_recs_returned > 0:
        completion_score = clamp(0.35)  # query produced useful info, small credit
    else:
        completion_score = 0.1

    # ── Weighted average ──────────────────────────────────────────────────────
    reward = (
        format_score     * 0.15 +
        accuracy_score   * 0.35 +
        quality_score    * 0.20 +
        efficiency_score * 0.15 +
        completion_score * 0.15
    )

    # Final safety clamp — must never be exactly 0.0 or 1.0
    return round(clamp(reward), 4)


def compute_stats(df) -> dict:
    """Compute dataset quality statistics."""
    if df is None or len(df) == 0:
        return {"missing_pct": 0.0, "balance_ratio": 0.0}
    missing_pct = float(df.isnull().mean().mean())
    label_counts = df["label"].value_counts(normalize=True)
    balance_ratio = float(label_counts.min()) if len(label_counts) > 1 else 1.0
    return {"missing_pct": missing_pct, "balance_ratio": balance_ratio}
