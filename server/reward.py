def compute(prev_accuracy, new_accuracy, prev_stats, new_stats, action, steps_taken, max_steps, budget_remaining, target_accuracy, relabeler_used):
    """
    CRITICAL REQUIREMENT: All reward components must be graders strictly between
    0.0 and 1.0 — exclusive. Neither 0.0 nor 1.0 are valid outputs.
    Valid range: (0.001 ... 0.999)

    Each sub-grader scores one independent aspect and returns a value in (0.0, 1.0).
    Final reward is a weighted average of all graders — also in (0.0, 1.0).
    """

    def clamp(v):
        """Clamp to strictly open interval (0.0, 1.0)."""
        return max(0.001, min(0.999, float(v)))

    # --- Grader 1: Format compliance (independent) ---
    # Did the agent produce a valid, well-formed action?
    valid_agents = ["cleaner", "augmenter", "balancer", "relabeler", "validator"]
    if not isinstance(action.get("agent"), str) or action.get("agent") not in valid_agents:
        format_score = 0.001  # invalid agent — minimum non-zero
    elif "target" not in action:
        format_score = 0.4    # valid agent but incomplete fields
    else:
        format_score = 0.999  # fully valid action format

    # --- Grader 2: Accuracy improvement ---
    # How much did accuracy improve toward target?
    delta_acc = new_accuracy - prev_accuracy
    remaining = max(0.001, target_accuracy - prev_accuracy)
    progress = delta_acc / remaining if remaining > 0 else 0.0
    accuracy_score = clamp(0.5 + progress * 0.49)  # neutral at 0.5, better if improving

    # --- Grader 3: Dataset quality improvement ---
    # Combined missing value reduction + balance improvement
    missing_improvement = prev_stats["missing_pct"] - new_stats["missing_pct"]
    balance_improvement = new_stats["balance_ratio"] - prev_stats["balance_ratio"]
    quality_delta = (missing_improvement + balance_improvement) / 2.0
    quality_score = clamp(0.5 + quality_delta * 2.0)

    # --- Grader 4: Efficiency ---
    # Did the agent improve anything at all? Penalize wasted steps.
    nothing_changed = (delta_acc <= 0 and missing_improvement <= 0 and balance_improvement <= 0)
    relabeler_overused = relabeler_used and budget_remaining < 3
    if nothing_changed:
        efficiency_score = 0.1   # wasted a step
    elif relabeler_overused:
        efficiency_score = 0.3   # used expensive tool recklessly
    else:
        # Reward using budget efficiently — more budget left = better
        efficiency_score = clamp(0.5 + (budget_remaining / max_steps) * 0.49)

    # --- Grader 5: Task completion ---
    # Did this action help reach the target threshold?
    if new_accuracy >= target_accuracy:
        # Success — reward scales with how much budget is left (efficiency bonus)
        completion_score = clamp(0.9 + (budget_remaining / max_steps) * 0.09)
    elif new_accuracy > prev_accuracy:
        completion_score = clamp(0.5 + (new_accuracy / target_accuracy) * 0.4)
    else:
        completion_score = 0.1  # no progress toward target

    # --- Weighted average — stays in (0.0, 1.0) by construction ---
    reward = (
        format_score     * 0.15 +
        accuracy_score   * 0.35 +
        quality_score    * 0.20 +
        efficiency_score * 0.15 +
        completion_score * 0.15
    )

    # Final safety clamp — must never be exactly 0.0 or 1.0
    reward = clamp(reward)

    return round(reward, 4)


def compute_stats(df):
    missing_pct = float(df.isnull().mean().mean())
    label_counts = df["label"].value_counts(normalize=True)
    balance_ratio = float(label_counts.min()) if len(label_counts) > 1 else 1.0
    return {"missing_pct": missing_pct, "balance_ratio": balance_ratio}
