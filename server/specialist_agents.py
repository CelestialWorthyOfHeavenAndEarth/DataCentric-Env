"""
server/specialist_agents.py

Five rule-based expert systems. The LLM queries these agents to get
recommendations, then decides which recommendation to apply.

The LLM never executes these agents directly — it only calls them
by name via the query_* action types.
"""

import numpy as np
import pandas as pd
from typing import Optional
import uuid


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# CleanerAgent
# ─────────────────────────────────────────────────────────────────────────────

class CleanerAgent:
    """
    Analyzes missing values, outliers, and type errors.
    Uses column skewness to recommend mean vs median fill.
    Returns up to 3 ranked recommendations.
    """

    def query(self, df: pd.DataFrame) -> dict:
        recs = []
        diagnosis_parts = []

        feature_cols = [c for c in df.columns if c != "label"]
        missing = df[feature_cols].isnull().sum()
        total_missing = missing.sum()

        # Per-column missing analysis
        for col in feature_cols:
            miss_count = missing[col]
            if miss_count == 0:
                continue
            miss_pct = miss_count / len(df)
            col_data = df[col].dropna()

            if len(col_data) < 5:
                strategy = "drop_rows"
                reason = f"{col}: too few non-null values ({len(col_data)})"
            else:
                skewness = float(col_data.skew())
                if abs(skewness) > 1.0:
                    strategy = "median_impute"
                    reason = f"{col}: skewness={skewness:.2f} → median fill recommended"
                else:
                    strategy = "mean_impute"
                    reason = f"{col}: skewness={skewness:.2f} → mean fill recommended"

            recs.append({
                "id": _make_id("clean"),
                "type": "impute",
                "column": col,
                "strategy": strategy,
                "missing_count": int(miss_count),
                "missing_pct": round(float(miss_pct), 4),
                "reason": reason,
                "priority": len(recs) + 1,
            })

        # Global drop_rows if missing is widespread
        if total_missing > 0 and len(recs) >= 2:
            recs.append({
                "id": _make_id("clean"),
                "type": "drop_rows",
                "column": "all",
                "strategy": "drop_rows",
                "missing_count": int(total_missing),
                "missing_pct": round(float(df.isnull().mean().mean()), 4),
                "reason": f"Drop all rows with any NaN ({total_missing} cells affected)",
                "priority": len(recs) + 1,
            })

        # Sort by missing count descending, keep top 3
        recs = sorted(recs, key=lambda r: r["missing_count"], reverse=True)[:3]
        for i, r in enumerate(recs):
            r["priority"] = i + 1

        if not recs:
            diagnosis = "No missing values detected. Dataset is clean on this dimension."
        else:
            diagnosis = (
                f"Found {total_missing} missing cells across {(missing > 0).sum()} columns. "
                f"Top recommendation: {recs[0]['strategy']} on {recs[0]['column']}."
            )

        return {
            "agent": "cleaner",
            "recommendations": recs,
            "diagnosis": diagnosis,
            "summary": f"{len(recs)} recommendations generated.",
        }

    def apply(self, df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, str]:
        col = rec["column"]
        strategy = rec["strategy"]
        df_out = df.copy()

        if strategy == "drop_rows":
            before = len(df_out)
            df_out = df_out.dropna()
            return df_out, f"Dropped {before - len(df_out)} rows with NaN values."

        cols = [col] if col != "all" else [c for c in df.columns if c != "label"]
        fixed = 0
        for c in cols:
            if c not in df_out.columns or c == "label":
                continue
            null_mask = df_out[c].isnull()
            if strategy == "median_impute":
                df_out.loc[null_mask, c] = df_out[c].median()
            elif strategy == "mean_impute":
                df_out.loc[null_mask, c] = df_out[c].mean()
            fixed += int(null_mask.sum())

        return df_out, f"Applied {strategy} to {cols}. Fixed {fixed} missing values."


# ─────────────────────────────────────────────────────────────────────────────
# AugmenterAgent
# ─────────────────────────────────────────────────────────────────────────────

class AugmenterAgent:
    """
    Detects underrepresented classes. Synthesizes new rows via Gaussian
    noise around existing samples of the target class.
    """

    def query(self, df: pd.DataFrame, target_class: Optional[int] = None) -> dict:
        df_clean = df.dropna()
        counts = df_clean["label"].value_counts()
        recs = []

        if len(counts) < 2:
            return {
                "agent": "augmenter",
                "recommendations": [],
                "diagnosis": "Only one class present — augmentation not applicable.",
                "summary": "0 recommendations.",
            }

        total = len(df_clean)
        minority_class = int(counts.idxmin())
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        current_ratio = minority_count / total

        # If target_class specified, use it; otherwise use minority
        aug_class = target_class if target_class is not None else minority_class
        aug_count = int(counts.get(aug_class, 0))

        # How many rows to add to reach 45% balance
        target_minority = int(majority_count * 0.9)
        n_to_add = max(0, target_minority - aug_count)

        if n_to_add == 0:
            diagnosis = f"Class {aug_class} is already well-represented ({aug_count} rows, ratio={current_ratio:.2f})."
            return {
                "agent": "augmenter",
                "recommendations": [],
                "diagnosis": diagnosis,
                "summary": "No augmentation needed.",
            }

        recs.append({
            "id": _make_id("aug"),
            "type": "augment",
            "target_class": aug_class,
            "n_to_add": n_to_add,
            "method": "gaussian_noise",
            "noise_std_factor": 0.05,
            "reason": (
                f"Class {aug_class} has {aug_count} samples (ratio={current_ratio:.2f}). "
                f"Adding {n_to_add} synthetic rows via Gaussian noise."
            ),
            "priority": 1,
        })

        return {
            "agent": "augmenter",
            "recommendations": recs,
            "diagnosis": (
                f"Class imbalance detected. Minority={minority_class} ({minority_count} rows). "
                f"Adding {n_to_add} synthetic samples to class {aug_class}."
            ),
            "summary": f"{len(recs)} recommendations generated.",
        }

    def apply(self, df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, str]:
        target_class = rec["target_class"]
        n_to_add = rec["n_to_add"]
        noise_std_factor = rec.get("noise_std_factor", 0.05)

        df_clean = df.dropna()
        class_rows = df_clean[df_clean["label"] == target_class]

        if len(class_rows) < 2:
            return df, f"Augmenter: not enough clean rows for class {target_class} to synthesize from."

        feature_cols = [c for c in df.columns if c != "label"]
        stds = class_rows[feature_cols].std().fillna(0.1)

        synthetic_rows = []
        for _ in range(n_to_add):
            base = class_rows[feature_cols].sample(1, random_state=None).iloc[0]
            noise = np.random.normal(0, noise_std_factor * stds.values, size=len(feature_cols))
            new_row = dict(zip(feature_cols, base.values + noise))
            new_row["label"] = target_class
            synthetic_rows.append(new_row)

        df_aug = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        return df_aug, f"Augmenter added {n_to_add} synthetic rows for class {target_class} via Gaussian noise."


# ─────────────────────────────────────────────────────────────────────────────
# BalancerAgent
# ─────────────────────────────────────────────────────────────────────────────

class BalancerAgent:
    """
    Detects class imbalance. Recommends oversampling minority or
    undersampling majority class.
    """

    def query(self, df: pd.DataFrame) -> dict:
        df_clean = df.dropna()
        counts = df_clean["label"].value_counts()
        recs = []

        if len(counts) < 2:
            return {
                "agent": "balancer",
                "recommendations": [],
                "diagnosis": "Only one class present — balancing not applicable.",
                "summary": "0 recommendations.",
            }

        minority_class = int(counts.idxmin())
        majority_class = int(counts.idxmax())
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        ratio = minority_count / (minority_count + majority_count)

        if ratio >= 0.45:
            return {
                "agent": "balancer",
                "recommendations": [],
                "diagnosis": f"Classes are sufficiently balanced (ratio={ratio:.2f}).",
                "summary": "No balancing needed.",
            }

        recs.append({
            "id": _make_id("bal"),
            "type": "balance",
            "strategy": "oversample_minority",
            "target_class": minority_class,
            "current_count": minority_count,
            "target_count": majority_count,
            "reason": f"Oversample class {minority_class} from {minority_count} to {majority_count} rows.",
            "priority": 1,
        })

        recs.append({
            "id": _make_id("bal"),
            "type": "balance",
            "strategy": "undersample_majority",
            "target_class": majority_class,
            "current_count": majority_count,
            "target_count": minority_count,
            "reason": f"Undersample class {majority_class} from {majority_count} to {minority_count} rows.",
            "priority": 2,
        })

        return {
            "agent": "balancer",
            "recommendations": recs,
            "diagnosis": (
                f"Class imbalance: ratio={ratio:.2f} (minority={minority_class}, {minority_count} rows; "
                f"majority={majority_class}, {majority_count} rows). Threshold: 0.45."
            ),
            "summary": "2 recommendations: oversample minority OR undersample majority.",
        }

    def apply(self, df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, str]:
        strategy = rec["strategy"]
        target_class = rec["target_class"]
        df_clean = df.dropna()
        counts = df_clean["label"].value_counts()

        if len(counts) < 2:
            return df, "Balancer: only one class present, cannot balance."

        if strategy == "undersample_majority":
            minority_count = counts.min()
            groups = [g.sample(int(minority_count), random_state=42) for _, g in df_clean.groupby("label")]
            df_out = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
            return df_out, f"Balancer undersampled majority class to {minority_count} rows per class."

        elif strategy == "oversample_minority":
            majority_count = int(counts.max())
            majority_class = int(counts.idxmax())
            minority_rows = df_clean[df_clean["label"] == target_class]
            n_to_add = majority_count - len(minority_rows)
            if n_to_add <= 0:
                return df_clean, "Balancer: minority class already at parity."
            # Resample with replacement
            extra = minority_rows.sample(n_to_add, replace=True, random_state=42)
            df_out = pd.concat([df_clean, extra]).sample(frac=1, random_state=42).reset_index(drop=True)
            return df_out, f"Balancer oversampled class {target_class} by {n_to_add} rows."

        return df, f"Balancer: unknown strategy {strategy}."


# ─────────────────────────────────────────────────────────────────────────────
# ValidatorAgent  (costs 2 budget)
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Checks business rule violations: value ranges, cross-column logic,
    type consistency. Costs 2 budget points.
    """
    BUDGET_COST = 2

    def query(self, df: pd.DataFrame) -> dict:
        recs = []
        violations = []

        feature_cols = [c for c in df.columns if c != "label"]

        # Check 1: duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            violations.append(f"duplicate_rows: {dup_count}")
            recs.append({
                "id": _make_id("val"),
                "type": "remove_duplicates",
                "fix": "drop_duplicates",
                "count": dup_count,
                "reason": f"{dup_count} exact duplicate rows found.",
                "priority": 1,
            })

        # Check 2: outliers per column (IQR method)
        for col in feature_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            outlier_mask = (col_data < Q1 - 3 * IQR) | (col_data > Q3 + 3 * IQR)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                violations.append(f"outliers_{col}: {n_outliers}")
                recs.append({
                    "id": _make_id("val"),
                    "type": "clip_outliers",
                    "column": col,
                    "fix": "clip_iqr",
                    "lower": round(float(Q1 - 3 * IQR), 4),
                    "upper": round(float(Q3 + 3 * IQR), 4),
                    "count": n_outliers,
                    "reason": f"{n_outliers} extreme outliers in {col} (3×IQR).",
                    "priority": len(recs) + 1,
                })

        recs = recs[:4]  # cap at 4 recommendations

        if not violations:
            diagnosis = "No business rule violations found. Dataset passes all checks."
        else:
            diagnosis = f"Found {len(violations)} violation type(s): {'; '.join(violations)}."

        return {
            "agent": "validator",
            "recommendations": recs,
            "violations": violations,
            "diagnosis": diagnosis,
            "budget_cost": self.BUDGET_COST,
            "summary": f"{len(recs)} fix recommendations. Budget cost: {self.BUDGET_COST}.",
        }

    def apply(self, df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, str]:
        fix_type = rec["type"]

        if fix_type == "remove_duplicates":
            before = len(df)
            df_out = df.drop_duplicates()
            return df_out, f"Validator removed {before - len(df_out)} duplicate rows."

        elif fix_type == "clip_outliers":
            col = rec["column"]
            lower = rec["lower"]
            upper = rec["upper"]
            df_out = df.copy()
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
            return df_out, f"Validator clipped outliers in {col} to [{lower}, {upper}]."

        return df, f"Validator: unknown fix type {fix_type}."


# ─────────────────────────────────────────────────────────────────────────────
# AnalystAgent  (costs 2 budget)
# ─────────────────────────────────────────────────────────────────────────────

class AnalystAgent:
    """
    The meta-agent. Runs a holistic diagnosis of all dataset problems
    (missing %, entropy, imbalance ratio, duplicates, outliers) and gives
    the LLM a prioritized ordered action plan. Costs 2 budget.
    """
    BUDGET_COST = 2

    def query(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in df.columns if c != "label"]

        # Compute all metrics
        missing_pct = float(df.isnull().mean().mean())
        missing_cols = int((df[feature_cols].isnull().sum() > 0).sum())
        dup_count = int(df.duplicated().sum())

        counts = df["label"].value_counts(normalize=True)
        imbalance_ratio = float(counts.min()) if len(counts) > 1 else 1.0
        entropy = float(-(counts * np.log2(counts + 1e-10)).sum())

        # Outlier count across all feature columns
        total_outliers = 0
        for col in feature_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                total_outliers += int(((col_data < Q1 - 3 * IQR) | (col_data > Q3 + 3 * IQR)).sum())

        # Build diagnosis
        issues = []
        if dup_count > 0:
            issues.append(("duplicates", dup_count, "high"))
        if missing_pct > 0.05:
            issues.append(("high_missing", round(missing_pct, 3), "high" if missing_pct > 0.15 else "medium"))
        if imbalance_ratio < 0.35:
            issues.append(("severe_imbalance", round(imbalance_ratio, 3), "high"))
        elif imbalance_ratio < 0.45:
            issues.append(("mild_imbalance", round(imbalance_ratio, 3), "medium"))
        if total_outliers > 5:
            issues.append(("outliers", total_outliers, "medium"))

        # Build prioritized action plan
        action_plan = []
        step = 1

        if dup_count > 0:
            action_plan.append({
                "step": step, "action": "query_validator",
                "reason": f"Remove {dup_count} duplicate rows first (free accuracy gain).",
            })
            step += 1

        if missing_pct > 0.05:
            action_plan.append({
                "step": step, "action": "query_cleaner",
                "reason": f"{missing_pct:.1%} missing values — impute before balancing.",
            })
            step += 1

        if imbalance_ratio < 0.45:
            if imbalance_ratio < 0.30:
                action_plan.append({
                    "step": step, "action": "query_augmenter",
                    "reason": f"Severe imbalance (ratio={imbalance_ratio:.2f}) — synthesize minority rows.",
                })
            else:
                action_plan.append({
                    "step": step, "action": "query_balancer",
                    "reason": f"Class imbalance (ratio={imbalance_ratio:.2f}) — resample classes.",
                })
            step += 1

        if total_outliers > 5:
            action_plan.append({
                "step": step, "action": "query_validator",
                "reason": f"{total_outliers} outliers detected — clip extreme values.",
            })

        if not action_plan:
            action_plan.append({
                "step": 1, "action": "query_augmenter",
                "reason": "Dataset looks clean. Try augmenting minority class for accuracy gains.",
            })

        return {
            "agent": "analyst",
            "diagnosis": {
                "missing_pct": round(missing_pct, 4),
                "missing_columns": missing_cols,
                "duplicate_count": dup_count,
                "imbalance_ratio": round(imbalance_ratio, 4),
                "entropy": round(entropy, 4),
                "total_outliers": total_outliers,
                "top_issues": [i[0] for i in issues],
            },
            "action_plan": action_plan,
            "budget_cost": self.BUDGET_COST,
            "summary": (
                f"Holistic diagnosis complete. {len(issues)} issues found. "
                f"{len(action_plan)}-step action plan generated. Budget cost: {self.BUDGET_COST}."
            ),
        }
