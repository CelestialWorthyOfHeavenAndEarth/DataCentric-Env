"""
server/specialist_agents.py — Domain-aware expert systems (v0.4).

KEY CHANGE from v0.3:
  All agents now receive domain_metadata alongside the DataFrame.
  This enables genuine domain reasoning, not just generic statistics:

  - CleanerAgent knows that Glucose=0 means missing in medical data
  - CleanerAgent knows capital-gain should be log-transformed in census data
  - ValidatorAgent knows which business rules apply to which domain
  - AnalystAgent integrates domain context into its prioritized action plan

The LLM sees these domain-informed recommendations and must understand
WHY they make sense — it cannot just blindly apply everything.
"""

import numpy as np
import pandas as pd
from typing import Optional
import uuid


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "label" and not c.startswith("_")]


# ─────────────────────────────────────────────────────────────────────────────
# CleanerAgent
# ─────────────────────────────────────────────────────────────────────────────

class CleanerAgent:
    """
    Analyzes missing values and data quality issues.
    DOMAIN-AWARE: uses domain_metadata to detect real-world patterns
    like zero-as-missing in medical data and skew in financial data.
    """

    def query(self, df: pd.DataFrame, domain_metadata: dict = None) -> dict:
        recs = []
        meta = domain_metadata or {}
        rules = meta.get("domain_rules", {})
        zeros_as_missing = rules.get("zeros_as_missing", [])
        log_candidates = rules.get("log_transform_candidates", [])
        redundant = rules.get("redundant_features", [])
        f_cols = _feature_cols(df)

        # ── Domain-specific: zeros as missing ─────────────────────────────────
        for col in zeros_as_missing:
            if col not in df.columns:
                continue
            zero_count = int((df[col] == 0).sum())
            if zero_count > 0:
                recs.append({
                    "id": _make_id("clean"),
                    "type": "zero_to_nan_impute",
                    "column": col,
                    "strategy": "zero_to_nan_then_median",
                    "zero_count": zero_count,
                    "zero_pct": round(zero_count / len(df), 4),
                    "reason": (
                        f"'{col}' has {zero_count} zero values "
                        f"({zero_count/len(df):.1%}). "
                        f"In {meta.get('domain', 'this domain')}, "
                        f"zero is medically/logically impossible — these are missing values."
                    ),
                    "domain_informed": True,
                    "priority": 1,
                })

        # ── Standard: NaN-based missing values ────────────────────────────────
        missing = df[f_cols].isnull().sum()
        total_missing = int(missing.sum())
        for col in f_cols:
            miss_count = int(missing.get(col, 0))
            if miss_count == 0:
                continue
            miss_pct = miss_count / len(df)
            col_data = df[col].dropna()
            if len(col_data) < 5:
                strategy = "drop_rows"
                reason = f"'{col}': too few non-null values ({len(col_data)}) — drop rows."
            else:
                skewness = float(col_data.skew())
                if abs(skewness) > 1.0:
                    strategy = "median_impute"
                    reason = f"'{col}': skewness={skewness:.2f} (right-skewed) → median fill."
                else:
                    strategy = "mean_impute"
                    reason = f"'{col}': skewness={skewness:.2f} (symmetric) → mean fill."
            recs.append({
                "id": _make_id("clean"),
                "type": "impute",
                "column": col,
                "strategy": strategy,
                "missing_count": miss_count,
                "missing_pct": round(miss_pct, 4),
                "reason": reason,
                "domain_informed": False,
                "priority": len(recs) + 1,
            })

        # ── Domain-specific: log transform for skewed financial features ───────
        for col in log_candidates:
            if col not in df.columns:
                continue
            zero_pct = float((df[col] == 0).sum() / len(df))
            skew = float(df[col].dropna().skew()) if df[col].dropna().std() > 0 else 0
            if skew > 2.0 or zero_pct > 0.5:
                recs.append({
                    "id": _make_id("clean"),
                    "type": "log_transform",
                    "column": col,
                    "strategy": "log1p",
                    "skewness": round(skew, 2),
                    "zero_pct": round(zero_pct, 4),
                    "reason": (
                        f"'{col}' is {zero_pct:.0%} zero with skewness={skew:.2f}. "
                        f"Log1p transform will reduce skew and help the classifier."
                    ),
                    "domain_informed": True,
                    "priority": len(recs) + 1,
                })

        # ── Domain-specific: redundant features ───────────────────────────────
        for col in redundant:
            if col in df.columns:
                recs.append({
                    "id": _make_id("clean"),
                    "type": "drop_redundant",
                    "column": col,
                    "strategy": "drop_column",
                    "reason": (
                        f"'{col}' is redundant — it encodes the same information "
                        f"as another feature. Dropping reduces noise."
                    ),
                    "domain_informed": True,
                    "priority": len(recs) + 1,
                })

        # Sort by priority, keep top 4
        recs = recs[:4]
        for i, r in enumerate(recs):
            r["priority"] = i + 1

        domain_hint = ""
        if meta.get("display_name"):
            domain_hint = f" [{meta['display_name']}]"

        if not recs:
            diagnosis = f"No missing values or known domain issues detected.{domain_hint}"
        else:
            diagnosis = (
                f"Found {total_missing} missing cells + {len([r for r in recs if r.get('domain_informed')])} "
                f"domain-specific issues.{domain_hint} "
                f"Top recommendation: {recs[0]['strategy']} on '{recs[0]['column']}'."
            )

        return {
            "agent": "cleaner",
            "recommendations": recs,
            "diagnosis": diagnosis,
            "domain": meta.get("domain", "generic"),
            "summary": f"{len(recs)} recommendations ({sum(1 for r in recs if r.get('domain_informed'))} domain-informed).",
        }

    def apply(self, df: pd.DataFrame, rec: dict, domain_metadata: dict = None) -> tuple[pd.DataFrame, str]:
        col = rec.get("column", "all")
        strategy = rec.get("strategy", "")
        rec_type = rec.get("type", "")
        df_out = df.copy()

        if rec_type == "zero_to_nan_impute" or strategy == "zero_to_nan_then_median":
            cols = [col] if col != "all" else _feature_cols(df)
            fixed = 0
            for c in cols:
                if c not in df_out.columns:
                    continue
                zero_mask = df_out[c] == 0
                df_out.loc[zero_mask, c] = np.nan
                median_val = df_out[c].median()
                df_out.loc[zero_mask, c] = median_val
                fixed += int(zero_mask.sum())
            return df_out, f"Converted {fixed} medically-impossible zeros to NaN, then median-imputed."

        if rec_type == "log_transform" or strategy == "log1p":
            if col in df_out.columns:
                df_out[col] = np.log1p(df_out[col].clip(lower=0))
                return df_out, f"Applied log1p transform to '{col}' (was heavily right-skewed)."

        if rec_type == "drop_redundant" or strategy == "drop_column":
            if col in df_out.columns:
                df_out = df_out.drop(columns=[col])
                return df_out, f"Dropped redundant feature '{col}'."

        if strategy == "drop_rows":
            before = len(df_out)
            df_out = df_out.dropna()
            return df_out, f"Dropped {before - len(df_out)} rows with NaN values."

        cols = [col] if col != "all" else _feature_cols(df)
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
    Synthesizes new rows for underrepresented classes.
    Now respects holdout constraints — doesn't synthesize rows that would
    inflate cross-val scores but fail on real holdout.
    Uses SMOTE-like interpolation rather than pure Gaussian noise.
    """

    def query(self, df: pd.DataFrame, target_class: Optional[int] = None, domain_metadata: dict = None) -> dict:
        df_clean = df.dropna()
        if len(df_clean) < 10:
            return {"agent": "augmenter", "recommendations": [], "diagnosis": "Too few clean rows.", "summary": "0 recommendations."}

        counts = df_clean["label"].value_counts()
        if len(counts) < 2:
            return {"agent": "augmenter", "recommendations": [], "diagnosis": "Only one class present.", "summary": "0 recommendations."}

        minority_class = int(counts.idxmin())
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        ratio = minority_count / (minority_count + majority_count)

        aug_class = target_class if target_class is not None else minority_class
        aug_count = int(counts.get(aug_class, 0))
        target_count = int(majority_count * 0.85)  # aim for 85% of majority
        n_to_add = max(0, target_count - aug_count)

        if n_to_add == 0:
            return {
                "agent": "augmenter",
                "recommendations": [],
                "diagnosis": f"Class {aug_class} already well-represented (ratio={ratio:.2f}).",
                "summary": "No augmentation needed.",
            }

        # Warn if augmentation is very large (may hurt holdout)
        warning = ""
        if n_to_add > 200:
            warning = " WARNING: Large augmentation may reduce holdout accuracy — consider balancing instead."

        rec = {
            "id": _make_id("aug"),
            "type": "augment",
            "target_class": aug_class,
            "n_to_add": min(n_to_add, 300),  # cap at 300
            "method": "interpolation",
            "reason": (
                f"Class {aug_class} has {aug_count} rows (ratio={ratio:.2f}). "
                f"Adding {min(n_to_add, 300)} synthetic rows via interpolation between existing samples.{warning}"
            ),
            "priority": 1,
        }

        return {
            "agent": "augmenter",
            "recommendations": [rec],
            "diagnosis": f"Imbalance detected: minority={minority_class} ({minority_count} rows, ratio={ratio:.2f}).",
            "summary": "1 recommendation.",
        }

    def apply(self, df: pd.DataFrame, rec: dict, domain_metadata: dict = None) -> tuple[pd.DataFrame, str]:
        target_class = rec["target_class"]
        n_to_add = rec["n_to_add"]
        df_clean = df.dropna()
        class_rows = df_clean[df_clean["label"] == target_class]
        f_cols = _feature_cols(df)

        if len(class_rows) < 2:
            return df, f"Not enough rows for class {target_class} to synthesize from."

        # SMOTE-like: interpolate between pairs of existing samples
        synthetic = []
        stds = class_rows[f_cols].std().fillna(0.01)
        for _ in range(n_to_add):
            if len(class_rows) >= 2:
                pair = class_rows[f_cols].sample(2, random_state=None)
                alpha = np.random.random()
                new_row = pair.iloc[0] * alpha + pair.iloc[1] * (1 - alpha)
            else:
                base = class_rows[f_cols].sample(1).iloc[0]
                noise = np.random.normal(0, 0.05 * stds.values, size=len(f_cols))
                new_row = dict(zip(f_cols, base.values + noise))
            row_dict = dict(zip(f_cols, new_row.values if hasattr(new_row, 'values') else new_row))
            row_dict["label"] = target_class
            synthetic.append(row_dict)

        df_out = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
        return df_out, f"Added {n_to_add} synthetic rows for class {target_class} via SMOTE-like interpolation."


# ─────────────────────────────────────────────────────────────────────────────
# BalancerAgent
# ─────────────────────────────────────────────────────────────────────────────

class BalancerAgent:
    """Resamples to fix class imbalance. Now exposes rationale for each strategy."""

    def query(self, df: pd.DataFrame, domain_metadata: dict = None) -> dict:
        df_clean = df.dropna()
        counts = df_clean["label"].value_counts()
        if len(counts) < 2:
            return {"agent": "balancer", "recommendations": [], "diagnosis": "Only one class.", "summary": "0 recs."}

        minority_class = int(counts.idxmin())
        majority_class = int(counts.idxmax())
        minority_count = int(counts.min())
        majority_count = int(counts.max())
        ratio = minority_count / (minority_count + majority_count)

        if ratio >= 0.45:
            return {"agent": "balancer", "recommendations": [], "diagnosis": f"Classes balanced (ratio={ratio:.2f}).", "summary": "No action needed."}

        recs = [
            {
                "id": _make_id("bal"),
                "type": "balance",
                "strategy": "oversample_minority",
                "target_class": minority_class,
                "current_count": minority_count,
                "target_count": majority_count,
                "tradeoff": "Increases training size. May introduce near-duplicates. Better for small datasets.",
                "reason": f"Oversample class {minority_class}: {minority_count} → {majority_count} rows.",
                "priority": 1,
            },
            {
                "id": _make_id("bal"),
                "type": "balance",
                "strategy": "undersample_majority",
                "target_class": majority_class,
                "current_count": majority_count,
                "target_count": minority_count,
                "tradeoff": "Reduces training size. Loses majority-class information. Better when majority is very large.",
                "reason": f"Undersample class {majority_class}: {majority_count} → {minority_count} rows.",
                "priority": 2,
            },
        ]

        return {
            "agent": "balancer",
            "recommendations": recs,
            "diagnosis": f"Imbalance: ratio={ratio:.2f} (minority={minority_class}: {minority_count}, majority={majority_class}: {majority_count}).",
            "summary": "2 strategies: oversample minority (preserves data) OR undersample majority (faster, loses info).",
        }

    def apply(self, df: pd.DataFrame, rec: dict, domain_metadata: dict = None) -> tuple[pd.DataFrame, str]:
        strategy = rec["strategy"]
        target_class = rec["target_class"]
        df_clean = df.dropna()
        counts = df_clean["label"].value_counts()
        if len(counts) < 2:
            return df, "Only one class — cannot balance."

        if strategy == "undersample_majority":
            minority_count = counts.min()
            groups = [g.sample(int(minority_count), random_state=42) for _, g in df_clean.groupby("label")]
            df_out = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
            return df_out, f"Undersampled majority to {minority_count} rows/class."

        elif strategy == "oversample_minority":
            majority_count = int(counts.max())
            minority_rows = df_clean[df_clean["label"] == target_class]
            n_to_add = majority_count - len(minority_rows)
            if n_to_add <= 0:
                return df_clean, "Minority already at parity."
            extra = minority_rows.sample(n_to_add, replace=True, random_state=42)
            df_out = pd.concat([df_clean, extra]).sample(frac=1, random_state=42).reset_index(drop=True)
            return df_out, f"Oversampled class {target_class} by {n_to_add} rows."

        return df, f"Unknown strategy: {strategy}."


# ─────────────────────────────────────────────────────────────────────────────
# ValidatorAgent (costs 2 budget)
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Business rule validation. Now DOMAIN-AWARE:
    - In medical domains, warns against aggressive IQR outlier removal
      (outliers may represent rare but real conditions)
    - In financial domains, applies domain-specific value range checks
    """
    BUDGET_COST = 2

    def query(self, df: pd.DataFrame, domain_metadata: dict = None) -> dict:
        recs = []
        violations = []
        meta = domain_metadata or {}
        domain = meta.get("domain", "generic")
        f_cols = _feature_cols(df)

        # Check: duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            violations.append(f"duplicate_rows: {dup_count}")
            recs.append({
                "id": _make_id("val"),
                "type": "remove_duplicates",
                "fix": "drop_duplicates",
                "count": dup_count,
                "reason": f"{dup_count} exact duplicate rows — removing is always safe.",
                "priority": 1,
            })

        # Check: outliers (domain-aware IQR threshold)
        iqr_multiplier = 5.0 if "medical" in domain else 3.0
        warning_note = " (using conservative 5× IQR — medical outliers may be real)" if "medical" in domain else ""

        for col in f_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10 or col_data.dtype == object:
                continue
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            outlier_mask = (col_data < Q1 - iqr_multiplier * IQR) | (col_data > Q3 + iqr_multiplier * IQR)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                violations.append(f"outliers_{col}: {n_outliers}")
                recs.append({
                    "id": _make_id("val"),
                    "type": "clip_outliers",
                    "column": col,
                    "fix": "clip_iqr",
                    "lower": round(float(Q1 - iqr_multiplier * IQR), 4),
                    "upper": round(float(Q3 + iqr_multiplier * IQR), 4),
                    "count": n_outliers,
                    "reason": f"{n_outliers} extreme outliers in '{col}' ({iqr_multiplier}×IQR){warning_note}.",
                    "priority": len(recs) + 1,
                })

        recs = recs[:4]
        diagnosis = (
            f"No violations found." if not violations
            else f"{len(violations)} violation(s): {'; '.join(violations)}."
        )
        return {
            "agent": "validator",
            "recommendations": recs,
            "violations": violations,
            "diagnosis": diagnosis,
            "domain_note": f"IQR threshold: {iqr_multiplier}× (domain={domain})",
            "budget_cost": self.BUDGET_COST,
            "summary": f"{len(recs)} recs. Budget cost: {self.BUDGET_COST}.",
        }

    def apply(self, df: pd.DataFrame, rec: dict, domain_metadata: dict = None) -> tuple[pd.DataFrame, str]:
        fix_type = rec["type"]
        if fix_type == "remove_duplicates":
            before = len(df)
            df_out = df.drop_duplicates()
            return df_out, f"Removed {before - len(df_out)} duplicate rows."
        elif fix_type == "clip_outliers":
            col, lower, upper = rec["column"], rec["lower"], rec["upper"]
            df_out = df.copy()
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
            return df_out, f"Clipped '{col}' outliers to [{lower}, {upper}]."
        return df, f"Unknown fix type: {fix_type}."


# ─────────────────────────────────────────────────────────────────────────────
# AnalystAgent (costs 2 budget)
# ─────────────────────────────────────────────────────────────────────────────

class AnalystAgent:
    """
    Holistic meta-analysis. DOMAIN-AWARE:
    Integrates domain knowledge into its diagnosis and action plan.
    Mentions published baseline so the agent knows how far it needs to go.
    """
    BUDGET_COST = 2

    def query(self, df: pd.DataFrame, domain_metadata: dict = None) -> dict:
        meta = domain_metadata or {}
        f_cols = _feature_cols(df)

        missing_pct = float(df.isnull().mean().mean())
        missing_cols = int((df[f_cols].isnull().sum() > 0).sum())
        dup_count = int(df.duplicated().sum())
        counts = df["label"].value_counts(normalize=True)
        imbalance_ratio = float(counts.min()) if len(counts) > 1 else 1.0

        total_outliers = 0
        for col in f_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10 or col_data.dtype == object:
                continue
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                total_outliers += int(((col_data < Q1 - 3 * IQR) | (col_data > Q3 + 3 * IQR)).sum())

        # Domain-specific issues
        domain_issues = []
        rules = meta.get("domain_rules", {})
        zeros_as_missing = rules.get("zeros_as_missing", [])
        for col in zeros_as_missing:
            if col in df.columns:
                zero_count = int((df[col] == 0).sum())
                if zero_count > 0:
                    domain_issues.append(f"'{col}' has {zero_count} impossible zeros (domain: missing data)")

        redundant = rules.get("redundant_features", [])
        for col in redundant:
            if col in df.columns:
                domain_issues.append(f"'{col}' is a redundant feature (domain-informed)")

        log_candidates = rules.get("log_transform_candidates", [])
        for col in log_candidates:
            if col in df.columns:
                skew = float(df[col].dropna().skew()) if df[col].dropna().std() > 0 else 0
                if skew > 2:
                    domain_issues.append(f"'{col}' is heavily right-skewed (skew={skew:.1f}) — log transform recommended")

        # Build action plan
        action_plan = []
        step = 1

        if domain_issues:
            action_plan.append({
                "step": step, "action": "query_cleaner",
                "reason": f"Domain-specific issues found: {'; '.join(domain_issues[:2])}. Start here.",
            })
            step += 1

        if dup_count > 0:
            action_plan.append({
                "step": step, "action": "query_validator",
                "reason": f"{dup_count} duplicates — free accuracy gain, always remove first.",
            })
            step += 1

        if missing_pct > 0.05 and not domain_issues:
            action_plan.append({
                "step": step, "action": "query_cleaner",
                "reason": f"{missing_pct:.1%} missing values — impute before resampling.",
            })
            step += 1

        if imbalance_ratio < 0.35:
            action_plan.append({
                "step": step, "action": "query_augmenter",
                "reason": f"Severe imbalance (ratio={imbalance_ratio:.2f}) — synthesize minority rows.",
            })
            step += 1
        elif imbalance_ratio < 0.45:
            action_plan.append({
                "step": step, "action": "query_balancer",
                "reason": f"Imbalance (ratio={imbalance_ratio:.2f}) — resample.",
            })
            step += 1

        if not action_plan:
            action_plan.append({
                "step": 1, "action": "query_validator",
                "reason": "Dataset looks clean statistically. Check business rule violations.",
            })

        # Published baseline context
        baseline = meta.get("published_baseline")
        baseline_note = (
            f" Published benchmark: {baseline:.1%} accuracy on this dataset."
            if baseline else ""
        )

        return {
            "agent": "analyst",
            "domain": meta.get("display_name", "Unknown dataset"),
            "diagnosis": {
                "missing_pct": round(missing_pct, 4),
                "missing_columns": missing_cols,
                "duplicate_count": dup_count,
                "imbalance_ratio": round(imbalance_ratio, 4),
                "total_outliers": total_outliers,
                "domain_issues": domain_issues,
                "known_issues": meta.get("known_issues", []),
            },
            "action_plan": action_plan,
            "budget_cost": self.BUDGET_COST,
            "summary": (
                f"Dataset: {meta.get('display_name', 'unknown')}. "
                f"{len(domain_issues)} domain-specific issues + "
                f"{int(missing_pct > 0.05)} missing + "
                f"{int(imbalance_ratio < 0.45)} imbalance."
                f"{baseline_note}"
            ),
        }
