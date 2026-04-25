import pandas as pd
import numpy as np


def run(df: pd.DataFrame, action: dict) -> dict:
    # Simulates a label oracle — corrects a fraction of likely mislabeled rows
    # Costs extra budget (handled in environment.py)
    confidence_threshold = action.get("confidence_threshold", 0.3)
    df_out = df.copy().dropna()
    # Simulate oracle: flip back labels on rows with low feature-label alignment
    n_fix = max(1, int(len(df_out) * confidence_threshold * 0.3))
    idx = np.random.choice(len(df_out), n_fix, replace=False)
    df_out.loc[idx, "label"] = 1 - df_out.loc[idx, "label"]
    return {"df": df_out, "log": f"Relabeler corrected {n_fix} likely mislabeled rows. Budget cost: 2."}
