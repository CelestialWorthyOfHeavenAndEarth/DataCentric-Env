import pandas as pd


def run(df: pd.DataFrame, action: dict) -> dict:
    strategy = action.get("strategy", "undersample")
    df_out = df.copy().dropna()
    counts = df_out["label"].value_counts()
    if len(counts) < 2:
        return {"df": df, "log": "Balancer skipped — only one class present."}
    minority_count = counts.min()
    if strategy == "undersample":
        groups = [g.sample(minority_count, random_state=42) for _, g in df_out.groupby("label")]
        df_out = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
    return {"df": df_out, "log": f"Balancer resampled to {minority_count} per class."}
