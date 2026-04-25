import pandas as pd


def run(df: pd.DataFrame, action: dict) -> dict:
    target_col = action.get("target", "all")
    strategy = action.get("strategy", "median_impute")
    df_out = df.copy()

    cols = [target_col] if target_col != "all" else df.columns[:-1].tolist()

    for col in cols:
        if col not in df_out.columns or col == "label":
            continue
        if strategy == "median_impute":
            df_out[col] = df_out[col].fillna(df_out[col].median())
        elif strategy == "mean_impute":
            df_out[col] = df_out[col].fillna(df_out[col].mean())
        elif strategy == "drop_rows":
            df_out = df_out.dropna(subset=[col])

    reduced = df.isnull().sum().sum() - df_out.isnull().sum().sum()
    return {
        "df": df_out,
        "log": f"Cleaner applied {strategy} to {cols}. Missing reduced by {reduced} cells."
    }
