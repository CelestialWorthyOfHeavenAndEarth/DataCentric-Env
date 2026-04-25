from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np


def run(df: pd.DataFrame, action: dict) -> dict:
    df_clean = df.dropna()
    if len(df_clean) < 20 or len(set(df_clean["label"])) < 2:
        return {"df": df, "log": "Augmenter skipped — insufficient clean data."}
    X = df_clean.drop("label", axis=1).values
    y = df_clean["label"].values
    try:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        df_out = pd.DataFrame(X_res, columns=df.columns[:-1])
        df_out["label"] = y_res
        added = len(df_out) - len(df_clean)
        return {"df": df_out, "log": f"Augmenter added {added} synthetic samples."}
    except Exception as e:
        return {"df": df, "log": f"Augmenter failed: {str(e)}"}
