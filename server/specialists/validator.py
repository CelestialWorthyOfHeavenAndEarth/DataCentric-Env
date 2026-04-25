import pandas as pd


def run(df: pd.DataFrame, action: dict) -> dict:
    issues = []
    missing_pct = df.isnull().mean().mean()
    if missing_pct > 0.05:
        issues.append(f"High missing rate: {missing_pct:.1%}")
    if df["label"].value_counts(normalize=True).min() < 0.2:
        issues.append("Class imbalance detected")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        issues.append(f"Removed {duplicates} duplicate rows")
    return {
        "df": df,
        "log": f"Validator report: {'; '.join(issues) if issues else 'No major issues found.'}"
    }
