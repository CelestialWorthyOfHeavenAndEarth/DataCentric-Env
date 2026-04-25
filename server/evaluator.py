from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


class Evaluator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200, random_state=42)

    def evaluate(self, df):
        """Returns accuracy score. Drops NaN rows and non-numeric columns."""
        clean = df.dropna()
        if len(clean) < 20:
            return 0.0
        # Drop non-numeric columns (e.g. _archetype metadata)
        feature_cols = [c for c in clean.columns if c != "label" and clean[c].dtype != object]
        X = clean[feature_cols].values
        y = clean["label"].values
        if len(set(y)) < 2:
            return 0.0
        scores = cross_val_score(self.model, X, y, cv=3, scoring="accuracy")
        return float(np.mean(scores))
