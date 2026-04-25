from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


class Evaluator:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200, random_state=42)

    def evaluate(self, df):
        """Returns accuracy score. Handles missing values by dropping rows."""
        clean = df.dropna()
        if len(clean) < 20:
            return 0.0
        X = clean.drop("label", axis=1).values
        y = clean["label"].values
        if len(set(y)) < 2:
            return 0.0
        scores = cross_val_score(self.model, X, y, cv=3, scoring="accuracy")
        return float(np.mean(scores))
