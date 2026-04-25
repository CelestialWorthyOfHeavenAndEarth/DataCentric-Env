"""
server/evaluator.py — Train/holdout evaluator (v0.5).

Trains on agent's modified data, tests on frozen holdout.
Also returns feature importance and regression explanations.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Evaluator:

    def __init__(self, holdout_df: pd.DataFrame):
        self.holdout_df = holdout_df
        self._feature_cols: list = None
        self._holdout_X: np.ndarray = None
        self._holdout_y: np.ndarray = None
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42, n_jobs=1)),
        ])
        self._last_feature_importance: dict = {}
        self._prepare_holdout()

    def _prepare_holdout(self):
        df = self.holdout_df.dropna()
        self._feature_cols = [
            c for c in df.columns
            if c != "label" and not c.startswith("_") and df[c].dtype != object
        ]
        if not self._feature_cols or len(df) < 5:
            self._holdout_X = None
            self._holdout_y = None
            return
        self._holdout_X = df[self._feature_cols].values.astype(float)
        self._holdout_y = df["label"].values

    def evaluate(self, train_df: pd.DataFrame) -> float:
        return self._run(train_df)["accuracy"]

    def evaluate_with_details(self, train_df: pd.DataFrame, prev_accuracy: float = None) -> dict:
        result = self._run(train_df)
        acc = result["accuracy"]
        explanation = None
        if prev_accuracy is not None and acc < prev_accuracy - 0.005:
            explanation = self._explain_regression(train_df, prev_accuracy, acc)
        return {
            "accuracy": acc,
            "feature_importance": result.get("feature_importance", {}),
            "regression_explanation": explanation,
        }

    def _run(self, train_df: pd.DataFrame) -> dict:
        if self._holdout_X is None or len(self._holdout_y) < 5:
            return {"accuracy": 0.0, "feature_importance": {}}

        train_clean = train_df.dropna()
        if len(train_clean) < 20:
            return {"accuracy": 0.0, "feature_importance": {}}

        available_cols = [
            c for c in self._feature_cols
            if c in train_clean.columns and train_clean[c].dtype != object
        ]
        if not available_cols:
            return {"accuracy": 0.0, "feature_importance": {}}

        X_train = train_clean[available_cols].values.astype(float)
        y_train = train_clean["label"].values
        if len(set(y_train)) < 2:
            return {"accuracy": 0.0, "feature_importance": {}}

        holdout_clean = self.holdout_df.dropna()
        available_holdout = [c for c in available_cols if c in holdout_clean.columns]
        if not available_holdout:
            return {"accuracy": 0.0, "feature_importance": {}}

        X_test = holdout_clean[available_holdout].values.astype(float)
        y_test = holdout_clean["label"].values
        if len(set(y_test)) < 2:
            return {"accuracy": 0.0, "feature_importance": {}}

        try:
            self._pipeline.fit(X_train, y_train)
            accuracy = float(self._pipeline.score(X_test, y_test))

            clf = self._pipeline.named_steps["clf"]
            coefs = clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_.mean(axis=0)
            importance = dict(zip(available_cols, [round(float(c), 4) for c in coefs]))
            sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            feature_importance = {
                "top_positive": [{"feature": k, "coef": v} for k, v in sorted_imp if v > 0][:4],
                "top_negative": [{"feature": k, "coef": v} for k, v in sorted_imp if v < 0][:4],
                "note": "Coefficients after StandardScaler — magnitude reflects relative importance.",
            }
            self._last_feature_importance = feature_importance
            return {"accuracy": accuracy, "feature_importance": feature_importance}
        except Exception:
            return {"accuracy": 0.0, "feature_importance": {}}

    def _explain_regression(self, train_df: pd.DataFrame, prev_acc: float, new_acc: float) -> dict:
        delta = round(new_acc - prev_acc, 4)
        n_rows = len(train_df.dropna())
        n_missing = int(train_df.isnull().sum().sum())
        label_counts = train_df["label"].value_counts()
        balance = float(label_counts.min() / label_counts.sum()) if len(label_counts) > 1 else 1.0

        likely_cause = "unknown"
        suggestion = "Try a different approach or rollback this step."

        if n_rows > 800:
            likely_cause = "large_augmentation_overfitting"
            suggestion = (
                "Large augmentation overfits the training set — synthetic rows don't generalise to holdout. "
                "Try 'query_balancer' with undersample_majority instead, or rollback."
            )
        elif n_missing / max(n_rows * train_df.shape[1], 1) > 0.15:
            likely_cause = "high_residual_missing"
            suggestion = "Many missing values remain. Apply 'query_cleaner' again on remaining columns."
        elif balance < 0.25:
            likely_cause = "worsened_class_imbalance"
            suggestion = (
                "Class imbalance got worse. The classifier is biased toward majority on holdout. "
                "Apply 'query_balancer' or 'query_augmenter'."
            )
        elif n_rows < 200:
            likely_cause = "too_few_training_rows"
            suggestion = "Row deletion left too few examples. Prefer imputation over drop_rows, or rollback."

        return {
            "accuracy_delta": delta,
            "likely_cause": likely_cause,
            "suggestion": suggestion,
            "training_stats": {
                "n_rows": n_rows,
                "n_missing_cells": n_missing,
                "class_balance": round(balance, 4),
            },
        }

    def baseline_accuracy(self) -> float:
        if self._holdout_y is None:
            return 0.0
        majority = np.bincount(self._holdout_y).max()
        return round(majority / len(self._holdout_y), 4)

    @property
    def last_feature_importance(self) -> dict:
        return self._last_feature_importance
