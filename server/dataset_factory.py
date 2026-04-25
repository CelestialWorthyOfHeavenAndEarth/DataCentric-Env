import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


class DatasetFactory:
    def generate(self, difficulty="easy"):
        """
        difficulty controls degradation severity:
          easy   — single issue, low severity
          medium — two issues, moderate severity
          hard   — three issues, high severity
        """
        X, y = make_classification(
            n_samples=500, n_features=10,
            n_informative=5, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["label"] = y

        params = self._difficulty_params(difficulty)
        df = self._inject_missing(df, params["missing_fraction"])
        df = self._inject_noise(df, params["noise_rate"])
        df = self._inject_imbalance(df, params["imbalance_ratio"])

        return df, params["target_accuracy"]

    def _difficulty_params(self, difficulty):
        return {
            "easy":   {"missing_fraction": 0.05, "noise_rate": 0.05, "imbalance_ratio": 0.8,  "target_accuracy": 0.80},
            "medium": {"missing_fraction": 0.15, "noise_rate": 0.15, "imbalance_ratio": 0.6,  "target_accuracy": 0.75},
            "hard":   {"missing_fraction": 0.30, "noise_rate": 0.25, "imbalance_ratio": 0.3,  "target_accuracy": 0.70},
        }[difficulty]

    def _inject_missing(self, df, fraction):
        mask = np.random.random(df.shape) < fraction
        df_copy = df.copy()
        for col in df.columns[:-1]:  # never corrupt label column
            df_copy.loc[mask[:, df.columns.get_loc(col)], col] = np.nan
        return df_copy

    def _inject_noise(self, df, rate):
        df_copy = df.copy()
        n_flip = int(len(df) * rate)
        idx = np.random.choice(len(df), n_flip, replace=False)
        df_copy.loc[idx, "label"] = 1 - df_copy.loc[idx, "label"]
        return df_copy

    def _inject_imbalance(self, df, ratio):
        minority = df[df["label"] == 1]
        majority = df[df["label"] == 0]
        minority_sample = minority.sample(frac=ratio, random_state=42)
        return pd.concat([majority, minority_sample]).sample(frac=1).reset_index(drop=True)
