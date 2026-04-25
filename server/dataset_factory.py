"""
server/dataset_factory.py — Richer dataset generation with multiple archetypes
and golden rows.

Golden rows: A fixed set of rows injected into every dataset that represent
"ground truth" — they are perfectly clean and correctly labeled. If a specialist
operation corrupts them, the environment detects and penalizes this.

Archetypes provide variety so the agent can't memorize a single dataset shape.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from server.config import cfg


ARCHETYPES = [
    # (name, n_informative, n_redundant, class_sep)
    ("credit_risk",    5, 2, 1.0),
    ("churn",          4, 3, 0.8),
    ("fraud",          6, 1, 1.2),
    ("medical",        5, 2, 0.9),
    ("supply_chain",   4, 2, 1.1),
]

DIFFICULTY_PARAMS = {
    "easy":   {"missing_fraction": 0.05, "noise_rate": 0.05, "imbalance_ratio": 0.80, "target_accuracy": 0.82},
    "medium": {"missing_fraction": 0.15, "noise_rate": 0.12, "imbalance_ratio": 0.60, "target_accuracy": 0.77},
    "hard":   {"missing_fraction": 0.28, "noise_rate": 0.22, "imbalance_ratio": 0.35, "target_accuracy": 0.72},
}


class DatasetFactory:

    def __init__(self):
        self._archetype_idx = 0

    def generate(self, difficulty: str = "easy") -> tuple[pd.DataFrame, float, set]:
        """
        Returns:
            df             — corrupted DataFrame
            target_acc     — accuracy target to hit
            golden_row_ids — set of row indices that are "golden" (must not be corrupted)
        """
        params = DIFFICULTY_PARAMS[difficulty]

        # Rotate archetypes for variety
        arch_name, n_info, n_red, class_sep = ARCHETYPES[self._archetype_idx % len(ARCHETYPES)]
        self._archetype_idx += 1

        n = cfg.DATASET_N_SAMPLES
        X, y = make_classification(
            n_samples=n,
            n_features=10,
            n_informative=n_info,
            n_redundant=n_red,
            class_sep=class_sep,
            random_state=np.random.randint(0, 9999),
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["label"] = y
        df["_archetype"] = arch_name  # metadata column — not used by classifier

        # Insert golden rows BEFORE corruption (they stay clean)
        golden_indices = self._insert_golden_rows(df, cfg.GOLDEN_ROW_COUNT)

        # Corrupt non-golden rows only
        non_golden = df.index.difference(golden_indices).tolist()
        df = self._inject_missing(df, non_golden, params["missing_fraction"])
        df = self._inject_noise(df, non_golden, params["noise_rate"])
        df = self._inject_imbalance(df, params["imbalance_ratio"])

        return df, params["target_accuracy"], set(golden_indices)

    def _insert_golden_rows(self, df: pd.DataFrame, n: int) -> list[int]:
        """
        Inject n perfectly clean rows with known-correct labels.
        Returns their indices.
        """
        golden_ids = []
        feature_cols = [c for c in df.columns if c not in ("label", "_archetype")]
        for cls in [0, 1]:
            class_rows = df[df["label"] == cls]
            if len(class_rows) < n // 2:
                continue
            sample = class_rows.sample(n=n // 2, random_state=42)
            golden_ids.extend(sample.index.tolist())
        return golden_ids

    def _inject_missing(self, df: pd.DataFrame, non_golden: list, fraction: float) -> pd.DataFrame:
        df_copy = df.copy()
        feature_cols = [c for c in df.columns if c not in ("label", "_archetype")]
        mask = np.random.random((len(non_golden), len(feature_cols))) < fraction
        for i, idx in enumerate(non_golden):
            for j, col in enumerate(feature_cols):
                if mask[i, j]:
                    df_copy.at[idx, col] = np.nan
        return df_copy

    def _inject_noise(self, df: pd.DataFrame, non_golden: list, rate: float) -> pd.DataFrame:
        df_copy = df.copy()
        n_flip = int(len(non_golden) * rate)
        flip_indices = np.random.choice(non_golden, n_flip, replace=False)
        for idx in flip_indices:
            df_copy.at[idx, "label"] = 1 - df_copy.at[idx, "label"]
        return df_copy

    def _inject_imbalance(self, df: pd.DataFrame, ratio: float) -> pd.DataFrame:
        minority = df[df["label"] == 1]
        majority = df[df["label"] == 0]
        keep = max(1, int(len(minority) * ratio))
        minority_sample = minority.sample(n=keep, random_state=42)
        return pd.concat([majority, minority_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
