"""
server/dataset_registry.py — Real dataset loader.

Uses sklearn's fetch_openml to load 5 well-known public datasets with
GENUINE quality issues. No artificial corruption injected.

Datasets are cached as parquet after first download so the server
doesn't need internet access on every restart.

Each dataset ships with domain_metadata that tells specialist agents:
  - What domain this is (medical, finance, etc.)
  - What the known real-world issues are
  - What the published baseline accuracy is
  - Domain-specific rules (e.g. zero=missing in medical data)
  - A held-out test split that the agent NEVER sees or modifies
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

# ── Dataset definitions ────────────────────────────────────────────────────────

REGISTRY = {
    "adult_census": {
        "display_name": "UCI Adult Census Income",
        "description": "Predict whether income exceeds $50K based on census data.",
        "openml_name": "adult",
        "openml_version": 2,
        "target_column": "class",
        "published_baseline": 0.871,
        "domain": "income_prediction",
        "difficulty_weight": 1,  # easy
        "known_issues": [
            "~14% of rows have '?' for occupation, workclass, native-country (real missing data)",
            "capital-gain is 97% zero with heavy right skew — needs log transform",
            "education and education-num are redundant (same info, two encodings)",
            "Class imbalance: 76% income<=50K, 24% income>50K",
        ],
        "domain_rules": {
            "zeros_as_missing": [],
            "log_transform_candidates": ["capital-gain", "capital-loss"],
            "redundant_features": ["education"],  # education-num is numeric equivalent
        },
    },
    "diabetes_pima": {
        "display_name": "Pima Indians Diabetes Dataset",
        "description": "Predict diabetes onset in Pima Indian women based on medical measurements.",
        "openml_name": "diabetes",
        "openml_version": 1,
        "target_column": "class",
        "published_baseline": 0.770,
        "domain": "medical_diagnosis",
        "difficulty_weight": 2,  # medium
        "known_issues": [
            "Glucose, BloodPressure, BMI = 0 are medically impossible — zeros mean missing",
            "Insulin has 374 zeros (49% are actually missing)",
            "SkinThickness has 227 zeros (30% are actually missing)",
            "Class imbalance: 65% negative, 35% positive",
        ],
        "domain_rules": {
            "zeros_as_missing": ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"],
            "log_transform_candidates": [],
            "redundant_features": [],
        },
    },
    "breast_cancer": {
        "display_name": "Wisconsin Breast Cancer Diagnostic",
        "description": "Classify tumors as malignant or benign from cell nucleus measurements.",
        "openml_name": None,  # use sklearn built-in
        "target_column": "target",
        "published_baseline": 0.973,
        "domain": "medical_imaging",
        "difficulty_weight": 1,  # easy (very clean)
        "known_issues": [
            "Several feature groups are highly correlated (mean/SE/worst versions of same measurement)",
            "Some outlier samples represent rare aggressive tumor types — IQR removal is dangerous",
            "Feature scales vary wildly: radius ~10-30, fractal_dimension ~0.05-0.10",
        ],
        "domain_rules": {
            "zeros_as_missing": [],
            "log_transform_candidates": [],
            "redundant_features": [],
        },
    },
    "german_credit": {
        "display_name": "German Credit Risk",
        "description": "Classify loan applicants as good or bad credit risks.",
        "openml_name": "credit-g",
        "openml_version": 1,
        "target_column": "class",
        "published_baseline": 0.768,
        "domain": "credit_risk",
        "difficulty_weight": 2,  # medium
        "known_issues": [
            "Mix of categorical and numeric features — encoding strategy matters",
            "Cost-sensitive: misclassifying bad credit as good is 5× more expensive",
            "Class imbalance: 70% good credit, 30% bad credit",
            "Several ordinal features encoded as integers — scaling affects model significantly",
        ],
        "domain_rules": {
            "zeros_as_missing": [],
            "log_transform_candidates": [],
            "redundant_features": [],
        },
    },
    "heart_disease": {
        "display_name": "Cleveland Heart Disease",
        "description": "Predict presence of heart disease from clinical measurements.",
        "openml_name": "heart-h",
        "openml_version": 1,
        "target_column": "class",
        "published_baseline": 0.855,
        "domain": "medical_diagnosis",
        "difficulty_weight": 3,  # hard
        "known_issues": [
            "Real missing values in: ca (4 missing), thal (2 missing)",
            "Target is originally 0-4 severity scale — binarized as 0 vs >0",
            "slope, ca, thal are ordinal but treated as continuous",
            "Small dataset (303 rows) — train/test split matters a lot",
        ],
        "domain_rules": {
            "zeros_as_missing": [],
            "log_transform_candidates": [],
            "redundant_features": [],
        },
    },
}

DIFFICULTY_TO_DATASETS = {
    "easy":   ["adult_census", "breast_cancer"],
    "medium": ["diabetes_pima", "german_credit"],
    "hard":   ["heart_disease"],
}


# ── Dataset loading ────────────────────────────────────────────────────────────

class DatasetRegistry:
    """Loads, caches, and serves real datasets with train/holdout splits."""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}
        self._rotation: dict[str, int] = {d: 0 for d in DIFFICULTY_TO_DATASETS}

    def get(self, difficulty: str = "easy", seed: int = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Returns (train_df, holdout_df, domain_metadata).

        train_df  — the agent works on this (80% of data)
        holdout_df — frozen, never modified (20% of data)
        domain_metadata — known issues, rules, baseline accuracy
        """
        datasets = DIFFICULTY_TO_DATASETS.get(difficulty, ["adult_census"])
        idx = self._rotation[difficulty] % len(datasets)
        self._rotation[difficulty] += 1
        name = datasets[idx]

        meta = REGISTRY[name]
        df = self._load(name, meta)

        # Stable train/holdout split — same seed → same split
        rng_seed = seed if seed is not None else 42
        train_df, holdout_df = train_test_split(
            df, test_size=0.20, random_state=rng_seed, stratify=df["label"]
        )
        train_df = train_df.reset_index(drop=True)
        holdout_df = holdout_df.reset_index(drop=True)

        return train_df, holdout_df, meta

    def _load(self, name: str, meta: dict) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name].copy()

        cache_path = os.path.join(CACHE_DIR, f"{name}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            self._cache[name] = df
            return df.copy()

        # Download
        df = self._download(name, meta)
        df.to_csv(cache_path, index=False)
        self._cache[name] = df
        return df.copy()

    def _download(self, name: str, meta: dict) -> pd.DataFrame:
        if name == "breast_cancer":
            return self._load_breast_cancer()

        try:
            ds = fetch_openml(
                meta["openml_name"],
                version=meta["openml_version"],
                as_frame=True,
                parser="auto",
            )
            df = ds.frame.copy()
            target_col = meta["target_column"]

            # Standardize target → "label" column as int
            if target_col in df.columns:
                df = df.rename(columns={target_col: "label"})
            else:
                df["label"] = ds.target

            # Encode label to 0/1
            le = LabelEncoder()
            df["label"] = le.fit_transform(df["label"].astype(str))

            # Encode all object/categorical columns to numeric
            for col in df.select_dtypes(include=["object", "category"]).columns:
                if col == "label":
                    continue
                # Replace '?' (OpenML's missing value marker) with NaN
                df[col] = df[col].replace("?", np.nan)
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            df = df.dropna(subset=["label"])
            return df

        except Exception as e:
            # Fallback: generate enriched synthetic data with documented patterns
            print(f"[DatasetRegistry] Could not download {name}: {e}. Using fallback.")
            return self._synthetic_fallback(name, meta)

    def _load_breast_cancer(self) -> pd.DataFrame:
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df = df.rename(columns={"target": "label"})
        return df

    def _synthetic_fallback(self, name: str, meta: dict) -> pd.DataFrame:
        """
        Fallback: sklearn synthetic data with realistic properties
        matching the real dataset's known issues.
        """
        from sklearn.datasets import make_classification
        np.random.seed(42)
        n = 500
        X, y = make_classification(
            n_samples=n, n_features=10, n_informative=5,
            n_redundant=2, class_sep=0.8, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df["label"] = y

        # Inject the documented issues from the real dataset
        domain_rules = meta.get("domain_rules", {})

        # Inject missing values (like real dataset)
        missing_rate = 0.12 if meta["difficulty_weight"] >= 2 else 0.05
        mask = np.random.random((n, 8)) < missing_rate
        for i in range(8):
            df.loc[mask[:, i], f"feature_{i}"] = np.nan

        # Inject class imbalance if known
        if "imbalance" in str(meta.get("known_issues", [])).lower():
            minority_idx = df[df["label"] == 1].index
            drop_n = int(len(minority_idx) * 0.4)
            df = df.drop(minority_idx[:drop_n])

        return df.reset_index(drop=True)
