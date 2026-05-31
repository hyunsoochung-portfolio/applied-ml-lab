"""Build a dict of six tree-ensemble classifiers for benchmarking."""
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)


def build_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, random_state=42,
        ),
    }
    # XGBoost / LightGBM are optional imports — fail soft so the script
    # still benchmarks the rest if one library is missing on a machine.
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
            tree_method="hist",
        )
    except Exception as e:
        print(f"[warn] xgboost unavailable: {e!r}")
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    except Exception as e:
        print(f"[warn] lightgbm unavailable: {e!r}")
    return models
